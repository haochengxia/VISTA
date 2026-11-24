/* vqonline.h: the deployable online selector (vq.py's OnlineVQ / OnlineV1) in C, keyed by
 * 64-bit object ids, for use inside other simulators (e.g. libCacheSim) or a live cache.
 *
 * The live cache starts as S3-FIFO. From the first request on, every candidate design runs in a
 * miniature simulation on a spatial sample of the keys (mix64(key) <= rate * 2^64; the rate is
 * raised so that a miniature cache holds at least min_mini objects). Requests are grouped into
 * windows of `period`; misses count from window `from_window`. At the start of window `start`
 * and of every window after it, the live cache switches to the candidate with the fewest sampled
 * misses so far (ties: the earlier candidate), unless the current design is within `margin`
 * (relative) of it. A switch is a takeover (vqcore.h: adopt for the first, then switch_design).
 *
 * Unlike vqsim, object records are reused: a table maps each key to a record while the object is
 * in the cache or the ghost, so memory is bounded by the cache and ghost sizes, not by the number
 * of objects ever seen. Usage:
 *
 *     VQOnline e;
 *     if (vqo_init(&e, configs, K, capacity, &VQO_V1)) ...;   // configs: vqsim config strings
 *     int miss = vqo_access(&e, key);                          // one request; 1 on a miss
 *     vqo_free(&e);
 */
#ifndef VQONLINE_H
#define VQONLINE_H
#include <stdint.h>

typedef struct VQTable VQTable;
static void vqt_release(VQTable *t, uint32_t slot);
#define VQ_RELEASE(s, o, y) vqt_release((VQTable *)(s)->owner, (y))
#include "vqcore.h"

/* -- a table of object records: key -> slot (linear probing, backward-shift deletion) ---------- */
struct VQTable {
    uint64_t *keys; uint32_t *vals; uint64_t mask; /* the map; vals[i] == UINT32_MAX: empty */
    Obj *o; uint64_t *key_of; uint32_t *free; uint32_t n, cap, nfree; /* records and free slots */
    uint64_t peak; /* most records in use at once */
};

static inline uint64_t vqt_hash(uint64_t key) { return mix64(key ^ 0x632BE59BD9B4E019ull); }

static void vqt_init(VQTable *t, uint64_t expect) {
    uint64_t size = 64; while (size < 2 * expect + 2) size <<= 1;
    t->mask = size - 1; t->keys = malloc(size * sizeof(uint64_t)); t->vals = malloc(size * sizeof(uint32_t));
    for (uint64_t i = 0; i < size; i++) t->vals[i] = UINT32_MAX;
    t->cap = (uint32_t)(expect + 16); t->n = 0; t->nfree = 0; t->peak = 0;
    t->o = malloc((uint64_t)t->cap * sizeof(Obj)); t->key_of = malloc((uint64_t)t->cap * sizeof(uint64_t)); t->free = malloc((uint64_t)t->cap * sizeof(uint32_t));
}

static void vqt_free(VQTable *t) { free(t->keys); free(t->vals); free(t->o); free(t->key_of); free(t->free); }

static void vqt_rehash(VQTable *t, uint64_t size) {
    uint64_t *keys = t->keys; uint32_t *vals = t->vals; uint64_t old = t->mask + 1;
    t->mask = size - 1; t->keys = malloc(size * sizeof(uint64_t)); t->vals = malloc(size * sizeof(uint32_t));
    for (uint64_t i = 0; i < size; i++) t->vals[i] = UINT32_MAX;
    for (uint64_t i = 0; i < old; i++) {
        if (vals[i] == UINT32_MAX) continue;
        uint64_t j = vqt_hash(keys[i]) & t->mask;
        while (t->vals[j] != UINT32_MAX) j = (j + 1) & t->mask;
        t->keys[j] = keys[i]; t->vals[j] = vals[i];
    }
    free(keys); free(vals);
}

/* The record of `key`, created (not in the cache, not in the ghost) if it has none. Records may
 * move when the table grows: re-read t->o after this call. */
static uint32_t vqt_slot(VQTable *t, uint64_t key) {
    uint64_t i = vqt_hash(key) & t->mask;
    for (; t->vals[i] != UINT32_MAX; i = (i + 1) & t->mask) if (t->keys[i] == key) return t->vals[i];
    uint32_t slot;
    if (t->nfree) slot = t->free[--t->nfree];
    else {
        if (t->n == t->cap) {
            t->cap = t->cap * 2;
            t->o = realloc(t->o, (uint64_t)t->cap * sizeof(Obj)); t->key_of = realloc(t->key_of, (uint64_t)t->cap * sizeof(uint64_t));
            t->free = realloc(t->free, (uint64_t)t->cap * sizeof(uint32_t));
        }
        slot = t->n++;
    }
    t->o[slot] = (Obj){0}; t->o[slot].stage = -1; t->key_of[slot] = key;
    t->keys[i] = key; t->vals[i] = slot;
    uint64_t live = (uint64_t)t->n - t->nfree;
    if (live > t->peak) t->peak = live;
    if (2 * live > t->mask + 1) vqt_rehash(t, 2 * (t->mask + 1));
    return slot;
}

static void vqt_release(VQTable *t, uint32_t slot) {
    uint64_t i = vqt_hash(t->key_of[slot]) & t->mask;
    while (t->vals[i] != slot) i = (i + 1) & t->mask;
    t->vals[i] = UINT32_MAX;
    for (uint64_t j = (i + 1) & t->mask; t->vals[j] != UINT32_MAX; j = (j + 1) & t->mask) { /* backward shift */
        uint64_t home = vqt_hash(t->keys[j]) & t->mask;
        if (((j - home) & t->mask) >= ((j - i) & t->mask)) { t->keys[i] = t->keys[j]; t->vals[i] = t->vals[j]; t->vals[j] = UINT32_MAX; i = j; }
    }
    t->free[t->nfree++] = slot;
}

static inline uint64_t vqt_live(const VQTable *t) { return (uint64_t)t->n - t->nfree; }

/* -- the selector ---------------------------------------------------------------------------- */
typedef struct { uint64_t period, start, from_window, min_mini; double rate, margin; } VQOParams;
static const VQOParams VQO_V1 = {500000, 4, 2, 200, 0.005, 0.0}; /* OnlineV1 (sandbox/vq.py) */

typedef struct {
    VQOParams p; int K; Config *cand, s3; uint64_t capacity, mcap, limit; int sample_all;
    VQTable lt; Sim live; int current; int64_t window; uint64_t now, switches;
    VQTable *mt; Sim *ms; uint64_t *misses;
} VQOnline;

/* 0 on success, -1 for bad parameters (capacity and period must be >= 1), else 1 + the index of the
 * first bad candidate config. */
static int vqo_init(VQOnline *e, const char *const *configs, int K, uint64_t capacity, const VQOParams *p) {
    memset(e, 0, sizeof *e);
    e->p = *p; e->K = K; e->capacity = capacity; e->current = -1; e->window = -1;
    if (K < 1 || p->period < 1 || capacity < 1) return -1;
    e->cand = calloc(K, sizeof(Config));
    for (int i = 0; i < K; i++) if (parse(configs[i], &e->cand[i]) || e->cand[i].gate >= 0) { free(e->cand); e->cand = NULL; return 1 + i; } /* gate: keyed by slot here, not supported */
    parse(S3_DEFAULT, &e->s3);
    double r = p->rate;
    if (r * (double)capacity < (double)p->min_mini) r = (double)p->min_mini / (double)capacity;
    if (r > 1) r = 1;
    e->sample_all = r >= 1.0; e->limit = e->sample_all ? UINT64_MAX : (uint64_t)(r * 18446744073709551615.0);
    e->mcap = (uint64_t)(r * (double)capacity + 0.5); if (e->mcap < 1) e->mcap = 1;
    double ghost = e->s3.ghost;
    for (int i = 0; i < K; i++) if (e->cand[i].ghost > ghost) ghost = e->cand[i].ghost;
    vqt_init(&e->lt, (uint64_t)((double)capacity * (1.0 + ghost)) + 1);
    sim_init(&e->live, &e->s3, capacity); e->live.owner = &e->lt;
    e->mt = calloc(K, sizeof(VQTable)); e->ms = calloc(K, sizeof(Sim)); e->misses = calloc(K, sizeof(uint64_t));
    for (int i = 0; i < K; i++) {
        vqt_init(&e->mt[i], (uint64_t)((double)e->mcap * (1.0 + e->cand[i].ghost)) + 1);
        sim_init(&e->ms[i], &e->cand[i], e->mcap); e->ms[i].owner = &e->mt[i];
    }
    return 0;
}

static void vqo_free(VQOnline *e) {
    if (!e->cand) return;
    sim_free(&e->live); vqt_free(&e->lt);
    for (int i = 0; i < e->K; i++) { sim_free(&e->ms[i]); vqt_free(&e->mt[i]); }
    free(e->mt); free(e->ms); free(e->misses); free(e->cand); e->cand = NULL;
}

static int vqo_leader(const VQOnline *e) {
    int leader = 0;
    for (int i = 1; i < e->K; i++) if (e->misses[i] < e->misses[leader]) leader = i;
    int c = e->current;
    if (c >= 0 && leader != c && (double)e->misses[leader] >= (1.0 - e->p.margin) * (double)e->misses[c]) return c;
    return leader;
}

static void vqo_switch(VQOnline *e, int leader) {
    Sim b; sim_init(&b, &e->cand[leader], e->capacity); b.owner = &e->lt;
    if (e->current < 0) adopt(&b, &e->live, e->lt.o); /* from S3-FIFO: hit times restart */
    else { switch_design(&b, &e->live, e->lt.o); e->switches++; }
    sim_free(&e->live); e->live = b; e->current = leader;
}

/* One request for `key`; returns 1 on a miss. */
static int vqo_access(VQOnline *e, uint64_t key) {
    uint64_t now = e->now++;
    int64_t window = (int64_t)(now / e->p.period);
    if (window != e->window) {
        e->window = window;
        if (e->current >= 0 || (uint64_t)window >= e->p.start) {
            int leader = vqo_leader(e);
            if (leader != e->current) vqo_switch(e, leader);
        }
    }
    uint32_t x = vqt_slot(&e->lt, key);
    int miss = step(&e->live, e->lt.o, x);
    if (e->sample_all || mix64(key) <= e->limit) {
        int count = (uint64_t)window >= e->p.from_window;
        for (int i = 0; i < e->K; i++) {
            uint32_t y = vqt_slot(&e->mt[i], key);
            if (step(&e->ms[i], e->mt[i].o, y) && count) e->misses[i]++;
        }
    }
    return miss;
}

/* Objects in the live cache. */
static inline uint64_t vqo_resident(const VQOnline *e) { return e->live.resident; }

/* Bytes allocated (rings, record arrays, maps; not malloc overhead): the live cache, and the
 * selector's miniature caches. With start beyond the trace, the live cache stays S3-FIFO, so
 * vqo_live_bytes is then S3-FIFO's footprint in this engine. */
static uint64_t vq_sim_bytes(const Sim *s) {
    uint64_t b = (uint64_t)s->g.cap * 2 * sizeof(uint32_t); /* ghost ring + entry numbers */
    for (int i = 0; i < s->k; i++) b += (uint64_t)s->q[i].cap * sizeof(uint32_t);
    return b;
}
static uint64_t vqt_bytes(const VQTable *t) {
    return (t->mask + 1) * (sizeof(uint64_t) + sizeof(uint32_t)) + (uint64_t)t->cap * (sizeof(Obj) + sizeof(uint64_t) + sizeof(uint32_t));
}
static uint64_t vqo_live_bytes(const VQOnline *e) { return vq_sim_bytes(&e->live) + vqt_bytes(&e->lt); }
static uint64_t vqo_mini_bytes(const VQOnline *e) {
    uint64_t b = 0;
    for (int i = 0; i < e->K; i++) b += vq_sim_bytes(&e->ms[i]) + vqt_bytes(&e->mt[i]);
    return b;
}

/* Object records in use (cache + ghost, then all miniature caches + ghosts). */
static inline uint64_t vqo_records(const VQOnline *e) { return vqt_live(&e->lt); }
static uint64_t vqo_mini_records(const VQOnline *e) { uint64_t n = 0; for (int i = 0; i < e->K; i++) n += vqt_live(&e->mt[i]); return n; }

#endif
