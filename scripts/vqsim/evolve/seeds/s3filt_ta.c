/* S3-FIFO with a learned filter and TinyLFU-style admission into main (settings 1, 2, 7). Setting 0 is S3-FIFO exactly (vqsim's semantics, as seeds/s3fifo.c).
 *
 * Offline, on the training traces: at the small queue's tail S3-FIFO promotes objects with >= 2 hits; only 24% of them
 * are requested again within one pass of main. A reuse table learned per workload over (the counter, time since the
 * last access relative to main's pass length, the object's request count) picks promotions at the same rate with 39%
 * precision; one table for all workloads only 30%. Main's tail rule is already near what such features allow.
 *
 * So the learned settings keep S3-FIFO's structure and main queue, and change only which small-queue objects move to
 * main: each object at the small tail gets a class (counter 0..3 x 10 buckets of log2(since last access / pass) x 8
 * buckets of log2(request count), the count from a 4-row count-min sketch of 8-bit counters halved every 8C requests),
 * and is promoted if its class's learned chance of a request within one pass is in the top q of recent small-tail
 * objects, q being the share S3-FIFO's rule would promote (times the setting's factor). Labels: a promoted object is
 * positive if requested within one pass of its promotion; an evicted one (in S3-FIFO's ghost) if it returns within one
 * pass; a ghost entry dropped before a pass has gone by gives no label. Counts halve every max(4096, 2C) labels; a
 * class with few labels is shrunk to the overall rate. Statistics are kept in every setting. */
#include <string.h>
#include "policy.h"

#define NCLS (4 * 10 * 8)
#define NBIN 64
enum { NONE = 0, SQ = 1, MQ = 2 };
typedef struct { uint64_t key; uint32_t gseq, ta, last; uint16_t cls; uint8_t freq; uint8_t stage : 2, ghost : 1, pend : 1; } Ent;
typedef struct { uint64_t *buf, head, tail, cap; } Ring;
typedef struct {
    uint64_t cap, target[2], gcap, glive, gnum, mask, gringcap, ghead, gtail, now, smask, sreset, nlab, period;
    int evicted, ghost_hit, learned, admit; double factor;
    double h[NCLS], n[NCLS], gh, gn, hist[NBIN], hall, hinc, q, pass;
    Ent *tab; Ring q2[2]; uint64_t *gk, *gk2; uint32_t *gs, *gs2; uint8_t *sk;
} S3;

static const double SHARE[8] = {0.10, 0.10, 0.10, 0.10, 0.02, 0.30, 0.30, 0.02};
static const double FACTOR[8] = {1, 1, 1, 0.5, 1, 1, 1, 1};
static const int LEARNED[8] = {0, 0, 1, 1, 0, 0, 1, 0};
static const int ADMIT[8] = {0, 1, 1, 0, 0, 0, 0, 1}; /* TinyLFU-style: main admits a newcomer only if its count beats main's tail */
int policy_knobs(void) { return 8; }
void policy_set_knob(void *arena, int k) {
    S3 *s = arena; s->learned = LEARNED[k]; s->factor = FACTOR[k]; s->admit = ADMIT[k];
    s->target[0] = (uint64_t)((double)s->cap * SHARE[k]); s->target[1] = s->cap - s->target[0];
}

static uint64_t rlen(const Ring *r) { return r->head - r->tail; }
static void rpush(Ring *r, uint64_t k) { r->buf[r->head++ % r->cap] = k; }
static uint64_t rpop(Ring *r) { return r->buf[r->tail++ % r->cap]; }

static uint64_t slots_for(uint64_t cap) { uint64_t m = 16, need = 3 * (cap + (uint64_t)((double)cap * 0.9) + 2) / 2 + 2; while (m < need) m <<= 1; return m; }
static uint64_t ringcap(uint64_t cap) { return 2 * (uint64_t)((double)cap * 0.9) + 16; }
static uint64_t sketch_w(uint64_t cap) { uint64_t w = 64; while (w < 4 * cap) w <<= 1; return w; }

size_t policy_bytes(uint64_t cap) {
    return sizeof(S3) + slots_for(cap) * sizeof(Ent) + 2 * (cap + 1) * 8 + 2 * ringcap(cap) * 12 + 4 * sketch_w(cap) + 64;
}

void policy_init(void *arena, uint64_t cap) {
    S3 *s = arena; char *p = (char *)arena + sizeof(S3);
    s->cap = cap; s->gcap = (uint64_t)((double)cap * 0.9); s->gringcap = ringcap(cap);
    policy_set_knob(s, 0);
    uint64_t m = slots_for(cap); s->mask = m - 1;
    s->tab = (Ent *)p; p += m * sizeof(Ent);
    for (int i = 0; i < 2; i++) { s->q2[i].buf = (uint64_t *)p; s->q2[i].cap = cap + 1; p += (cap + 1) * 8; }
    s->gk = (uint64_t *)p; p += s->gringcap * 8; s->gk2 = (uint64_t *)p; p += s->gringcap * 8;
    s->gs = (uint32_t *)p; p += s->gringcap * 4; s->gs2 = (uint32_t *)p; p += s->gringcap * 4;
    s->sk = (uint8_t *)p; s->smask = sketch_w(cap) - 1;
    s->period = cap * 2 > 4096 ? cap * 2 : 4096; s->pass = (double)cap; s->q = 0.1;
}

/* count-min sketch of request counts, halved every 8C requests */
static uint64_t hrow(uint64_t key, int i) { uint64_t x = key * (0x9E3779B97F4A7C15ull + 2 * (uint64_t)i + 1); return x ^ (x >> 29); }
static void sk_add(S3 *s, uint64_t key) {
    for (int i = 0; i < 4; i++) { uint8_t *c = &s->sk[i * (s->smask + 1) + (hrow(key, i) & s->smask)]; if (*c < 255) (*c)++; }
    if (++s->sreset >= 8 * s->cap + 64) { s->sreset = 0; for (uint64_t j = 0; j < 4 * (s->smask + 1); j++) s->sk[j] >>= 1; }
}
static int sk_get(const S3 *s, uint64_t key) {
    int m = 255; for (int i = 0; i < 4; i++) { int c = s->sk[i * (s->smask + 1) + (hrow(key, i) & s->smask)]; if (c < m) m = c; }
    return m;
}

static Ent *find(S3 *s, uint64_t key) {
    for (uint64_t i = key & s->mask;; i = (i + 1) & s->mask) { if (!s->tab[i].key) return NULL; if (s->tab[i].key == key) return &s->tab[i]; }
}
static Ent *add(S3 *s, uint64_t key) {
    uint64_t i = key & s->mask; while (s->tab[i].key) i = (i + 1) & s->mask;
    memset(&s->tab[i], 0, sizeof(Ent)); s->tab[i].key = key; return &s->tab[i];
}
static void del(S3 *s, uint64_t key) {
    uint64_t i = key & s->mask; while (s->tab[i].key != key) i = (i + 1) & s->mask;
    s->tab[i].key = 0;
    for (uint64_t j = (i + 1) & s->mask; s->tab[j].key; j = (j + 1) & s->mask) {
        uint64_t h = s->tab[j].key & s->mask;
        if (((j - h) & s->mask) >= ((j - i) & s->mask)) { s->tab[i] = s->tab[j]; s->tab[j].key = 0; i = j; }
    }
}

static void label(S3 *s, int c, int hit) {
    s->n[c] += 1; s->gn += 1; if (hit) { s->h[c] += 1; s->gh += 1; }
    if (++s->nlab >= s->period) { s->nlab = 0; for (int i = 0; i < NCLS; i++) { s->h[i] /= 2; s->n[i] /= 2; } s->gh /= 2; s->gn /= 2; }
}

static void ghost_add(S3 *s, uint64_t key) {
    while (s->glive >= s->gcap) {
        uint64_t pos = s->gtail++ % s->gringcap; Ent *e = find(s, s->gk[pos]);
        if (e && e->ghost && e->gseq == s->gs[pos]) {
            if (e->pend && (double)(s->now - e->ta) >= s->pass) label(s, e->cls, 0); /* gone for a pass: no return */
            e->ghost = 0; e->pend = 0; s->glive--; if (e->stage == NONE) del(s, e->key);
        }
    }
    if (s->ghead - s->gtail >= s->gringcap) {
        uint64_t w = 0, len = s->ghead - s->gtail;
        for (uint64_t i = 0; i < len; i++) {
            uint64_t pos = (s->gtail + i) % s->gringcap; Ent *e = find(s, s->gk[pos]);
            if (e && e->ghost && e->gseq == s->gs[pos]) { s->gk2[w] = s->gk[pos]; s->gs2[w] = s->gs[pos]; w++; }
        }
        uint64_t *tk = s->gk; s->gk = s->gk2; s->gk2 = tk; uint32_t *ts = s->gs; s->gs = s->gs2; s->gs2 = ts;
        s->gtail = 0; s->ghead = w;
    }
    Ent *e = find(s, key);
    e->ghost = 1; e->gseq = (uint32_t)s->gnum;
    s->gk[s->ghead % s->gringcap] = key; s->gs[s->ghead % s->gringcap] = (uint32_t)s->gnum; s->ghead++; s->gnum++; s->glive++;
}

/* a promoted object's label resolves at its first request (or main's tail) */
static void resolve(S3 *s, Ent *e) { if (e->pend) { label(s, e->cls, (double)(s->now - e->ta) <= s->pass); e->pend = 0; } }

void policy_hit(void *arena, uint64_t key) {
    S3 *s = arena; s->now++; sk_add(s, key);
    Ent *e = find(s, key);
    if (e->freq < 3) e->freq++;
    if (e->stage == MQ) resolve(s, e);
    e->last = (uint32_t)s->now;
}

int policy_admit(void *arena, uint64_t key) {
    S3 *s = arena; s->now++; sk_add(s, key);
    Ent *e = find(s, key);
    s->ghost_hit = 0;
    if (e && e->ghost) { resolve(s, e); e->ghost = 0; s->glive--; s->ghost_hit = 1; }
    return 1;
}

static int pick(const S3 *s) {
    for (int i = 0; i < 2; i++) if (rlen(&s->q2[i]) >= s->target[i] && rlen(&s->q2[i]) > 0) return i;
    for (int i = 0; i < 2; i++) if (rlen(&s->q2[i]) > 0) return i;
    return -1;
}

static int cls_of(S3 *s, Ent *e) {
    double r = (double)(s->now - e->last + 1) / (s->pass + 1);
    int b = 0; while (r >= 1.0 / 16 && b < 9) { r /= 2; b++; } /* log2 buckets from 1/32 of a pass */
    int c = sk_get(s, e->key), cb = 0; while (c > 1 && cb < 7) { c >>= 1; cb++; }
    return (e->freq * 10 + b) * 8 + cb;
}

/* small tail: S3-FIFO's rule, or the learned rule at S3-FIFO's promotion rate (times the setting's factor) */
static int promote(S3 *s, Ent *e, int f) {
    int c = cls_of(s, e); e->cls = (uint16_t)c;
    int s3 = f >= 2;
    s->q += ((double)s3 - s->q) / 1024;
    double g = s->gn > 0 ? s->gh / s->gn : 0.1, rate = (s->h[c] + 8 * g) / (s->n[c] + 8);
    int bin = (int)(rate * NBIN); if (bin >= NBIN) bin = NBIN - 1;
    /* recent small-tail objects by predicted rate, decaying by 1/4096 per object (a growing increment, rescaled) */
    if (s->hinc == 0) s->hinc = 1;
    s->hinc /= 1.0 - 1.0 / 4096; s->hist[bin] += s->hinc; s->hall += s->hinc;
    if (s->hinc > 1e200) { for (int i = 0; i < NBIN; i++) s->hist[i] /= s->hinc; s->hall /= s->hinc; s->hinc = 1; }
    if (!s->learned || s->gn < 256) return s3;
    double want = s->q * s->factor, above = 0;
    for (int i = NBIN - 1; i > bin; i--) above += s->hist[i];
    return above < want * s->hall; /* in the top `want` share of recent small-tail objects */
}

uint64_t policy_victim(void *arena) {
    S3 *s = arena;
    s->evicted = 1;
    int k = pick(s);
    int64_t guard = 4 * (int64_t)s->cap + 16;
    for (;;) {
        if (rlen(&s->q2[k]) == 0) k = pick(s);
        uint64_t key = rpop(&s->q2[k]); Ent *e = find(s, key); int f = e->freq;
        guard--;
        if (k == 0) {
            int up = promote(s, e, f);
            if (up && s->admit && rlen(&s->q2[1]) >= s->target[1] && rlen(&s->q2[1]) > 0
                && sk_get(s, key) <= sk_get(s, s->q2[1].buf[s->q2[1].tail % s->q2[1].cap])) up = 0;
            if (guard >= 0 && up) { e->freq = 0; e->stage = MQ; e->pend = 1; e->ta = (uint32_t)s->now; rpush(&s->q2[1], key); continue; }
            e->stage = NONE; e->freq = 0; e->pend = 1; e->ta = (uint32_t)s->now;
            if (s->gcap > 0 && !e->ghost) ghost_add(s, key); else { e->pend = 0; if (!e->ghost) del(s, key); }
            return key;
        }
        s->pass += ((double)(s->now - e->ta) - s->pass) / 64; /* ta: time it entered main or was last reinserted */
        resolve(s, e);
        if (guard >= 0 && f >= 1) { e->freq = (uint8_t)(f - 1); e->ta = (uint32_t)s->now; rpush(&s->q2[1], key); continue; }
        e->stage = NONE; e->freq = 0;
        if (!e->ghost) del(s, key);
        return key;
    }
}

void policy_insert(void *arena, uint64_t key) {
    S3 *s = arena; Ent *e = find(s, key); if (!e) e = add(s, key);
    int to = s->ghost_hit ? 1 : 0;
    if (to && s->admit && s->evicted && rlen(&s->q2[1]) >= s->target[1] && rlen(&s->q2[1]) > 0
        && sk_get(s, key) <= sk_get(s, s->q2[1].buf[s->q2[1].tail % s->q2[1].cap])) to = 0;
    if (!s->ghost_hit && !s->evicted && rlen(&s->q2[0]) >= s->target[0]) to = 1;
    e->stage = (uint8_t)(to ? MQ : SQ); e->freq = 0; e->pend = 0; e->ta = (uint32_t)s->now; e->last = (uint32_t)s->now;
    rpush(&s->q2[to], key); s->ghost_hit = 0;
}
