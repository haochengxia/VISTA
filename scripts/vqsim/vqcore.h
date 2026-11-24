/* vqcore.h: the virtual-queue FIFO design engine shared by vqsim.c (dense ids, evaluation) and
 * vqonline.h (keyed, deployable). Semantics are documented in vqsim.c; the sandbox's vq.py
 * implements the same ones (tests check they agree exactly).
 *
 * Object records live in an array indexed by uint32 ids. vqsim gives every object of a trace its
 * own record; vqonline.h reuses records, so it defines VQ_RELEASE(sim, objs, id) before including
 * this file: it is called when an object leaves both the cache and the ghost.
 *
 * VQ_RD (defined by vqsim.c only) adds the experimental reuse-distance rule (`rd=`, `rdmode=`; see
 * below). It widens the object record, so vqonline.h, the libCacheSim port and v1 do not use it. */
#ifndef VQCORE_H
#define VQCORE_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef VQ_RELEASE
#define VQ_RELEASE(s, o, y) ((void)0)
#endif
#define VQ_FN static __attribute__((unused))

#define MAXK 8
typedef struct { double size, delay, age; int count, promote, up, onup, down, feeds; } Stage;
typedef struct { int k, maxf, entry, overflow, gentry, gfreq, handover, asmall, amain, gate, gmode, rdmode; uint64_t until; double ghost, gwin, rd; Stage st[MAXK]; } Config;

/* All per-object state in one 16-byte record: a request touches one cache line. With VQ_RD the record
 * also keeps the object's last gap between accesses and when it entered its stage (24 bytes). */
#ifdef VQ_RD
#define VQ_NOGAP UINT32_MAX
typedef struct { uint32_t gstamp, counted, touched; int8_t stage; uint8_t freq; uint8_t rescued, pad; uint32_t gap, pushed; } Obj;
#else
typedef struct { uint32_t gstamp, counted, touched; int8_t stage; uint8_t freq; uint16_t pad; } Obj;
#endif

VQ_FN int parse(const char *s, Config *c) {
    memset(c, 0, sizeof *c);
    c->maxf = 3; c->entry = 0; c->overflow = -1; c->gentry = -1; c->gfreq = 0; c->ghost = 0.9; c->asmall = -1; c->amain = -1; c->gate = -1; c->gmode = 0; c->gwin = 10;
    c->rd = 0; c->rdmode = 3;
    char buf[4096]; strncpy(buf, s, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    for (char *tok = strtok(buf, " \t\n"); tok; tok = strtok(NULL, " \t\n")) {
        if (!strncmp(tok, "st=", 3)) {
            if (c->k >= MAXK) return -1;
            Stage *x = &c->st[c->k++];
            if (sscanf(tok + 3, "%lf,%d,%d,%d,%d,%d,%d,%lf,%lf", &x->size, &x->count, &x->promote, &x->up, &x->onup, &x->down, &x->feeds, &x->delay, &x->age) < 7) return -1;
        } else if (!strncmp(tok, "maxf=", 5)) c->maxf = atoi(tok + 5);
        else if (!strncmp(tok, "entry=", 6)) c->entry = atoi(tok + 6);
        else if (!strncmp(tok, "overflow=", 9)) c->overflow = atoi(tok + 9);
        else if (!strncmp(tok, "ghost=", 6)) c->ghost = atof(tok + 6);
        else if (!strncmp(tok, "gentry=", 7)) c->gentry = atoi(tok + 7);
        else if (!strncmp(tok, "gfreq=", 6)) c->gfreq = atoi(tok + 6);
        else if (!strncmp(tok, "handover=", 9)) c->handover = atoi(tok + 9);
        else if (!strncmp(tok, "asmall=", 7)) c->asmall = atoi(tok + 7);
        else if (!strncmp(tok, "amain=", 6)) c->amain = atoi(tok + 6);
        else if (!strncmp(tok, "until=", 6)) c->until = strtoull(tok + 6, NULL, 10);
        else if (!strncmp(tok, "gate=", 5)) c->gate = atoi(tok + 5);
        else if (!strncmp(tok, "gwin=", 5)) c->gwin = atof(tok + 5);
        else if (!strncmp(tok, "gmode=", 6)) c->gmode = atoi(tok + 6);
#ifdef VQ_RD
        else if (!strncmp(tok, "rd=", 3)) c->rd = atof(tok + 3);
        else if (!strncmp(tok, "rdmode=", 7)) c->rdmode = atoi(tok + 7);
#endif
        else return -1;
    }
    if (c->k < 1 || c->maxf < 1 || c->maxf > 15 || c->ghost < 0) return -1;
    if (c->overflow < 0) c->overflow = c->k - 1;
    if (c->gentry < 0) c->gentry = c->k - 1;
    if (c->asmall < 0) c->asmall = c->entry;
    if (c->amain < 0) c->amain = c->k - 1;
    for (int i = 0; i < c->k; i++) {
        Stage *x = &c->st[i];
        if (x->up < 0 || x->up >= c->k || x->down >= c->k || x->down < -1 || x->onup < 0 || x->onup > 3) return -1;
        if (x->up == i && x->onup == 1 && x->promote > 0) return -1; /* reinsertion must lower the counter */
    }
    if (c->entry >= c->k || c->overflow >= c->k || c->gentry >= c->k || c->asmall >= c->k || c->amain >= c->k) return -1;
    if (c->gate < -1 || c->gate >= c->k || c->gate == c->entry || c->gwin <= 0 || c->gmode < 0 || c->gmode > 2) return -1;
    if (c->rd < 0 || c->rdmode < 0 || c->rdmode > 15) return -1;
    /* Moves that need no hit and keep the counter (a demotion, a pass with promote 0, a promotion
     * with onup=keep) must not form a cycle, or objects could circle without ever leaving. */
    int edge[MAXK][MAXK] = {{0}};
    for (int i = 0; i < c->k; i++) {
        const Stage *x = &c->st[i];
        if (x->down >= 0) edge[i][x->down] = 1;
        if (c->gate >= 0 && c->gmode != 1 && i != c->gate) edge[i][c->gate] = 1; /* a rescue needs no hit */
        if (x->up != i && (x->promote == 0 || x->onup == 1)) edge[i][x->up] = 1;
    }
    for (int m = 0; m < c->k; m++) for (int i = 0; i < c->k; i++) for (int j = 0; j < c->k; j++) if (edge[i][m] && edge[m][j]) edge[i][j] = 1;
    for (int i = 0; i < c->k; i++) if (edge[i][i]) return -1;
    return 0;
}

typedef struct { uint32_t *buf; uint64_t head, tail, cap; } Ring; /* push at head, pop at tail */
VQ_FN void rpush(Ring *r, uint32_t v) { r->buf[r->head++ % r->cap] = v; }
VQ_FN uint32_t rpop(Ring *r) { return r->buf[r->tail++ % r->cap]; }
VQ_FN uint64_t rlen(const Ring *r) { return r->head - r->tail; }
VQ_FN uint32_t rat(const Ring *r, uint64_t i) { return r->buf[(r->tail + i) % r->cap]; } /* i-th from the tail */

/* gate (TinyLFU-style admission, off by default): objects compete with the object at the tail of
 * stage `gate` (the main stage) on a count-min sketch of recent request frequencies (4 rows of
 * saturating 4-bit counts, width >= C, all halved after gwin * C requests).
 *   gmode=0 rescue: an object about to leave the cache from any other stage moves to `gate`
 *           instead if the sketch rates it higher; the scan goes on there.
 *   gmode=1 filter: a promotion into `gate` from another stage goes ahead only if the sketch rates
 *           the object higher; otherwise the object is treated as not qualifying (demoted if its
 *           stage demotes, else it leaves). This is W-TinyLFU's admission test.
 *   gmode=2 both. */
typedef struct VQSketch { uint8_t *t; uint64_t mask, adds, reset_at; } VQSketch;
VQ_FN void sk_init(VQSketch *k, uint64_t capacity, double gwin) {
    uint64_t w = 16; while (w < capacity) w <<= 1;
    k->mask = w - 1; k->t = calloc(4 * w, 1); k->adds = 0; k->reset_at = (uint64_t)(gwin * (double)(capacity ? capacity : 1)) + 1;
}
VQ_FN void sk_free(VQSketch *k) { free(k->t); k->t = NULL; }
VQ_FN uint64_t sk_h(uint64_t x, int row) { x += 0x9E3779B97F4A7C15ull * (uint64_t)(row + 1); x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31); }
VQ_FN void sk_add(VQSketch *k, uint64_t x) {
    for (int r = 0; r < 4; r++) { uint8_t *c = &k->t[(uint64_t)r * (k->mask + 1) + (sk_h(x, r) & k->mask)]; if (*c < 15) (*c)++; }
    if (++k->adds >= k->reset_at) { for (uint64_t i = 0; i < 4 * (k->mask + 1); i++) k->t[i] >>= 1; k->adds /= 2; }
}
VQ_FN int sk_est(const VQSketch *k, uint64_t x) {
    int m = 15;
    for (int r = 0; r < 4; r++) { int c = k->t[(uint64_t)r * (k->mask + 1) + (sk_h(x, r) & k->mask)]; if (c < m) m = c; }
    return m;
}

/* rd (reuse distance, VQ_RD only, off by default): a qualifying move is refused when the object's
 * predicted wait for its next request exceeds rd times the residence time of the stage it would move
 * to. Time is the insertion clock. The wait is (last gap - time since the last request) while that is
 * positive, else the time since the last request (an overdue object is guessed to wait as long again).
 * A stage's residence time is a running mean (1/32) of the time its objects spend in it, pop to push.
 * rdmode is a bitmask: 1 promotions (up to another stage), 2 reinsertions (up to the same stage),
 * 4 ghost admission (a ghost hit whose gap exceeds rd times the residence time of `gentry` enters
 * `entry` instead), 8 rescue (an object that does not qualify moves up anyway if its predicted wait is
 * within the bound and its gap is known; once per request). Default 3. */
#define VQ_RD_ALPHA (1.0 / 32)

typedef struct {
    const Config *c; uint64_t capacity, target[MAXK], delay[MAXK], age[MAXK], gcap, glive, resident;
    double res[MAXK]; /* VQ_RD: running mean residence time per stage */
    uint32_t clock, gnum; int evicted, k;
    Ring q[MAXK], g; uint32_t *gseq; /* ghost ring of keys, and each slot's entry number */
    void *owner; /* for VQ_RELEASE: whoever maps keys to object records */
    struct VQSketch *sk; /* gate: frequency sketch, fed by the caller (vqsim: every request) */
} Sim;

VQ_FN void sim_init(Sim *s, const Config *c, uint64_t capacity) {
    memset(s, 0, sizeof *s);
    s->c = c; s->k = c->k; s->capacity = capacity;
    double used = 0;
    for (int i = 0; i < c->k; i++) {
        s->q[i].cap = capacity + 1; s->q[i].buf = malloc(s->q[i].cap * sizeof(uint32_t));
        if (i < c->k - 1) { s->target[i] = (uint64_t)(capacity * c->st[i].size); used += (double)s->target[i]; }
        s->delay[i] = (uint64_t)(c->st[i].delay * capacity); s->age[i] = (uint64_t)(c->st[i].age * capacity);
    }
    s->target[c->k - 1] = capacity > (uint64_t)used ? capacity - (uint64_t)used : 0;
    for (int i = 0; i < c->k; i++) s->res[i] = s->target[i] > 0 ? (double)s->target[i] : 1.0;
    s->gcap = (uint64_t)(capacity * c->ghost);
    s->g.cap = 2 * s->gcap + 16; s->g.buf = malloc(s->g.cap * sizeof(uint32_t)); s->gseq = malloc(s->g.cap * sizeof(uint32_t));
}

VQ_FN void sim_free(Sim *s) {
    for (int i = 0; i < s->k; i++) free(s->q[i].buf);
    free(s->g.buf); free(s->gseq);
}

VQ_FN void ghost_add(Sim *s, Obj *o, uint32_t y) {
    Ring *g = &s->g;
    while (s->glive >= s->gcap) { uint64_t pos = g->tail; uint32_t z = rpop(g); if (o[z].gstamp == s->gseq[pos % g->cap] + 1) { o[z].gstamp = 0; s->glive--; VQ_RELEASE(s, o, z); } }
    if (rlen(g) >= g->cap) { /* compact stale entries */
        uint64_t w = 0, len = rlen(g); uint32_t *nb = malloc(g->cap * sizeof(uint32_t)), *ns = malloc(g->cap * sizeof(uint32_t));
        for (uint64_t i = 0; i < len; i++) { uint64_t pos = (g->tail + i) % g->cap; uint32_t z = g->buf[pos]; if (o[z].gstamp == s->gseq[pos] + 1) { nb[w] = z; ns[w] = s->gseq[pos]; w++; } }
        free(g->buf); free(s->gseq); g->buf = nb; s->gseq = ns; g->tail = 0; g->head = w;
    }
    s->gseq[g->head % g->cap] = s->gnum; o[y].gstamp = s->gnum + 1; s->gnum++; rpush(g, y); s->glive++;
}

VQ_FN int pick(const Sim *s) {
    for (int i = 0; i < s->k; i++) if (rlen(&s->q[i]) >= s->target[i] && rlen(&s->q[i]) > 0) return i;
    for (int i = 0; i < s->k; i++) if (rlen(&s->q[i]) > 0) return i;
    return -1;
}

VQ_FN void evict_one(Sim *s, Obj *o) {
    const Config *c = s->c;
    s->evicted = 1;
    int k = pick(s);
    int64_t guard = 4 * (int64_t)s->capacity + 16; /* a design whose moves cycle without hits is cut short */
    for (;;) {
        if (rlen(&s->q[k]) == 0) k = pick(s); /* this stage drained without an eviction: pick again */
        uint32_t y = rpop(&s->q[k]);
        const Stage *st = &c->st[k];
        Obj *p = &o[y];
        int f = p->freq;
        guard--;
#ifdef VQ_RD
        s->res[k] += ((double)(s->clock - p->pushed) - s->res[k]) * VQ_RD_ALPHA;
#endif
        if (guard >= 0) {
            int idle = s->age[k] && s->clock - p->touched >= s->age[k];
            int up = f >= st->promote && !idle && !(st->promote == 0 && st->up == k);
            if (up && c->gate >= 0 && c->gmode >= 1 && st->up == c->gate && k != c->gate && s->sk && rlen(&s->q[c->gate]) > 0
                && sk_est(s->sk, y) <= sk_est(s->sk, rat(&s->q[c->gate], 0))) up = 0; /* gate filter: refused */
#ifdef VQ_RD
            if (c->rd > 0 && p->gap != VQ_NOGAP && (up ? (c->rdmode & (st->up == k ? 2 : 1)) : ((c->rdmode & 8) && !p->rescued && !idle))) {
                uint32_t e = s->clock - p->touched;
                double wait = e < p->gap ? (double)(p->gap - e) : (double)e;
                int near = wait <= c->rd * s->res[st->up];
                if (up && !near) up = 0; /* refused: it would not be requested before it reaches the tail again */
                else if (!up && near) { up = 1; p->rescued = 1; } /* rescue: it moves up as if it qualified, counter as is */
            }
#endif
            if (up) {
                int to = st->up;
                if (st->onup == 0) p->freq = 0; else if (st->onup == 2) p->freq = f > 0 ? f - 1 : 0; else if (st->onup == 3) p->freq = f >> 1;
                p->stage = (int8_t)to; rpush(&s->q[to], y);
#ifdef VQ_RD
                p->pushed = s->clock;
#endif
                continue;
            }
            if (st->down >= 0) { /* demotion: the receiving stage may now be over its target, pick again */
                p->stage = (int8_t)st->down; rpush(&s->q[st->down], y);
#ifdef VQ_RD
                p->pushed = s->clock;
#endif
                k = pick(s);
                continue;
            }
        }
        if (c->gate >= 0 && c->gmode != 1 && k != c->gate && s->sk && guard >= 0 && rlen(&s->q[c->gate]) > 0
            && sk_est(s->sk, y) > sk_est(s->sk, rat(&s->q[c->gate], 0))) { /* gate rescue: admitted after all */
            p->stage = (int8_t)c->gate; rpush(&s->q[c->gate], y);
#ifdef VQ_RD
            p->pushed = s->clock;
#endif
            k = pick(s);
            continue;
        }
        p->stage = -1; p->freq = 0; s->resident--; /* leave the cache */
        if (st->feeds && s->gcap > 0 && !p->gstamp) ghost_add(s, o, y);
        else if (!p->gstamp) VQ_RELEASE(s, o, y);
        return;
    }
}

/* One request; returns 1 on a miss. */
VQ_FN inline int step(Sim *s, Obj *o, uint32_t x) {
    const Config *c = s->c;
    Obj *p = &o[x];
    if (p->stage >= 0) { /* hit */
        int sx = p->stage;
        if (c->st[sx].count && p->freq < c->maxf && (!s->delay[sx] || s->clock - p->counted > s->delay[sx])) { p->freq++; p->counted = s->clock; }
#ifdef VQ_RD
        p->gap = s->clock - p->touched; p->rescued = 0;
#endif
        p->touched = s->clock;
        return 0;
    }
    int ghost_hit = 0;
    if (p->gstamp) { p->gstamp = 0; s->glive--; ghost_hit = 1; }
    if (s->capacity == 0) return 1;
#ifdef VQ_RD
    uint32_t gap = ghost_hit ? s->clock - p->touched : VQ_NOGAP; /* a ghost hit's gap since its last request */
#endif
    while (s->resident >= s->capacity) evict_one(s, o);
    int to = ghost_hit ? c->gentry : c->entry;
#ifdef VQ_RD
    if (ghost_hit && c->rd > 0 && (c->rdmode & 4) && (double)gap > c->rd * s->res[c->gentry]) to = c->entry;
#endif
    if (!ghost_hit && !s->evicted && rlen(&s->q[to]) >= s->target[to]) to = c->overflow;
    p->stage = (int8_t)to; p->freq = ghost_hit ? (uint8_t)c->gfreq : 0; rpush(&s->q[to], x); s->resident++;
    s->clock++; p->counted = p->touched = s->clock;
#ifdef VQ_RD
    p->gap = gap; p->pushed = s->clock; p->rescued = 0;
#endif
    return 1;
}

/* Online mode only: where a takeover puts the old cache's objects. 0 = by stage (objects of the
 * old entry stage to asmall, all others to amain); 1 = by counter (objects hit since they entered
 * or last moved to amain, the others to asmall); 2 = all to amain. Modes 1 and 2 place the old
 * stages from the last to the first, so the old main's objects end up oldest. 3 (VQ_RD only, switches
 * between designs; the takeover at the split is mode 0) = by stage, but the objects bound for amain
 * are merged across the old stages in the order they entered them, so FIFO age carries over. */
static int takeover = 0;
VQ_FN int takeover_to(const Config *c, int from_entry, int freq) {
    if (takeover == 1) return freq > 0 ? c->amain : c->asmall;
    if (takeover == 2) return c->amain;
    return from_entry ? c->asmall : c->amain;
}

/* The design in `b` takes over S3-FIFO's live cache in `a` (vq.py's adopt). */
VQ_FN void adopt(Sim *b, const Sim *a, Obj *o) {
    const Config *c = b->c;
    for (int k = 0; k < 2; k++) {
        int src = takeover ? 1 - k : k;
        for (uint64_t i = 0; i < rlen(&a->q[src]); i++) {
            uint32_t y = rat(&a->q[src], i);
            Obj *p = &o[y];
            int to = takeover_to(c, src == 0, p->freq);
            p->stage = (int8_t)to; p->freq = c->st[to].count ? (uint8_t)(p->freq < c->maxf ? p->freq : c->maxf) : 0;
#ifdef VQ_RD
            p->pushed = 0;
            if (c->rd > 0) { p->counted = 0; p->touched -= a->clock; } else
#endif
            p->counted = p->touched = 0;
            rpush(&b->q[to], y);
        }
    }
    b->resident = a->resident; b->evicted = a->evicted; b->clock = 0;
    uint64_t live = 0, len = rlen(&a->g); /* the incumbent's ghost, oldest first; keep the newest gcap */
    for (uint64_t i = 0; i < len; i++) { uint64_t pos = (a->g.tail + i) % a->g.cap; if (o[a->g.buf[pos]].gstamp == a->gseq[pos] + 1) live++; }
    uint64_t drop = live > b->gcap ? live - b->gcap : 0, seen = 0;
    for (uint64_t i = 0; i < len; i++) {
        uint64_t pos = (a->g.tail + i) % a->g.cap; uint32_t z = a->g.buf[pos];
        if (o[z].gstamp != a->gseq[pos] + 1) continue;
        o[z].gstamp = 0;
#ifdef VQ_RD
        if (c->rd > 0) o[z].touched -= a->clock;
#endif
        if (seen++ >= drop) { b->gseq[b->g.head % b->g.cap] = b->gnum; o[z].gstamp = b->gnum + 1; b->gnum++; rpush(&b->g, z); b->glive++; }
        else VQ_RELEASE(b, o, z);
    }
}

static const char *S3_DEFAULT = "maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0";

#ifdef VQ_RD
static int cmp_u64(const void *x, const void *y) { uint64_t u = *(const uint64_t *)x, v = *(const uint64_t *)y; return u < v ? -1 : u > v; }
#endif

VQ_FN void switch_design(Sim *b, const Sim *a, Obj *o) {
    const Config *c = b->c;
#ifdef VQ_RD
    if (takeover == 3) {
        const Ring *e = &a->q[a->c->entry];
        for (uint64_t j = 0; j < rlen(e); j++) { /* the old entry stage, tail first */
            uint32_t y = rat(e, j); Obj *p = &o[y];
            p->stage = (int8_t)c->asmall; p->freq = c->st[c->asmall].count ? (uint8_t)(p->freq < c->maxf ? p->freq : c->maxf) : 0;
            rpush(&b->q[c->asmall], y);
        }
        uint64_t n = 0; for (int i = 0; i < a->k; i++) if (i != a->c->entry) n += rlen(&a->q[i]);
        uint64_t *key = malloc((n ? n : 1) * sizeof *key); uint32_t *ids = malloc((n ? n : 1) * sizeof *ids); n = 0;
        for (int i = 0; i < a->k; i++) if (i != a->c->entry)
            for (uint64_t j = 0; j < rlen(&a->q[i]); j++) { ids[n] = rat(&a->q[i], j); key[n] = ((uint64_t)o[ids[n]].pushed << 32) | n; n++; }
        qsort(key, n, sizeof *key, cmp_u64); /* oldest entry first; ties keep the stage order */
        for (uint64_t j = 0; j < n; j++) {
            uint32_t y = ids[key[j] & 0xffffffffu]; Obj *p = &o[y];
            p->stage = (int8_t)c->amain; p->freq = c->st[c->amain].count ? (uint8_t)(p->freq < c->maxf ? p->freq : c->maxf) : 0;
            rpush(&b->q[c->amain], y); /* `pushed` kept: the age it carries over */
        }
        free(key); free(ids);
    } else
#endif
    for (int k = 0; k < a->k; k++) {
        int i = takeover ? a->k - 1 - k : k;
        for (uint64_t j = 0; j < rlen(&a->q[i]); j++) {
            uint32_t y = rat(&a->q[i], j);
            Obj *p = &o[y];
            int to = takeover_to(c, i == a->c->entry, p->freq);
            p->stage = (int8_t)to; p->freq = c->st[to].count ? (uint8_t)(p->freq < c->maxf ? p->freq : c->maxf) : 0;
#ifdef VQ_RD
            p->pushed = a->clock;
#endif
            rpush(&b->q[to], y);
        }
    }
    b->resident = a->resident; b->evicted = a->evicted; b->clock = a->clock;
    uint64_t live = 0, len = rlen(&a->g);
    for (uint64_t i = 0; i < len; i++) { uint64_t pos = (a->g.tail + i) % a->g.cap; if (o[a->g.buf[pos]].gstamp == a->gseq[pos] + 1) live++; }
    uint64_t drop = live > b->gcap ? live - b->gcap : 0, seen = 0;
    for (uint64_t i = 0; i < len; i++) {
        uint64_t pos = (a->g.tail + i) % a->g.cap; uint32_t z = a->g.buf[pos];
        if (o[z].gstamp != a->gseq[pos] + 1) continue;
        o[z].gstamp = 0;
        if (seen++ >= drop) { b->gseq[b->g.head % b->g.cap] = b->gnum; o[z].gstamp = b->gnum + 1; b->gnum++; rpush(&b->g, z); b->glive++; }
        else VQ_RELEASE(b, o, z);
    }
}

VQ_FN uint64_t mix64(uint64_t x) { x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31); }

#endif
