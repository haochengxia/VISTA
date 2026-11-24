/* S3-FIFO with a learned main-queue rule. Settings: the small queue's share x {S3-FIFO's rule, the learned rule};
 * setting 0 is S3-FIFO exactly (vqsim's semantics, as seeds/s3fifo.c).
 *
 * The learned rule replaces "reinsert a main-queue object if it was hit (counter >= 1)". When an object reaches main's
 * tail it falls in a class: hits during this pass (0..3+), passes survived in main (0..3+), entered main from the ghost
 * (0/1), and whether its last hit came in the second half of the pass (0/1). The object is kept (reinserted, pass hits
 * reset) if its class's measured chance of a hit in the next pass beats that of objects newly entering main; else it
 * leaves. Labels are free: a kept object shows at its next tail arrival whether it was hit; for a sample (1/4) of the
 * objects that leave main, a request within one main pass counts as the hit they would have had. Statistics are kept
 * in every setting (so a setting change finds them ready), halve every max(4096, 2C) decisions, and a class decides
 * only after 32 observations (before that, S3-FIFO's rule). */
#include <string.h>
#include "policy.h"

enum { NONE = 0, SQ = 1, MQ = 2 };
typedef struct { uint64_t key; uint32_t tp, ta; uint8_t freq, ph, cls; uint8_t stage : 2, r : 2, g : 1, pend : 1, ghost : 1, lab : 1; } Ent;
typedef struct { uint64_t *buf, head, tail, cap; } Ring;
typedef struct {
    uint64_t cap, target[2], gcap, glive, gnum, mask, gringcap, ghead, gtail, now, lcap, lhead, ltail, ndec, period;
    int evicted, ghost_hit, learned;
    double h[64], n[64], eh, en, tm;
    Ent *tab; Ring q[2]; uint64_t *gk, *gk2, *lk; uint32_t *gs, *gs2, *ls;
} S3;

static const double SHARE[4] = {0.10, 0.02, 0.30, 0.60};
int policy_knobs(void) { return 8; }
void policy_set_knob(void *arena, int k) {
    S3 *s = arena; s->learned = k >= 4;
    s->target[0] = (uint64_t)((double)s->cap * SHARE[k & 3]); s->target[1] = s->cap - s->target[0];
}

static uint64_t rlen(const Ring *r) { return r->head - r->tail; }
static void rpush(Ring *r, uint64_t k) { r->buf[r->head++ % r->cap] = k; }
static uint64_t rpop(Ring *r) { return r->buf[r->tail++ % r->cap]; }
static uint64_t mix(uint64_t x) { x ^= x >> 31; x *= 0x7fb5d329728ea185ull; x ^= x >> 27; return x; }

static uint64_t entries(uint64_t cap) { return cap + (uint64_t)((double)cap * 0.9) + cap / 2 + 64; }
static uint64_t slots_for(uint64_t cap) { uint64_t m = 16, need = 3 * entries(cap) / 2 + 2; while (m < need) m <<= 1; return m; }
static uint64_t ringcap(uint64_t cap) { return 2 * (uint64_t)((double)cap * 0.9) + 16; }
static uint64_t lringcap(uint64_t cap) { return cap / 2 + 64; }

size_t policy_bytes(uint64_t cap) {
    return sizeof(S3) + slots_for(cap) * sizeof(Ent) + 2 * (cap + 1) * 8 + 2 * ringcap(cap) * 12 + lringcap(cap) * 12 + 64;
}

void policy_init(void *arena, uint64_t cap) {
    S3 *s = arena; char *p = (char *)arena + sizeof(S3);
    s->cap = cap; s->gcap = (uint64_t)((double)cap * 0.9); s->gringcap = ringcap(cap); s->lcap = lringcap(cap);
    policy_set_knob(s, 0);
    uint64_t m = slots_for(cap); s->mask = m - 1;
    s->tab = (Ent *)p; p += m * sizeof(Ent);
    for (int i = 0; i < 2; i++) { s->q[i].buf = (uint64_t *)p; s->q[i].cap = cap + 1; p += (cap + 1) * 8; }
    s->gk = (uint64_t *)p; p += s->gringcap * 8; s->gk2 = (uint64_t *)p; p += s->gringcap * 8;
    s->gs = (uint32_t *)p; p += s->gringcap * 4; s->gs2 = (uint32_t *)p; p += s->gringcap * 4;
    s->lk = (uint64_t *)p; p += s->lcap * 8; s->ls = (uint32_t *)p;
    s->period = cap * 2 > 4096 ? cap * 2 : 4096; s->tm = (double)cap;
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
static void maybe_del(S3 *s, Ent *e) { if (e->stage == NONE && !e->ghost && !e->lab) del(s, e->key); }

static void ghost_add(S3 *s, uint64_t key) {
    while (s->glive >= s->gcap) {
        uint64_t pos = s->gtail++ % s->gringcap; Ent *e = find(s, s->gk[pos]);
        if (e && e->ghost && e->ta == s->gs[pos]) { e->ghost = 0; s->glive--; maybe_del(s, e); }
    }
    if (s->ghead - s->gtail >= s->gringcap) {
        uint64_t w = 0, len = s->ghead - s->gtail;
        for (uint64_t i = 0; i < len; i++) {
            uint64_t pos = (s->gtail + i) % s->gringcap; Ent *e = find(s, s->gk[pos]);
            if (e && e->ghost && e->ta == s->gs[pos]) { s->gk2[w] = s->gk[pos]; s->gs2[w] = s->gs[pos]; w++; }
        }
        uint64_t *tk = s->gk; s->gk = s->gk2; s->gk2 = tk; uint32_t *ts = s->gs; s->gs = s->gs2; s->gs2 = ts;
        s->gtail = 0; s->ghead = w;
    }
    Ent *e = find(s, key);
    e->ghost = 1; e->ta = (uint32_t)s->gnum;
    s->gk[s->ghead % s->gringcap] = key; s->gs[s->ghead % s->gringcap] = (uint32_t)s->gnum; s->ghead++; s->gnum++; s->glive++;
}

/* a label for class c: hit (1) or not (0), with weight w */
static void label(S3 *s, int c, int hit, double w) { s->n[c] += w; if (hit) s->h[c] += w; }
static void lab_add(S3 *s, Ent *e, int c) { /* a sampled object leaving main: watch for its return within one pass */
    if (s->lhead - s->ltail >= s->lcap) {
        uint64_t pos = s->ltail++ % s->lcap; Ent *o = find(s, s->lk[pos]);
        if (o && o->lab && o->ta == s->ls[pos]) { label(s, o->cls, 0, 4); o->lab = 0; maybe_del(s, o); }
    }
    e->lab = 1; e->cls = (uint8_t)c; e->ta = (uint32_t)s->now;
    s->lk[s->lhead % s->lcap] = e->key; s->ls[s->lhead % s->lcap] = (uint32_t)s->now; s->lhead++;
}

void policy_hit(void *arena, uint64_t key) {
    S3 *s = arena; s->now++;
    Ent *e = find(s, key);
    if (e->freq < 3) e->freq++;
    if (e->ph < 3) e->ph++;
    e->ta = (uint32_t)s->now; /* last hit */
}

int policy_admit(void *arena, uint64_t key) {
    S3 *s = arena; s->now++;
    Ent *e = find(s, key);
    s->ghost_hit = 0;
    if (e && e->lab) { /* a sampled main leaver came back: a hit if within one pass */
        label(s, e->cls, (double)(s->now - e->ta) <= s->tm, 4); e->lab = 0;
    }
    if (e && e->ghost) { e->ghost = 0; s->glive--; s->ghost_hit = 1; }
    return 1;
}

static int pick(const S3 *s) {
    for (int i = 0; i < 2; i++) if (rlen(&s->q[i]) >= s->target[i] && rlen(&s->q[i]) > 0) return i;
    for (int i = 0; i < 2; i++) if (rlen(&s->q[i]) > 0) return i;
    return -1;
}

static void enter_main(S3 *s, Ent *e, int from_ghost) { e->stage = MQ; e->r = 0; e->g = (uint8_t)from_ghost; e->pend = 0; e->ph = 0; e->tp = (uint32_t)s->now; }

/* at main's tail: learn from the object's last pass, then decide; returns 1 to keep */
static int main_tail(S3 *s, Ent *e) {
    double pass = (double)(s->now - e->tp);
    s->tm += (pass - s->tm) / 64;
    int ph = e->ph, late = ph > 0 && (double)(e->ta - e->tp) > pass / 2;
    if (e->pend) label(s, e->cls, ph > 0, 1);
    else if (e->r == 0) { s->en += 1; if (ph > 0) s->eh += 1; }
    if (++s->ndec >= s->period) { s->ndec = 0; for (int i = 0; i < 64; i++) { s->h[i] /= 2; s->n[i] /= 2; } s->eh /= 2; s->en /= 2; }
    int c = (((ph < 3 ? ph : 3) * 4 + e->r) * 2 + e->g) * 2 + late;
    int keep = e->freq >= 1; /* S3-FIFO's rule */
    if (s->learned && s->n[c] >= 32 && s->en >= 32) keep = s->h[c] / s->n[c] > s->eh / s->en;
    e->cls = (uint8_t)c;
    return keep;
}

uint64_t policy_victim(void *arena) {
    S3 *s = arena;
    s->evicted = 1;
    int k = pick(s);
    int64_t guard = 4 * (int64_t)s->cap + 16;
    for (;;) {
        if (rlen(&s->q[k]) == 0) k = pick(s);
        uint64_t key = rpop(&s->q[k]); Ent *e = find(s, key); int f = e->freq;
        guard--;
        if (k == 0) {
            if (guard >= 0 && f >= 2) { e->freq = 0; enter_main(s, e, 0); rpush(&s->q[1], key); continue; }
            e->stage = NONE; e->freq = 0;
            if (s->gcap > 0 && !e->ghost) ghost_add(s, key); else maybe_del(s, e);
            return key;
        }
        int keep = main_tail(s, e);
        if (guard >= 0 && keep) {
            e->freq = s->learned ? 0 : (uint8_t)(f - 1); /* S3-FIFO: f >= 1 here */
            if (s->learned && f == 0) e->freq = 0;
            e->pend = 1; if (e->r < 3) e->r++; e->ph = 0; e->tp = (uint32_t)s->now;
            rpush(&s->q[1], key);
            continue;
        }
        int c = e->cls;
        e->stage = NONE; e->freq = 0; e->pend = 0;
        if (!e->lab && (mix(key) & 3) == 0) lab_add(s, e, c); else maybe_del(s, e);
        return key;
    }
}

void policy_insert(void *arena, uint64_t key) {
    S3 *s = arena; Ent *e = find(s, key); if (!e) e = add(s, key);
    int to = s->ghost_hit ? 1 : 0;
    if (!s->ghost_hit && !s->evicted && rlen(&s->q[0]) >= s->target[0]) to = 1;
    e->freq = 0; e->ph = 0; e->ta = (uint32_t)s->now;
    if (to == 1) enter_main(s, e, s->ghost_hit); else { e->stage = SQ; e->tp = (uint32_t)s->now; }
    rpush(&s->q[to], key); s->ghost_hit = 0;
}
