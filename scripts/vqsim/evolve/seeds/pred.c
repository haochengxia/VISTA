/* PRED (a probe, not a candidate): evicts, from 64 sampled cached objects, the one whose next access is predicted
 * furthest away, with next = last access + the object's inter-access gap (EWMA). An object whose gap is unknown (seen
 * once) uses the running mean of observed first gaps; one overdue by more than its gap counts as dead and goes first.
 * History (last access, gap) is kept for 2C + 64 recently evicted objects, so a returning object keeps its gap. */
#include <string.h>
#include "policy.h"

#define NOPOS 0xFFFFFFFFu
typedef struct { uint64_t key; uint32_t last, gap, pos, hseq; } Ent;
typedef struct {
    uint64_t cap, mask, hcap, hhead, htail, rng, n, now;
    double d0;
    Ent *tab; uint64_t *keys, *hk; uint32_t *hs;
} P;

static uint64_t slots_for(uint64_t cap) { uint64_t m = 16, need = 3 * (3 * cap + 66) / 2; while (m < need) m <<= 1; return m; } /* load <= 2/3 */
size_t policy_bytes(uint64_t cap) { uint64_t h = 2 * cap + 64; return sizeof(P) + slots_for(cap) * sizeof(Ent) + cap * 8 + h * 12 + 64; }

void policy_init(void *arena, uint64_t cap) {
    P *p = arena; char *q = (char *)arena + sizeof(P);
    p->cap = cap; p->hcap = 2 * cap + 64; uint64_t m = slots_for(cap); p->mask = m - 1; p->rng = 0x9E3779B97F4A7C15ull; p->d0 = 0;
    p->tab = (Ent *)q; q += m * sizeof(Ent); p->keys = (uint64_t *)q; q += cap * 8;
    p->hk = (uint64_t *)q; q += p->hcap * 8; p->hs = (uint32_t *)q;
}

static Ent *find(P *p, uint64_t key) {
    for (uint64_t i = key & p->mask;; i = (i + 1) & p->mask) { if (!p->tab[i].key) return NULL; if (p->tab[i].key == key) return &p->tab[i]; }
}
static Ent *add(P *p, uint64_t key) {
    uint64_t i = key & p->mask; while (p->tab[i].key) i = (i + 1) & p->mask;
    memset(&p->tab[i], 0, sizeof(Ent)); p->tab[i].key = key; p->tab[i].pos = NOPOS; return &p->tab[i];
}
static void del(P *p, uint64_t key) {
    uint64_t i = key & p->mask; while (p->tab[i].key != key) i = (i + 1) & p->mask;
    p->tab[i].key = 0;
    for (uint64_t j = (i + 1) & p->mask; p->tab[j].key; j = (j + 1) & p->mask) {
        uint64_t h = p->tab[j].key & p->mask;
        if (((j - h) & p->mask) >= ((j - i) & p->mask)) { p->tab[i] = p->tab[j]; p->tab[j].key = 0; i = j; }
    }
}
static void touch(P *p, Ent *e) { /* a request for e at time now */
    if (e->last || e->gap) {
        uint32_t d = (uint32_t)(p->now - e->last);
        if (!e->gap) { p->d0 = p->d0 ? 0.99 * p->d0 + 0.01 * d : d; e->gap = d ? d : 1; }
        else e->gap = (uint32_t)(((uint64_t)e->gap + d) / 2) + 1;
    }
    e->last = (uint32_t)p->now;
}

void policy_hit(void *arena, uint64_t key) { P *p = arena; p->now++; touch(p, find(p, key)); }

int policy_admit(void *arena, uint64_t key) {
    P *p = arena; p->now++;
    Ent *e = find(p, key);
    if (!e) { e = add(p, key); e->last = (uint32_t)p->now; e->hseq = 0; return 1; }
    touch(p, e); e->hseq = 0; /* no longer in history */
    return 1;
}

static uint64_t rnd(P *p) { p->rng ^= p->rng << 13; p->rng ^= p->rng >> 7; p->rng ^= p->rng << 17; return p->rng; }

uint64_t policy_victim(void *arena) {
    P *p = arena; uint64_t best = 0; double bs = -1e300;
    int samples = p->n < 64 ? (int)p->n : 64;
    for (int s = 0; s < samples; s++) {
        uint64_t key = p->keys[rnd(p) % p->n]; Ent *e = find(p, key);
        double gap = e->gap ? e->gap : (p->d0 > 0 ? p->d0 : (double)p->cap);
        double age = (double)(p->now - e->last);
        double sc = age > 2 * gap ? 1e18 + age : gap - age; /* dead first, else the furthest predicted next access */
        if (sc > bs) { bs = sc; best = key; }
    }
    Ent *e = find(p, best);
    uint32_t pos = e->pos; p->keys[pos] = p->keys[--p->n]; if (pos < p->n) find(p, p->keys[pos])->pos = pos;
    e->pos = NOPOS;
    /* into history */
    if (p->hhead - p->htail >= p->hcap) {
        uint64_t hp = p->htail++ % p->hcap; Ent *o = find(p, p->hk[hp]);
        if (o && o->pos == NOPOS && o->hseq == p->hs[hp]) del(p, o->key);
    }
    e = find(p, best); /* del may have moved it */
    e->hseq = (uint32_t)(p->hhead + 1); p->hk[p->hhead % p->hcap] = best; p->hs[p->hhead % p->hcap] = e->hseq; p->hhead++;
    return best;
}

void policy_insert(void *arena, uint64_t key) {
    P *p = arena; Ent *e = find(p, key);
    e->pos = (uint32_t)p->n; p->keys[p->n++] = key;
}
