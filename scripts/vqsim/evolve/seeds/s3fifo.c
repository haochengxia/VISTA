/* S3-FIFO in the policy.h contract, with vqsim's semantics (vqcore.h, S3_DEFAULT): small FIFO S (10%), main FIFO M,
 * ghost FIFO of keys (0.9 x capacity); counters capped at 3; S objects with >= 2 hits go to M (counter reset), others
 * leave into the ghost; M objects with >= 1 hit are reinserted (counter - 1), others leave; ghost hits enter M; before
 * the first eviction, new objects overflow into M once S is at its target. The scan stays on a stage until an object
 * leaves (re-picking only if the stage empties); after 4C + 16 moves without an eviction the next object leaves. */
#include <string.h>
#include "policy.h"

typedef struct { uint64_t key; uint32_t gseq; int8_t stage; uint8_t freq, ghost, pad; } Ent; /* metadata of a cached or ghost object */
typedef struct { uint64_t *buf, head, tail, cap; } Ring;                                   /* FIFO of keys: push at head, pop at tail */
typedef struct {
    uint64_t cap, target[2], gcap, glive, gnum, mask, gringcap, ghead, gtail;
    int evicted, ghost_hit;
    Ent *tab; Ring q[2]; uint64_t *gk, *gk2; uint32_t *gs, *gs2; /* ghost ring and its compaction buffer */
} S3;

static uint64_t rlen(const Ring *r) { return r->head - r->tail; }
static void rpush(Ring *r, uint64_t k) { r->buf[r->head++ % r->cap] = k; }
static uint64_t rpop(Ring *r) { return r->buf[r->tail++ % r->cap]; }

static uint64_t slots_for(uint64_t cap) { uint64_t m = 16, need = 2 * (cap + (uint64_t)((double)cap * 0.9) + 2); while (m < need) m <<= 1; return m; }
static uint64_t ringcap(uint64_t cap) { return 2 * (uint64_t)((double)cap * 0.9) + 16; }

size_t policy_bytes(uint64_t cap) {
    return sizeof(S3) + slots_for(cap) * sizeof(Ent) + 2 * (cap + 1) * 8 + 2 * ringcap(cap) * 12 + 64;
}

void policy_init(void *arena, uint64_t cap) {
    S3 *s = arena; char *p = (char *)arena + sizeof(S3);
    s->cap = cap; s->target[0] = (uint64_t)((double)cap * 0.1); s->target[1] = cap - s->target[0];
    s->gcap = (uint64_t)((double)cap * 0.9); s->gringcap = ringcap(cap);
    uint64_t m = slots_for(cap); s->mask = m - 1;
    s->tab = (Ent *)p; p += m * sizeof(Ent);
    for (int i = 0; i < 2; i++) { s->q[i].buf = (uint64_t *)p; s->q[i].cap = cap + 1; p += (cap + 1) * 8; }
    s->gk = (uint64_t *)p; p += s->gringcap * 8; s->gk2 = (uint64_t *)p; p += s->gringcap * 8;
    s->gs = (uint32_t *)p; p += s->gringcap * 4; s->gs2 = (uint32_t *)p;
}

static Ent *find(S3 *s, uint64_t key) {
    for (uint64_t i = key & s->mask;; i = (i + 1) & s->mask) { if (!s->tab[i].key) return NULL; if (s->tab[i].key == key) return &s->tab[i]; }
}
static Ent *add(S3 *s, uint64_t key) {
    uint64_t i = key & s->mask; while (s->tab[i].key) i = (i + 1) & s->mask;
    memset(&s->tab[i], 0, sizeof(Ent)); s->tab[i].key = key; s->tab[i].stage = -1; return &s->tab[i];
}
static void del(S3 *s, uint64_t key) { /* backward-shift deletion */
    uint64_t i = key & s->mask; while (s->tab[i].key != key) i = (i + 1) & s->mask;
    s->tab[i].key = 0;
    for (uint64_t j = (i + 1) & s->mask; s->tab[j].key; j = (j + 1) & s->mask) {
        uint64_t h = s->tab[j].key & s->mask;
        if (((j - h) & s->mask) >= ((j - i) & s->mask)) { s->tab[i] = s->tab[j]; s->tab[j].key = 0; i = j; }
    }
}

static void ghost_add(S3 *s, uint64_t key) {
    while (s->glive >= s->gcap) { /* drop the oldest live entry */
        uint64_t pos = s->gtail++ % s->gringcap; Ent *e = find(s, s->gk[pos]);
        if (e && e->ghost && e->gseq == s->gs[pos]) { e->ghost = 0; s->glive--; if (e->stage < 0) del(s, e->key); }
    }
    if (s->ghead - s->gtail >= s->gringcap) { /* compact stale entries into the other buffer, oldest first */
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

void policy_hit(void *arena, uint64_t key) {
    Ent *e = find(arena, key);
    if (e->freq < 3) e->freq++;
}

int policy_admit(void *arena, uint64_t key) {
    S3 *s = arena; Ent *e = find(s, key);
    s->ghost_hit = 0;
    if (e && e->ghost) { e->ghost = 0; s->glive--; s->ghost_hit = 1; } /* before any eviction, as vqsim does */
    return 1;
}

static int pick(const S3 *s) {
    for (int i = 0; i < 2; i++) if (rlen(&s->q[i]) >= s->target[i] && rlen(&s->q[i]) > 0) return i;
    for (int i = 0; i < 2; i++) if (rlen(&s->q[i]) > 0) return i;
    return -1;
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
        if (guard >= 0 && f >= (k == 0 ? 2 : 1)) { /* S: promote to M, reset; M: reinsert, decrement */
            e->freq = k == 0 ? 0 : (uint8_t)(f - 1); e->stage = 1; rpush(&s->q[1], key);
            continue;
        }
        e->stage = -1; e->freq = 0;
        if (k == 0 && s->gcap > 0 && !e->ghost) ghost_add(s, key); else if (!e->ghost) del(s, key);
        return key;
    }
}

void policy_insert(void *arena, uint64_t key) {
    S3 *s = arena; Ent *e = find(s, key); if (!e) e = add(s, key);
    int to = s->ghost_hit ? 1 : 0;
    if (!s->ghost_hit && !s->evicted && rlen(&s->q[0]) >= s->target[0]) to = 1;
    e->stage = (int8_t)to; e->freq = 0; rpush(&s->q[to], key); s->ghost_hit = 0;
}
