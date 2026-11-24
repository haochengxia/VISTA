/* polsim: runs one cache policy (policy.h) cold over a whole trace of dense uint32 ids, as vqsim's cold runs do.
 *   cc -O2 -o polsim polsim.c policy.o
 *   polsim <ids.u32> <split> <windows> <capacity>...
 * Prints, per capacity: <misses before split> <misses after split> <window misses...> (vqsim's format), or
 * "error <why>" if the policy breaks the contract (arena over budget, a victim that is not cached).
 *
 * Knobs (optional): a policy with policy_knobs() = K > 1 settings is run with K miniature copies, one per setting, on
 * a spatial sample of the keys (rate 0.005, raised so a miniature holds at least 200 objects), as vqsim's online
 * selector does; the live cache starts at setting 0 and, at the start of each window from 3 on, is set to the setting
 * whose miniature has the fewest sampled misses from window 2 on (ties to the lower setting). POLSIM_KNOB=k instead
 * fixes the live cache at setting k (no miniatures). Knob changes go to stderr as "knobs <capacity> <changes> <final>". */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "policy.h"

static uint64_t mix64(uint64_t x) { x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31); }

/* the cache's contents: an open-addressing set of keys (0 = empty), backward-shift deletion */
typedef struct { uint64_t *k, mask, n; } Set;
static void set_init(Set *s, uint64_t cap) { uint64_t m = 16; while (m < 2 * cap + 2) m <<= 1; s->k = calloc(m, 8); s->mask = m - 1; s->n = 0; }
static uint64_t set_slot(const Set *s, uint64_t key) { uint64_t i = key & s->mask; while (s->k[i] && s->k[i] != key) i = (i + 1) & s->mask; return i; }
static int set_has(const Set *s, uint64_t key) { return s->k[set_slot(s, key)] == key; }
static void set_add(Set *s, uint64_t key) { s->k[set_slot(s, key)] = key; s->n++; }
static int set_del(Set *s, uint64_t key) {
    uint64_t i = set_slot(s, key);
    if (s->k[i] != key) return 0;
    s->k[i] = 0; s->n--;
    for (uint64_t j = (i + 1) & s->mask; s->k[j]; j = (j + 1) & s->mask) { /* shift back what the hole would cut off */
        uint64_t h = s->k[j] & s->mask;
        if (((j - h) & s->mask) >= ((j - i) & s->mask)) { s->k[i] = s->k[j]; s->k[j] = 0; i = j; }
    }
    return 1;
}

#pragma weak policy_knobs
#pragma weak policy_set_knob

/* one cache: the policy's arena and the contents the driver checks victims against */
typedef struct { void *arena; Set s; uint64_t cap; } Cache;
static int cache_open(Cache *c, uint64_t cap, size_t *need) {
    *need = policy_bytes(cap);
    if (*need > POLICY_BUDGET(cap)) return -1;
    size_t sz = (*need + 63) / 64 * 64 + 64;
    c->arena = aligned_alloc(64, sz); memset(c->arena, 0, sz); c->cap = cap;
    policy_init(c->arena, cap); set_init(&c->s, cap); return 0;
}
static void cache_close(Cache *c) { free(c->arena); free(c->s.k); }
/* one request: 0 hit, 1 miss, -1 a victim that is not cached */
static int cache_req(Cache *c, uint64_t key) {
    if (set_has(&c->s, key)) { policy_hit(c->arena, key); return 0; }
    if (!c->cap || !policy_admit(c->arena, key)) return 1;
    while (c->s.n >= c->cap) if (!set_del(&c->s, policy_victim(c->arena))) return -1;
    set_add(&c->s, key); policy_insert(c->arena, key); return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--knobs")) { printf("%d\n", policy_knobs ? policy_knobs() : 1); return 0; }
    if (argc < 5) { fprintf(stderr, "usage: polsim <ids.u32> <split> <windows> <capacity>... | polsim --knobs\n"); return 2; }
    int fd = open(argv[1], O_RDONLY); struct stat sb;
    if (fd < 0 || fstat(fd, &sb)) { perror(argv[1]); return 1; }
    uint64_t n = (uint64_t)sb.st_size / 4, split = strtoull(argv[2], NULL, 10); int windows = atoi(argv[3]);
    const uint32_t *ids = n ? mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : NULL;
    if (windows < 1 || (n && ids == MAP_FAILED)) { fprintf(stderr, "bad input\n"); return 2; }
    int K = policy_knobs ? policy_knobs() : 1; const char *fix = getenv("POLSIM_KNOB");
    if (K < 1 || K > 8 || (K > 1 && !policy_set_knob)) { printf("error knobs: policy_knobs() must be 1..8, with policy_set_knob\n"); return 0; }
    if (fix && (atoi(fix) < 0 || atoi(fix) >= K)) { fprintf(stderr, "POLSIM_KNOB out of range\n"); return 2; }
    for (int a = 4; a < argc; a++) {
        uint64_t cap = strtoull(argv[a], NULL, 10); size_t need;
        Cache live, mini[8]; int nm = fix || K == 1 ? 0 : K, cur = fix ? atoi(fix) : 0, changes = 0;
        if (cache_open(&live, cap, &need)) { printf("error budget %zu > %zu\n", need, POLICY_BUDGET(cap)); fflush(stdout); continue; }
        if (K > 1) policy_set_knob(live.arena, cur);
        double r = 0.005; if (r * (double)cap < 200) r = 200.0 / (double)cap; if (r > 1) r = 1;
        uint64_t mcap = (uint64_t)(r * (double)cap + 0.5), limit = (uint64_t)(r * 18446744073709551615.0); if (mcap < 1) mcap = 1;
        int bad = 0;
        for (int i = 0; i < nm && !bad; i++) {
            if (cache_open(&mini[i], mcap, &need)) { printf("error budget %zu > %zu (miniature)\n", need, POLICY_BUDGET(mcap)); bad = 3; nm = i; }
            else policy_set_knob(mini[i].arena, i);
        }
        if (bad) { for (int i = 0; i < nm; i++) cache_close(&mini[i]); cache_close(&live); fflush(stdout); continue; }
        uint64_t before = 0, after = 0, *win = calloc((size_t)windows, 8), score[8] = {0}; int w_prev = 0;
        for (uint64_t t = 0; t < n && !bad; t++) {
            uint64_t key = mix64(ids[t]); int w = (int)((t * (uint64_t)windows) / n);
            if (!key) { bad = 1; break; }
            if (nm && w != w_prev && w >= 3) { /* a decision point: fewest sampled misses from window 2 on */
                int best = 0; for (int i = 1; i < nm; i++) if (score[i] < score[best]) best = i;
                if (best != cur) { policy_set_knob(live.arena, best); cur = best; changes++; }
            }
            w_prev = w;
            if (nm && (r >= 1 || mix64(key) <= limit))
                for (int i = 0; i < nm; i++) { int m = cache_req(&mini[i], key); if (m < 0) { bad = 2; break; } if (w >= 2) score[i] += (uint64_t)m; }
            if (bad) break;
            int m = cache_req(&live, key);
            if (m < 0) { bad = 2; break; }
            if (m) { if (t < split) before++; else after++; win[w]++; }
        }
        if (bad) printf("error %s\n", bad == 1 ? "key" : "victim");
        else { printf("%lu %lu", (unsigned long)before, (unsigned long)after); for (int w = 0; w < windows; w++) printf(" %lu", (unsigned long)win[w]); putchar('\n'); }
        if (K > 1) fprintf(stderr, "knobs %lu %d %d\n", (unsigned long)cap, changes, cur);
        fflush(stdout); fflush(stderr);
        for (int i = 0; i < nm; i++) cache_close(&mini[i]);
        cache_close(&live); free(win);
    }
    return 0;
}
