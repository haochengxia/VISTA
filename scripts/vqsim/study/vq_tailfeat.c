/* vq_tailfeat: runs S3-FIFO (vqsim's semantics) over a trace of dense uint32 ids and records, for a sample of the
 * decisions at the queues' tails from the split on, what the cache knows about the object and when it is next
 * requested (offline, from the trace). For studying what predicts reuse beyond S3-FIFO's counter.
 *   cc -O2 -o vq_tailfeat vq_tailfeat.c
 *   vq_tailfeat <ids.u32> <split> <capacity> <one-in-K sample> <out.bin>
 * Record (Rec below): stage (0 small, 1 main), S3-FIFO's counter, hits this pass, passes survived in main, entered main
 * from the ghost, requests so far for the object, now, insertion time, last access, the gap before the last access
 * (0 if none), main's pass length (EWMA), and the next request time (0xFFFFFFFF if none). */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct { uint8_t stage, freq, ph, r, g, pad[3]; uint32_t count, now, ins, last, gap, pass, next; } Rec;
typedef struct { uint32_t *buf; uint64_t head, tail, cap; } Ring;
static uint64_t rlen(const Ring *r) { return r->head - r->tail; }
static void rpush(Ring *r, uint32_t x) { r->buf[r->head++ % r->cap] = x; }
static uint32_t rpop(Ring *r) { return r->buf[r->tail++ % r->cap]; }

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: vq_tailfeat <ids.u32> <split> <capacity> <K> <out.bin>\n"); return 2; }
    int fd = open(argv[1], O_RDONLY); struct stat sb; if (fd < 0 || fstat(fd, &sb)) { perror(argv[1]); return 1; }
    uint64_t n = (uint64_t)sb.st_size / 4, split = strtoull(argv[2], NULL, 10), cap = strtoull(argv[3], NULL, 10), K = strtoull(argv[4], NULL, 10);
    const uint32_t *ids = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint32_t nobj = 0; for (uint64_t t = 0; t < n; t++) if (ids[t] >= nobj) nobj = ids[t] + 1;
    uint32_t *next = malloc(n * 4), *seen = malloc((size_t)nobj * 4);
    memset(seen, 0xFF, (size_t)nobj * 4);
    for (uint64_t t = n; t-- > 0;) { next[t] = seen[ids[t]]; seen[ids[t]] = (uint32_t)t; }
    free(seen);
    int8_t *stage = malloc(nobj); uint8_t *freq = calloc(nobj, 1), *ph = calloc(nobj, 1), *rr = calloc(nobj, 1), *gg = calloc(nobj, 1), *ghost = calloc(nobj, 1);
    uint32_t *count = calloc(nobj, 4), *ins = calloc(nobj, 4), *last = calloc(nobj, 4), *gap = calloc(nobj, 4), *gseq = calloc(nobj, 4), *ptime = calloc(nobj, 4);
    memset(stage, -1, nobj);
    uint64_t target0 = (uint64_t)((double)cap * 0.1), target[2] = {target0, cap - target0}, gcap = (uint64_t)((double)cap * 0.9);
    Ring q[2]; for (int i = 0; i < 2; i++) { q[i].cap = cap + 1; q[i].buf = malloc((cap + 1) * 4); q[i].head = q[i].tail = 0; }
    Ring gq = {malloc((4 * cap + 64) * 4), 0, 0, 4 * cap + 64}; uint32_t *gqs = malloc((4 * cap + 64) * 4); uint64_t glive = 0, gnum = 1;
    FILE *out = fopen(argv[5], "wb"); uint64_t rng = 0x9E3779B97F4A7C15ull, used = 0, misses = 0; int evicted = 0;
    double pass = (double)cap;
    for (uint64_t t = 0; t < n; t++) {
        uint32_t x = ids[t];
        if (stage[x] >= 0) { /* hit */
            if (freq[x] < 3) freq[x]++;
            if (ph[x] < 3) ph[x]++;
            gap[x] = (uint32_t)t - last[x]; last[x] = (uint32_t)t; count[x]++;
            continue;
        }
        misses++;
        int ghost_hit = 0;
        if (ghost[x]) { ghost[x] = 0; glive--; ghost_hit = 1; }
        if (count[x]) gap[x] = (uint32_t)t - last[x];
        last[x] = (uint32_t)t; count[x]++;
        if (!cap) continue;
        while (used >= cap) { /* evict one */
            evicted = 1;
            int k = -1; for (int i = 0; i < 2 && k < 0; i++) if (rlen(&q[i]) >= target[i] && rlen(&q[i]) > 0) k = i;
            for (int i = 0; i < 2 && k < 0; i++) if (rlen(&q[i]) > 0) k = i;
            int64_t guard = 4 * (int64_t)cap + 16;
            for (;;) {
                if (rlen(&q[k]) == 0) { k = -1; for (int i = 0; i < 2 && k < 0; i++) if (rlen(&q[i]) >= target[i] && rlen(&q[i]) > 0) k = i; for (int i = 0; i < 2 && k < 0; i++) if (rlen(&q[i]) > 0) k = i; }
                uint32_t y = rpop(&q[k]); int f = freq[y]; guard--;
                if (k == 1) pass += ((double)(t - ptime[y]) - pass) / 64;
                if (t >= split && K) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    if (rng % K == 0) {
                        Rec rec = {(uint8_t)k, (uint8_t)f, ph[y], rr[y], gg[y], {0}, count[y], (uint32_t)t, ins[y], last[y], gap[y], (uint32_t)pass, 0};
                        uint64_t nt = (uint64_t)t; uint32_t nx = 0xFFFFFFFFu; /* next request after now: walk from the last access */
                        for (uint32_t z = last[y]; z != 0xFFFFFFFFu; z = next[z]) if (z >= nt) { nx = z; break; }
                        rec.next = nx; fwrite(&rec, sizeof rec, 1, out);
                    }
                }
                if (guard >= 0 && f >= (k == 0 ? 2 : 1)) {
                    if (k == 0) { freq[y] = 0; rr[y] = 0; gg[y] = 0; } else { freq[y] = (uint8_t)(f - 1); if (rr[y] < 3) rr[y]++; }
                    stage[y] = 1; ph[y] = 0; ptime[y] = (uint32_t)t; rpush(&q[1], y); continue;
                }
                stage[y] = -1; freq[y] = 0; used--;
                if (k == 0 && gcap > 0 && !ghost[y]) {
                    while (glive >= gcap) { uint64_t p = gq.tail++ % gq.cap; uint32_t z = gq.buf[p]; if (ghost[z] && gseq[z] == gqs[p]) { ghost[z] = 0; glive--; } }
                    if (gq.head - gq.tail >= gq.cap) { /* compact */
                        uint64_t w = 0, len = gq.head - gq.tail; uint32_t *nb = malloc(gq.cap * 4), *ns = malloc(gq.cap * 4);
                        for (uint64_t i = 0; i < len; i++) { uint64_t p = (gq.tail + i) % gq.cap; uint32_t z = gq.buf[p]; if (ghost[z] && gseq[z] == gqs[p]) { nb[w] = z; ns[w] = gqs[p]; w++; } }
                        free(gq.buf); free(gqs); gq.buf = nb; gqs = ns; gq.tail = 0; gq.head = w;
                    }
                    ghost[y] = 1; gseq[y] = (uint32_t)gnum; gq.buf[gq.head % gq.cap] = y; gqs[gq.head % gq.cap] = (uint32_t)gnum; gq.head++; gnum++; glive++;
                }
                break;
            }
        }
        int to = ghost_hit ? 1 : 0;
        if (!ghost_hit && !evicted && rlen(&q[0]) >= target[0]) to = 1;
        stage[x] = (int8_t)to; freq[x] = 0; ph[x] = 0; ins[x] = (uint32_t)t; ptime[x] = (uint32_t)t; rr[x] = 0; gg[x] = (uint8_t)ghost_hit;
        rpush(&q[to], x); used++;
    }
    fclose(out);
    fprintf(stderr, "misses %lu\n", (unsigned long)misses);
    return 0;
}
