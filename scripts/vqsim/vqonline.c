/* vqonline: run the deployable online selector (vqonline.h) over a trace of 64-bit keys.
 *
 *   cc -O2 -o vqonline scripts/vqsim/vqonline.c
 *   vqonline <keys.u64> <candidates.txt> <capacity> [period=N] [start=W] [from=W] [rate=R] [min=M] [margin=X]
 *
 * keys.u64: little-endian uint64 object ids, one per request. candidates.txt: one vqsim config per
 * line (blank lines and lines starting with # are skipped). Defaults are OnlineV1's.
 * Output: misses switches leader records peak_records mini_records peak_mini_records seconds live_bytes mini_bytes */
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#include "vqonline.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: vqonline <keys.u64> <candidates.txt> <capacity> [period= start= from= rate= min= margin=]\n"); return 2; }
    int fd = open(argv[1], O_RDONLY); struct stat sb;
    if (fd < 0 || fstat(fd, &sb)) { perror(argv[1]); return 1; }
    uint64_t n = sb.st_size / 8;
    const uint64_t *keys = n ? mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : NULL;
    FILE *f = fopen(argv[2], "r"); if (!f) { perror(argv[2]); return 1; }
    char line[8192], **cfgs = NULL; int K = 0;
    while (fgets(line, sizeof line, f)) {
        char *s = line; while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\n' || !*s) continue;
        cfgs = realloc(cfgs, (K + 1) * sizeof(char *)); cfgs[K++] = strdup(s);
    }
    fclose(f);
    uint64_t capacity = strtoull(argv[3], NULL, 10);
    VQOParams p = VQO_V1;
    for (int i = 4; i < argc; i++) {
        const char *a = argv[i];
        if (!strncmp(a, "period=", 7)) p.period = strtoull(a + 7, NULL, 10);
        else if (!strncmp(a, "start=", 6)) p.start = strtoull(a + 6, NULL, 10);
        else if (!strncmp(a, "from=", 5)) p.from_window = strtoull(a + 5, NULL, 10);
        else if (!strncmp(a, "rate=", 5)) p.rate = atof(a + 5);
        else if (!strncmp(a, "min=", 4)) p.min_mini = strtoull(a + 4, NULL, 10);
        else if (!strncmp(a, "margin=", 7)) p.margin = atof(a + 7);
        else { fprintf(stderr, "unknown option %s\n", a); return 2; }
    }
    VQOnline e;
    int bad = vqo_init(&e, (const char *const *)cfgs, K, capacity, &p);
    if (bad) { fprintf(stderr, bad < 0 ? "bad parameters\n" : "bad config on candidate %d\n", bad - 1); return 1; }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t misses = 0;
    for (uint64_t t = 0; t < n; t++) misses += (uint64_t)vqo_access(&e, keys[t]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t peak_mini = 0; for (int i = 0; i < e.K; i++) peak_mini += e.mt[i].peak;
    printf("%lu %lu %d %lu %lu %lu %lu %.3f %lu %lu\n", (unsigned long)misses, (unsigned long)e.switches, e.current,
           (unsigned long)vqo_records(&e), (unsigned long)e.lt.peak, (unsigned long)vqo_mini_records(&e), (unsigned long)peak_mini,
           (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec),
           (unsigned long)vqo_live_bytes(&e), (unsigned long)vqo_mini_bytes(&e));
    vqo_free(&e);
    for (int i = 0; i < K; i++) free(cfgs[i]);
    free(cfgs);
    return 0;
}
