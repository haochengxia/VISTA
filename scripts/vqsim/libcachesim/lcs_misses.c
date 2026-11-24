/* lcs_misses: exact miss counts and run time of libCacheSim algorithms on an oracleGeneral trace,
 * unit object size (for checking OnlineV1 against vqonline and timing it next to other algorithms).
 *   cc -O2 -I<libCacheSim>/libCacheSim/include $(pkg-config --cflags glib-2.0) lcs_misses.c <libCacheSim>/_build/liblibCacheSim.a \
 *      $(pkg-config --libs glib-2.0) -lzstd -lstdc++ -lm -lpthread -ldl -o lcs_misses
 *   lcs_misses <trace.oracleGeneral> <capacity> <algo>[:<params>] ...
 * Output, one line per algorithm: <algo> <capacity> <requests> <misses> <seconds> */
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "libCacheSim/cache.h"
#include "libCacheSim/evictionAlgo.h"
#include "libCacheSim/reader.h"

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: lcs_misses <trace.oracleGeneral> <capacity> <algo>[:<params>] ...\n"); return 2; }
  reader_init_param_t rp = default_reader_init_params();
  rp.ignore_obj_size = true;
  long capacity = atol(argv[2]);
  for (int a = 3; a < argc; a++) {
    char name[256]; strncpy(name, argv[a], sizeof name - 1); name[sizeof name - 1] = 0;
    char *params = strchr(name, ':'); if (params) *params++ = 0;
    reader_t *reader = open_trace(argv[1], ORACLE_GENERAL_TRACE, &rp);
    common_cache_params_t cp = default_common_cache_params();
    cp.cache_size = capacity;
    cache_t *cache = create_cache_by_name(name, cp, params);
    if (!cache) { fprintf(stderr, "unknown algorithm %s\n", name); return 1; }
    request_t *req = new_request();
    long n = 0, misses = 0;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    while (read_one_req(reader, req) == 0) { n++; if (!cache->get(cache, req)) misses++; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("%s %ld %ld %ld %.3f\n", argv[a], capacity, n, misses, (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
    fflush(stdout);
    free_request(req); cache->cache_free(cache); close_trace(reader);
  }
  return 0;
}
