/* OnlineV1 for libCacheSim: CacheLab's version-1 policy (an online selector over 32 FIFO-based
 * designs, starting from S3-FIFO; sandbox/vq.py OnlineV1, docs/cachelab-v1.md) as an eviction
 * algorithm. The engine is scripts/vqsim/vqonline.h, exact with vq.py.
 *
 * The engine owns its own state, so the cache is a black box behind get(): libCacheSim's
 * simulator (bin/cachesim) only calls get(). find/insert/evict/to_evict/remove are not supported.
 * Objects are unit size: run with --ignore-obj-size 1; the cache size is a number of objects.
 *
 * Parameters (comma-separated key=value; defaults are OnlineV1's): period=500000, start=4,
 * from=2, rate=0.005, min=200, margin=0.
 *
 * Install into a libCacheSim checkout with scripts/vqsim/libcachesim/install.sh. */
#include "libCacheSim/evictionAlgo.h"

#include "vqonline.h"
#include "vqonline_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

static bool OnlineV1_get(cache_t *cache, const request_t *req) {
  VQOnline *e = (VQOnline *)cache->eviction_params;
  cache->n_req += 1;
  bool hit = vqo_access(e, (uint64_t)req->obj_id) == 0;
  cache->n_obj = (int64_t)vqo_resident(e);
  cache->occupied_byte = (int64_t)vqo_resident(e);
  return hit;
}

static int64_t OnlineV1_get_occupied_byte(const cache_t *cache) { return (int64_t)vqo_resident((const VQOnline *)cache->eviction_params); }
static int64_t OnlineV1_get_n_obj(const cache_t *cache) { return (int64_t)vqo_resident((const VQOnline *)cache->eviction_params); }

static void OnlineV1_unsupported(const char *what) {
  fprintf(stderr, "OnlineV1 supports get() only (called %s); it keeps its own state\n", what);
  abort();
}
static cache_obj_t *OnlineV1_find(cache_t *cache, const request_t *req, bool update_cache) {
  (void)cache; (void)req; (void)update_cache;
  OnlineV1_unsupported("find");
  return NULL;
}
static cache_obj_t *OnlineV1_insert(cache_t *cache, const request_t *req) {
  (void)cache; (void)req;
  OnlineV1_unsupported("insert");
  return NULL;
}
static cache_obj_t *OnlineV1_to_evict(cache_t *cache, const request_t *req) {
  (void)cache; (void)req;
  OnlineV1_unsupported("to_evict");
  return NULL;
}
static void OnlineV1_evict(cache_t *cache, const request_t *req) {
  (void)cache; (void)req;
  OnlineV1_unsupported("evict");
}
static bool OnlineV1_remove(cache_t *cache, const obj_id_t obj_id) {
  (void)cache; (void)obj_id;
  OnlineV1_unsupported("remove");
  return false;
}

static void OnlineV1_free(cache_t *cache) {
  VQOnline *e = (VQOnline *)cache->eviction_params;
  vqo_free(e);
  free(e);
  cache_struct_free(cache);
}

static void OnlineV1_parse_params(VQOParams *p, const char *params) {
  if (params == NULL || !*params) return;
  char *copy = strdup(params), *save = NULL;
  for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
    while (*tok == ' ') tok++;
    char *eq = strchr(tok, '=');
    if (eq == NULL) { fprintf(stderr, "OnlineV1: bad parameter %s\n", tok); abort(); }
    *eq = '\0';
    const char *key = tok, *value = eq + 1;
    if (!strcasecmp(key, "period")) p->period = strtoull(value, NULL, 10);
    else if (!strcasecmp(key, "start")) p->start = strtoull(value, NULL, 10);
    else if (!strcasecmp(key, "from")) p->from_window = strtoull(value, NULL, 10);
    else if (!strcasecmp(key, "rate")) p->rate = atof(value);
    else if (!strcasecmp(key, "min")) p->min_mini = strtoull(value, NULL, 10);
    else if (!strcasecmp(key, "margin")) p->margin = atof(value);
    else if (!strcasecmp(key, "print")) { printf("OnlineV1 parameters: period, start, from, rate, min, margin\n"); exit(0); }
    else { fprintf(stderr, "OnlineV1: unknown parameter %s\n", key); abort(); }
  }
  free(copy);
}

cache_t *OnlineV1_init(const common_cache_params_t ccache_params, const char *cache_specific_params) {
  cache_t *cache = cache_struct_init("OnlineV1", ccache_params, cache_specific_params);
  cache->cache_init = OnlineV1_init;
  cache->cache_free = OnlineV1_free;
  cache->get = OnlineV1_get;
  cache->find = OnlineV1_find;
  cache->insert = OnlineV1_insert;
  cache->evict = OnlineV1_evict;
  cache->remove = OnlineV1_remove;
  cache->to_evict = OnlineV1_to_evict;
  cache->get_occupied_byte = OnlineV1_get_occupied_byte;
  cache->get_n_obj = OnlineV1_get_n_obj;
  cache->can_insert = cache_can_insert_default;
  cache->obj_md_size = 0;

  VQOParams p = VQO_V1;
  OnlineV1_parse_params(&p, cache_specific_params);
  VQOnline *e = (VQOnline *)malloc(sizeof(VQOnline));
  int bad = vqo_init(e, VQO_V1_CANDIDATES, VQO_V1_K, (uint64_t)ccache_params.cache_size, &p);
  if (bad) { fprintf(stderr, "OnlineV1: cannot start (code %d; the cache size must be at least 1 object)\n", bad); abort(); }
  cache->eviction_params = e;
  return cache;
}

#ifdef __cplusplus
}
#endif
