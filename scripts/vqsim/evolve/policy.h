/* policy.h: the contract between a cache eviction policy and polsim (the driver).
 *
 * The driver owns the cache's contents and counts hits and misses; the policy keeps its own metadata and decides.
 * All the policy's state lives in one arena the driver allocates and zeroes: at most POLICY_BUDGET(capacity) bytes.
 * A policy uses no heap, no files and no large static data (the harness checks the object file).
 * Objects are identified by 64-bit keys (hashes of the trace's ids; never 0).
 *
 * Per request for `key`:
 *   cached     -> policy_hit(arena, key)
 *   not cached -> if (policy_admit(arena, key)) {            (0 = do not cache it: the request still misses)
 *                     while (the cache holds `capacity` objects) the driver evicts policy_victim(arena),
 *                         which must be a cached key (the driver checks);
 *                     policy_insert(arena, key);
 *                 }
 * The cache never holds more than `capacity` objects. Time is the request count; policies may keep their own clocks.
 *
 * Optional knob: a policy may define policy_knobs() (number of settings, 1..8) and policy_set_knob(arena, k). The
 * driver then runs a miniature copy of the policy per setting on a small spatial sample of the keys and, at regular
 * points, sets the live cache to the setting whose copy misses least so far. Setting 0 is the start (and the tie
 * winner). policy_set_knob is called on a running cache, so a setting must take effect on the existing contents (no
 * rebuild); it is also called once right after policy_init. */
#ifndef POLICY_H
#define POLICY_H
#include <stddef.h>
#include <stdint.h>
#define POLICY_BUDGET(capacity) ((size_t)256 * (size_t)(capacity) + ((size_t)1 << 20))
size_t policy_bytes(uint64_t capacity);
void policy_init(void *arena, uint64_t capacity);
void policy_hit(void *arena, uint64_t key);
int policy_admit(void *arena, uint64_t key);
uint64_t policy_victim(void *arena);
void policy_insert(void *arena, uint64_t key);
int policy_knobs(void);                    /* optional */
void policy_set_knob(void *arena, int k);  /* optional */
#endif
