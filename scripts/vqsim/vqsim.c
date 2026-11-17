/* vqsim: a fast simulator for a family of FIFO-queue cache designs ("virtual queues").
 *
 * An object lives in one of K FIFO stages. Hits never move objects (lazy promotion); they only
 * raise a saturating counter, and only in stages that count. When the cache is full, the first
 * stage (in index order) at or over its target size is scanned from its tail: an object whose
 * counter reaches the stage's threshold goes "up" (to any stage, itself = reinsertion), with its
 * counter reset, kept, decremented or halved; otherwise it goes "down" to another stage or leaves
 * the cache (into a ghost FIFO of keys if the stage feeds it). The scan stays on that stage until
 * an object leaves or one is demoted (then the stage is picked again). New objects enter one
 * stage, ghost hits another. S3-FIFO is K=2:
 *   S: 10%, counts, freq>=2 -> M (reset), else evict to ghost;  M: rest, freq>=1 -> M (dec), else evict.
 * The sandbox's vq.py implements the same semantics in Python (tests check they agree exactly).
 *
 * Input: a trace of dense object ids (uint32, 0..n_obj-1) and a jobs file, one job per line:
 *   <capacity> <split> <windows> <config>
 * config: space-separated key=value: maxf, entry, overflow, ghost (fraction of capacity), gentry,
 *   gfreq, and per stage st=<size>,<count>,<promote>,<up>,<onup>,<down>,<feeds_ghost>[,<delay>[,<age>]]
 *   delay (D-FR, VLDB'26): a hit counts only if more than delay*C insertions passed since the last counted one;
 *   age (AGE, VLDB'26): at the tail, an object idle for at least age*C insertions is not promoted.
 *   Time is the insertion clock: an object crosses a FIFO cache in about C insertions.
 *   (size: fraction; the last stage's size is the remainder; onup 0 reset 1 keep 2 dec 3 half;
 *   down -1 = leave the cache).
 *   Run options (also key=value in the config):
 *   handover=1  S3-FIFO (default knobs) serves requests [0, split), then this design takes over its
 *               live cache, as vq.py's adopt does: S objects -> stage `asmall` (default entry),
 *               M objects -> stage `amain` (default the last), counters kept (capped; 0 if the stage
 *               does not count), the ghost's newest entries kept.
 *   until=N     stop after N requests (e.g. the observed prefix only); windows then split [0, N).
 * Output per job: <misses before split> <misses after split> <window misses...>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define MAXK 8
typedef struct { double size, delay, age; int count, promote, up, onup, down, feeds; } Stage;
typedef struct { int k, maxf, entry, overflow, gentry, gfreq, handover, asmall, amain; uint64_t until; double ghost; Stage st[MAXK]; } Config;

/* All per-object state in one 16-byte record: a request touches one cache line. */
typedef struct { uint32_t gstamp, counted, touched; int8_t stage; uint8_t freq; uint16_t pad; } Obj;

static int parse(const char *s, Config *c) {
    memset(c, 0, sizeof *c);
    c->maxf = 3; c->entry = 0; c->overflow = -1; c->gentry = -1; c->gfreq = 0; c->ghost = 0.9; c->asmall = -1; c->amain = -1;
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
    /* Moves that need no hit and keep the counter (a demotion, a pass with promote 0, a promotion
     * with onup=keep) must not form a cycle, or objects could circle without ever leaving. */
    int edge[MAXK][MAXK] = {{0}};
    for (int i = 0; i < c->k; i++) {
        const Stage *x = &c->st[i];
        if (x->down >= 0) edge[i][x->down] = 1;
        if (x->up != i && (x->promote == 0 || x->onup == 1)) edge[i][x->up] = 1;
    }
    for (int m = 0; m < c->k; m++) for (int i = 0; i < c->k; i++) for (int j = 0; j < c->k; j++) if (edge[i][m] && edge[m][j]) edge[i][j] = 1;
    for (int i = 0; i < c->k; i++) if (edge[i][i]) return -1;
    return 0;
}

typedef struct { uint32_t *buf; uint64_t head, tail, cap; } Ring; /* push at head, pop at tail */
static void rpush(Ring *r, uint32_t v) { r->buf[r->head++ % r->cap] = v; }
static uint32_t rpop(Ring *r) { return r->buf[r->tail++ % r->cap]; }
static uint64_t rlen(const Ring *r) { return r->head - r->tail; }
static uint32_t rat(const Ring *r, uint64_t i) { return r->buf[(r->tail + i) % r->cap]; } /* i-th from the tail */

typedef struct {
    const Config *c; uint64_t capacity, target[MAXK], delay[MAXK], age[MAXK], gcap, glive, resident;
    uint32_t clock, gnum; int evicted, k;
    Ring q[MAXK], g; uint32_t *gseq; /* ghost ring of keys, and each slot's entry number */
} Sim;

static void sim_init(Sim *s, const Config *c, uint64_t capacity) {
    memset(s, 0, sizeof *s);
    s->c = c; s->k = c->k; s->capacity = capacity;
    double used = 0;
    for (int i = 0; i < c->k; i++) {
        s->q[i].cap = capacity + 1; s->q[i].buf = malloc(s->q[i].cap * sizeof(uint32_t));
        if (i < c->k - 1) { s->target[i] = (uint64_t)(capacity * c->st[i].size); used += (double)s->target[i]; }
        s->delay[i] = (uint64_t)(c->st[i].delay * capacity); s->age[i] = (uint64_t)(c->st[i].age * capacity);
    }
    s->target[c->k - 1] = capacity > (uint64_t)used ? capacity - (uint64_t)used : 0;
    s->gcap = (uint64_t)(capacity * c->ghost);
    s->g.cap = 2 * s->gcap + 16; s->g.buf = malloc(s->g.cap * sizeof(uint32_t)); s->gseq = malloc(s->g.cap * sizeof(uint32_t));
}

static void sim_free(Sim *s) {
    for (int i = 0; i < s->k; i++) free(s->q[i].buf);
    free(s->g.buf); free(s->gseq);
}

static void ghost_add(Sim *s, Obj *o, uint32_t y) {
    Ring *g = &s->g;
    while (s->glive >= s->gcap) { uint64_t pos = g->tail; uint32_t z = rpop(g); if (o[z].gstamp == s->gseq[pos % g->cap] + 1) { o[z].gstamp = 0; s->glive--; } }
    if (rlen(g) >= g->cap) { /* compact stale entries */
        uint64_t w = 0, len = rlen(g); uint32_t *nb = malloc(g->cap * sizeof(uint32_t)), *ns = malloc(g->cap * sizeof(uint32_t));
        for (uint64_t i = 0; i < len; i++) { uint64_t pos = (g->tail + i) % g->cap; uint32_t z = g->buf[pos]; if (o[z].gstamp == s->gseq[pos] + 1) { nb[w] = z; ns[w] = s->gseq[pos]; w++; } }
        free(g->buf); free(s->gseq); g->buf = nb; s->gseq = ns; g->tail = 0; g->head = w;
    }
    s->gseq[g->head % g->cap] = s->gnum; o[y].gstamp = s->gnum + 1; s->gnum++; rpush(g, y); s->glive++;
}

static int pick(const Sim *s) {
    for (int i = 0; i < s->k; i++) if (rlen(&s->q[i]) >= s->target[i] && rlen(&s->q[i]) > 0) return i;
    for (int i = 0; i < s->k; i++) if (rlen(&s->q[i]) > 0) return i;
    return -1;
}

static void evict_one(Sim *s, Obj *o) {
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
        if (guard >= 0) {
            int idle = s->age[k] && s->clock - p->touched >= s->age[k];
            if (f >= st->promote && !idle && !(st->promote == 0 && st->up == k)) {
                int to = st->up;
                if (st->onup == 0) p->freq = 0; else if (st->onup == 2) p->freq = f > 0 ? f - 1 : 0; else if (st->onup == 3) p->freq = f >> 1;
                p->stage = (int8_t)to; rpush(&s->q[to], y);
                continue;
            }
            if (st->down >= 0) { /* demotion: the receiving stage may now be over its target, pick again */
                p->stage = (int8_t)st->down; rpush(&s->q[st->down], y);
                k = pick(s);
                continue;
            }
        }
        p->stage = -1; p->freq = 0; s->resident--; /* leave the cache */
        if (st->feeds && s->gcap > 0 && !p->gstamp) ghost_add(s, o, y);
        return;
    }
}

/* One request; returns 1 on a miss. */
static inline int step(Sim *s, Obj *o, uint32_t x) {
    const Config *c = s->c;
    Obj *p = &o[x];
    if (p->stage >= 0) { /* hit */
        int sx = p->stage;
        if (c->st[sx].count && p->freq < c->maxf && (!s->delay[sx] || s->clock - p->counted > s->delay[sx])) { p->freq++; p->counted = s->clock; }
        p->touched = s->clock;
        return 0;
    }
    int ghost_hit = 0;
    if (p->gstamp) { p->gstamp = 0; s->glive--; ghost_hit = 1; }
    if (s->capacity == 0) return 1;
    while (s->resident >= s->capacity) evict_one(s, o);
    int to = ghost_hit ? c->gentry : c->entry;
    if (!ghost_hit && !s->evicted && rlen(&s->q[to]) >= s->target[to]) to = c->overflow;
    p->stage = (int8_t)to; p->freq = ghost_hit ? (uint8_t)c->gfreq : 0; rpush(&s->q[to], x); s->resident++;
    s->clock++; p->counted = p->touched = s->clock;
    return 1;
}

/* Online mode only: where a takeover puts the old cache's objects. 0 = by stage (objects of the
 * old entry stage to asmall, all others to amain); 1 = by counter (objects hit since they entered
 * or last moved to amain, the others to asmall); 2 = all to amain. Modes 1 and 2 place the old
 * stages from the last to the first, so the old main's objects end up oldest. */
static int takeover = 0;
static int takeover_to(const Config *c, int from_entry, int freq) {
    if (takeover == 1) return freq > 0 ? c->amain : c->asmall;
    if (takeover == 2) return c->amain;
    return from_entry ? c->asmall : c->amain;
}

/* The design in `b` takes over S3-FIFO's live cache in `a` (vq.py's adopt). */
static void adopt(Sim *b, const Sim *a, Obj *o) {
    const Config *c = b->c;
    for (int k = 0; k < 2; k++) {
        int src = takeover ? 1 - k : k;
        for (uint64_t i = 0; i < rlen(&a->q[src]); i++) {
            uint32_t y = rat(&a->q[src], i);
            Obj *p = &o[y];
            int to = takeover_to(c, src == 0, p->freq);
            p->stage = (int8_t)to; p->freq = c->st[to].count ? (uint8_t)(p->freq < c->maxf ? p->freq : c->maxf) : 0;
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
        if (seen++ >= drop) { b->gseq[b->g.head % b->g.cap] = b->gnum; o[z].gstamp = b->gnum + 1; b->gnum++; rpush(&b->g, z); b->glive++; }
    }
}

static const char *S3_DEFAULT = "maxf=3 entry=0 overflow=1 ghost=0.9 gentry=1 gfreq=0 st=0.1,1,2,1,0,-1,1 st=0,1,1,1,2,-1,0";

/* The incumbent's state at the split, kept for consecutive handover jobs of the same cell. */
static struct { uint64_t capacity, split; Obj *o; Sim sim; Config cfg; uint64_t before; uint64_t *win; int windows; int valid; } snap;

static void run(const uint32_t *ids, uint64_t n, uint32_t n_obj, uint64_t capacity, uint64_t split, int windows, const Config *c, Obj *o, FILE *out) {
    uint64_t end = c->until && c->until < n ? c->until : n;
    if (split > end) split = end;
    takeover = 0; /* handover jobs always use the stage mapping */
    uint64_t before = 0, after = 0;
    uint64_t *win = calloc(windows > 0 ? windows : 1, sizeof(uint64_t));
    Sim s;
    uint64_t t0 = 0;
    if (c->handover && split > 0) {
        if (!(snap.valid && snap.capacity == capacity && snap.split == split && snap.windows == windows)) {
            if (snap.valid) { sim_free(&snap.sim); free(snap.win); }
            parse(S3_DEFAULT, &snap.cfg);
            Sim a; sim_init(&a, &snap.cfg, capacity);
            for (uint32_t i = 0; i < n_obj; i++) { o[i] = (Obj){0}; o[i].stage = -1; }
            uint64_t *w = calloc(windows > 0 ? windows : 1, sizeof(uint64_t)), b = 0;
            for (uint64_t t = 0; t < split; t++) if (step(&a, o, ids[t])) { b++; if (windows > 0) w[(t * (uint64_t)windows) / end]++; }
            if (!snap.o) snap.o = malloc((uint64_t)n_obj * sizeof(Obj));
            memcpy(snap.o, o, (uint64_t)n_obj * sizeof(Obj));
            snap.sim = a; snap.sim.c = &snap.cfg; snap.capacity = capacity; snap.split = split; snap.windows = windows; snap.before = b; snap.win = w; snap.valid = 1;
        } else memcpy(o, snap.o, (uint64_t)n_obj * sizeof(Obj));
        before = snap.before;
        for (int w = 0; w < windows; w++) win[w] = snap.win[w];
        sim_init(&s, c, capacity);
        adopt(&s, &snap.sim, o);
        t0 = split;
    } else {
        for (uint32_t i = 0; i < n_obj; i++) { o[i] = (Obj){0}; o[i].stage = -1; }
        sim_init(&s, c, capacity);
    }
    for (uint64_t t = t0; t < end; t++) {
        if (step(&s, o, ids[t])) {
            if (t < split) before++; else after++;
            if (windows > 0) win[(t * (uint64_t)windows) / end]++;
        }
    }
    fprintf(out, "%lu %lu", (unsigned long)before, (unsigned long)after);
    for (int w = 0; w < windows; w++) fprintf(out, " %lu", (unsigned long)win[w]);
    fputc('\n', out);
    free(win); sim_free(&s);
}

/* -- online re-choice: the live cache switches design while it runs ----------------------------
 * A block in the jobs file:
 *   online <capacity> <split> <windows> [rate=R] [min=M] [margin=X] [horizon=H] [start=W] [schedule=...]
 *   <config of candidate 0>
 *   ...
 *   end
 * S3-FIFO serves [0, split); at the split the live cache takes the first design and, at the start
 * of each later window, the leader. Without a schedule the leader is chosen by miniature
 * simulations: each candidate runs on a spatial sample of the objects (rate R, raised so that the
 * miniature cache holds at least M objects; R=1 is an exact shadow), from a cold start; the leader
 * has the fewest sampled misses so far from window 2 on (or over the last H windows), and the live
 * cache leaves its design only for a leader at least X (relative) better; with start=W the
 * policy runs from window W instead of taking over at the split. With a schedule (window:candidate pairs,
 * in order; "s:i" applies at the split, "w:i" at the start of window w, which may come before the
 * split: the policy then runs from there instead of taking over at the split) the switches are
 * given, which isolates the cost of switching.
 * A switch is a takeover between designs: objects of the old entry stage go to the new design's
 * asmall stage, all others to its amain stage (stage by stage, tail first), counters capped (0 in
 * a stage that does not count), hit times kept, the ghost's newest entries kept.
 * Output: <misses before split> <misses after split> <switches> <window misses...> */

static void switch_design(Sim *b, const Sim *a, Obj *o) {
    const Config *c = b->c;
    for (int k = 0; k < a->k; k++) {
        int i = takeover ? a->k - 1 - k : k;
        for (uint64_t j = 0; j < rlen(&a->q[i]); j++) {
            uint32_t y = rat(&a->q[i], j);
            Obj *p = &o[y];
            int to = takeover_to(c, i == a->c->entry, p->freq);
            p->stage = (int8_t)to; p->freq = c->st[to].count ? (uint8_t)(p->freq < c->maxf ? p->freq : c->maxf) : 0;
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
    }
}

static uint64_t mix64(uint64_t x) { x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31); }

static void run_online(const uint32_t *ids, uint64_t n, uint32_t n_obj, char *head, char **cfgs, int K, Obj *o, FILE *out) {
    unsigned long capacity, split; int windows, off = 0;
    double rate = 0.01, margin = 0; uint64_t min_mini = 1000; int sched_w[256], sched_c[256], ns = 0, start = -1, horizon = 0;
    takeover = 0;
    if (sscanf(head, "online %lu %lu %d %n", &capacity, &split, &windows, &off) < 3 || windows < 9 || K < 1) { fprintf(out, "error bad-online\n"); return; }
    for (char *tok = strtok(head + off, " \t\n"); tok; tok = strtok(NULL, " \t\n")) {
        if (!strncmp(tok, "rate=", 5)) rate = atof(tok + 5);
        else if (!strncmp(tok, "min=", 4)) min_mini = strtoull(tok + 4, NULL, 10);
        else if (!strncmp(tok, "margin=", 7)) margin = atof(tok + 7);
        else if (!strncmp(tok, "start=", 6)) start = atoi(tok + 6);
        else if (!strncmp(tok, "horizon=", 8)) horizon = atoi(tok + 8);
        else if (!strncmp(tok, "takeover=", 9)) takeover = atoi(tok + 9);
        else if (!strncmp(tok, "schedule=", 9)) {
            for (char *q = tok + 9; *q && ns < 256;) {
                int w, i, used = 0;
                if (*q == 's') { if (sscanf(q + 1, ":%d%n", &i, &used) < 1) break; w = -1; used += 1; } /* s:i = at the split */
                else if (sscanf(q, "%d:%d%n", &w, &i, &used) < 2) break;
                sched_w[ns] = w; sched_c[ns] = i; ns++; q += used; if (*q == ',') q++;
            }
        } else { fprintf(out, "error bad-online\n"); return; }
    }
    Config *cand = calloc(K, sizeof(Config));
    for (int i = 0; i < K; i++) if (parse(cfgs[i], &cand[i])) { fprintf(out, "error bad-config %d\n", i); free(cand); return; }
    for (int i = 0; i < ns; i++) if (sched_c[i] < 0 || sched_c[i] >= K) { fprintf(out, "error bad-online\n"); free(cand); return; }
    /* miniature simulations (none with a schedule) */
    int mini = ns == 0;
    double r = rate; if (mini && r * capacity < (double)min_mini) r = (double)min_mini / (double)capacity; if (r > 1) r = 1;
    int32_t *smap = NULL; uint32_t n_s = 0; Obj **mo = NULL; Sim *ms = NULL; uint64_t *mwin = NULL;
    uint64_t mcap = (uint64_t)(r * capacity + 0.5); if (mcap < 1) mcap = 1;
    if (mini) {
        smap = malloc((uint64_t)n_obj * sizeof(int32_t));
        uint64_t limit = (uint64_t)(r * 18446744073709551615.0);
        for (uint32_t x = 0; x < n_obj; x++) smap[x] = (r >= 1 || mix64(x) <= limit) ? (int32_t)n_s++ : -1;
        mo = malloc(K * sizeof(Obj *)); ms = malloc(K * sizeof(Sim)); mwin = calloc((uint64_t)K * windows, sizeof(uint64_t));
        for (int i = 0; i < K; i++) {
            mo[i] = malloc((uint64_t)(n_s ? n_s : 1) * sizeof(Obj));
            for (uint32_t x = 0; x < n_s; x++) { mo[i][x] = (Obj){0}; mo[i][x].stage = -1; }
            sim_init(&ms[i], &cand[i], mcap);
        }
    }
    Config s3; parse(S3_DEFAULT, &s3);
    for (uint32_t i = 0; i < n_obj; i++) { o[i] = (Obj){0}; o[i].stage = -1; }
    Sim live; sim_init(&live, &s3, capacity);
    int current = -1, switches = 0, next_sched = 0;
    uint64_t before = 0, after = 0, *win = calloc(windows, sizeof(uint64_t));
    int w_prev = -1;
    for (uint64_t t = 0; t < n; t++) {
        int w = (int)((t * (uint64_t)windows) / n);
        int point = ns ? (t == split || w != w_prev)
                       : start >= 0 ? (w != w_prev && w >= start) : (t == split || (t > split && w != w_prev));
        if (point) { /* a decision point: the split, then each window start (a schedule may start earlier) */
            int leader = current;
            if (ns) {
                while (next_sched < ns && ((sched_w[next_sched] < 0 && t >= split) || (sched_w[next_sched] >= 0 && sched_w[next_sched] <= w && w != w_prev)))
                    leader = sched_c[next_sched++];
            } else { /* fewest sampled misses so far (from window 2, or over the last `horizon` windows) */
                uint64_t best = UINT64_MAX, mine = UINT64_MAX;
                int lo = horizon > 0 && w - horizon > 2 ? w - horizon : 2;
                for (int i = 0; i < K; i++) {
                    uint64_t m = 0; for (int v = lo; v <= w; v++) m += mwin[(uint64_t)i * windows + v];
                    if (m < best) { best = m; leader = i; }
                    if (i == current) mine = m;
                }
                /* hysteresis: leave the current design only for a leader at least `margin` better */
                if (current >= 0 && leader != current && (double)best >= (1.0 - margin) * (double)mine) leader = current;
            }
            if (leader >= 0 && leader != current) {
                Sim b; sim_init(&b, &cand[leader], capacity);
                if (current < 0) adopt(&b, &live, o); else switch_design(&b, &live, o);
                if (current >= 0) switches++;
                sim_free(&live); live = b; current = leader;
            }
        }
        w_prev = w;
        uint32_t x = ids[t];
        if (step(&live, o, x)) { if (t < split) before++; else after++; win[w]++; }
        if (mini && smap[x] >= 0) for (int i = 0; i < K; i++) if (step(&ms[i], mo[i], (uint32_t)smap[x])) mwin[(uint64_t)i * windows + w]++;
    }
    takeover = 0; /* handover jobs (run) keep the stage mapping */
    fprintf(out, "%lu %lu %d", (unsigned long)before, (unsigned long)after, switches);
    for (int w = 0; w < windows; w++) fprintf(out, " %lu", (unsigned long)win[w]);
    fputc('\n', out);
    sim_free(&live); free(win);
    if (mini) { for (int i = 0; i < K; i++) { free(mo[i]); sim_free(&ms[i]); } free(mo); free(ms); free(mwin); free(smap); }
    free(cand);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: vqsim <ids.u32> <jobs>\n"); return 2; }
    int fd = open(argv[1], O_RDONLY); struct stat sb; if (fd < 0 || fstat(fd, &sb)) { perror(argv[1]); return 1; }
    uint64_t n = sb.st_size / 4; const uint32_t *ids = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint32_t n_obj = 0; for (uint64_t i = 0; i < n; i++) if (ids[i] + 1 > n_obj) n_obj = ids[i] + 1;
    Obj *o = malloc((uint64_t)(n_obj ? n_obj : 1) * sizeof(Obj));
    FILE *jobs = fopen(argv[2], "r"); if (!jobs) { perror(argv[2]); return 1; }
    char line[8192];
    while (fgets(line, sizeof line, jobs)) {
        if (!strncmp(line, "online", 6)) { /* a block: header, candidate configs, end */
            char head[8192]; strncpy(head, line, sizeof head - 1); head[sizeof head - 1] = 0;
            char **cfgs = NULL; int K = 0;
            while (fgets(line, sizeof line, jobs) && strncmp(line, "end", 3)) { cfgs = realloc(cfgs, (K + 1) * sizeof(char *)); cfgs[K++] = strdup(line); }
            run_online(ids, n, n_obj, head, cfgs, K, o, stdout); fflush(stdout);
            for (int i = 0; i < K; i++) free(cfgs[i]);
            free(cfgs);
            continue;
        }
        unsigned long capacity, split; int windows, off = 0; Config c;
        if (sscanf(line, "%lu %lu %d %n", &capacity, &split, &windows, &off) < 3) continue;
        if (parse(line + off, &c)) { printf("error bad-config\n"); fflush(stdout); continue; }
        run(ids, n, n_obj, capacity, split, windows, &c, o, stdout); fflush(stdout);
    }
    return 0;
}
