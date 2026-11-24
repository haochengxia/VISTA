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
 *   gate=G      TinyLFU-style admission into stage G (vqcore.h; not in vqonline.h yet): a
 *               frequency sketch compares objects with G's tail object. gmode=0 (default) rescues
 *               objects leaving the cache from other stages, gmode=1 filters promotions into G,
 *               gmode=2 does both; gwin=W halves the sketch every W*C requests (default 10).
 *   rd=X        reuse-distance rule (vqcore.h, VQ_RD; vqsim only): a move up is refused when the object's
 *               predicted wait for its next request is over X times the residence time of the stage it
 *               would enter. rdmode (bitmask, default 3): 1 promotions, 2 reinsertions, 4 ghost admission
 *               (a long gap enters `entry` instead of `gentry`), 8 rescue (a near object moves up anyway).
 *               With rd, a handover keeps the objects' access times (shifted to the new clock).
 *   Online takeover=3 (VQ_RD): switches merge the objects bound for amain by when they entered their stage.
 * Output per job: <misses before split> <misses after split> <window misses...>
 */
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define VQ_RD
#include "vqcore.h"

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
    VQSketch sk = {0}; /* gate: fed every request from the start (the prefix too) */
    if (c->gate >= 0) { sk_init(&sk, capacity, c->gwin); for (uint64_t t = 0; t < t0; t++) sk_add(&sk, ids[t]); s.sk = &sk; }
    for (uint64_t t = t0; t < end; t++) {
        if (s.sk) sk_add(s.sk, ids[t]);
        if (step(&s, o, ids[t])) {
            if (t < split) before++; else after++;
            if (windows > 0) win[(t * (uint64_t)windows) / end]++;
        }
    }
    fprintf(out, "%lu %lu", (unsigned long)before, (unsigned long)after);
    for (int w = 0; w < windows; w++) fprintf(out, " %lu", (unsigned long)win[w]);
    fputc('\n', out);
    free(win); sim_free(&s); sk_free(&sk);
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
 * Gated candidates (gate=, which must share gwin) consult one frequency sketch fed every request
 * from the start (the live cache, whatever its design) and one fed the sampled requests (the
 * miniatures, sized to the miniature cache).
 * Pool screening (pool=FILE screen=M, miniatures only, not with start or a schedule): the selector also logs
 * the sampled requests of the prefix; at the split it replays that log through a miniature of every design
 * in FILE (one config per line; designs equal to a candidate are skipped), one at a time, and adds the M
 * with the fewest sampled misses from window 2 on (ties to the earlier line) as candidates K..K+M-1, warm,
 * before the split's choice. Gated pool designs must share the candidates' gwin.
 * logcap=N bounds the log to the last N sampled requests. If the prefix had more, the replay of those N
 * scores everyone alike: the first quarter warms the miniature up, misses after it count; the given
 * candidates are replayed too (for their score only), and the leader is then chosen by that score plus
 * the sampled misses from the split's window on. wrapmode=1 instead keeps the given candidates' history
 * (the leader rule from window 2, as with a full log) and gives each screened design the given leader's
 * per-window misses before the split's window scaled by (its replay score + 1) / (that leader's + 1).
 * trial=H (miniatures only, not with start or a schedule): switches are tried before they are made. A miniature
 * mirror of the live cache runs on the sampled requests from the start and takes over whenever the live cache does
 * (S3-FIFO, then the same designs at the same points), so it carries the live cache's path. It is snapshotted at the
 * last H decision points (before the split: H, ..., 1 window lengths before it). When the leader differs from the
 * live design, the oldest snapshot is replayed through the sampled requests since then as if the live cache had
 * switched to the leader there; the switch goes ahead only if that replay has fewer misses than the mirror had.
 * trialsplit=0: the first takeover (from S3-FIFO) is not tried, only the switches between designs after it.
 * reseed=1 (miniatures only, not with start or a schedule): at the split, after its choice, every miniature takes over
 * the mirror's cache (S3-FIFO's, sampled) as the live cache takes over S3-FIFO's, so from the split on each candidate's
 * sampled misses include what taking over costs it; the leader rule is unchanged.
 * replay=L (not with trial or reseed): the live cache logs its last L*C requests; at a takeover the next design
 * rebuilds its own state by replaying them and is reconciled with what is cached (replay_takeover), instead of the
 * stage mapping. VQSIM_CHECK=1 checks the live records after each takeover.
 * Output: <misses before split> <misses after split> <switches> <window misses...> */

/* Replays the logged sampled prefix (`kept` entries of a ring of `cap` slots, the oldest at `first`) through a
 * fresh miniature of `c` (state left in *s and o). Full log: every miss is added to win[its window]. Wrapped log:
 * misses of entries from offset `from` on, before window `wsplit` to *prior, in it to win[wsplit]. A gated design
 * gets its own sketch fed the same sampled requests, so a full replay ends equal to the shared miniature sketch;
 * the caller keeps the shared one. */
static void replay_prefix(const Config *c, uint64_t mcap, uint32_t n_s, const uint32_t *lid, const uint8_t *lw, uint64_t cap,
                          uint64_t first, uint64_t kept, int wrapped, int wsplit, double gwin, Sim *s, Obj *o, uint64_t *win, uint64_t *prior) {
    for (uint32_t x = 0; x < n_s; x++) { o[x] = (Obj){0}; o[x].stage = -1; }
    sim_init(s, c, mcap);
    VQSketch tsk = {0};
    if (c->gate >= 0) { sk_init(&tsk, mcap, gwin); s->sk = &tsk; }
    uint64_t from = wrapped ? kept / 4 : 0;
    for (uint64_t l = 0; l < kept; l++) {
        uint64_t j = (first + l) % cap;
        if (s->sk) sk_add(s->sk, lid[j]);
        if (step(s, o, lid[j]) && l >= from) {
            if (!wrapped || lw[j] >= wsplit) win[lw[j]]++; else (*prior)++;
        }
    }
    sk_free(&tsk); s->sk = NULL;
}


/* A deep copy of s into d (d's buffers are not freed). */
static void sim_copy(Sim *d, const Sim *s) {
    *d = *s;
    for (int i = 0; i < s->k; i++) { d->q[i].buf = malloc(s->q[i].cap * sizeof(uint32_t)); memcpy(d->q[i].buf, s->q[i].buf, s->q[i].cap * sizeof(uint32_t)); }
    d->g.buf = malloc(s->g.cap * sizeof(uint32_t)); memcpy(d->g.buf, s->g.buf, s->g.cap * sizeof(uint32_t));
    d->gseq = malloc(s->g.cap * sizeof(uint32_t)); memcpy(d->gseq, s->gseq, s->g.cap * sizeof(uint32_t));
}
/* replay takeover (online replay=L): the next design rebuilds its own state from the last L*C requests (`rl`, a ring of
 * `rlcap` ids, `rln` pushed so far) through a fresh instance with scratch records `ro` (all clear on entry and exit),
 * then it is reconciled with what is cached: replayed objects that are not cached go to its ghost (after the replay's
 * own uncached ghost entries), and cached objects it did not keep go to the old end of asmall (from the old entry
 * stage) or amain with counter 0. Its clock and the kept objects' times are the replay's. */
static void replay_takeover(Sim *live, Obj *o, const Config *D, const uint32_t *rl, uint64_t rlcap, uint64_t rln, Obj *ro) {
    uint64_t cap = live->capacity, kept = rln < rlcap ? rln : rlcap, first = rln < rlcap ? 0 : rln % rlcap;
    Sim b; sim_init(&b, D, cap); b.sk = live->sk;
    for (uint64_t l = 0; l < kept; l++) step(&b, ro, rl[(first + l) % rlcap]);
    for (uint64_t i = 0, len = rlen(&live->g); i < len; i++) { /* the old ghost is dropped */
        uint64_t pos = (live->g.tail + i) % live->g.cap; uint32_t z = live->g.buf[pos];
        if (o[z].gstamp == live->gseq[pos] + 1) o[z].gstamp = 0;
    }
    Sim nw; sim_init(&nw, D, cap); nw.sk = live->sk;
    for (int i = 0; i < live->k; i++) /* cached, not kept by the replay: oldest, counter 0 */
        for (uint64_t j = 0; j < rlen(&live->q[i]); j++) {
            uint32_t y = rat(&live->q[i], j);
            if (ro[y].stage >= 0) continue;
            int to = i == live->c->entry ? D->asmall : D->amain;
            Obj *p = &o[y]; p->stage = (int8_t)to; p->freq = 0; p->counted = p->touched = 0; p->gstamp = 0;
#ifdef VQ_RD
            p->gap = VQ_NOGAP; p->pushed = 0; p->rescued = 0;
#endif
            rpush(&nw.q[to], y);
        }
    uint32_t *drop = malloc((cap + 1) * sizeof *drop); uint64_t nd = 0;
    for (int i = 0; i < b.k; i++) /* the replay's cached objects, in its order */
        for (uint64_t j = 0; j < rlen(&b.q[i]); j++) {
            uint32_t y = rat(&b.q[i], j);
            if (o[y].stage >= 0) { /* cached: kept with the replay's record */
                Obj keep = ro[y]; keep.gstamp = 0; o[y] = keep; rpush(&nw.q[i], y);
            } else if (o[y].stage < 0) drop[nd++] = y;
        }
    for (uint64_t i = 0, len = rlen(&b.g); i < len; i++) { /* the replay's ghost, then what it held that is not cached */
        uint64_t pos = (b.g.tail + i) % b.g.cap; uint32_t z = b.g.buf[pos];
        if (ro[z].gstamp != b.gseq[pos] + 1 || o[z].stage >= 0 || nw.gcap == 0) continue;
        o[z].touched = ro[z].touched; o[z].gstamp = 0; ghost_add(&nw, o, z);
    }
    for (uint64_t i = 0; i < nd && nw.gcap > 0; i++) { uint32_t z = drop[i]; o[z].touched = ro[z].touched; o[z].gstamp = 0; ghost_add(&nw, o, z); }
    free(drop);
    nw.resident = live->resident; nw.evicted = live->evicted; nw.clock = b.clock;
    memcpy(nw.res, b.res, sizeof nw.res);
    for (uint64_t l = 0; l < kept; l++) { uint32_t y = rl[(first + l) % rlcap]; ro[y] = (Obj){0}; ro[y].stage = -1; }
    sim_free(&b); sim_free(live); *live = nw;
}
/* VQSIM_CHECK: the live cache's records agree with its queues and ghost */
static void check_live(const Sim *s, const Obj *o, uint32_t n_obj) {
    uint64_t res = 0;
    for (int i = 0; i < s->k; i++) for (uint64_t j = 0; j < rlen(&s->q[i]); j++) { uint32_t y = rat(&s->q[i], j); res++; if (o[y].stage != i) fprintf(stderr, "check failed: stage\n"); }
    if (res != s->resident) fprintf(stderr, "check failed: resident %lu %lu\n", (unsigned long)res, (unsigned long)s->resident);
    uint64_t gl = 0, inr = 0; for (uint32_t y = 0; y < n_obj; y++) { if (o[y].stage >= 0) inr++; if (o[y].gstamp) gl++; }
    if (inr != res) fprintf(stderr, "check failed: records %lu %lu\n", (unsigned long)inr, (unsigned long)res);
    if (gl != s->glive) fprintf(stderr, "check failed: ghost %lu %lu\n", (unsigned long)gl, (unsigned long)s->glive);
}
/* trial: the live mirror at a decision point; `off` is its first request in the trial buffer, `miss` the mirror's misses before it */
typedef struct { Sim s; Obj *o; VQSketch sk; uint64_t off, miss; int cur; } Snap;

static void run_online(const uint32_t *ids, uint64_t n, uint32_t n_obj, char *head, char **cfgs, int K, Obj *o, FILE *out) {
    unsigned long capacity, split; int windows, off = 0;
    double rate = 0.01, margin = 0; uint64_t min_mini = 1000; int sched_w[256], sched_c[256], ns = 0, start = -1, horizon = 0;
    char pool_path[4096] = ""; int screen = 0, wrapmode = 0, trial = 0, trialsplit = 1, reseed = 0; double replay = 0; uint64_t logcap = 0;
    takeover = 0;
    if (sscanf(head, "online %lu %lu %d %n", &capacity, &split, &windows, &off) < 3 || windows < 9 || K < 1) { fprintf(out, "error bad-online\n"); return; }
    for (char *tok = strtok(head + off, " \t\n"); tok; tok = strtok(NULL, " \t\n")) {
        if (!strncmp(tok, "rate=", 5)) rate = atof(tok + 5);
        else if (!strncmp(tok, "min=", 4)) min_mini = strtoull(tok + 4, NULL, 10);
        else if (!strncmp(tok, "margin=", 7)) margin = atof(tok + 7);
        else if (!strncmp(tok, "start=", 6)) start = atoi(tok + 6);
        else if (!strncmp(tok, "horizon=", 8)) horizon = atoi(tok + 8);
        else if (!strncmp(tok, "takeover=", 9)) takeover = atoi(tok + 9);
        else if (!strncmp(tok, "pool=", 5)) { strncpy(pool_path, tok + 5, sizeof pool_path - 1); }
        else if (!strncmp(tok, "screen=", 7)) screen = atoi(tok + 7);
        else if (!strncmp(tok, "logcap=", 7)) logcap = strtoull(tok + 7, NULL, 10);
        else if (!strncmp(tok, "wrapmode=", 9)) wrapmode = atoi(tok + 9);
        else if (!strncmp(tok, "trial=", 6)) trial = atoi(tok + 6);
        else if (!strncmp(tok, "trialsplit=", 11)) trialsplit = atoi(tok + 11);
        else if (!strncmp(tok, "reseed=", 7)) reseed = atoi(tok + 7);
        else if (!strncmp(tok, "replay=", 7)) replay = atof(tok + 7);
        else if (!strncmp(tok, "schedule=", 9)) {
            for (char *q = tok + 9; *q && ns < 256;) {
                int w, i, used = 0;
                if (*q == 's') { if (sscanf(q + 1, ":%d%n", &i, &used) < 1) break; w = -1; used += 1; } /* s:i = at the split */
                else if (sscanf(q, "%d:%d%n", &w, &i, &used) < 2) break;
                sched_w[ns] = w; sched_c[ns] = i; ns++; q += used; if (*q == ',') q++;
            }
        } else { fprintf(out, "error bad-online\n"); return; }
    }
    /* the pool to screen at the split (its designs equal to a candidate dropped) */
    Config *pcfg = NULL; int P = 0, *pline = NULL, fl = -1; /* pline: each kept pool design's line in FILE (diagnostics) */
    if (screen > 0 && (ns || start >= 0 || !pool_path[0] || windows > 255)) { fprintf(out, "error bad-online\n"); return; }
    if (trial < 0 || trial > 64 || ((trial > 0 || reseed) && (ns || start >= 0)) || replay < 0 || (replay > 0 && (trial || reseed))) { fprintf(out, "error bad-online\n"); return; }
    if (screen > 0) {
        FILE *pf = fopen(pool_path, "r"); if (!pf) { fprintf(out, "error bad-pool\n"); return; }
        char pl[8192];
        while (fgets(pl, sizeof pl, pf)) {
            fl++; size_t L = strlen(pl); while (L && (pl[L - 1] == '\n' || pl[L - 1] == ' ' || pl[L - 1] == '\r')) pl[--L] = 0;
            if (!L) continue;
            int dup = 0;
            for (int i = 0; i < K && !dup; i++) { size_t m = strlen(cfgs[i]); while (m && (cfgs[i][m - 1] == '\n' || cfgs[i][m - 1] == ' ' || cfgs[i][m - 1] == '\r')) m--; dup = m == L && !strncmp(cfgs[i], pl, L); }
            if (dup) continue;
            pcfg = realloc(pcfg, (P + 1) * sizeof(Config)); pline = realloc(pline, (P + 1) * sizeof(int)); pline[P] = fl;
            if (parse(pl, &pcfg[P])) { fprintf(out, "error bad-pool %d\n", P); fclose(pf); free(pcfg); return; }
            P++;
        }
        fclose(pf);
        if (screen > P) screen = P;
    }
    int K0 = K, KM = K + screen; /* candidates 0..K0-1 given; K0..KM-1 screened from the pool at the split */
    Config *cand = calloc(KM, sizeof(Config));
    double gwin = 0; /* gated candidates share one sketch of the request stream (live) and one of the sample (miniatures) */
    for (int i = 0; i < K; i++) {
        if (parse(cfgs[i], &cand[i]) || (cand[i].gate >= 0 && gwin > 0 && cand[i].gwin != gwin)) { fprintf(out, "error bad-config %d\n", i); free(cand); free(pcfg); return; }
        if (cand[i].gate >= 0) gwin = cand[i].gwin;
    }
    for (int j = 0; j < P; j++) {
        if (pcfg[j].gate >= 0 && gwin > 0 && pcfg[j].gwin != gwin) { fprintf(out, "error bad-pool %d\n", j); free(cand); free(pcfg); return; }
        if (pcfg[j].gate >= 0) gwin = pcfg[j].gwin;
    }
    for (int i = 0; i < ns; i++) if (sched_c[i] < 0 || sched_c[i] >= K) { fprintf(out, "error bad-online\n"); free(cand); free(pcfg); return; }
    /* miniature simulations (none with a schedule) */
    int mini = ns == 0;
    double r = rate; if (mini && r * capacity < (double)min_mini) r = (double)min_mini / (double)capacity; if (r > 1) r = 1;
    int32_t *smap = NULL; uint32_t n_s = 0; Obj **mo = NULL; Sim *ms = NULL; uint64_t *mwin = NULL;
    uint64_t mcap = (uint64_t)(r * capacity + 0.5); if (mcap < 1) mcap = 1;
    VQSketch lsk = {0}, msk = {0};
    if (gwin > 0) { sk_init(&lsk, capacity, gwin); if (mini) sk_init(&msk, mcap, gwin); }
    if (mini) {
        smap = malloc((uint64_t)n_obj * sizeof(int32_t));
        uint64_t limit = (uint64_t)(r * 18446744073709551615.0);
        for (uint32_t x = 0; x < n_obj; x++) smap[x] = (r >= 1 || mix64(x) <= limit) ? (int32_t)n_s++ : -1;
        mo = malloc(KM * sizeof(Obj *)); ms = malloc(KM * sizeof(Sim)); mwin = calloc((uint64_t)KM * windows, sizeof(uint64_t));
        for (int i = 0; i < K; i++) {
            mo[i] = malloc((uint64_t)(n_s ? n_s : 1) * sizeof(Obj));
            for (uint32_t x = 0; x < n_s; x++) { mo[i][x] = (Obj){0}; mo[i][x].stage = -1; }
            sim_init(&ms[i], &cand[i], mcap);
            if (gwin > 0) ms[i].sk = &msk;
        }
    }
    Config s3; parse(S3_DEFAULT, &s3);
    for (uint32_t i = 0; i < n_obj; i++) { o[i] = (Obj){0}; o[i].stage = -1; }
    Sim live; sim_init(&live, &s3, capacity);
    if (gwin > 0) live.sk = &lsk;
    int current = -1, switches = 0, next_sched = 0;
    uint64_t rlcap = replay > 0 ? (uint64_t)(replay * (double)capacity) : 0, rln = 0; /* replay: the request log and scratch records */
    if (replay > 0 && rlcap < 1) rlcap = 1;
    uint32_t *rl = rlcap ? malloc(rlcap * sizeof *rl) : NULL; Obj *ro = NULL;
    if (rlcap) { ro = malloc((uint64_t)(n_obj ? n_obj : 1) * sizeof(Obj)); for (uint32_t x = 0; x < n_obj; x++) { ro[x] = (Obj){0}; ro[x].stage = -1; } }
    /* trial: the live mirror (lm, lmo), its last `trial` snapshots, the sampled requests since the oldest (tb), a scratch replay */
    Sim lm = {0}; Obj *lmo = NULL, *fo = NULL; Snap *snaps = NULL; int nsn = 0; VQSketch fsk = {0};
    uint32_t *tb = NULL; uint64_t tbn = 0, tbcap = 0, lm_miss = 0, wl = n / windows ? n / windows : 1;
    int lmon = trial || reseed; /* the mirror runs */
    if (lmon) {
        lmo = malloc((uint64_t)(n_s ? n_s : 1) * sizeof(Obj));
        for (uint32_t x = 0; x < n_s; x++) { lmo[x] = (Obj){0}; lmo[x].stage = -1; }
        sim_init(&lm, &s3, mcap); if (gwin > 0) lm.sk = &msk;
    }
    if (trial) {
        fo = malloc((uint64_t)(n_s ? n_s : 1) * sizeof(Obj)); if (gwin > 0) sk_init(&fsk, mcap, gwin);
        snaps = calloc(trial, sizeof(Snap));
    }
    uint64_t before = 0, after = 0, *win = calloc(windows, sizeof(uint64_t));
    int w_prev = -1;
    uint32_t *lid = NULL; uint8_t *lw = NULL; uint64_t nl = 0, lcap = 0; /* screening: the sampled prefix (a ring if logcap) */
    if (screen > 0 && logcap > 0) { lcap = logcap; lid = malloc(lcap * sizeof *lid); lw = malloc(lcap); }
    uint64_t *prior = calloc(KM, sizeof(uint64_t)); int wlo = 2; /* wrapped log: the leader's score is prior + misses from window wlo */
    for (uint64_t t = 0; t < n; t++) {
        int w = (int)((t * (uint64_t)windows) / n);
        if (screen > 0 && t == split) { /* score every pool design on the logged prefix, keep the best `screen` warm */
            uint64_t *score = malloc((uint64_t)(P ? P : 1) * sizeof(uint64_t)), *tw = calloc(windows, sizeof(uint64_t));
            Obj *to = malloc((uint64_t)(n_s ? n_s : 1) * sizeof(Obj)); Sim tmp;
            int wrapped = logcap > 0 && nl > logcap;
            uint64_t kept = wrapped ? logcap : nl, first = wrapped ? nl % logcap : 0, rcap = lcap ? lcap : 1;
            if (wrapped && !wrapmode) wlo = w;
            for (int j = 0; j < P; j++) {
                memset(tw, 0, windows * sizeof(uint64_t)); uint64_t pr = 0;
                replay_prefix(&pcfg[j], mcap, n_s, lid, lw, rcap, first, kept, wrapped, w, gwin, &tmp, to, tw, &pr); sim_free(&tmp);
                score[j] = pr; for (int v = wlo; v <= w; v++) score[j] += tw[v];
            }
            int lead0 = 0; uint64_t lead_score = 0; /* wrapmode 1: the given leader so far, and its replay score */
            if (wrapped && !wrapmode) for (int i = 0; i < K0; i++) { /* the given candidates: scored on the same replay, state kept */
                memset(tw, 0, windows * sizeof(uint64_t));
                replay_prefix(&cand[i], mcap, n_s, lid, lw, rcap, first, kept, wrapped, w, gwin, &tmp, to, tw, &prior[i]); sim_free(&tmp);
            }
            if (wrapped && wrapmode) {
                uint64_t best = UINT64_MAX;
                for (int i = 0; i < K0; i++) { uint64_t m = 0; for (int v = 2; v <= w; v++) m += mwin[(uint64_t)i * windows + v]; if (m < best) { best = m; lead0 = i; } }
                memset(tw, 0, windows * sizeof(uint64_t));
                replay_prefix(&cand[lead0], mcap, n_s, lid, lw, rcap, first, kept, wrapped, w, gwin, &tmp, to, tw, &lead_score); sim_free(&tmp);
            }
            char *taken = calloc(P ? P : 1, 1);
            for (int m = 0; m < screen; m++) {
                int bj = -1; for (int j = 0; j < P; j++) if (!taken[j] && (bj < 0 || score[j] < score[bj])) bj = j;
                taken[bj] = 1; int i = K0 + m; cand[i] = pcfg[bj];
                mo[i] = malloc((uint64_t)(n_s ? n_s : 1) * sizeof(Obj));
                replay_prefix(&cand[i], mcap, n_s, lid, lw, rcap, first, kept, wrapped, w, gwin, &ms[i], mo[i], &mwin[(uint64_t)i * windows], &prior[i]);
                if (wrapped && wrapmode) { /* history before the split's window: the given leader's, scaled */
                    double rho = ((double)prior[i] + 1) / ((double)lead_score + 1); prior[i] = 0;
                    for (int v = 0; v < w; v++) mwin[(uint64_t)i * windows + v] = (uint64_t)(rho * (double)mwin[(uint64_t)lead0 * windows + v] + 0.5);
                }
                if (gwin > 0) ms[i].sk = &msk;
                if (getenv("VQSIM_SCREEN_LOG")) fprintf(stderr, "screen %d: pool line %d (after dedup), file line %d, %lu sampled misses\n", m, bj, pline[bj], (unsigned long)score[bj]);
            }
            K = KM;
            free(score); free(tw); free(to); free(taken); free(lid); free(lw); lid = NULL; lw = NULL;
        }
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
                if (lo < wlo) lo = wlo;
                for (int i = 0; i < K; i++) {
                    uint64_t m = lo == wlo ? prior[i] : 0; for (int v = lo; v <= w; v++) m += mwin[(uint64_t)i * windows + v];
                    if (m < best) { best = m; leader = i; }
                    if (i == current) mine = m;
                }
                /* hysteresis: leave the current design only for a leader at least `margin` better */
                if (current >= 0 && leader != current && (double)best >= (1.0 - margin) * (double)mine) leader = current;
                if (getenv("VQSIM_TRACE")) fprintf(stderr, "decide w=%d live=%d leader=%d best=%lu live_score=%lu\n", w, current, leader, (unsigned long)best, (unsigned long)mine);
            }
            if (trial && leader >= 0 && leader != current && (current >= 0 || trialsplit)) { /* would switching at the oldest snapshot have beaten the mirror since? */
                uint64_t h = 0, stay = 0; int pass = 0;
                if (nsn > 0 && tbn > snaps[0].off) {
                    Snap *p = &snaps[0];
                    memcpy(fo, p->o, (uint64_t)n_s * sizeof(Obj));
                    Sim F; sim_init(&F, &cand[leader], mcap);
                    if (gwin > 0) { memcpy(fsk.t, p->sk.t, 4 * (fsk.mask + 1)); fsk.adds = p->sk.adds; F.sk = &fsk; }
                    if (p->cur < 0) adopt(&F, &p->s, fo); else switch_design(&F, &p->s, fo);
                    for (uint64_t l = p->off; l < tbn; l++) { if (gwin > 0) sk_add(&fsk, tb[l]); h += step(&F, fo, tb[l]) ? 1 : 0; }
                    sim_free(&F);
                    stay = lm_miss - p->miss; pass = h < stay;
                }
                if (getenv("VQSIM_TRACE")) fprintf(stderr, "trial w=%d live=%d leader=%d h=%lu stay=%lu pass=%d\n", w, current, leader, (unsigned long)h, (unsigned long)stay, pass);
                if (!pass) leader = current;
            }
            if (reseed && t == split) /* every miniature takes over the mirror's cache (still S3-FIFO's) */
                for (int i = 0; i < K; i++) {
                    memcpy(mo[i], lmo, (uint64_t)n_s * sizeof(Obj));
                    Sim b; sim_init(&b, &cand[i], mcap); b.sk = ms[i].sk;
                    adopt(&b, &lm, mo[i]); sim_free(&ms[i]); ms[i] = b;
                }
            if (leader >= 0 && leader != current && rlcap) {
                replay_takeover(&live, o, &cand[leader], rl, rlcap, rln, ro);
                if (getenv("VQSIM_CHECK")) check_live(&live, o, n_obj);
                if (current >= 0) switches++;
                current = leader;
            }
            if (leader >= 0 && leader != current) {
                Sim b; sim_init(&b, &cand[leader], capacity); b.sk = live.sk;
                if (current < 0) adopt(&b, &live, o); else switch_design(&b, &live, o);
                if (lmon) { /* the mirror follows */
                    Sim b2; sim_init(&b2, &cand[leader], mcap); b2.sk = lm.sk;
                    if (current < 0) adopt(&b2, &lm, lmo); else switch_design(&b2, &lm, lmo);
                    sim_free(&lm); lm = b2;
                }
                if (current >= 0) switches++;
                sim_free(&live); live = b; current = leader;
            }
        }
        if (trial && (point || (t < split && split - t <= (uint64_t)trial * wl && (split - t) % wl == 0))) { /* snapshot the mirror */
            if (nsn == trial) { /* drop the oldest, and the requests only it needed */
                sim_free(&snaps[0].s); free(snaps[0].o); free(snaps[0].sk.t);
                memmove(snaps, snaps + 1, (uint64_t)(trial - 1) * sizeof(Snap)); nsn--;
                uint64_t d = nsn ? snaps[0].off : tbn;
                memmove(tb, tb + d, (tbn - d) * sizeof *tb); tbn -= d; for (int i = 0; i < nsn; i++) snaps[i].off -= d;
            }
            Snap *p = &snaps[nsn++];
            sim_copy(&p->s, &lm); p->o = malloc((uint64_t)(n_s ? n_s : 1) * sizeof(Obj)); memcpy(p->o, lmo, (uint64_t)n_s * sizeof(Obj));
            p->sk = msk; if (gwin > 0) { p->sk.t = malloc(4 * (msk.mask + 1)); memcpy(p->sk.t, msk.t, 4 * (msk.mask + 1)); } else p->sk.t = NULL;
            p->off = tbn; p->miss = lm_miss; p->cur = current;
        }
        w_prev = w;
        uint32_t x = ids[t];
        if (gwin > 0) { sk_add(&lsk, x); if (mini && smap[x] >= 0) sk_add(&msk, (uint32_t)smap[x]); }
        if (step(&live, o, x)) { if (t < split) before++; else after++; win[w]++; }
        if (rlcap) rl[rln++ % rlcap] = x;
        if (mini && smap[x] >= 0) for (int i = 0; i < K; i++) if (step(&ms[i], mo[i], (uint32_t)smap[x])) mwin[(uint64_t)i * windows + w]++;
        if (lmon && smap[x] >= 0) {
            uint32_t y = (uint32_t)smap[x];
            if (step(&lm, lmo, y)) lm_miss++;
            if (nsn > 0) { if (tbn == tbcap) { tbcap = tbcap ? 2 * tbcap : 4096; tb = realloc(tb, tbcap * sizeof *tb); } tb[tbn++] = y; }
        }
        if (screen > 0 && t < split && mini && smap[x] >= 0) {
            if (!logcap && nl == lcap) { lcap = lcap ? 2 * lcap : 4096; lid = realloc(lid, lcap * sizeof *lid); lw = realloc(lw, lcap); }
            lid[nl % lcap] = (uint32_t)smap[x]; lw[nl % lcap] = (uint8_t)w; nl++;
        }
    }
    free(lid); free(lw); free(pcfg); free(pline); free(prior);
    takeover = 0; /* handover jobs (run) keep the stage mapping */
    fprintf(out, "%lu %lu %d", (unsigned long)before, (unsigned long)after, switches);
    for (int w = 0; w < windows; w++) fprintf(out, " %lu", (unsigned long)win[w]);
    fputc('\n', out);
    sim_free(&live); free(win); sk_free(&lsk); sk_free(&msk);
    if (lmon) { sim_free(&lm); free(lmo); }
    free(rl); free(ro);
    if (trial) { free(fo); for (int i = 0; i < nsn; i++) { sim_free(&snaps[i].s); free(snaps[i].o); free(snaps[i].sk.t); } free(snaps); free(tb); sk_free(&fsk); }
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
