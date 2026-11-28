/* rad0017: typed rank allocation, publication and retirement; frozen 2.2 projection. */
#pragma once

struct ArInst {
    int  rank = 0, world = 0;
    long long group = 0;      /* the tensor-parallel group it rendezvoused in (rv_for) */
    long my_scratch = 0, my_flags = 0, my_seq = 0, locpack = 0;
    Rendezvous* pending_rv = nullptr;  /* BC_VTP14_N3_RANK_PENDING */
    int allocation_device = -1;
    /* A 2-rank host-mailbox instance's VRAM mirror of its slots, 0 otherwise (r4d_ar_seq.h). */
    long my_stage = 0;
    long peer_scratch[R4D_MAX_PEERS] = { 0 };
    long peer_flags  [R4D_MAX_PEERS] = { 0 };
    long slot_bytes = 0;      /* per-slot capacity, 16-byte aligned                  */
    long slot16     = 0;      /* the same in 16-byte words                           */
    long max_numel  = 0;      /* the declared bound the sizing was computed from     */
    int  esize      = 2;
    int  dtype      = 0;      /* libr4d code: 0 bf16, 1 fp16, 2 fp32                 */
    int  max_blocks = 0;      /* what the flag array was sized for; a hard bound      */
    /* Block count a launch may use: a tuning, unlike the max_blocks capacity. Each block drives its
     * own flag handshake, so more blocks help only until the link saturates, then add round trips
     * (at a 2048-token chunk the capacity allows 936 blocks; 24 is used). */
    int  nb_cap     = 0;
    /* Exchange buffers: 0 in each rank's VRAM, written by peers over P2P; 1 in pinned host memory
     * each card reaches over its own link (see build). */
    int  host       = 0;
    long gate_ctr   = 0;      /* the blocks that have seen their peer's bytes land (r4d_ar_seq.h) */
    /* The eight-rank two-shot's per-rank shard weights (r4d_ar_shards.h), fixed at build so the
     * scratch was sized for them and a recorded launch keeps its table. */
    int  shard_weighted = 0;
    int  shard_w[R4D_MAX_RANKS] = { 0 };
    /* Its piece count, 0 for the size rule (r4d_ar8_auto_pieces); a launch's `pieces` parameter
     * overrides it. */
    int  pieces = 0;
    /* Messages under these sizes (elements) run even and in one piece, whatever is configured. */
    long long weights_min = R4D_AR8_MIN_ELEMS_DEFAULT;
    long long pieces_min  = R4D_AR8_MIN_ELEMS_DEFAULT;
};

/* A parameter that may be declared as a range: init gets one band's frozen geometry, and buffers
 * are sized for the band's upper bound. */
long long pgeti_hi(const RadParam* p, int n_p, const char* k, long long dflt) {
    if (!p || !k) return dflt;
    for (int i = 0; i < n_p; ++i) {
        if (std::strcmp(p[i].key, k) != 0) continue;
        if (p[i].kind == RAD_P_INT)   return p[i].ival;
        if (p[i].kind == RAD_P_RANGE) return p[i].ihi;
    }
    return dflt;
}

long align16(long v) { return (v + 15) & ~15L; }

/* The eight-rank two-shot's settings for the next build: the engine's defaults unless
 * r4d_ar8_test_settings set others. */
std::mutex     g_ar8_m;
R4DAr8Settings g_ar8;

R4DAr8Settings ar8_settings() {
    std::lock_guard<std::mutex> lk(g_ar8_m);
    return g_ar8;
}

/* This rank's card's slowest PCIe hop, from sysfs by the runtime's bus id. */
bool ar8_own_link(R4DArLink* l) {
    int dev = 0;
    char bus[64] = { 0 };
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetPCIBusId(bus, (int)sizeof bus, dev) != hipSuccess)
        return false;
    unsigned dom = 0, b = 0, d = 0, f = 0;
    if (std::sscanf(bus, "%x:%x:%x.%x", &dom, &b, &d, &f) != 4) return false;
    char bdf[32];
    std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.%x", dom, b, d, f);
    return r4d_ar_link_bottleneck("", bdf, l);
}

/* Card-to-card traffic of the collectives. Ctx::ar_bytes is each rank's own
 * payload; what a card receives depends on the route libr4d picks for the message, which only this
 * file knows. Each entry below adds, for its rank, the bytes it sends to every other rank, so
 * g_ar_sent[r][p] is what went from card r to card p and a column's sum is what arrived at a card.
 * By construction, not by measuring the wire:
 *   one-shot    the whole message to every peer.
 *   two-shot    to peer p, this rank's part of p's shard (the push) and then this rank's own reduced
 *               shard (the assemble): shard_p + shard_r bytes. Even shards make that half a message
 *               a peer at four ranks; a rank that owns no shard sends only its parts of the others'.
 *   coded       the two-shot with both pushes coded at about 12.2 bits a bf16 value; the bytes are
 *               the data's, so this is an estimate at 12.2 / 16 of the plain two-shot's.
 *   lossy       the tiered-int8 two-shot at 38 bytes a 32-value group, the two-rank wht6 at 6 bits
 *               a value.
 *   gather      this rank's piece to every peer.
 * The fused collectives (the quantising reductions, hc_write) are not followed: they are counted as
 * calls in g_ar_uncounted, so a reader can tell whether the table is the whole of the traffic.
 * These entries run when a pass is issued in full. A pass the engine plays back from a tape does not
 * come through here, so the engine takes the difference a recording made to its rank's counts
 * (r4d_ar_traffic_rank before and after) and adds it again for every play (Ctx::record_pass).
 * Relaxed atomics: a rank's thread is the only writer of its row. */
enum { kTrOneShot = 0, kTrTwoShot, kTrCoded, kTrLossy, kTrGather, kTrRoutes };
std::atomic<long long> g_ar_sent[R4D_MAX_RANKS][R4D_MAX_RANKS];
std::atomic<long long> g_ar_route_calls[R4D_MAX_RANKS][kTrRoutes], g_ar_route_bytes[R4D_MAX_RANKS][kTrRoutes];
std::atomic<long long> g_ar_uncounted[R4D_MAX_RANKS];

void traffic_route(int rank, int route, long long payload) {
    g_ar_route_calls[rank][route].fetch_add(1, std::memory_order_relaxed);
    g_ar_route_bytes[rank][route].fetch_add(payload, std::memory_order_relaxed);
}
/* The whole message, or a rank's piece, to every peer; `num` / `den` scales a coded wire. */
void traffic_all(const ArInst* in, long numel, int route, long num = 1, long den = 1) {
    if (!in || in->rank < 0 || in->rank >= R4D_MAX_RANKS || in->world > R4D_MAX_RANKS) return;
    const long long b = (long long)numel * in->esize;
    for (int p = 0; p < in->world; ++p)
        if (p != in->rank) g_ar_sent[in->rank][p].fetch_add(b * num / den, std::memory_order_relaxed);
    traffic_route(in->rank, route, b);
}
void traffic_twoshot(const ArInst* in, long numel, const int* weights, int route, long num = 1, long den = 1) {
    if (!in || in->rank < 0 || in->rank >= R4D_MAX_RANKS || in->world > R4D_MAX_RANKS) return;
    long off[R4D_MAX_RANKS + 1];
    r4d_ar_shard_table((long)numel * in->esize / 16, in->world, weights, off);
    const long long mine = (long long)(off[in->rank + 1] - off[in->rank]) * 16;
    for (int p = 0; p < in->world; ++p) {
        if (p == in->rank) continue;
        const long long theirs = (long long)(off[p + 1] - off[p]) * 16;
        g_ar_sent[in->rank][p].fetch_add((theirs + mine) * num / den, std::memory_order_relaxed);
    }
    traffic_route(in->rank, route, (long long)numel * in->esize);
}
/* The fused entries have their instance under other names, so they are counted against rank 0's
 * row: the count says whether there were any, not whose. */
void traffic_uncounted_rank() { g_ar_uncounted[0].fetch_add(1, std::memory_order_relaxed); }

/* The eight-rank two-shot's choice for one message, a function of its size and the instance
 * alone: a recorded launch keeps the table and piece count it was recorded with, and replays of
 * the same size agree with it. */
const int* ar8_weights_for(const ArInst* in, long numel) {
    return in->shard_weighted && numel >= in->weights_min ? in->shard_w : nullptr;
}
long long ar8_pieces_for(const ArInst* in, long numel, long long asked) {
    if (numel < in->pieces_min) return 1;
    return asked > 0 ? asked : r4d_ar8_auto_pieces(numel);
}

/* The eight-rank two-shot's settings, fixed before anything is sized: each rank publishes its own
 * card's link, and after one barrier every rank derives the same weights from the same eight
 * entries. */
int ar8_settle(ArInst* in, int rank, int world) {
    Rendezvous& rv = rv_for(in->group);
    static std::once_flag said_links, said_unknown, said_set;
    const R4DAr8Settings set = ar8_settings();
    R4DArLink own;
    if (set.weights == R4D_AR8_WEIGHTS_AUTO && !ar8_own_link(&own)) own = R4DArLink();
    {
        std::lock_guard<std::mutex> lk(rv.m);
        rv.ar8_link[rank] = own;
    }
    const int rc = rv_barrier(rv, world);
    if (rc != RAD_OK) return rc;

    const R4DArLink* all = rv.ar8_link;
    in->pieces = set.pieces;
    in->weights_min = set.weights_min;
    in->pieces_min = set.pieces_min;

    int w[R4D_MAX_RANKS] = { 0 };
    if (set.weights == R4D_AR8_WEIGHTS_SET) {
        std::memcpy(w, set.w, sizeof(set.w));
        std::call_once(said_set, [&w] {
            std::fprintf(stderr, "[r4d] eight-rank two-shot shards weighted "
                         "%d,%d,%d,%d,%d,%d,%d,%d (r4d_ar8_test_settings)\n", w[0], w[1], w[2],
                         w[3], w[4], w[5], w[6], w[7]);
        });
    } else if (set.weights == R4D_AR8_WEIGHTS_AUTO) {
        double gbs[R4D_MAX_RANKS] = { 0 };
        for (int r = 0; r < world; ++r) gbs[r] = r4d_ar_link_gbs(all[r]);
        if (!r4d_ar_weights_from_rates(gbs, world, w)) {
            std::call_once(said_unknown, [all, world] {
                std::string who;
                for (int r = 0; r < world; ++r)
                    if (r4d_ar_link_gbs(all[r]) <= 0.0) who += " " + std::to_string(r);
                std::fprintf(stderr, "[r4d] eight-rank two-shot: no PCIe link speed in sysfs for "
                             "rank%s; shards stay even\n", who.c_str());
            });
            return RAD_OK;
        }
        std::call_once(said_links, [all, world, &w] {
            std::string l;
            for (int r = 0; r < world; ++r) {
                char one[96];
                std::snprintf(one, sizeof one, "%s r%d %.1f GT/s x%d %.1f GB/s w%d",
                              r ? "," : "", r, all[r].gts, all[r].width,
                              r4d_ar_link_gbs(all[r]), w[r]);
                l += one;
            }
            std::fprintf(stderr, "[r4d] eight-rank two-shot shard weights from the slowest PCIe "
                         "hop of each card:%s\n", l.c_str());
        });
        bool same = true;
        for (int r = 1; r < world; ++r) same = same && w[r] == w[0];
        if (same) return RAD_OK;   /* equal links: the even split, on the unweighted path */
    } else {
        return RAD_OK;
    }
    in->shard_weighted = 1;
    std::memcpy(in->shard_w, w, sizeof(in->shard_w));
    return RAD_OK;
}

/* The four-rank exact two-shot's shard weights, fixed before anything is sized: each rank publishes its
 * own card's slowest PCIe hop, and after one barrier every rank derives the same weights from the same
 * four entries by the duplex model (r4d_ar_weights_duplex, r4d_ar_links.h). On equal links that is the
 * even split. With two cards at half the others' rate (x8 and x4 slots, x8, x4, x8, x4) it
 * gives the slow pair no shard, 1,0,1,0 in effect: each message then reaches them once instead of 1.5
 * times, the fast pair takes what they give up (1 + 2 * share of the message each), and every link's
 * time is the same where the even split waits on the slow pair; on such a board that table is faster than
 * the even split. A link sysfs does not describe leaves the shards even. Weights apply from the first
 * element: the one-shot below `ts_min` carries the small messages, so only two-shot messages see them.
 * The table changes which link a word crosses, never its fold, so the result has the same bits
 * (r4d_ar_shards.h). Only the plain exact two-shot takes it: the coded lane form, the tiered-int8 wire
 * and the one-shot keep their own layouts. */
int ar4_settle(ArInst* in, int rank, int world) {
    Rendezvous& rv = rv_for(in->group);
    static std::once_flag said;
    R4DArLink own;
    if (!ar8_own_link(&own)) own = R4DArLink();
    {
        std::lock_guard<std::mutex> lk(rv.m);
        rv.ar8_link[rank] = own;
    }
    const int rc = rv_barrier(rv, world);
    if (rc != RAD_OK) return rc;
    double gbs[R4D_MAX_RANKS] = { 0 };
    for (int r = 0; r < world; ++r) gbs[r] = r4d_ar_link_gbs(rv.ar8_link[r]);
    int w[R4D_MAX_RANKS] = { 0 };
    if (!r4d_ar_weights_duplex(gbs, world, w)) return RAD_OK;   /* a rate unknown: even */
    bool same = true;
    for (int r = 1; r < world; ++r) same = same && w[r] == w[0];
    if (same) return RAD_OK;                                        /* equal links: even */
    const R4DArLink* all = rv.ar8_link;
    std::call_once(said, [all, world, &w] {
        std::string l;
        for (int r = 0; r < world; ++r) {
            char one[96];
            std::snprintf(one, sizeof one, "%s r%d %.1f GT/s x%d %.1f GB/s w%d", r ? "," : "", r,
                          all[r].gts, all[r].width, r4d_ar_link_gbs(all[r]), w[r]);
            l += one;
        }
        std::fprintf(stderr, "[r4d] four-rank exact two-shot shard weights from the slowest PCIe hop of "
                     "each card:%s\n", l.c_str());
    });
    in->shard_weighted = 1;
    std::memcpy(in->shard_w, w, sizeof(in->shard_w));
    in->weights_min = 0;
    return RAD_OK;
}

/* Sequence counter allocation in words: 2 * nbw (r4d_ar_seq.h), then r4d_ar_link_landed's block
 * counter, padded to a cache line. */
long kSeqWords(long nbw) { return 2L * nbw + 32; }

/* Behind a flags allocation's flags and check words: a 2-rank instance's wake word
 * (r4d_ar_link_wake, r4d_ar_seq.h), on a line of its own. */
constexpr long kWakeBytes = 64;

void free_rank_buffers(ArInst* in) {  /* BC_VTP14_N3_RANK_FREE */
    /* Free only this rank's own allocations; peer pointers belong to the peer. */
    if (in->host) {
        r4d_ar_host_free(in->my_scratch);
        r4d_ar_host_free(in->my_flags);
    } else {
        r4d_ar_ipc_free(in->my_scratch);
        r4d_ar_ipc_free(in->my_flags);
    }
    r4d_ar_ipc_free(in->my_seq);
    r4d_ar_ipc_free(in->locpack);
    r4d_ar_ipc_free(in->my_stage);
}

void free_failed_rank(void* instance) {
    auto* in = static_cast<ArInst*>(instance);
    int previous = -1;
    const bool restore = hipGetDevice(&previous) == hipSuccess && previous != in->allocation_device;
    if (hipSetDevice(in->allocation_device) != hipSuccess)
        std::fprintf(stderr, "[r4d] could not bind device %d for failed all-reduce cleanup\n",
                     in->allocation_device);
    free_rank_buffers(in);
    if (restore) (void)hipSetDevice(previous);
    delete in;
}

using DeferRankCleanup = int (*)(void*, void*, void (*)(void*));
using ForgetRankCleanup = void (*)(void*, void*);

struct RankCleanupBridge {
    DeferRankCleanup defer = reinterpret_cast<DeferRankCleanup>(
        dlsym(RTLD_DEFAULT, "rad_bc_defer_rank_cleanup"));
    ForgetRankCleanup forget = reinterpret_cast<ForgetRankCleanup>(
        dlsym(RTLD_DEFAULT, "rad_bc_forget_rank_cleanup"));
};

const RankCleanupBridge& rank_cleanup_bridge() {
    static const RankCleanupBridge bridge;
    return bridge;
}

/* Build the world. `max_numel` is the largest message any launch on this instance carries.
 *
 * One sizing rule for the whole family: a slot holds a full message. The one-shots need exactly
 * that; the two-shot's two regions (2 * message / world_size) and the tiered-int8 wire's two
 * images (2 * ceil(shard/32) * 38 bytes) are smaller. RadInitFn is not told which row resolved,
 * so sizing for the maximum avoids reconstructing which kernel this instance serves. The one
 * exception is a weighted eight-rank two-shot whose largest weight is over half the sum: its two
 * regions of the longest shard outgrow a message, and the slot grows to hold them.
 *
 * Cost: at a 4096-token chunk on a 5120-wide residual, 40 MiB a slot and 80 MiB for the pair. One
 * set serves every op with the same (rank, world, dtype, bound); see the cache below. */
int build(ArInst* in, long long group, int rank, int world, long long max_numel, int dtype_code,
          int host) {
    if (world < 2 || world > R4D_MAX_RANKS) return RAD_E_SHAPE;
    if (rank < 0 || rank >= world) return RAD_E_INVAL;
    if (max_numel <= 0) return RAD_E_INVAL;

    in->group = group;
    in->rank = rank;
    in->world = world;
    in->host = host;
    in->dtype = dtype_code;
    in->esize = r4d_ar_esize(dtype_code);
    in->max_numel = (long)max_numel;
    in->slot_bytes = align16((long)max_numel * in->esize);
    /* A two-rank bf16 slot over P2P also holds the coded form of its largest message (the fused
     * writes' coded push, r4d_ar_oneshot_2rank_exact.hip): 16.125 bytes a 16-byte word, 1/128
     * more than the message. */
    if (world == 2 && dtype_code == 0 && !host) {
        const long coded = align16(r4d_ar_coded2_bytes(in->slot_bytes / 16));
        if (coded > in->slot_bytes) in->slot_bytes = coded;
    }
    in->slot16 = in->slot_bytes / 16;
    if (world == 8) {
        const int rc8 = ar8_settle(in, rank, world);
        if (rc8 != RAD_OK) return rc8;
    } else if (world == 4) {
        const int rc4 = ar4_settle(in, rank, world);
        if (rc4 != RAD_OK) return rc4;
    }
    if (in->shard_weighted) {
        const long need16 =
            2 * r4d_ar_shard_max_bound((long)max_numel * in->esize / 16, world, in->shard_w);
        /* Sized only if some message this instance carries is weighted. */
        if (max_numel >= in->weights_min && need16 > in->slot16) {
            in->slot16 = need16;
            in->slot_bytes = need16 * 16;
        }
    }

    int maxb = 0, nb_design = 0, max_peers = 0;
    if (world == 2) {
        maxb = r4d_ar_max_blocks();
    } else {
        r4d_ar_wide_dims(&maxb, &nb_design, &max_peers);
        if (world - 1 > max_peers) return RAD_E_SHAPE;
    }
    if (maxb <= 0) return RAD_E_DEVICE;
    in->max_blocks = maxb;
    /* R4D_AR_MAX_BLOCKS: a ceiling on every collective's block count, for two engines sharing one
     * set of cards. Each block of a collective spins until its twin on the peer card has pushed, and
     * a prefill-sized ar_rmsnorm_quant_fp8 launches up to 256 blocks of 16 waves, twice what an
     * R9700 holds. One engine survives that (both cards dispatch blocks in the same order, so the
     * lowest unfinished block is resident on both); two engines do not: one engine's spinning blocks
     * fill a card while their twins wait behind the other engine's kernel on the peer card, which
     * waits for a slot on the full card. Not capped: the hyper-connection gathers of the MoE paths
     * (r4d_ar_gather_hc_write without its inverse scratch, r4d_ar_hc_write_wht6), whose LDS table
     * needs a minimum block count, which may exceed the ceiling (dense models do not use them).
     * Two TP2 engines on two R9700s (Qwen3.8-27B and Gemma 3 27B) hung within a minute of
     * concurrent prefill; MES then failed to evict the queues and the failed reset froze the
     * host. Capped in both engines so that their spinners together cannot
     * fill a card, the cycle cannot close. The flag and counter arrays keep their full size (maxb);
     * only the launch counts shrink, and each element's sum is the same. Unset, nothing changes. */
    if (const char* e = std::getenv("R4D_AR_MAX_BLOCKS")) {
        const int cap = std::atoi(e);
        if (cap > 0 && cap < in->max_blocks) in->max_blocks = cap;
    }
    /* Launch ceilings: 24 at two ranks, the library's design count at width. The compressed
     * 2-rank wire uses up to 48, set at its own call site: it moves three eighths of the bytes, so
     * it needs more blocks to saturate the link. */
    in->nb_cap = (world == 2) ? 24 : (nb_design > 0 ? nb_design : maxb);
    if (in->nb_cap > in->max_blocks) in->nb_cap = in->max_blocks;

    /* Slots: one per (parity, source rank) at width; two at world 2, where there is one remote
     * writer. Flags: one word per (block, source rank) at width, one per block at world 2.
     * Sequence counters: a handshake and a parity word per block of capacity (r4d_ar_seq.h),
     * device-resident so a captured graph replays. */
    const int  nslots     = (world == 2) ? 2 : 2 * world;
    const long flag_words = (long)maxb * ((world == 2) ? 1 : world);

    char handle[R4D_AR_HANDLE_BYTES];
    int  hlen = 0;
    int  dev = 0;

    /* Fine-grained: the peer writes the scratch and this rank reads it bypassing L2, so a
     * coarse-grained allocation could leave the peer's bytes in a cache this rank never
     * invalidates, and the launches run without an acquire fence. r4d_ar_ipc_alloc throws when it
     * cannot allocate fine-grained memory; there is no coarse fallback.
     *
     * Without peer access the scratch and flags move to pinned host memory and nothing else
     * changes: kernels touch other ranks only through those two, so no card addresses another's
     * VRAM. A message then crosses two links instead of one; only rows requiring device.p2p == 0
     * take this path (r4d_rows.cpp). Sequence counters stay in VRAM, since only their own rank
     * reads them, and coarse-grained: only this rank's kernels touch them, with device- and
     * agent-scope atomics, which then complete in L2 rather than past it -- the first returning
     * atomic of every launch (r4d_ar_seq_take) is on the push's path. */
    // BC_VTP14_N3_UNPUBLISHED: clean up any acquired buffers if allocation or setup fails.
    std::unique_ptr<ArInst, decltype(&free_rank_buffers)> unpublished(in, &free_rank_buffers);
    R4D_TRY("ar_ipc_alloc",
        if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
        if (host) {
            in->my_scratch = r4d_ar_host_alloc((long)nslots * in->slot_bytes);
            in->my_flags = r4d_ar_host_alloc((flag_words + 2L * maxb) * 4 + kWakeBytes);
        } else {
            in->my_scratch = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 1, handle, &hlen);
            in->my_flags = r4d_ar_ipc_alloc((flag_words + 2L * maxb) * 4 + kWakeBytes, 1, handle, &hlen);
        }
        in->my_seq = r4d_ar_ipc_alloc(kSeqWords(maxb) * 4, 0, handle, &hlen));

    /* A host mailbox is read over the link past every cache, so the 2-rank kernels' push check
     * copies what it reads into this VRAM mirror for the reduce, crossing the link once
     * (r4d_ar_seq.h). Coarse-grained: only this rank's kernels touch it, each block its own words.
     * The wide family has no push check and reads its slots once. Host-memory slots free the same
     * VRAM over P2P, so the mirror keeps the footprint where the P2P instance has it. */
    if (host && world == 2) {
        R4D_TRY("ar_ipc_alloc(stage)",
            in->my_stage = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 0, handle, &hlen));
    }

    /* This rank's own packed copy, for the rotated 6-bit wire only: the reduce must fold exactly
     * the bytes it sent. The 6-bit payload is 3/8 of the bf16 message and the f32 scales 1/32 of
     * it, laid out from half the slot; a row padded to whole 512-element chunks
     * (ar_rmsnorm_quant_fp8) adds up to 7 groups' scales a row, so the scales get a sixteenth
     * (enough for any row of 512 or more) plus a page. ar_rmsnorm_quant checks the fit. */
    if (in->dtype != 2) {
        R4D_TRY("ar_ipc_alloc(locpack)",
            in->locpack = r4d_ar_ipc_alloc(in->slot_bytes / 2 + in->slot_bytes / 16 + 4096, 0,
                                  handle, &hlen));
    }

    in->gate_ctr   = in->my_seq + 2L * maxb * 4;   /* the word after the sequence counters (r4d_ar_seq.h) */

    Rendezvous& rv = rv_for(group);
    {
        std::lock_guard<std::mutex> lk(rv.m);
        if (rv.failed) return RAD_E_STATE;
        if (rv.owner != 0) {
            const auto& bridge = rank_cleanup_bridge();
            const int rc = bridge.defer && bridge.forget
                ? bridge.defer(&rv, in, free_failed_rank) : RAD_E_STATE;
            if (rc != RAD_OK) {
                rv.failed = 1;
                rv.cv.notify_all();
                return rc;
            }
            in->allocation_device = dev;
            in->pending_rv = &rv;
        }
        rv.scratch[rank] = in->my_scratch;
        rv.flags[rank]   = in->my_flags;
        rv.seq[rank]     = in->my_seq;
        rv.dev[rank]     = dev;
        // Published peer pointers require group-level unwind on failure.
        unpublished.release();
    }
    int rc = rv_barrier(rv, world);
    if (rc != RAD_OK) return rc;

    /* Peer access is per context, so each rank enables its own, after the publish barrier where
     * peer device ordinals are known. A host can expose more HIP devices than ranks (e.g. an
     * integrated GPU), so the ordinal comes from the peer rather than the rank. A peer on this
     * rank's own card needs none (the runtime refuses to enable it): r4d_selftest runs eight ranks
     * on one card that way, each on its own stream. */
    int j = 0;
    for (int r = 0; r < world; ++r) {
        if (r == rank) continue;
        if (!rv.scratch[r] || !rv.flags[r]) return RAD_E_STATE;
        in->peer_scratch[j] = rv.scratch[r];
        in->peer_flags[j]   = rv.flags[r];
        ++j;
        if (!host && rv.dev[r] != dev)
            R4D_TRY("ar_ipc_enable_peer", r4d_ar_ipc_enable_peer((long)rv.dev[r]));
    }

    /* Second barrier: no rank may launch until every rank has enabled peer access, or the first
     * message would be pushed into a mapping the receiver has not opened. */
    return rv_barrier(rv, world);
}

/* ------------------------------------------------------------------ one world, shared
 *
 * One instance per (rank, world, dtype, message bound), not per op. The core calls init per
 * resolved instance, and a 64-layer tensor-parallel model declares 128 identical all-reduces.
 * Scratch is sized by the largest message (at a 2048-token chunk on a 5120-wide residual, ~53 MiB
 * per instance), so per-op instances would cost 6.7 GiB a card. It comes from hipMalloc, outside
 * the VRAM budget, so over-allocation shows up as the KV cache or a prefill chunk failing to fit.
 *
 * Sharing is safe. Slots are double-buffered by the parity of the launch on this instance, and a
 * rank's all-reduces run on one stream in declared order, so all-reduce k+2 (the next user of k's
 * slot) cannot launch until k+1 completed, which needed the peer's k+1 flag, set only after the
 * peer's k kernel finished reading that slot. This needs the launch's parity, not a block's: ops
 * sharing an instance use different block counts (r4d_ar_seq.h).
 *
 * The rendezvous stays symmetric because every rank declares the same graph in the same order:
 * the Nth all-reduce misses the cache on every rank or on none. */
struct CacheKey {
    long long group;          /* the tensor-parallel group: another engine's ranks never share */
    int       rank, world, dtype;
    long long numel;
    int       host;
    bool operator==(const CacheKey& o) const {
        return group == o.group && rank == o.rank && world == o.world && dtype == o.dtype &&
               numel == o.numel && host == o.host;
    }
};

struct CacheSlot {
    CacheKey key;
    ArInst*  in   = nullptr;
    int      refs = 0;
};

std::mutex             g_cache_mu;
std::vector<CacheSlot> g_cache;

int acquire(const CacheKey& k, ArInst** out) {
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (CacheSlot& s : g_cache)
            if (s.key == k && s.in) { ++s.refs; *out = s.in; return RAD_OK; }
    }
    /* Built outside the lock: build() rendezvouses with the other ranks, and holding the mutex
     * across that barrier would deadlock with the peer's lookup. Keys include the rank and each
     * rank has one declare thread, so no two builders race for one key. */
    ArInst* in = new (std::nothrow) ArInst();
    if (!in) return RAD_E_NOMEM;
    // BC_VTP14_N3_FAILED_RANK: once published, failed instances stay live until the owner retires them.
    int rc = build(in, k.group, k.rank, k.world, k.numel, k.dtype, k.host);
    if (rc == RAD_OK) {
        try {
            std::lock_guard<std::mutex> lk(g_cache_mu);
            g_cache.push_back(CacheSlot{ k, in, 1 });
        } catch (const std::bad_alloc&) {
            rc = RAD_E_NOMEM;
        }
    }
    Rendezvous* rv = in->pending_rv;
    if (rc != RAD_OK) {
        if (rv) {
            std::lock_guard<std::mutex> lk(rv->m);
            rv->failed = 1;
            rv->cv.notify_all();
        } else {
            delete in;
        }
        return rc;
    }
    if (rv) {
        rank_cleanup_bridge().forget(rv, in);
        in->pending_rv = nullptr;
    }
    *out = in;
    return RAD_OK;
}

int acquire_at_least(const CacheKey& k, ArInst** out) {
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (CacheSlot& s : g_cache)
            if (s.key.group == k.group && s.key.rank == k.rank &&
                s.key.world == k.world && s.key.dtype == k.dtype &&
                s.key.host == k.host && s.key.numel >= k.numel && s.in) {
                ++s.refs;
                *out = s.in;
                return RAD_OK;
            }
    }
    return acquire(k, out);
}

void release(ArInst* in) {
    if (!in) return;
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (size_t i = 0; i < g_cache.size(); ++i) {
            if (g_cache[i].in != in) continue;
            if (--g_cache[i].refs > 0) return;
            g_cache.erase(g_cache.begin() + (long)i);
            break;
        }
    }
    free_rank_buffers(in);  /* BC_VTP14_N3_RELEASE */
    delete in;
}
