#pragma once

// Expert description, model shape, backend acceptance and the shared-memory segment shape. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {


uint64_t helper_budget_bytes() {
    const char* e = std::getenv("R3_MOE_WORKER_MIB");
    if (!e || !*e) return 28672ull * 1024ull * 1024ull;
    char* end = nullptr;
    errno = 0;
    const unsigned long long mib = std::strtoull(e, &end, 10);
    if (errno || !end || *end || mib == 0 ||
        mib > std::numeric_limits<uint64_t>::max() / (1024ull * 1024ull)) return 0;
    return (uint64_t)mib * 1024ull * 1024ull;
}

uint32_t env_rows(const char* e, uint32_t fallback) {
    if (!e || !*e) return fallback;
    char* end = nullptr;
    errno = 0;
    const unsigned long v = std::strtoul(e, &end, 10);
    if (errno || !end || *end || v == 0) return fallback;
    return (uint32_t)std::min<unsigned long>(v, kMaxRows);
}
/* The largest step handed off (the engine reads the same variable) and the small, pre-queued lane size. */
uint32_t helper_max_rows() { return env_rows(std::getenv("R3_MOE_WORKER_MAX_ROWS"), 64); }
uint32_t helper_small_rows() { return std::min(env_rows(std::getenv("R3_MOE_WORKER_SMALL_ROWS"), 64), helper_max_rows()); }

/* The pre-queued small-lane sizes (RAD12): R3_MOE_WORKER_SMALL_LANES="4,8,16,24" (up to kSmallClasses, strictly ascending,
 * each within the large-lane capacity), else the single size R3_MOE_WORKER_SMALL_ROWS gives (the production behaviour). */
struct SmallClasses {
    std::array<uint32_t, r3::moe_shm::kSmallClasses> rows{};
    uint32_t count = 0;
};
SmallClasses helper_small_classes() {
    SmallClasses sc;
    const char* list = std::getenv("R3_MOE_WORKER_SMALL_LANES");
    if (!list || !*list) {
        sc.rows[0] = helper_small_rows();
        sc.count = 1;
        return sc;
    }
    const uint32_t cap = helper_max_rows();
    const char* p = list;
    while (*p && sc.count < r3::moe_shm::kSmallClasses) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v <= 0 || (uint32_t)v > cap || (sc.count && (uint32_t)v <= sc.rows[sc.count - 1])) {
            std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SMALL_LANES=%s must be up to %u ascending sizes within %u rows\n",
                         list, (unsigned)r3::moe_shm::kSmallClasses, cap);
            return SmallClasses{};
        }
        sc.rows[sc.count++] = (uint32_t)v;
        p = end;
        if (*p == ',') ++p;
        else break;
    }
    if (*p) {
        std::fprintf(stderr, "r3_moe_worker: R3_MOE_WORKER_SMALL_LANES=%s has more than %u sizes or trailing text\n", list,
                     (unsigned)r3::moe_shm::kSmallClasses);
        return SmallClasses{};
    }
    return sc;
}

/* The lanes this helper serves: with several helpers each serves only its own rank slot's lanes, and a disabled small
 * class (beyond `classes`) has none. Only these get a stream, a hardware queue, buffers and a worker thread. */
std::array<bool, r3::moe_shm::kLanes> served_lanes(uint32_t helper_count, uint32_t helper_index, uint32_t classes) {
    using namespace r3::moe_shm;
    std::array<bool, kLanes> s{};
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        s[lane] = (helper_count <= 1 || lane % kMaxSlots == helper_index) &&
                  !(lane_is_small(lane) && lane_class(lane) >= classes);
    return s;
}

int64_t canonical_expert_bytes(const Container& f, uint32_t l, uint32_t e) {
    int64_t n = 0;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        const RadFileEntry* w = f.weight(l, e, slot);
        if (!w) return -1;
        for (uint32_t p = 0; p < w->n_planes; ++p) {
            const uint64_t b = f.plane(w, (int)p)->bytes;
            if (b > (uint64_t)std::numeric_limits<int64_t>::max() - (uint64_t)n) return -1;
            n += (int64_t)b;
        }
    }
    return n;
}

/* What the container says an expert is (RCS09/RAD11): its weight format, scale type and tile, read from the encoding, never assumed.
 * W4NL64 = 4-bit codes through a lookup table with scales per 64 columns; FP8_E4M3_BLOCK = E4M3 codes with a scale per tile.
 * Anything else is MOE_FMT_NONE and is refused by name before any expert is laid out. */
uint32_t scale_dtype_of(uint32_t dtype) {
    switch (dtype) {
        case RAD_F8E4M3: return MOE_SCALE_E4M3;
        case RAD_F16: return MOE_SCALE_F16;
        case RAD_BF16: return MOE_SCALE_BF16;
        case RAD_F32: return MOE_SCALE_F32;
        default: return MOE_SCALE_NONE;
    }
}

bool describe_expert(const Container& f, uint32_t l, uint32_t e, MoeExpertDesc* out) {
    const RadFileEntry* gu = f.weight(l, e, 0);
    const RadFileEntry* dn = f.weight(l, e, 1);
    if (!gu || !dn || !out) return false;
    const RadEncoding* enc = f.encoding(gu);
    if (!enc || !rad_enc_valid(enc)) return false;
    MoeExpertDesc d{};
    const int codes = rad_enc_find(enc, "codes");
    const int table = rad_enc_find(enc, "table");
    const int scale0 = rad_enc_find_level(enc, "scale", 0);
    const int scale1 = rad_enc_find_level(enc, "scale", 1);
    if (codes >= 0 && scale0 >= 0) {
        const RadEncPlane& cp = enc->plane[codes];
        const RadEncPlane& sp = enc->plane[scale0];
        if (rad_dtype_bits(cp.dtype) == 4 && table >= 0) {
            /* the two scale levels relayout to one E4M3 byte per 64 columns (the helper checks this geometry at upload) */
            (void)scale1;
            d.weight = MoeWeightDesc{MOE_FMT_W4NL64, MOE_ACT_FP8_E4M3_GROUP, MOE_SCALE_E4M3, 0, (uint32_t)sp.block[1], 0};
        } else if (cp.dtype == RAD_F8E4M3 && table < 0) {
            d.weight = MoeWeightDesc{MOE_FMT_FP8_E4M3_BLOCK, MOE_ACT_FP8_E4M3_GROUP, scale_dtype_of(sp.dtype),
                                     (uint32_t)sp.block[0], (uint32_t)sp.block[1], 0};
        }
    }
    int64_t gr = 1, gc = 1, dr = 1, dc = 1;
    rad_enc_view(gu->rank, gu->shape, &gr, &gc);
    rad_enc_view(dn->rank, dn->shape, &dr, &dc);
    d.gu_rows = (uint64_t)gr; d.gu_cols = (uint64_t)gc;
    d.dn_rows = (uint64_t)dr; d.dn_cols = (uint64_t)dc;
    *out = d;
    return true;
}

/* The shape the container implies (RCS09): layers, routed experts, hidden, expert intermediate width. top_k is the engine's. */
bool container_shape(const Container& f, MoeShape* s) {
    if (!f.experts || !f.hidden) return false;
    *s = MoeShape{f.layers, f.experts, f.hidden, f.top_k, f.intermediate, 0, 0};
    return true;
}

const char* format_name(uint32_t fmt) {
    return fmt == MOE_FMT_W4NL64 ? "w4nl64" : fmt == MOE_FMT_FP8_E4M3_BLOCK ? "fp8_e4m3_block" : "unrecognised";
}

/* Whether the backend for this architecture can run this container's experts, and why not when it cannot. */
bool backend_accepts(const Container& f, const char* arch, bool verbose) {
    MoeShape shape{};
    MoeExpertDesc d{};
    if (!container_shape(f, &shape) || !describe_expert(f, 0, 0, &d)) {
        std::fprintf(stderr, "r3_moe_worker: the container's first expert could not be described\n");
        return false;
    }
    shape.top_k = 1;      // the container does not carry the engine's top-k; the check here is the format and geometry
    MoeDeviceCaps caps{};
    std::strncpy(caps.arch, arch, sizeof(caps.arch) - 1);
    caps.wave_size = 32;
    caps.has_wmma_f16 = std::strncmp(arch, "gfx11", 5) == 0;
    const r3::moe_backend::Backend* backend = r3::moe_backend::backend_for(arch);
    const int rc = backend ? r3::moe_backend::probe(*backend, shape, d.weight, caps) : (int)MOE_UNSUPPORTED_ARCH;
    if (verbose || rc != MOE_OK)
        std::fprintf(stderr,
                     "r3_moe_worker: container: %u layers, %u experts, hidden %u, expert width %u, weight format %s "
                     "(scale dtype %u, tile %ux%u), %s backend: %s\n",
                     shape.layers, shape.experts, shape.hidden, shape.intermediate, format_name(d.weight.format),
                     d.weight.scale_dtype, d.weight.block_n, d.weight.block_k, backend ? backend->name : arch,
                     rc == MOE_OK ? "supported" : rc == MOE_UNSUPPORTED_FORMAT ? "weight format not supported"
                     : rc == MOE_UNSUPPORTED_ARCH ? "architecture not supported" : "geometry not supported");
    return rc == MOE_OK;
}

/* Bytes one expert takes on the device. With a GPU this is the backend contract's size for the container's own description of the
 * expert (moe_backend.h), the same number for every weight format; without one it is the container's stored size. */
int64_t laid_out_bytes(const Container& f, r3_moe_gpu::Runner* gpu, uint32_t l, uint32_t e) {
    if (!gpu) return canonical_expert_bytes(f, l, e);
    MoeExpertDesc d{};
    if (!describe_expert(f, l, e, &d)) return -1;
    return r3::moe_backend::expert_bytes(d);
}

/* The shared-memory shape and layout for this container (RCS09-03): hidden, experts and expert width are the container's own; the
 * row capacity, top-k and activation group are the protocol's until the engine negotiates them. A shape other than the one the
 * execution paths run is refused by the backend probe before this (moe_backend.h), not here. */
bool segment_shape(const Container& f, r3::moe_shm::LayoutShape* shape, r3::moe_shm::Layout* lay) {
    MoeShape cs{};
    if (!container_shape(f, &cs)) {
        std::fprintf(stderr, "r3_moe_worker: the container's shape could not be read\n");
        return false;
    }
    *shape = r3::moe_shm::LayoutShape{cs.hidden, cs.top_k ? cs.top_k : r3::moe_shm::kTopK, cs.experts, r3::moe_shm::kMaxRows,
                                      r3::moe_shm::kActGroup, cs.intermediate};
    if (!r3::moe_shm::layout_for(*shape, lay)) {
        std::fprintf(stderr, "r3_moe_worker: the shared-memory layout for this shape is invalid\n");
        return false;
    }
    return true;
}
} // namespace
