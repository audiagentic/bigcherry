#pragma once

// The .rad container reader and the process signal flag. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {
using namespace r3::moe_shm;
/* The stop word: set by the signal handler and by any failing thread, and futex-woken so the main thread sleeps on it
 * instead of polling. A lock-free atomic store plus the raw futex syscall are async-signal-safe. */
std::atomic<uint32_t> g_stop_word{0};
static_assert(std::atomic<uint32_t>::is_always_lock_free && sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
inline bool stopping() { return g_stop_word.load(std::memory_order_acquire) != 0; }
inline void request_stop() {
    g_stop_word.store(1, std::memory_order_release);
    (void)::syscall(SYS_futex, reinterpret_cast<uint32_t*>(&g_stop_word), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}
void on_signal(int) { request_stop(); }

float bf16(uint16_t v) {
    uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f;
}
uint16_t to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}
float f16(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    if (!e) return std::ldexp((s ? -1.f : 1.f) * (float)m, -24);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    return std::ldexp((s ? -1.f : 1.f) * (1024.f + (float)m), (int)e - 25);
}
float f8e4m3(uint8_t b) {
    const float sign = (b & 0x80) ? -1.f : 1.f;
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 0) return sign * std::ldexp((float)m, -9);
    if (e == 15 && m == 7) return NAN; /* OCP E4M3FN's sole NaN payload. */
    return sign * std::ldexp(1.f + (float)m / 8.f, e - 7);
}

uint64_t ubits(const uint8_t* row, int bits, int64_t col) {
    const uint64_t bit = (uint64_t)col * (uint64_t)bits;
    const uint64_t byte = bit >> 3, shift = bit & 7;
    uint64_t x = row[byte];
    if (shift + (uint64_t)bits > 8) x |= (uint64_t)row[byte + 1] << 8;
    return (x >> shift) & ((1ull << bits) - 1);
}
int64_t sbits(uint64_t x, int bits) {
    const uint64_t sign = 1ull << (bits - 1);
    return (int64_t)((x ^ sign) - sign);
}

/* The helper deliberately consumes only Radiance's installed container ABI (rad_format.h and
 * rad_encoding.h): no core/private reader or kernel layout is linked into this reference path. */
struct Container {
    int fd = -1;
    size_t bytes = 0;
    RadFileHeader hdr{};
    const RadFileHeader* h = &hdr;
    std::vector<RadFileEntry> dir_store;
    std::vector<RadFilePlane> plane_store;
    std::vector<RadEncoding> enc_store;
    std::vector<RadFileProfile> profile_store;
    const RadFileEntry* dir = nullptr;
    const RadFilePlane* planes = nullptr;
    const RadEncoding* encs = nullptr;
    const RadFileProfile* profile = nullptr;
    uint32_t layers = 0;
    uint32_t experts = 0;      /* routed experts per layer: the highest expert id in the directory + 1 */
    uint32_t hidden = 0;       /* model hidden width: rows of an expert's down matrix */
    uint32_t top_k = 0;        /* routed experts per token (metadata n_expert_used); 0 when the container does not say */
    uint32_t intermediate = 0; /* expert width: columns of an expert's down matrix */
    std::vector<std::array<const RadFileEntry*, 2>> expert;

    ~Container() { if (fd >= 0) close(fd); }
    bool range(uint64_t off, uint64_t n) const { return off <= bytes && n <= bytes - off; }
    bool pread_full(uint64_t off, void* dst, size_t n) const {
        if (!range(off, n)) return false;
        auto* p = static_cast<uint8_t*>(dst);
        while (n) {
            const ssize_t got = ::pread(fd, p, n, (off_t)off);
            if (got < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (got == 0) return false;
            p += (size_t)got;
            off += (uint64_t)got;
            n -= (size_t)got;
        }
        return true;
    }
    bool open(const char* path) {
        fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) || st.st_size < (off_t)sizeof(RadFileHeader)) return false;
        bytes = (size_t)st.st_size;
        if (!pread_full(0, &hdr, sizeof(hdr))) return false;
        if (hdr.magic != RAD_MAGIC || hdr.version != RAD_FORMAT_VER || hdr.file_bytes > bytes ||
            !range(hdr.dir_off, hdr.dir_count * sizeof(RadFileEntry)) ||
            !range(hdr.plane_off, hdr.plane_count * sizeof(RadFilePlane)) ||
            !range(hdr.enc_off, hdr.enc_count * sizeof(RadEncoding)) ||
            (hdr.prof_count && !range(hdr.prof_off, hdr.prof_count * sizeof(RadFileProfile)))) return false;
        dir_store.resize((size_t)hdr.dir_count);
        plane_store.resize((size_t)hdr.plane_count);
        enc_store.resize((size_t)hdr.enc_count);
        profile_store.resize((size_t)hdr.prof_count);
        if ((hdr.dir_count && !pread_full(hdr.dir_off, dir_store.data(), dir_store.size() * sizeof(RadFileEntry))) ||
            (hdr.plane_count && !pread_full(hdr.plane_off, plane_store.data(), plane_store.size() * sizeof(RadFilePlane))) ||
            (hdr.enc_count && !pread_full(hdr.enc_off, enc_store.data(), enc_store.size() * sizeof(RadEncoding))) ||
            (hdr.prof_count && !pread_full(hdr.prof_off, profile_store.data(), profile_store.size() * sizeof(RadFileProfile))))
            return false;
        {   // the model's routes per token, from the container's metadata
            std::vector<char> strs;
            std::vector<RadFileKV> meta;
            if (hdr.str_bytes && hdr.str_bytes < (1ull << 28) && hdr.meta_count && hdr.meta_count < (1ull << 20) &&
                range(hdr.str_off, hdr.str_bytes) && range(hdr.meta_off, hdr.meta_count * sizeof(RadFileKV))) {
                strs.resize((size_t)hdr.str_bytes + 1, 0);
                meta.resize((size_t)hdr.meta_count);
                if (pread_full(hdr.str_off, strs.data(), (size_t)hdr.str_bytes) &&
                    pread_full(hdr.meta_off, meta.data(), meta.size() * sizeof(RadFileKV)))
                    for (const RadFileKV& kv : meta)
                        if (kv.type == RAD_P_INT && kv.key < hdr.str_bytes && std::strcmp(&strs[(size_t)kv.key], "n_expert_used") == 0 &&
                            kv.v.i > 0 && kv.v.i <= 64)
                            top_k = (uint32_t)kv.v.i;
            }
        }
        dir = dir_store.data();
        planes = plane_store.data();
        encs = enc_store.data();
        profile = profile_store.empty() ? nullptr : profile_store.data();
        int32_t max_layer = -1, max_expert = -1;
        for (uint64_t i = 0; i < hdr.dir_count; ++i)
            if (dir[i].layer >= 0 && dir[i].expert >= 0) {
                max_layer = std::max(max_layer, dir[i].layer);
                max_expert = std::max(max_expert, dir[i].expert);
            }
        layers = (uint32_t)(max_layer + 1);
        experts = (uint32_t)(max_expert + 1);
        for (uint64_t i = 0; i < hdr.dir_count && !hidden; ++i)
            if (dir[i].layer >= 0 && dir[i].expert >= 0 && dir[i].slot == 1) {
                int64_t nr = 1, nc = 1;
                rad_enc_view(dir[i].rank, dir[i].shape, &nr, &nc);
                if (nr > 0 && nc > 0 && nr < (1ll << 20) && nc < (1ll << 20)) { hidden = (uint32_t)nr; intermediate = (uint32_t)nc; }
            }
        if (!hidden) return false;
        if (!layers || !experts || (uint64_t)layers * experts > (1ull << 24)) return false;   /* a directory this large is not a model */
        expert.resize((size_t)layers * experts);
        for (uint64_t i = 0; i < hdr.dir_count; ++i) {
            const auto& e = dir[i];
            if (e.layer < 0 || e.expert < 0 || e.expert >= (int)experts || e.slot < 0 || e.slot >= 2) continue;
            if ((uint32_t)e.layer >= layers || e.enc >= hdr.enc_count || e.plane_first + e.n_planes > hdr.plane_count) return false;
            for (uint32_t p = 0; p < e.n_planes; ++p)
                if (!range(planes[e.plane_first + p].offset, planes[e.plane_first + p].bytes)) return false;
            expert[(size_t)e.layer * experts + (uint32_t)e.expert][e.slot] = &e;
        }
        return layers != 0;
    }
    const RadFileEntry* weight(uint32_t layer, uint32_t exp, uint32_t slot) const {
        if (layer >= layers || exp >= experts || slot >= 2) return nullptr;
        return expert[(size_t)layer * experts + exp][slot];
    }
    const RadEncoding* encoding(const RadFileEntry* e) const { return e ? &encs[e->enc] : nullptr; }
    const RadFilePlane* plane(const RadFileEntry* e, int p) const { return &planes[e->plane_first + p]; }
};

} // namespace
