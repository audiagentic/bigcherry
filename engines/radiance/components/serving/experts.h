#pragma once
// The helpers' expert ownership, GPU-free (RSC03 slice 2): the .rad container reader and the owned-expert selection
// (heat, budget, balance, tail placement) the helper process runs, moved here unchanged so bc-serve plans with the same
// code the helper selects with. Only Radiance's installed container ABI (rad_format.h, rad_encoding.h) is read.
#include "moe_backend.h"
#include "moe_remote_shm.h"
#include "rad_encoding.h"
#include "rad_format.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace bc::experts {

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

/* What selection reads from the helper's settings: the heat ranking, the placement, the shares, the R3_MOE_WORKER_BALANCE
 * split and the weight formats the card runs. The helper fills it from its manifest section; bc-serve from the resolved
 * helper sections. */
struct OwnerPolicy {
    std::string heat, placement, formats;
    std::string shares;                    // R3_MOE_WORKER_SHARES text ("w0,w1,.."; "" = equal), parsed where the worker parsed it
    bool balance = false;
    std::vector<double> balance_speeds;    // two positive speeds; checked when balance is on
};

struct Owned {
    std::vector<std::pair<uint32_t,uint32_t>> expert;
    uint64_t bytes = 0;
};

int64_t canonical_expert_bytes(const Container& f, uint32_t l, uint32_t e);
bool describe_expert(const Container& f, uint32_t l, uint32_t e, MoeExpertDesc* out);
bool container_shape(const Container& f, MoeShape* s);
bool helper_backend(const char* arch, const char* formats, r3::moe_backend::Backend* out);
bool backend_accepts(const Container& f, const char* arch, const char* formats, bool verbose);
/* Bytes one expert takes on the device: with the device layout, the backend contract's size for the container's own
 * description of the expert (moe_backend.h); without, the container's stored size. */
int64_t laid_out_bytes(const Container& f, bool device_layout, uint32_t l, uint32_t e);
/* helper `helper` of `helpers`' owned experts, in allocation order; `arch` is its card's ("host" without one). */
/* layout_check (may be empty) runs after the budget and backend checks and before anything else, as the worker's device-layout
 * agreement did; false fails the selection. */
bool select_owned(const Container& f, const OwnerPolicy& p, const std::string& arch, bool device_layout,
                  const std::function<bool()>& layout_check, const uint64_t* budgets, uint32_t helper, uint32_t helpers,
                  Owned* out);

/* Every helper's owned experts at once, as each helper selects them (select_owned with its own index), and the experts no
 * helper owns, which stay with the engine (its card or its host pool). Fails with the failing helper named. */
struct OwnerMap {
    std::vector<Owned> helpers;
    std::vector<std::pair<uint32_t, uint32_t>> unowned;   // in (layer, expert) order
    uint64_t routed = 0;                                   // routed experts the container holds
};
bool plan_owners(const Container& f, const std::vector<OwnerPolicy>& policies, const std::vector<std::string>& archs,
                 const std::vector<uint64_t>& budgets, OwnerMap* out, std::string* why);

}  // namespace bc::experts
