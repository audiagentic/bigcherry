/* bc_rank_devices.cpp -- BC_VTP14_N1: see bc_rank_devices.h. */
#include "bc_rank_devices.h"

#include "rad_abi.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace rad {

namespace {
bool hex_run(const std::string& s, size_t at, size_t n) {
    if (at + n > s.size()) return false;
    for (size_t i = at; i < at + n; ++i)
        if (!std::isxdigit((unsigned char)s[i])) return false;
    return true;
}
}  /* namespace */

bool bc_parse_bdf(const std::string& s, std::string* canonical) {
    /* dddd:bb:dd.f -- 12 characters, the spelling /sys/bus/pci/devices and the HIP backend use. */
    if (s.size() != 12 || s[4] != ':' || s[7] != ':' || s[10] != '.') return false;
    if (!hex_run(s, 0, 4) || !hex_run(s, 5, 2) || !hex_run(s, 8, 2)) return false;
    if (s[11] < '0' || s[11] > '7') return false;
    /* A device number is five bits. */
    const int dev = std::stoi(s.substr(8, 2), nullptr, 16);
    if (dev > 0x1f) return false;
    if (canonical) {
        *canonical = s;
        for (char& c : *canonical) c = (char)std::tolower((unsigned char)c);
    }
    return true;
}

bool bc_parse_bdf_list(const std::string& csv, std::vector<std::string>* out, std::string* why) {
    out->clear();
    size_t start = 0;
    for (;;) {
        const size_t comma = csv.find(',', start);
        const std::string item = csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        std::string canon;
        if (item.empty()) {
            if (why) *why = "an empty card address in '" + csv + "'";
            return false;
        }
        if (!bc_parse_bdf(item, &canon)) {
            if (why) *why = "'" + item + "' is not a PCI address (domain:bus:device.function, e.g. 0000:03:00.0)";
            return false;
        }
        out->push_back(canon);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return true;
}

int bc_rank_devices_plan(const std::vector<BcDeviceInfo>& inventory, const std::vector<std::string>& want,
                         int tp, std::vector<int>* ordinals, std::string* why) {
    ordinals->clear();
    auto refuse = [&](std::string msg) {
        if (why) *why = std::move(msg);
        ordinals->clear();
        return RAD_E_DEVICE;
    };
    if ((int)want.size() != tp)
        return refuse("names " + std::to_string(want.size()) + " card(s) for --tp " + std::to_string(tp) +
                      "; give exactly one card per rank, rank 0 first");
    bool any_pci = false;
    for (const BcDeviceInfo& d : inventory) any_pci |= !d.pci.empty();
    if (!any_pci)
        return refuse("this device backend reports no PCI addresses (the host backend has none), so a card "
                      "cannot be named; run without --rank-devices");
    for (size_t r = 0; r < want.size(); ++r) {
        std::string canon;
        if (!bc_parse_bdf(want[r], &canon))
            return refuse("rank " + std::to_string(r) + ": '" + want[r] + "' is not a PCI address");
        for (size_t q = 0; q < r; ++q)
            if (want[q] == canon)
                return refuse("ranks " + std::to_string(q) + " and " + std::to_string(r) + " both name card " + canon +
                              "; every rank needs its own card");
        const BcDeviceInfo* hit = nullptr;
        for (const BcDeviceInfo& d : inventory)
            if (d.pci == canon) {
                if (hit)
                    return refuse("card " + canon + " is reported twice by the device backend");
                hit = &d;
            }
        if (!hit) {
            std::string seen;
            for (const BcDeviceInfo& d : inventory)
                seen += (seen.empty() ? "" : ", ") + (d.pci.empty() ? std::string("?") : d.pci) + " " + d.arch;
            return refuse("rank " + std::to_string(r) + ": no visible card at " + canon + " (visible: " + seen +
                          "); ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES may hide it");
        }
        if (hit->integrated)
            return refuse("rank " + std::to_string(r) + ": " + canon + " is integrated graphics; a rank placed on "
                          "one runs slowly and wrongly");
        ordinals->push_back(hit->ordinal);
    }
    return RAD_OK;
}

std::vector<std::string> bc_rank_archs(const std::vector<BcDeviceInfo>& inventory,
                                       const std::vector<int>& ordinals) {
    std::vector<std::string> out;
    for (int o : ordinals)
        for (const BcDeviceInfo& d : inventory)
            if (d.ordinal == o && std::find(out.begin(), out.end(), d.arch) == out.end()) out.push_back(d.arch);
    return out;
}

int bc_rank_collectives_agree(const std::vector<std::vector<BcCollStep>>& per_rank, std::string* why) {
    if (per_rank.size() < 2) return RAD_OK;
    const std::vector<BcCollStep>& ref = per_rank[0];
    for (size_t r = 1; r < per_rank.size(); ++r) {
        const std::vector<BcCollStep>& got = per_rank[r];
        const size_t n = std::min(ref.size(), got.size());
        for (size_t i = 0; i < n; ++i) {
            if (ref[i].op == got[i].op && ref[i].kernel == got[i].kernel) {
                if (ref[i].geom.empty() || got[i].geom.empty() || ref[i].geom == got[i].geom) continue;
                if (why)
                    *why = "collective " + std::to_string(i) + " (" + ref[i].op + ", " + ref[i].kernel +
                           "): rank 0 launches it as [" + ref[i].geom + "] (" + ref[i].plugin + "), rank " +
                           std::to_string(r) + " as [" + got[i].geom + "] (" + got[i].plugin + "); a block whose "
                           "twin on a peer is never launched spins forever, so every rank must use one launch "
                           "geometry (check each library's tuned choice for this kernel)";
                return RAD_E_STATE;
            }
            if (why)
                *why = "collective " + std::to_string(i) + ": rank 0 issues " + ref[i].op + " with " + ref[i].kernel +
                       " (" + ref[i].plugin + "), rank " + std::to_string(r) + " issues " + got[i].op + " with " +
                       got[i].kernel + " (" + got[i].plugin + "); the ranks' peers meet by arrival order, so "
                       "every rank must issue the same collective kernels in the same order";
            return RAD_E_STATE;
        }
        if (ref.size() != got.size()) {
            if (why)
                *why = "rank 0 issues " + std::to_string(ref.size()) + " collective(s), rank " + std::to_string(r) +
                       " issues " + std::to_string(got.size()) + "; a rank with an extra collective waits forever "
                       "for peers that never arrive";
            return RAD_E_STATE;
        }
    }
    return RAD_OK;
}

}  /* namespace rad */
