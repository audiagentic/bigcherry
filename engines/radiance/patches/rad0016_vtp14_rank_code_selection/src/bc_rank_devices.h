/* bc_rank_devices.h -- BC_VTP14_N1: an explicit, ordered rank -> physical card assignment
 * (--rank-devices), and the startup checks a tensor-parallel world whose ranks run different card
 * architectures needs. BigCherry managed patch rad0016; pure functions over plain inputs, so the
 * refusals are tested without a card (tests/bc_rank_devices_test.cpp).
 *
 * A world of one architecture keeps radiance's own device choice (engine_bringup.cpp
 * usable_devices). A mixed world is only ever an explicit list: no `--tp N` alone reaches it. */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* One card as the device backend reports it. `pci` is "dddd:bb:dd.f" (lower case), "" when the
 * backend has none (the host backend). */
struct BcDeviceInfo {
    int         ordinal = -1;
    std::string pci;
    std::string arch;
    std::string name;
    bool        integrated = false;
};

/* "0000:03:00.0" (any case) -> its lower-case canonical form; false for anything else. A short
 * "03:00.0" is refused: the domain is part of a card's identity. */
bool bc_parse_bdf(const std::string& s, std::string* canonical);

/* Split "a,b,c" into canonical addresses. False with `why` on an empty item or a malformed address. */
bool bc_parse_bdf_list(const std::string& csv, std::vector<std::string>* out, std::string* why);

/* The ranks' device ordinals, rank i on want[i]. Refused (RAD_E_DEVICE, reason in `why`) when the
 * list does not name exactly `tp` cards, names one twice, names a card the backend does not
 * expose, names an integrated GPU, or the backend reports no PCI addresses. */
int bc_rank_devices_plan(const std::vector<BcDeviceInfo>& inventory, const std::vector<std::string>& want,
                         int tp, std::vector<int>* ordinals, std::string* why);

/* The distinct architectures of the chosen cards, in rank order of first appearance. */
std::vector<std::string> bc_rank_archs(const std::vector<BcDeviceInfo>& inventory,
                                       const std::vector<int>& ordinals);

/* One collective a rank's program issues: its op, the selected kernel, the library it came from and
 * its frozen launch geometry (Resolved::geom, with the tuned axes -- nblocks, nthreads, drain, acq --
 * folded in). Ranks of different architectures take a collective from different libraries (libr3 on
 * gfx1100, libr4d on gfx1201), so the library may differ; the kernel is the protocol and the geometry
 * is its launch shape, so neither may: a block whose twin on a peer is never launched spins forever.
 * An empty geom is not compared. */
struct BcCollStep {
    std::string op;
    std::string kernel;
    std::string plugin;
    std::string geom;
};

/* RAD_OK when every rank issues the same collectives in the same order with the same kernels and
 * launch geometry; otherwise RAD_E_STATE with the first difference named (rank, position, op, kernel,
 * geometry). A
 * collective's peers meet by arrival order, so any difference would hang or corrupt the exchange. */
int bc_rank_collectives_agree(const std::vector<std::vector<BcCollStep>>& per_rank, std::string* why);

}  /* namespace rad */
