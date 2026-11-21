/* bc_rank_engine.cpp -- BC_VTP14_N1: see bc_rank_engine.h. Also the Registry members for the rank list and the per-library
 * rank mask (declared by the managed registry.h edit). */
#include "bc_rank_engine.h"

#include "../device/device.h"
#include "../rad_internal.h"
#include "registry.h"

namespace rad {

/* The rank list is fixed before any row exists, so the masks rows carry never change while rank threads select. */
int Registry::set_code_devices(const std::vector<int>& devices) {
    if (shared() || !set_->rows.empty() || devices.empty() || devices.size() > 32) return RAD_E_INVAL;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] < 0) return RAD_E_INVAL;
        for (size_t j = 0; j < i; ++j)
            if (devices[j] == devices[i]) return RAD_E_INVAL;
    }
    set_->code_devices = devices;
    set_->code_device = devices[0];
    return RAD_OK;
}

uint32_t Registry::rank_bits(int rank) const {
    if (set_->code_devices.empty()) return 0;
    const size_t n = set_->code_devices.size();
    const uint32_t all = n >= 32 ? ~0u : ((1u << n) - 1u);
    if (rank < 0) return all;
    if ((size_t)rank >= n) return ~0u;   /* more bits than any device row carries: no match */
    return 1u << rank;
}

/* With a rank list, a library's code is checked against each rank's card. Its device rows stay when any rank runs them,
 * carrying the mask; selection offers a row only to those ranks. A library no rank runs loses its device rows exactly as on
 * the one-device path: the mask is 0 and `why` names each rank that cannot run it. */
uint32_t Registry::rank_code_mask(const std::string& plugin, size_t code_from, size_t code_to, int n_device,
                                  std::string* why) const {
    uint32_t mask = 0;
    std::string skipped;
    for (size_t i = 0; i < set_->code_devices.size(); ++i) {
        std::string why_i;
        int runs = 0;
        if (set_->no_device_code) why_i = *why;
        else if (set_->code_probe) runs = set_->code_probe(plugin, set_->code_devices[i], &why_i);
        else runs = rad::dev_code_runs(code_from, code_to, set_->code_devices[i], &why_i);
        if (runs > 0) {
            mask |= 1u << i;
        } else {
            skipped += fmt("%srank %zu (device %d: %s)", skipped.empty() ? "" : ", ", i,
                           set_->code_devices[i], why_i.empty() ? "no code object for it" : why_i.c_str());
        }
    }
    if (mask != 0 && !skipped.empty())
        RAD_INFO("kernel plugin '%s': its %d device kernel(s) are offered only to the ranks whose "
                 "card runs them; not to %s", plugin.c_str(), n_device, skipped.c_str());
    if (mask == 0) *why = skipped;
    return mask;
}

std::vector<BcDeviceInfo> bc_device_inventory() {
    std::vector<BcDeviceInfo> inventory;
    for (int i = 0, n = rad_dev_count(); i < n; ++i) {
        RadDeviceProps p{};
        if (rad_dev_props(i, &p) < 0) continue;
        inventory.push_back({ i, dev_pci_address(i), p.arch, p.name, dev_integrated(i) });
    }
    return inventory;
}

int bc_admit_rank_devices(const Config& cfg, BcRankDevicesAdmission* out) {
    BcRankDevicesRequest request;
    request.want = cfg.rank_devices;
    request.tp = cfg.tp;
    request.wire_exact = cfg.tp_wire_exact != 0;
    request.wire_unvalidated = cfg.tp_wire_unvalidated;
    request.fleet_devices = cfg.devices;
    const int status = bc_rank_devices_admit(request, bc_device_inventory(), out);
    for (const std::string& w : out->warnings) RAD_WARN("%s", w.c_str());
    if (status != RAD_OK) RAD_ERR("%s", out->refusal.c_str());
    return status;
}

BcProgramReceipt bc_program_receipt(const Program& program) {
    BcProgramReceipt receipt;
    for (const OpInfo& op : program.ops)
        for (const Band& b : op.bands) {
            for (int dm = 0; dm < RAD_N_DOMAINS; ++dm)
                if (const KernelRow* row = b.dom[dm].row)
                    ++receipt.bands_by_library[row->plugin + (dm == RAD_DOMAIN_HOST ? " (host)" : "")];
            const KernelRow* row = b.dom[RAD_DOMAIN_DEVICE].row;
            if (row && row->info && row->collective)
                receipt.collectives.push_back({ op.op, row->info->name, row->plugin,
                                                b.span + ": " + b.dom[RAD_DOMAIN_DEVICE].geom.str() });
        }
    return receipt;
}

std::string bc_rank_receipt_line(int rank, int device, const std::string& pci, const std::string& arch,
                                 const BcProgramReceipt& receipt) {
    std::string libs;
    for (const auto& [lib, n] : receipt.bands_by_library)
        libs += fmt("%s%s %d", libs.empty() ? "" : ", ", lib.c_str(), n);
    const std::vector<BcCollStep>& c = receipt.collectives;
    return fmt("rank receipt: rank %d on device %d %s %s; op bands by library: %s; %zu collective "
               "band(s)%s%s", rank, device, pci.c_str(), arch.c_str(),
               libs.empty() ? "none" : libs.c_str(), c.size(), c.empty() ? "" : ", first ",
               c.empty() ? "" : (c[0].kernel + " (" + c[0].plugin + ")").c_str());
}

std::vector<BcCollStep> bc_report_rank_program(const Program& program, int rank, int device) {
    const BcProgramReceipt receipt = bc_program_receipt(program);
    RadDeviceProps dp{};
    (void)rad_dev_props(device, &dp);
    RAD_INFO("%s", bc_rank_receipt_line(rank, device, dev_pci_address(device), dp.arch, receipt).c_str());
    return receipt.collectives;
}

}  /* namespace rad */
