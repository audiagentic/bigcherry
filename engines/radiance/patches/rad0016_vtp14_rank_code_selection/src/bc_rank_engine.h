/* bc_rank_engine.h -- BC_VTP14_N1: the engine-side parts of --rank-devices that need the engine's own types (the
 * configuration, the declared program, the device backend). BigCherry managed patch rad0016. The decisions they make are
 * the card-free ones in bc_rank_devices.h; this file only gathers their inputs and logs their answers. */
#pragma once

#include "bc_rank_devices.h"
#include "rad_core.h"

#include <map>
#include <string>
#include <vector>

namespace rad {

/* The cards the device backend reports, in backend order (a card whose properties cannot be read is skipped). */
std::vector<BcDeviceInfo> bc_device_inventory();

/* The engine's --rank-devices admission from its configuration: gathers the inventory, decides with
 * bc_rank_devices_admit and logs the warnings (in order) and the refusal. RAD_OK or the admission's error status. */
int bc_admit_rank_devices(const Config& cfg, BcRankDevicesAdmission* out);

/* What one rank's declared program runs: its op bands counted by the library each came from ("<library>" and
 * "<library> (host)"), and the collectives it issues, in program order, with their frozen launch geometry. */
struct BcProgramReceipt {
    std::map<std::string, int> bands_by_library;
    std::vector<BcCollStep>    collectives;
};
BcProgramReceipt bc_program_receipt(const Program& program);

/* The log line a rank's receipt is reported as (rank, card, libraries, collectives). */
std::string bc_rank_receipt_line(int rank, int device, const std::string& pci, const std::string& arch,
                                 const BcProgramReceipt& receipt);

/* Reports rank `rank`'s receipt on `device` and returns its collectives for the cross-rank agreement check. */
std::vector<BcCollStep> bc_report_rank_program(const Program& program, int rank, int device);

}  /* namespace rad */
