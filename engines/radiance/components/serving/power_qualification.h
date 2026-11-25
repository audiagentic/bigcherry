#pragma once
// PWR08 S3: what the S10 resolver needs from the power controller, derived from facts the controller already
// produces. No second probe and no qualification store: a setting is qualified when a durable receipt of an earlier
// apply on the same BDF and board identity shows the exact readback and the exact restore.
#include "power_transaction.h"
#include "v2/s10_resolver.h"

#include <filesystem>
#include <optional>
#include <string_view>

namespace bc::serving::power {

// The transaction's own capability read (Hardware::inspect), in S10 control names.
v2::S10DeviceProbe probe_of(const CardSnapshot& card);

// Run id of a receipt under `receipts` that qualifies `settings` for `pci` on `identity`, or nothing. A receipt
// qualifies when it is an apply that ended restored, the card has an applied and a restored event and no failure or
// rollback event, the request carried exactly these settings, the post-apply values read back equal them and the
// stop values equal the start values. The workload recorded in a receipt is evidence only.
std::optional<std::string> qualifying_receipt(const std::filesystem::path& receipts, const PciAddress& pci,
                                              const v2::PowerSettings& settings, std::string_view identity);

} // namespace bc::serving::power
