#pragma once
/* The manifest's owner map checked against the GPU-free planner (spec §4; RSC08 slice S1).
 *
 * spec.plan.owner_map holds each helper's owned experts, their count, their device bytes and the digests. The reader
 * (v1/manifest.cpp) checks what the document says about itself; only the planner knows what the model's experts really
 * take, so the bytes and the selection are checked here against bc::experts::plan_owners for the placement the manifest
 * itself declares (handoff.placement, each helper's tier/share/budget/formats and its effective balance settings).
 *
 * Used by the producer (bc-serve, before it writes a manifest) and by a helper at start-up (S4a: it verifies its own
 * initial owned set before GPU work). GPU-free; needs the model container. */
#include "experts.h"
#include "manifest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace bc::serving {

/* One bc::experts::OwnerPolicy per helper, in manifest order, built only from the manifest: the heat file and strategy
 * from handoff.placement, the shares from each helper's `share`, and the formats from its `formats`; balance and its
 * speeds (milli, so 1000 is 1.0) from the helper's effective settings. */
std::vector<bc::experts::OwnerPolicy> owner_policies(const v2::Manifest& m);
/* The helpers' cards (gfx targets, from spec.facts.gpus by address) and budgets in bytes, in manifest order. */
std::vector<std::string> helper_archs(const v2::Manifest& m);
std::vector<uint64_t> helper_budgets(const v2::Manifest& m);

/* Recomputes the owner map with the shared planner and compares it with the manifest's, helper by helper: the owned
 * experts and their order, the count, and the exact device bytes. BC_CFG_OWNERSHIP_MISMATCH naming the first
 * disagreeing pointer (or the planner's refusal) otherwise. It does not rely on the reader having run: before it plans,
 * it refuses (naming the field) an owner map that does not have one set per helper, a count that is not its experts', a
 * helper GPU absent from the facts (unresolved reference), a budget that does not fit 64 bits of bytes, a share above the
 * selector's 1,000,000 or all shares 0, a tier that contradicts the strategy (tail: the last helper is the tail helper,
 * every other is head; roundrobin: all head), and a balance the selector cannot run (not exactly two helpers of equal
 * budget, or with the tail strategy). The digests are not rechecked here: the reader verifies them
 * against the document's own experts, so equal experts mean equal digests. */
v2::Status verify_owner_map(const v2::Manifest& m, const bc::experts::Container& model);

}  // namespace bc::serving
