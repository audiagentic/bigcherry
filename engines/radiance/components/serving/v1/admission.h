#pragma once
/* Start-up check of a live engine memory plan against the manifest's expected one (spec §5.2, amendment 5; RSC08
 * acceptance). Typed, structured: invalid input is BC_CFG_INVALID_VALUE, stable drift BC_CFG_PLAN_MISMATCH, and a
 * live split that cannot hold what the plan requires BC_CFG_ADMISSION_REFUSED. The shared vectors in
 * docs/design/serving-manifest-v1/admission-vectors.json are the cases it must reproduce. */
#include "error.h"
#include "serving_types.h"

#include <cstdint>

namespace bc::serving::v1 {

/* kv_required = max(one full-length session's KV, ceil(kv_ceiling * ppm / 10^6)), without a product past 2^63. */
int64_t kv_required(int64_t kv_ceiling, int64_t kv_one_session, int64_t kv_backing_floor_ppm);

/* Checks, in order: ppm in [0, 10^6) and every VRAM term a byte count (invalid value); stable facts by exact
 * equality - host consumers (name, bytes, residence, inventory), checkpoints, one session's KV, host_pool_mib, statics,
 * expert floor and ceiling, KV ceiling, card count and expert-plane geometry, card capacity as a minimum (plan
 * mismatch); then admission on the live split - experts >= expert_floor, claimable >= statics + experts + stage +
 * kv_required, kv_resolved >= kv_required (admission refused). */
Status check_engine_plan(const EnginePlan& expected, const EnginePlan& live, int64_t kv_backing_floor_ppm);

}  // namespace bc::serving::v1
