/* bc_serving_engine.h -- the engine's side of the owner's admission (BigCherry managed patch rad0070, RSC02).
 *
 *   radiance --config FILE [--bc-admission-fd FD] [--plan-memory F --plan-memory-ranks R] [--bc-heat-launch-nonce HEX32]
 *            [--bc-step-phase-dir DIR]
 *
 * main() hands its whole stock body to bc_serving_engine_entry as a callable. The owner's flags are taken first
 * (bc_serving_managed.h), then rad0011's; the callable then sees only `--config FILE`, so the stock entry and the stock
 * Fleet run unchanged. Without the owner's flag nothing is installed and every getter reports "not a managed start". */
#ifndef BC_SERVING_ENGINE_H
#define BC_SERVING_ENGINE_H

#include <cstdint>
#include <string>
#include <vector>

#include "callback_boundary.h"

namespace bc::serving::v2 { struct EnginePlan; }

int bc_serving_engine_entry(int argc, char** argv, const bc::serving::callback::Guarded<int(int, char**)>& run_stock);

/* The expected memory plan the engine must admit at start-up, and the kv_backing_floor_ppm that goes with it. True only for
 * a managed serving start; false for any other start and for the measuring start (--plan-memory), which has no plan. */
bool bc_serving_expected_plan(bc::serving::v2::EnginePlan* expected, int64_t* floor_ppm);

/* The selected calibrated KV backing floor, including for a measuring start before an expected plan exists. Zero only
 * when not selected. */
int64_t bc_serving_kv_backing_floor_ppm();
/* The ordered rank BDFs of the admission, not device ordinals. */
std::vector<std::string> bc_serving_engine_bdfs();

#endif /* BC_SERVING_ENGINE_H */
