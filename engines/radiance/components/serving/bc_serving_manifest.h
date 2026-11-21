/* bc_serving_manifest.h -- BigCherry's start-up extension point: the engine starts from a serving manifest v1 (the
 * document bc-serve --write-manifest writes) instead of command-line flags.
 *
 *   engine --manifest PATH [--role engine] [-v|-vv] [--bc-heat-launch-nonce HEX32]
 *
 * main() calls bc_serving_engine_argv before config_parse, so radiance's own parser and its own validation and
 * errors see the manifest's engine-bound values as the equivalent flags; no other flag may accompany --manifest.
 * After a successful parse, bc_serving_mark_given relabels the startup report's option sources. All policy
 * lives in core/bc_serving_manifest.cpp; --manifest is deliberately not a row of radiance's option table. */
#ifndef BC_SERVING_MANIFEST_H
#define BC_SERVING_MANIFEST_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "callback_boundary.h"

namespace rad { struct Config; class Engine; class StallWatch; }
namespace bc::serving::v2 { struct EnginePlan; struct Manifest; enum class PlanUse; }

/* The argv radiance should parse. Without --manifest, argv is left unchanged and true is returned. With it, *argc
 * and *argv are replaced by a synthetic argv (storage lives for the process). On a malformed manifest or a stray
 * flag a message is printed to stderr and false is returned: the caller exits 2, as for any config error. */
bool bc_serving_engine_argv(int* argc, char*** argv);

/* One typed production entry: preserve the vendor's YAML entrypoint exactly
 * as a callable and let owned serving admission handle only managed starts. */
int bc_serving_engine_entry(int argc, char** argv, const bc::serving::callback::Guarded<int(int, char**)>& run_stock);

/* After rad0011 extracts launch flags, prepare the single native Config from the strict
 * v3 snapshot. Returns 1 when prepared, 0 for other starts, and -1 on refusal. */
int bc_serving_take_native_config(rad::Config* out);

/* Serve the admitted engine behind a one-model HTTP front on fleet-capable Radiance. */
int bc_serving_managed_front(rad::Engine& engine, std::unique_ptr<rad::StallWatch> watch);

/* After a successful parse: every option the manifest produced must be recorded in `given` as "command line" (an
 * option another option swallowed as its argument is not), else a message is printed and false is returned: the
 * caller exits 2. Then those sources are relabelled "manifest <first 12 hex of metadata.content_digest>". Returns true
 * and does nothing when the engine was not started from a manifest. */
bool bc_serving_mark_given(std::map<std::string, std::string>* given);

/* The expected memory plan the engine must admit at start-up, and the kv_backing_floor_ppm that goes with it, taken from the
 * manifest's measured plan (engine_view; spec §5.2). True only when the engine was started from a measured manifest (a
 * serving start); false for a start without --manifest and for the measuring start (--plan-memory), which has no plan to
 * check. A serving start with no plan cannot get here: the strict reader refuses an unmeasured manifest for it. */
bool bc_serving_expected_plan(bc::serving::v2::EnginePlan* expected, int64_t* floor_ppm);
/* BDFs are from the same strictly validated engine view, not device ordinals. */
std::vector<std::string> bc_serving_engine_bdfs();

#endif /* BC_SERVING_MANIFEST_H */
