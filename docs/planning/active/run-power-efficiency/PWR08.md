---
id: PWR08
order: 0
plan: run-power-efficiency
state: in_progress
created-at: '2026-11-10T13:32:38.583551+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# PWR08: named power profiles in the serving manifest (bigcherry/v2)

## Description

Owner decision (2026-11-10): the selected power profile is part of the serving manifest. GPT review (req_main_eb983900e21b) established that serving-manifest v1 is closed (spec.md section 8: any added field is bigcherry/v2; helper runtime schema is closed; the strict reader rejects unknown fields), so this is a v2 change, migrated producer/readers/plugins/fixtures/docs together, no v1 translator. Minimal v2 surface: spec.intent.power {profile, overrides by role} (operator intent, class policy) and spec.plan.power {mode observe|apply, profile, resolved per-role/BDF settings} (derived by the C++ resolver from layout roles, profile, host overrides and, for selection=calibrated, a qualified calibration record). Not in the manifest: live sensors, applied values, receipts (run evidence, PWR07/PWR09). runtime.power_save stays where it is with its meaning. Named reusable profiles live in config/power-profiles.toml (stock/efficient/performance/quiet, role-level, no BDFs); host calibration by BDF stays in environment.local.toml; measured records under BC_WORK_ROOT/power-calibration keyed by BDF, driver, topology, workload, build. A cached measurement never activates itself: selection needs explicit policy intent.

## Steps

1. Decide scope with the other known v1 gaps (RSC09 launcher literals, RSC10 margin refit) so there is ONE v2 migration, not several. 2. Registry successor of serving-v1.toml, generated schema/codec, field reference. 3. Profile file schema and role resolution in the C++ resolver (engine, helper head, helper tail from tier; per-BDF overrides; unknown role = error; duplicate BDF = error). 4. Provenance records through the existing manifest_writer pointer mapping. 5. Shared reader, semantic validation, fixtures, negative tests. 6. State the memory-plan cache key's relationship to power (power does not change memory geometry; performance calibration keys include the resolved power plan). 7. Migrate every consumer in one change (bc-serve, engine, plugins, launcher plan-check, tools tests, docs).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free: the same resolved power plan appears in unmeasured and measured manifests; changing the profile file after manifest generation changes nothing that is applied; unknown/duplicate roles and missing BDFs fail resolution; observation mode has an explicit empty settings map per role; registry fingerprints regenerate and CI generator-diff passes; all consumers reject v1 and accept v2 (no translator).

## Effort & Risk



## Standards

No legacy / no backward compatibility: v2 replaces v1, no aliases.

## Acceptance Criteria



## Notes

Depends on PWR07 (probe) and feeds PWR09. Challenged against GPT's proposal: keep spec.facts.power out of the first v2 unless the resolver needs a typed capability fact to refuse a profile (decide in step 1; stable driver/range metadata only, never an authority to write: the controller re-probes immediately before applying). CPU EPP/governor and non-AMDGPU intent can be designed in the schema but stay observe-only until a CPU-policy controller and ownership model exist. The v2 cost is high (producer, readers, plugins, bindings, fixtures, docs together): do it once, with every other pending v1 gap.

## Change Log

- 2026-11-10T13:32:38.583551+00:00 (created-by): Created by agent


## Update 2026-11-10 (PWR06 R9700 GFXOFF finding)

Kernel and firmware are host prerequisites, never manifest settings (PWR06: the GFXOFF fix is an amdgpu driver change). A proven runtime optimisation for the R9700 idle floor, if any, becomes a profile control here.


## Implementation status: 2026-11-10

Merged: #824 named profile file/reader with stock, efficient, quiet, performance and the data-only XTX -125 mV candidate; #825 inactive per-BDF resolver with qualified-selection and fresh-probe gates. Both are observe-first. The XTX -125 mV candidate needs a same-session A/B and explicit qualifying record before apply; no R9700 or 6900 XT optimum is validated. Still blocked: S9 v2 registry/generator, S10 C++ resolver/writer integration and S13 single active v2 migration. The Python supervisor S12 only consumes a validated v2 power plan and does not activate profiles.
