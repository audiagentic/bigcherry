---
id: RSC04
order: 4
plan: run-serving-configuration
state: in_progress
created-at: '2026-11-02T00:53:10.689343+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P2
work: S
---

# Retire legacy launch surfaces once their callers migrate

## Description

After RSC02/RSC03, RSC12 and RCS10/RHP09, retire the bash launch and Python serving start paths and the duplicated runtime-profile argument spellings. **Do not merge a deletion until RSC11 S15 passes on hardware on the same build for dense and helper configurations and release acceptance succeeds.** RSP01 phase 2 owns the Python-free release asset/installer and workflow; do not edit those files here. There is no backward-compatibility shim: callers adopt the native RSC12 config/manifest contract or remain explicitly blocked.

## Steps

1. Record the exact callers, native commands, flags and remaining gates (inventory below). RSC02 step 6 first, independent of hardware.
2. Test production `bc-serve serve/start` call graphs using fake engine/helper processes and real config-loader, writer, strict v2 reader, measuring plan and guards. Add genuinely missing typed functionality; never turn old positional flags into a second settings system.
3. Execute the S15 short leased same-build A/B acceptance runbook (dense + helper, semantic execution projection and RHP12 gates). A card-free dry run is necessary but not proof of GPU parity.
4. Migrate eligible callers and tests without removing the launcher; CI must enforce that migrated paths invoke the real native owner and do not silently use the shell. Separate release installer ownership stays with RSP01.
5. After S15 and RSP01 acceptance, merge the separately reviewed S16 deletion PR, deleting the launcher, `lib/plan-check.sh`, `lib/sdk-check.sh` if orphaned, Python serving supervisor/CLI only once no callers remain, layout files and old spellings only after the RSC12 adapter parity. Keep historical experiment patches and records as historical evidence, not executable dependencies.

## Detailed Solution & Technical Design

### Verified inventory, 2026-11-18, origin/main 9549aa603

The direct-hit query is `git grep -n -e launch-flashnext-best.sh -e plan-check.sh -- tools/lab/radiance tools/tests tools/bigcherry config/serving-settings.toml .github/workflows/radiance-release-bundle.yml`; also grep indirect `$L`, `$LAUNCH` and the Python entry points. A hit in historical documentation or an experiment patch is not a runtime call. In the lab tree there are **13 direct .sh/.py callers**, not 16 active executable direct callers on this revision; do not assume the planning estimate is an exact count.

| Caller / surface | Required native replacement or disposition | Block to removal |
| --- | --- | --- |
| `launch-flashnext-best.sh` (start/stop, layout/build/release, canary, measuring, heat, live policy, power, gfxoff) | `bc-serve serve CONFIG --configs engines/radiance/configs --run-dir DIR --engine-binary BUILD/bin/radiance --site SITE --model-root ROOT --power-profile NAME`; detached `start CONFIG` only on non-leased hosts; `bc-serve stop/status --run-dir DIR`. Release mode belongs to RSP01 native tagged release. Typed service/handoff settings belong in configs or recorded `--set`. | **Yes**: S15 dense/helper parity and CLI/lab gaps; no deletion now. |
| `lib/plan-check.sh` (sole source caller: launcher) | None: C++ owner handles measuring start, exit-10 plan-cache miss, exit-11 derived-pool remeasure, strict measured v2 writer, live admission before GPU child. | **Yes**, until launcher deletion, real planner tests and S15. |
| `lib/sdk-check.sh` (sole source caller: launcher) | `bc-serve` native `sdk_check.cpp` plus typed environment/system checks. | **Yes** until launcher removed and SDK negative tests prove parity. |
| `tools/lab/radiance/lab-env.sh` and `lab-build.sh` (not under `lib/`) | No serving start replacement; lab-side environment/build selection may remain for experiments. Never ship these in the engine release. | **No** for launcher deletion if they no longer start it. |
| `burst-idle.sh`, `burst-trace.sh` | Native named helper config or recorded `--set` variant, `serve/start/stop --run-dir`; keep trace/client tooling. | **Yes** pending per-run mode/trace parity. |
| `flashnext-parity.sh` | `bc-serve serve flashnext-4gpu --run-dir DIR`, with config-driven histogram period, semantic projection and matching benchmark client. Lease wraps foreground owner for its whole lifetime. | **Yes** pending short A/B/hardware parity. |
| `flashnext-probe.sh` | `bc-serve serve CONFIG --run-dir DIR --engine-binary ...`, typed tracing and log-path receipts, `stop --run-dir`; no implicit inherited R3_* start configuration. | **Yes**: tracing and log-discovery parity. |
| `heat-placement/rhp01-coverage.sh` | **ARCHIVED**, not migrated: it needs the routing_evidence lab experiment that the native observe-only mode correctly refuses; no new native lab mode. Immutable `archive/lab-rhp01-coverage`. | **No**: intentionally retired under owner rescope 2026-11-18. |
| `pwr01-baseline.sh`, `pwr03-idle-diag.sh` | Native `--power-profile NAME` and `--allow-power-apply` only with owner-held GPU lease; native gfxoff floor/loaded-idle receipts; keep samplers. | **Yes** pending measured power/gfxoff A/B. |
| `rad20-burst-profile.sh`, `rad20-profile.sh`, `rad20-queue-probe.sh` | Native `serve CONFIG` plus **explicit, typed** profiling/child-wrapper capability if needed; `HELPER_WRAP`, `HELPER0_WRAP` and arbitrary `R3_*` environment are not valid config transports. | **Yes**: must not migrate by forwarding untrusted inherited environment. |
| `rad20-helper-kill.sh` | Native `serve CONFIG`; kill a test-owned helper, verify owner stops the set and restores power; `status/stop --run-dir`. | **Yes** until session-fault card-free test and hardware parity. |
| `heat-placement/heat-watch.py` | Advisory now emits `bc-serve stop --run-dir DIR` then `serve CONFIG --run-dir DIR --set handoff.heat=...`; typed TOML value and shell-safe path, never starts processes itself. | **No**: card-free tests check native recommendation. |
| `rhp20-forced-move.py` | Native `serve CONFIG` with explicit lab `--calibration-run --live-policy FILE --calibration-prompts-a/b --calibration-store` **only** for calibration; experiment moves require typed RSC12 lab experiment evidence, not a magic `--experiment`. | **Yes**: production experiment mode missing. |
| `tools/bigcherry/cli/serve.py`, `serve_v2.py`, `serving/supervisor.py`, `serving/runtime_adapters.py`, `serving/activation.py` | `bc-serve serve/start/stop/status` + native `--explain` (`explain CONFIG` implemented in PR #1078); `tools/tests/serving/*` are behavioural oracles, not release dependencies. | **Yes** for source deletion until Python call sites and tests migrate; **no** Python may appear in RSP01 engine asset. |
| `bigcherry host-check`, `bigcherry gfxoff` and `tools/bigcherry/hardware/gfxoff.py` | Native `sdk_check.cpp`, `power_diagnostics.cpp`, session gfxoff floor/loaded-idle and refusal; standalone Python diagnostics can remain in developer tools, **no replacement needed in release**. | **No** for release if not packaged; negative startup tests still needed. |
| `tools/bigcherry/engine_adapters/radiance_plan.py` via `radiance.py` (campaign/benchmark) | C++ `bc-serve serve CONFIG` for online runs; `bc-serve --write-manifest` / strict v2 writer for offline manifests via one native planner; keep Python-only orchestration outside engine asset until consumer migration. | **Yes** for source retirement and S14 campaign/benchmark receipt parity; not a reason to ship Python. |
| `tools/bigcherry/release/install.py` | RSP01 Python-free tagged digest-trusted asset installer and `bc-serve` native release start; **do not edit here**. | **Yes**, owned by RSP01 phase 2. |
| `.github/workflows/radiance-release-bundle.yml` | RSP01 emits one Python-free engine asset, not the lab launcher. **No edit in this lane.** | **Yes**, RSP01 release/rollback gates. |
| `config/serving-settings.toml` (**40** launcher owner entries) | Named RSC12 native config plus `bc-serve` ownership metadata for each active profile; preserve historical records without inventing duplicate budgets. | **Yes** for reference cleanup after S15 and consumer migration. |
| `engines/radiance/layouts/*.toml`, old `server-args` and `[layout.NAME.settings]` | RSC12 `configs/{flashnext,qwen27b}.toml`, strict native loader, named variants, semantic parity; remove old spellings together at RSC12 step 6. | **Yes** until every layout/campaign consumer uses configs. |
| `tools/tests/lab/{test_launch_flashnext_cleanup,test_launch_flashnext_release,test_mtp_replay_guard,test_plan_check_launcher,test_helper_profile_parity,test_rhp16_calibrate,test_rhp18_policy_preflight,test_rhp19_calibration_watchdog}.py` | Native C++ declared CTest targets (`session_test`, `session_guards_test`, `session_stored_test`, calibration/power tests) and card-free CLI acceptance; launcher tests remain until equivalence proof. | **Yes**: migrate each semantic assertion, never delete a test merely because it reads bash. |
| `tools/tests/campaign/test_rsc03_resources.py`, `tools/tests/release/test_release_install.py`, `engines/radiance/components/serving/serving_test.cpp` | RSC03 C++ derivation and RSC12 config semantic parity without reading soon-to-be-deleted launcher source; release tests transition in RSP01. | **Yes**: remove launcher text-oracle dependency after independently grounded fixture. |
| Docs, READMEs, `experiments/radiance/*/change.patch`, historical JSON and evidence | Preserve historical references as evidence; update *active* runbooks to native commands only when tested. | **No** executable dependency, but active instructions must be updated at S16. |

### Mode equivalence and open native gates

| Launcher mode | Native state as of inventory | Acceptance gate |
| --- | --- | --- |
| `start --layout NAME --build DIR`, `stop` | Named `serve/start CONFIG --run-dir DIR --engine-binary DIR/bin/radiance`, `stop --run-dir`; config chooses helper layout. | Real resolver + writer + session-owner fake engine/helper test, then S15 GPU. |
| `--release TAG|latest` | No direct `bc-serve` CLI equivalent; RSP01 installer and tag digest trust. | RSP01 acceptance and rollback. |
| `--port`, `--heat`, `--record-heat`, `--record-period`, `--kv-dtype`, `--ngram-placement disk`, `--num-spec`, `--max-model-len`, `--budget` | Native registered settings and RSC12 config/provenance (e.g. `service`, `engine`, `handoff`, `memory`); no shell-flag aliases. The registered native settings cover each listed family; producer-shaped card-free `bc_typed_settings_test` now checks named-config provenance, strict unmeasured writer fields and typed refusals. S15 still must prove serving launch parity. | Real semantic projection parity; typed config and artifact checks. |
| `--ngram-device INDEX` | Explicitly not carried over; typed field required before use. | Refuse unsupported use, do not emulate. |
| `--live-policy FILE`, `--experiment FILE` | Foreground native `serve CONFIG --lab-experiment FILE` supports strict-v2 `placement.mode=observe` only, without stored-manifest reuse. Shape, forced churn, profiler and arbitrary lab experiments remain refused until RSC12 keys their measurement inputs; `--calibration-run --live-policy` remains its separate ABBA mode. | Real native resolver + writer/reader and rejection CTest; S15 hardware, RSC12 typed expansion. |
| `--calibrate --calibration-prompts-a/b` | Foreground native `serve --calibration-run --live-policy ... --calibration-prompts-a/b --calibration-store`. | ABBA evidence and two distinct real prompt files, guard test. |
| `--gpu-power` / `--no-gpu-power` | Named `--power-profile` and guarded `--allow-power-apply`, only within lease held by the owner. | Durable admission before first GPU child and restore after verified exits. |
| MTP replay, live policy report, gfxoff floor/loaded idle | Native owner guard/power hooks exist; exact lab receipts and log discovery still require parity. | Real producer/consumer fixtures, injected fault cases, S15. |
| HELPER_WRAP/trace/profiler env hooks | No typed native replacement. | Explicit native laboratory interface or retire the experiment with evidence; no inherited environment shim. |

## Code Samples & Guidance



## Files

`docs/planning/active/run-serving-configuration/{RSC04,RSC02,RSC11,RSC12}.md`; native serving component, RSC12 configs, lab caller/test tree. RSP01 release files are listed only as cross-lane dependencies.

## Validation

- `git grep` audit includes direct and indirect callers and catches config owner text and tests.
- Every declared CTest has a real build/utility target and TIMEOUT; fake-engine tests use real session/planner/manifest production path, never a fabricated manifest. All CI tests remain card-free and must not rely on chmod/permission failures (CI can run as root).
- C++17 HIP reachability for changed shared worker headers; strict g++/clang++, ASan/UBSan, clang-format, clang-tidy on positive diff hunks. Full Python suite, ruff, patch-lint, forbidden terms and bounded CI before each merge.
- S15: one build, same-session A/B with same semantic launch projection, dense + helper, RHP12 ABBA four clean pairs, safety/canary/power/quality/performance/release gates. Proof required before S16 deletion.
- S16 deletion proof: `git grep` has zero active runtime and test callers, not merely zero string mentions; allow historical evidence references explicitly.

## Effort & Risk



## Standards



## Acceptance Criteria

The 'Derived, not hand-picked' literal inventory is empty; every active caller uses native `bc-serve`; no Python, bash launcher or plan-check in the engine release; S15 and RSP01 proven; then and only then remove the launch surfaces and obsolete aliases.

## Notes

Source: GPT configuration assessment 2026-11-02; RSC02 step 6 caller audit on 2026-11-18. No hardware acceptance has been executed. No source is deleted in this inventory PR.

Source: GPT configuration assessment 2026-11-02; RSC02 step 6 caller audit on 2026-11-18. No hardware acceptance has been executed. No source is deleted in this inventory PR.

Owner rescope 2026-11-18 (engine first): the core production runtime path comes first; tooling only if the engine needs it at runtime; tests only if absolutely needed; Python that is not part of the production engine runtime (lab scripts, campaign planner radiance_plan.py, gfxoff.py and host-check diagnostics, repository tooling, remaining tests) stays. No new lab-only native modes, no profiling/child-wrapper capability, no migration of launcher-text tests. The inventory table above is therefore a ceiling: rows marked lab-only are resolved by classification below, not by native replacements.

Production parity inventory (engine-first lane, PR #1118 and #1119 merged or open): build/source selection, card/layout/resourcing, SDK check (native sdk_check.cpp; the refusal currently skips silently when release metadata is missing: add a negative test before removing sdk-check.sh), plan-check.sh behaviours (measuring start, exit-10 plan-cache miss, exit-11 remeasure, strict v2 writer, owner map, cache validation: all native), processes (ordered helper-first roles, role/manifest binding, shared memory, health/watch, PID-start-time ownership), runtime admission (health, canary, MTP replay, durable power journal/receipt, restore after children exit, gfxoff floor and loaded idle), handoff (heat file, histogram, live policy, calibration key, strict S10 provenance), stop/status/stored manifest. Only live hardware parity is open for all of these (S15).

Critical gap for the installed engine: the release archive holds configs/flashnext.toml, configs/qwen27b.toml, bin/radiance, radiance_home/bin/bc-serve and bc-release, but native config_sources() uses only an explicit --configs or BIGCHERRY_CONFIGS: `serve/start CONFIG` from an installed PREFIX does not auto-select PREFIX/configs, and `bc-serve start` does not itself re-verify the release. Close both before the Python installer is deleted.

S16 classification: MIGRATE only standing-procedure scripts (flashnext-parity.sh for the bump parity gate, pwr01-baseline.sh); ARCHIVE rhp01-coverage.sh at archive/lab-rhp01-coverage (requires the forbidden routing_evidence lab experiment, do not add a native mode) to `bc-serve serve CONFIG --run-dir`; ARCHIVE to immutable tags archive/lab-<name> and delete from the tree: burst-idle.sh, burst-trace.sh, flashnext-probe.sh, pwr03-idle-diag.sh, rad20-burst-profile.sh, rad20-profile.sh, rad20-queue-probe.sh, rad20-helper-kill.sh, rhp20-forced-move.py (each with an immutable-tag historical TOOL_DISPOSITION row); DELETE after S15 and RSP01: launch-flashnext-best.sh, lib/plan-check.sh, lib/sdk-check.sh, serving/supervisor.py, runtime_adapters.py, activation.py, cli/serve.py, serve_v2.py and serving-only Python modules once callers are gone, tools/bigcherry/release/install.py after the installed-tag start gate. Launcher-text tests are deleted with their subjects (test_launch_flashnext_cleanup, test_launch_flashnext_release, test_plan_check_launcher, test_helper_profile_parity, test_rhp16_calibrate, test_rhp18_policy_preflight, test_rhp19_calibration_watchdog, test_mtp_replay_guard; test_runtime_sdk_check after the native SDK negative test): port only an assertion that guards a production behaviour with no native test. Direct blockers from git grep: release/install.py references the launcher; tests/release/test_release_install.py checks the launcher command; tests/campaign/test_rsc03_resources.py reads launcher text. Deletion proof: git grep shows zero executable references. DO NOT MERGE the deletion until the leased S15 A/B (tools/lab/radiance/rsc02_s15_ab.py, runbook docs/reference/engines/radiance/RSC02_S15_ACCEPTANCE.md) passes on hardware for dense and helper configurations.



## Status 2026-11-19

S15 passed on hardware and S16 landed (#1131): `launch-flashnext-best.sh`, `lib/plan-check.sh`, `lib/sdk-check.sh`, `burst-idle.sh`, `flashnext-probe.sh`, `tools/bigcherry/cli/serve.py` and `serving/supervisor.py` are gone from main (checked 2026-11-19). `flashnext-parity.sh` is migrated to `bc-serve serve` and is the bump parity gate. Still present and to be classified before closing: `config/serving-settings.toml` (the launcher-owner entries), `engines/radiance/layouts/` (now example/test data; confirm nothing active reads it), and active runbooks that still name the launcher. Moving to in_progress for those leftovers.

## Change Log

- 2026-11-02T00:53:10.689343+00:00 (created-by): Created by agent
- 2026-11-18 (inventory): Reconciled direct callers and S15/S16 retirement gates; state intentionally stays pending until hardware proof.

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:56.311686+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-18T04:42:21.572347+00:00 (updated-by): Updated: section:notes
- 2026-11-19T07:37:11.732116+00:00 (updated-by): Updated: section:notes
- 2026-11-19T07:37:51.117086+00:00 (state-transition): State: pending → in_progress
