---
id: RSC11
order: 0
plan: run-serving-configuration
state: in_progress
created-at: '2026-11-10T13:54:06.725791+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Serving completion programme: native YAML launch, admission and manifest retirement

## Description

One existing programme owns the remaining serving acceptance and native launch migration. RUM04 completed authored YAML and native option parsing but deliberately retained v3 launch. Owner now requires no executable manifest input. Retain C++ owner, native parser, resource and helper planners, power transaction and evidence; migrate directly to pinned Radiance --config and native Fleet through managed local enhancements only. S17 historical outstanding gates and S18 native cutover are both required. No new RUM migration programme.

## Steps

S17: reconcile existing evidence for residual obligations, retain genuinely outstanding work with its owner. S18: migrate authored serving policy to YAML; extract only actual resource/helper/power inputs from launch manifest; switch existing owner and engine/helper paths to native YAML; migrate affected callers and remove obsolete launch reader/writer; qualify the changed paths together. No dual launch or separate programme. Use existing owner status, cache, transport and controller. Do not depend on unrelated cleanup or kernel migration.

## Detailed Solution & Technical Design

docs/design/radiance-native-yaml-launch.md is the complete implementation design and audit. Native YAML launch is not a v4 manifest. Retain only actual derived per-consumer runtime facts across process boundary, with existing typed domain interfaces and sealed descriptors/transport where needed; capture output can reuse v3 evidence only. Native Fleet::run and stock front replace the parallel managed front; preserve single-model admission and no reload/park/shared cards. No upstream edits; all changes managed packages/owned code. RUM10 deprecated, RUM14/15 and RCS kernel migration separate.

## Code Samples & Guidance



## Files



## Validation

Reuse current CI and targeted existing parser/extension, admission, helper, cache, owner and power tests; add only missing cases for changed behavior. Cheap validation of approved YAML roster, exact pinned composition/build, existing Gate E dense TP3 and helper/MoE lifecycle, measured reuse and relevant parity/quality. Share one real minimum-cap apply/readback/restore run with PWR10. Additional GPU runs only for materially different affected paths or unresolved regression/noise. Zero executable manifest launch callers, capture not a launch input. Preserve existing required safety/quality/no-regression gates; no blanket full-roster GPU matrix, repeated historical soaks or independent research as S18 prerequisites.

## Effort & Risk

Risks in priority order: partial/repeated manifest migrations; internal R3_* reads giving split configuration; power writes without all-device atomicity and safe stop; wrong-device selection from two inventories; host OOM / stale budgets; silent throughput regression; supervisor/cutover gaps leaving servers running; hardware portability.

## Standards

No launch manifest, compatibility reader, shim, legacy flag/env, backport, duplicate schema, scheduler or allocator. One native config entry, one owner; prior matched release is rollback. Managed patches against pinned source and owned enhancements only. Evidence record output is not a launch contract.

## Acceptance Criteria

S18: actual native --config owner/engine/helper serving, measurement, quality/experiment and named measured reuse. No executable launch manifest, writer serialization dependency, general renamed envelope or old launch input. Native Fleet single-model preload and front; managed reload/parking/multimodel refused. Preserve existing resource, helper, safety, numerical and performance requirements. Cheap parser/extension checks for approved YAML roster and one shared actual installed dense TP3/helper cohort, measured reuse, relevant quality/parity and qualified minimum-cap apply/restore. Additional cases only for materially different affected paths; existing valid required evidence reused. Independent S17 research remains with its owners, but unresolved safety/functional regression is a blocker. No mandatory upstream bump or full-roster GPU matrix. Current approved pin/build/ABI/layout and actual receipts; capture output only.

## Notes

**Unpublished P1 prototype disposition (2026-11-13):** The original local `rad0014_plan_memory_v2_inputs` sidecar commit `006b7ce14` is preserved at GitHub `archive/rsc11-s13-p1-engine-native-20261113` and is **superseded, not merged**. Current `rad0011` already emits the authoritative post-rank-split rank-demand JSON consumed by native P1; the old three-field sidecar lacks `claimable_bytes` and `evidence` and has a different schema. No production or GPU testing is claimed for `rad0014`. See [unpublished branch reconciliation](../../../evidence/2026-11-13-unpublished-branch-reconciliation/README.md). Retain the active RSC11 S11/S13 admission and hardware gates.


Lanes: engine (S1,S5), resolver/contract (S7,S8,S10; S9), power (S3,S4,S12), python serving/tooling (S2,S6,S11,S14), integration lead (S0,S13,S15,S16,S17). Supersedes the piecemeal RSC02/RSC04 sequencing.

CONSOLIDATION 2026-11-22 NATIVE-OWNERS: receiving RSC04,RSC09,RSC10 These are uncompleted acceptance requirements, not new release/quality passes. Historical source plans remain available as superseded records. Keep legacy and blocked/no-route behavior fail closed; do not add parallel native control planes or tools.

2026-11-25 S17 reconciliation: stock-only native S10 input and mixed-TP role-map restriction remain despite S13/S15/S16 and subsequent YAML migration. PWR08 owns restoration; RSC02 owns native transaction and transferred PWR09 acceptance; PWR07 observations, WSL08 detached lease, PWR10 real profile qualification, RSP01 installed release. No repeat manifest cutover or new format.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. No new RUM16: RSC11 owns the cross-cutover and RSC01/02/03/05/06/12 and PWR07/08/10 own implementation. S17 older manifest-specific acceptance terms are mapped to typed runtime plan/receipt equivalents, never a demand to retain manifest launch. Do not reopen completed RUM04 or RUM09 wholesale.

S15 prior proof source: docs/reference/engines/radiance/RSC02_S15_ACCEPTANCE.md. Card-free fake owner is not hardware parity; actual maintained ABBA lifecycle adapter must exercise both launch paths, not a direct engine proxy. Historical plan at b0bf406bf preserves the earlier wording.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. S18 and S17 acceptance consume MEN24 A–H order and MEN15/16/17 exact candidate gates; no second serving or upstream migration programme. RSC01/02/05/06 own native parser/Fleet/resource/helper seam tests and coupling declarations, PWR07/08/10 owner policy, RSP01 installed bundle. Coordinate RCS24 update of bump docs/skills on actual cutover.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. S18 full supported deployment roster now 18 at audit, dynamically enumerate exact candidate and installed configs; 16 historical RUM04 proof remains. Native --check parser only, separate actual row/format and memory inspection; explicit plan-only managed seam before weights, native Fleet reserve/headroom/NUMA semantics. Gate E current two layouts through 2020.11.5 is install/canary/stop only, not native launch/quality/power proof. S18 full native cutover matrix; later bump two existing Gate E plus affected configs/independent required cohorts, no repeated unrelated historical trials. RCS25 discrete before full build, native launch and power one coordinated cutover.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. Make README authoritative S17/S18 index (old TOML/v2/replay directions removed). Refresh actual launch versus read-only evidence versus unrelated artifact caller inventory, including bench/workloads.py, cli/workload.py, gate-e.sh, parity/PWR01/helpers, calibration-ingest and generated fixtures. S18-A field/port/variant/roster; B typed measured cache and host recheck, helper/power and policy eligibility; C actual native online/offline launch plus owner status and lab/campaign/installed assets; D full supported matrix and existing S17 proof. No dual launch. Current direct native model port stays authored; no hidden override. Maintain exact run provenance and S17 structured preweight one retry; RSC14 contention tests. Evidence-only flags not blanket-deleted, no launch from capture. No claim completed from historical Gate E.

Owner scope correction 2026-11-25: docs/design/radiance-native-yaml-launch.md section Scope and acceptance rule governs. Preserve behavior; smallest native launch and YAML policy change, no generalized machinery. Reuse existing tests/evidence, one shared dense/helper installed acceptance plus materially different affected paths only; cheap parser checks across approved YAMLs. Independent outstanding work stays with its owner, not a blanket S18 prerequisite.

docs/design/radiance-native-yaml-launch.md §4.2 (code seam review, 2026-11-25): Code review at c41676c90, upstream pinned b6b8751. Primary §4.2 names concrete symbols/ordering; implement in existing owners, no further design document. S18 is ready for scoped implementation: native parser + owned inputs, extract current owner planner/measurement/power/helper facts, replace launch and callers in one cutover, shared representative acceptance. Preserve Fleet one-model preload, actual aliases, native offline score, existing pool convergence versus live shortfall retry, helper bootstrap before shared-memory creation. Remove full launch codec linkage only after extracting actual consumers; CMake/patch requires and hashes of managed source snippets updated with callers. No compatibility path, blanket GPU matrix or requirement to complete independent S17 research before S18; retain unresolved safety/functional blockers. Use existing Gate E and acceptance cohort, no new mandatory pin bump to prove migration on current approved pin.

2026-11-25 current-pin implementation decision: docs/design/radiance-native-yaml-launch.md §4.3 controls. Finish S18 on approved Radiance b6b8751 (2.2.0); do not wait for a bump, cherry-pick newer Fleet or extend scope. Native single-model Fleet is already present; reuse actual source behavior. Shortest current-pin cutover: use matched parser/raw YAML, existing owner plan/power/helper extraction and narrow native main hook, then native Fleet single-model online and native offline measure/quality. No new upstream feature dependency. Pinned Fleet single-model bypasses broker and spare queues; no authored headroom map in installed configs, so current scalar memory budget plus existing measured admission is the baseline, no new budget enhancement unless actual failing required path proves need. Guard native admin/automatic re-demand/engine death so owner sees failure rather than unadmitted restart. First existing fake-child and parser/admission tests and one canonical integrated build; use shared stock dense TP3 + helper cohort and same measured reuse, actual quality and min-cap restore evidence. Only changed behavior gets new tests. Prior manifest-based run is baseline, not new native proof. Stop when current capabilities pass; optional parking, NUMA, queue and newer per-card optimizations wait for qualified bump. No new plan, mandatory bump, repeated broad experiments or intermediate dual launch.

2026-11-26 roadmap reconciliation at 165d6f724: VTP34 C01–C24 audit history retained. Own residual C08 ngram typed device selection if supported (default dormant, explicit unsupported refusal), C11/C15/C16/C17/C19 one native YAML authority, remove duplicate raw argv/Python/launcher defaults; actual lab and heat consumers move with cutover; C13 typed diagnostics on existing common build only where needed, keep reference lane; C14 authored helper-policy data migration with RSC06 and calibrated records RCS10; C21 typed power capability PWR07/08. C06 request to load archs.toml at runtime is superseded by explicit RSP01 decision to embed immutable hardware capability in engine, not a mutable second authority. C04/07/12/23/24 fixed historical checks retained, no repeat by default. C03 retired rad0013 under completed RUM08; C09 existing rad0008 promoted default and setting (RCS10 owns any changed derived choice, actual A/B); C18/C20 RCS12; C22 unrelated llama.cpp conditional PA49. No manifest resurrection. Completion requires every retained audit finding resolved or explicit scoped refusal/decision, no new core code for supported model/topology.

## Status 2026-11-19

S13 (v2 cutover, v1 deleted, #891), S15 (hardware acceptance, evidence 2026-11-19-rsc02-s15-acceptance) and S16 (launcher, Python supervisor and installer deleted, #1131) are done, so RSC07 and RSC08 are superseded by this item's single v2 contract. What is left under RSC11 is acceptance depth, not cutover: the RHP12 four-pair ABBA gates per layout and S17 close-out. The production start with the live policy on had never been exercised end to end before 2026-11-19 and needed three fixes (#1174 heat guard, #1176 receipt, #1177 replay); treat 'never exercised' guards as the risk class when closing S17.

## Decisions

1. v1 is closed and has now shipped (bc-radiance-1.17.0-r15 contains manifest-only launching): no power amendment in v1. Exactly ONE v2 contract, activated once (S13); preparation PRs add generator/tests/inactive v2 code; the final cutover switches bc-serve, radiance, r3_moe_worker, plugins, validation, launcher adapters and campaign generation together and rejects v1 (previous releases stay on their own tagged v1: release rollback, not a shim). The unmerged RAD49 diagnostic 'amendment 18' (db199c2eb on fix/rad49-tp-tail-kl) is not merged into v1; its field is assessed for v2 or left out.
2. v2 content: spec.intent.power {profile, role_overrides}; spec.plan.power {mode observe|apply, profile, profile_digest, devices[{role,bdf,settings}], cpu{}}; host pool / helper budgets derived (RSC03), the five handoff fields derived or calibrated (RSC09, fields stay), host_ledger_margin.mib refit with provenance (RSC10, no second margin field), typed native config without R3_* (RSC06/07), manifest BDFs authoritative (RSC05). Receipts carry observations (not the manifest). Serving profile and power profile are distinct CLI options (--profile / --power-profile).
3. Python supervisor `bigcherry serve start|stop|status|explain` is long-lived owner of the session: lease for the serving lifetime, ordered helpers, readiness, canary, MTP guard, plan measurement, one bounded retry, power transaction, restore only after verified child exit. It calls the C++ resolver/validator; no second resolver. The shell stop_all must not restore power while a process survives (current behaviour is unsafe).

OWNER DECISION 2026-11-14 (replaces decision 3): the supervisor is `bc-serve start|stop|status|explain` in C++, not a
Python `bigcherry serve`; one language for the start path. Inputs become readable per-model configs with named variants
(RSC12). Scope: RSC02 (rescoped), RSC12 (new). Slices S6 (supervisor skeleton), S11 (plan measurement and retry),
S12 (power lifecycle) and S14 (campaign and benchmark integration) are implemented in the C++ session owner
(`bc-serve serve`); the merged Python supervisor stays as the behavioural reference whose fake-process, lease and power
tests the C++ owner must pass. S13 stays the one coordinated v2 switch; S15 acceptance and S16 retirement keep their order.

## Gaps by item (code-based)

RSC01 partial (core delivered; client integration, remove duplicate paths) | RSC02 not started as the command | RSC03 partial (9,216 host pool fallback, 12,288 fallback+calibration, 22,528 in layouts.toml recipes) | RSC04 not started (release/install.py still emits the shell launcher) | RSC05 partial (Python layout_cmds._resolve uses environment inventory) | RSC06 partial (R3_* reads left in moe_remote_hook.cpp 130-161/1277-1307, libr3/r3_moe_settings.h, moe_mailbox_ops.hip 909-912; rad0009 patch setenv bridge; manifest_argv_test asserts it) | RSC07 partial (same bridge) | RSC08 partial (contract delivered; one-release cutover open) | RSC09 not started (five literals) | RSC10 not started as recovery path (structured host-room refusal + one retry).

## Slices (small PRs; lane owner; dependency)

S0 freeze baseline (lead) | S1 engine-side env removal: typed settings, hostile-R3_* negative test (engine) after S0 | S2 one device inventory: Python serving decisions via C++ BDF inventory (python) after S0 | S3 GPU probe + receipt types (power; probes landed: #810, #812) | S4 transactional power core PWR09 after S3 | S5 structured host-room refusal, engine half of RSC10 (engine) after S1 interface | S6 `bc-serve serve` session owner skeleton with fakes (C++, owner decision 2026-11-14; the Python supervisor's tests as reference) after S0 | S7 RSC03 derivations prepared for v2 (resolver) after S2 | S8 RSC09 five handoff rules/calibrations (resolver) after S7 design | S9 v2 registry/generator scaffold, inactive (contract), parallel | S10 v2 resolver/writer: named profile -> resolved power plan, inactive after S3 S7 S8 S9 | S11 plan measurement + RSC10 retry in the C++ session owner after S5 S6 | S12 power lifecycle in the C++ session owner after S4 S6 S11 | S13 single active v2 cutover after S1 S2 S7-S12 | S14 campaign + benchmark receipts after S2/S13 | S15 hardware and release acceptance, RHP12 cohorts, GPU lease after S13 S14 | S16 retire launcher and legacy surfaces (RSC04) after S15 | S17 close and reconcile RSC01-10, PWR06-09.
Single integrator for serving.cpp, manifest_writer.cpp and the generated registry (S7, S8, S9, S10, S13 conflict there).

## Gates (RHP12 definition, ABBA at least 4 clean pairs)

contract integrity (one fingerprint, v1/malformed/mixed rejected) | configuration isolation (hostile R3_* inert) | layout identity | memory admission (at most one retry) | power portability and safety (fault-injection) | serving lifecycle | numerical and performance (stock power profile first, power arms after) | release usability | benchmark evidence (manifest digest + power plan + observed state per cell).

## GPU lease policy

Almost everything card-free; first leased session only after v2 simulator, fake supervision, fake power transaction and component tests pass; then one known layout, then materially different layouts (4-card head/tail, 2xXTX, TP engine + helpers, engine-only).

## RSC11 S7: offline v2 resource derivation (2026-11-10)

The preparatory GPU-free Python ledger is `tools/bigcherry/engine_adapters/radiance_resources.py`. A caller obtains `radiance_inventory.Layout` from bc-serve's BDF inventory and calls `Layout.derive_resources(model, targets, host, worker_footprints_bytes=...)`. The result carries a host pool, host KV tier, per-helper expert budgets and a byte-exact explanation. Inputs are mandatory: no guessed model footprint, ownership plan, or worker footprint. Re-resolve on device/owner-map/model/context/RAM change.

The shipped `engines/radiance/layouts.toml` now omits all literal helper `budget-mib` settings, allowing the existing v1 bc-serve worker-footprint-based budget derivation to take effect. The v1 manifest schema, C++ resolver, manifest writer and generated registry have not been changed.

**Required v2 handoff, for S10/S13:** bc-serve must publish the engine-measured host embedding, unowned expert, staging, n-gram, checkpoint and other resident bytes; the actual owner-map digest; host-pool margin calibration; KV bytes per token and the context/retention target; live MemAvailable, effective cgroup headroom and pin availability; and each selected helper BDF's complete non-expert worker memory (lanes, pointer banks, slot map, incoming slot) from `worker_footprint.h`. The output must carry provenance and be invalidated/recomputed with inventory, owner map, model or targets. The v2 writer should use those values to populate the existing resource budget concepts, enforce RAM/pin admission and compare with engine `--plan-memory`; fail closed if any required measurement is missing. No v2 contract fields are introduced by S7.

**Unresolved until the single v2 cutover:** `serving.cpp` still holds the v1 host-pool/host-KV defaults and engine-only recipe-specific pool sizes remain in layouts.toml. The v1 Bash helper launcher now calls the GPU-free Python launch adapter on every start: bc-serve selects actual BDF roles, helper footprints and owner map; the RAD directory and current host memory facts feed the shared resource ledger. The adapter prints a breakdown and injects computed host-pool and host-KV MiB without restoring the old constants. The C++ v1 defaults remain for non-launcher callers until the single S13 cutover, when model/plan sizes and KV geometry must be carried through a typed v2 resource interface. The v1 manifest schema is unchanged; S16 removes the Bash surface.

## Power S12 and v2 dependency status: 2026-11-10

Merged: S6 Python supervisor #820, S7 preparatory resource derivation #821, PWR07/PWR09 base #810/#812/#815/#816, named profile definitions #824 and inactive per-BDF qualification resolver #825. S12 card-free supervisor adapter consumes only the immutable v2 manifest power plan, persists a receipt, applies before starting children and restores after exit with verified lease; it does not enable the production start command. The injected S11 measured plan/validator and S9 v2 registry are not active. S10 C++ manifest resolver/writer and S13 all-consumer v2 cutover remain the activation gate; no v1 fallback, no profile writes by default, no changes to serving.cpp, manifest_writer.cpp or generated contract. S15 hardware qualification and S16 shell retirement follow later.

## 2026-11-11 power evidence for S9/S10/S12/S13/S15 — no v2 activation

Owner-run `runs/pwr-sclk` (12 XTX SCLK/offset arms; `pwr-phases@1`, `prefill-doc-44k@1`, `decode-6stream@1` with indexed result IDs) shows **uncapped -125 mV remains withdrawn** as a current-build apply-profile saving, while **2700 MHz OD_SCLK max with -125 mV** is a **candidate**, *not qualified*. Candidate name `xtx-cap2700-uv125`, discovery row IDs `587b81b5fa3b4835` (power-phase), `b69a3764952f49b2` (doc prefill), `d9177d830ee64d89` (six-stream engine decode); control SCLK-only rows `9331ee86e4b34adc`, `8e2d6ec68b3f4196`, `706d606817fe4288`. These must **not** populate an active apply profile: all profile values in v2 must come from a **separate validating run** with two-XTX BDF identity, fresh probe, digest and immutable manifest/receipt, replicated control, stable thermal/health, output identity, 40-minute soak, host amdgpu error check and verified restoration. Soak and identity are pending; container `dmesg` inaccessible. Neither R9700 nor 6900 XT was tuned.

The future supervisor/manifest must keep **decode and prefill** energy and speed as **distinct** workload-matched measurements. `pwr-phases@1` explicitly labels its power source `sysfs:power1_average` and CPU energy `not_sampled`; sum-of-board-median W / same-phase tok/s is only a GPU-only **proxy J/token**, not integrated energy and never to be paired with separate doc-44k or six-stream throughput. Stock and default power intent remain observe-only; S9/S10/S13 cutover gates and S15 hardware qualification do not advance because of these docs. Future hardware arms: XTX MCLK (device-reported 97–1250 MHz, re-probe), reversible pp_power_profile_mode study (BOOTUP_DEFAULT vs COMPUTE/POWER_SAVING), and separate R9700 power-cap experiment near reported 210 W minimum, with no default adopted. Detailed raw table, V/F CSV medians, sensor caveats and run links: `docs/reference/power-tuning-evidence-2026-11.md`.

## S13 addendum preparation (12:05Z, 2026-11-11)

The inactive P1/P2 calibration gate has been merged as PR #867. The
follow-up slice in the S13 worktree provides
`tools/bigcherry/serving/calibration_rebind.py` (immutable, audited
old-to-new build scope rebinds) and
`tools/bigcherry/serving/s8_lab.py` (GPU-lease-checked hardware candidate
sweep producing `s8_calibration_measurement` files). Requirements, exact
runner ABI, no-rebind conditions, and example commands are in
`docs/design/serving-manifest-v2/s13-calibration-rebind-and-run.md`.
Neither module activates v2 or treats synthetic tests as hardware evidence.
Missing qualified records, missing rebinding proof, changed measured scope,
unclassifiable source diffs, or unverified runner output still fail closed.
The native hardware runner for applying these new v2 tunables and the final
single active v2 production cutover remain S13 integration work; S15 hardware
runs remain lead-owned.

## Change Log

- 2026-11-19T07:37:07.525404+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4321
- 2026-11-22T03:32:52.501052+00:00 (updated-by): Updated: section:notes
- RV4510
- 2026-11-22T10:35:22.894292+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- RV4579
- 2026-11-25T01:03:40.126592+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4588
- 2026-11-25T01:35:31.897852+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:validation, section:standards, section:acceptance_criteria, section:notes
- 2026-11-25T01:44:23.913317+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- RV4613
- 2026-11-25T02:08:23.102972+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4642
- 2026-11-25T02:45:29.138905+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- 2026-11-25T02:53:06.747780+00:00 (updated-by): Updated: section:validation
- RV4659
- 2026-11-25T03:26:13.110103+00:00 (updated-by): Updated: section:notes
- 2026-11-25T03:34:04.609044+00:00 (updated-by): Updated: section:steps
- 2026-11-25T04:30:18.338487+00:00 (updated-by): Updated: section:steps, section:validation, section:notes
- RV4676
- 2026-11-25T05:42:24.534829+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:43:37.451316+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-11-25T05:53:50.605139+00:00 (updated-by): Updated: section:notes
- 2026-11-25T22:03:19.966623+00:00 (updated-by): Updated: section:notes
