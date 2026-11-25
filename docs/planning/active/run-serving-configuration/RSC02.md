---
id: RSC02
order: 2
plan: run-serving-configuration
state: in_progress
created-at: '2026-11-02T00:53:02.723581+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: M
---

# bc-serve native YAML owner: launch, lifecycle, measured-cache reuse and power

## Description

Retain existing C++ bc-serve sole session owner and process guards. Native YAML is the only executable config; named starts re-resolve YAML and reuse validated measurements, not stored launch manifests. Actual engine invokes pinned radiance --config with sealed raw YAML, native Fleet online and native offline Engine path. Helpers use same YAML plus assigned role and existing transport/admission facts. Keep all old launcher functionality and residual recovery gates. No separate native supervisor or legacy launch route.

## Steps

1. Freeze exact original YAML/root and matched parser inspection; ordered selected devices, typed owner PowerPlan and resource/transport facts. 2. Admit/apply power once under lease before measuring helpers/engine or final children, preserve exact restore. 3. Reuse existing measured plan cache or measure native YAML with rad0011; fresh room/identity readback and bounded refit. 4. Start helpers by role then native --config engine; native Fleet preload/front, verify pre-weight admission and canary. 5. Stop/drain/reap, exact restore, receipt and lease release; preserve guard/heat/lab/quality/cancel/partial-start and release selection. 6. Remove --manifest, --write-manifest, PinnedManifest, stored launch replay and writer child as execution authority, move all callers together. Evidence capture optional/output-only; RSC11 S18 gate.

## Detailed Solution & Technical Design

docs/design/radiance-native-yaml-launch.md sections 3–6 and 8. session.cpp/h EngineStart, pinned descriptor, ChildKind, plan/replay and PowerHook become typed owner state. main.cpp native config CLI, native_fleet_config/NativeConfigSnapshot seal raw YAML. Retire bc_serving_manifest alternative Engine/front and argv, narrow managed native load/Config/plan hooks at actual pinned source, no duplicate front. Reuse existing guarded callback, owner journal, device masks and run records. ManifestPower -> typed owner PowerPlan, no manifest authorization. Maintain exact preexisting caps/OD and physical cross-run recovery; WSL08 owns detached lease lifetime.

## Code Samples & Guidance



## Files



## Validation

- Card-free lifecycle tests with fake processes (ported from the Python supervisor's): ordered start; a helper or the
  engine failing during start and while serving; partial start cleaned up; repeated start/stop; stop of a dead set; a
  second start refused while one runs; stop through the control socket only; power restored only after every child
  exited, journal kept when restore fails; inherited environment ignored; old `--manifest` launch refused; stale measurement key refused or remeasured, naming why.
- Hardware (gpu lease held by the owner for the session): one dense config (qwen27b TP3) and one helper config
  (flashnext-4gpu) start, serve, survive a killed helper (set stopped, power restored), stop; same-build native versus prior qualified launch path passes RHP12 gates.

## Effort & Risk



## Standards



## Acceptance Criteria

- `bc-serve start CONFIG` starts every configured model, dense and helper layouts alike; `bc-serve stop` stops it
  and restores the host's power state.
- Named YAML start revalidates current inputs and reuses only compatible measured data; old manifest launch refused.
- No process receives configuration through configuration flags other than native `--config` and assigned helper role, or config environment variables; derived runtime facts use narrow typed admission.
- The bash launcher and `tools/lab/radiance/lib/plan-check.sh` are deleted; no caller references them.
- `bc-serve explain` (dry run) prints resolved YAML intent and the process plan without starting anything.

Native power lifecycle independent acceptance (transferred from superseded PWR09/PWR04): selected-BDF whole-layout preflight and exact original readback before first write; verified owner/lease and explicit qualified intent; durable journal and receipt, bounded apply/readback, reverse rollback, children reaped before restoration, exact prior values restored before lease release; failure and restore failure visible, no success on partial apply. Prove unresolved-journal protection across different run directories and physical devices; reconcile archived observe-only stale-journal semantics with current SessionPower::begin, no new mutation until authorized recovery. Privilege boundary prevents unauthorized sysfs writes; no promise of SIGKILL or expiry restoration without independent verified exit/lease recovery (WSL08). Native fresh YAML -> measurement -> measured start -> stop and named measurement reuse tested with nonstock mixed TP plan; verify same policy, room recheck, fake fault/permission/missing lease/partial apply/child failure and live selected-device readback/restore. Preserve stock and zero-write observe; label admission/no-op separately from actual writes. Foreground leased serve first; detached apply not enabled until WSL08 proof. No runtime phase switching until quiescence/dwell/rollback separately proven. Existing bc_power_transaction_test, bc_manifest_power_test, session/stored/cache tests and installed release gate; no duplicate implementation.

## Notes

Depends on RSC01. Source: GPT configuration assessment 2026-11-02.

Depends on RSC01 (loader), RSC06 (helpers read the manifest) and RSC07 (engine reads the manifest). Owner decision 2026-11-02 (recorded on RSC01): no command-line converter and no fallback path.



BACKLOG TRIAGE 2026-11-04 absorbed items (superseded into this one; their notes keep the full history): MEN14: Remaining cutover (one workflow, lab launcher retired) is RSC02/RSC04 (bc-serve as the only start-up path); both exist and are open.; MEN18: `bigcherry remote run|status|wait|list` exist (tools/bigcherry cli); leftover launch utilities belong to RSC02.



CORRECTION 2026-11-04: the earlier note that MEN14, MEN18 were absorbed into this item is withdrawn. An independent revalidation found each either still has open scope of its own (reopened) or was delivered outright. This item's scope is unchanged.

OWNER DIRECTION 2026-11-14: rescoped. The command is `bc-serve start|stop|status|explain` in C++ (radiance's language),
not `bigcherry serve` in Python; it starts a named config from RSC12 (one readable file per model, named variants) or a
stored manifest. Depends on RSC12 for configs; steps 1-4 can start on today's layouts. GPT is the backup reviewer.

STEP 1 2026-11-14 (feat/rsc02-session-owner): `bc-serve serve|start|stop|status` (components/serving/session.cpp),
engine-only configs. Owner process with session lock, control socket and record; plan step runs the build's own
`--write-manifest` (exit 10 measure with the engine alone, exit 11 derived pool), same plan-extra as the bash engine-only
start; engine from the manifest with a scrubbed environment; /health, canary; stop SIGINT -> SIGKILL, verified for the
whole process group; processes recorded with their start time, so an owner that died leaves a session `stop` refuses to
call stopped and `serve` refuses to start over. Same-build check (an engine from another build is refused: the plan key
is compiled into both). The qwen27b config now states the bash launcher's engine-only values (host pool 0, KV tier 0,
n-gram on disk, p2p off); the parity test includes them. Card-free: 17 lifecycle cases (session_test.cpp) and the plan
step (serving_test.cpp). Hardware (lease around serve): qwen27b-tp3-int8 and -556 measured on cache miss, served,
canary, stopped clean; engine SIGKILLed while serving -> session failed, lease released, no KFD processes. A bc-serve
newer than its engine was refused by the engine (BC_CFG_PLAN_MISMATCH): the reason for the same-build check. GPT review
req_rsc02-session-owner_f98e1286bafc (2 P0, 5 P1, 1 P2, all fixed). Next: step 2 (helpers); the remaining layouts become
configs first (owner order 2026-11-14).

2026-11-25 native power parity audit: #895 auto-applied configured cards under lease in retired launcher; #891 stock-only S10 input remains; #1131 removed launcher. Initial S15 serving parity did not prove nonstock native apply. PWR08 owns resolution, PWR07 probe/receipt; the remaining PWR09 gates transferred here because #1400 superseded the Python lifecycle plan after #1386 transferred PWR04 obligations into it. No second supervisor or new manifest. Keep native YAML authority and AGENT_INSTANCE guard (#1369).

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. The older examples/step 6/S15/S16 records are evidence of prior cutover, not callable syntax or a reason to repeat the Bash migration. Preserve all previous acceptance as semantic gates; stored replay now means named YAML + measured-cache validation, never run-record import.

S15 evidence source: docs/reference/engines/radiance/RSC02_S15_ACCEPTANCE.md and completed RSC04 caller inventory at 9549aa603; S14 lifecycle adapter must actually exercise both owners, not direct engine benchmarking. Historical pre-S18 text remains in Git at b0bf406bf.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Native cutover must update serving/component.toml coupling flags, old expected-manifest and option-registry, test targets and managed patch requires; not relax lint. Declare exact native load/Config, Fleet preload/front/readiness/drain, pre-weight plan, ordered BDF seams, actual signatures/lifetime/ownership and fail-closed fixtures. Use component_coupling.py plus compiled/semantic tests; source file existence is insufficient. Existing C++ owner and measured resource/cache admission preserved. MEN15 standalone CPU smoke and separate fake owner tests; MEN17 actual candidate lifecycle.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Native main offline predicate only KLD/calib/diagnose/debug_graph, not local plan-only. Explicit smallest managed measure operation at same native Engine::plan/preweight boundary, native --config online Fleet; prove actual hook before reservations/allocations/weights, no second allocator/front or synthetic argv. Raw snapshot path/root relative semantics and native reload guard tested. Candidate changes Fleet queue cleanup, host fill and loader copy drain; stop/reap all children and preserve staging lifetime before restore. Existing owner guards/FD lifetime sufficient; no extra tamper scheme.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. Extract session.cpp recheck_native_host_room from full manifest and retain actual live cgroup/pin/MemAvailable, helper residency, rank/checkpoint checks before native weight load. Native --config preserves authored server.port; readiness must bind actual owner/child identity and port, not another process answering /health. Existing session.json/control socket supplies actual run identity to MEN20/21, no registry. Migrate actual helper/measure/final/offline quality paths, writer child, stored reader and active callers; keep evidence-only read commands with no launch authority. WSL08 detached apply independently gated.

## Status 2026-11-19 (after S15 and S16)

- Done: `bc-serve serve|start|stop|status|explain` is the only start path. S13 switched serving to manifest v2 and deleted v1 (#891); S15 hardware acceptance passed (`docs/evidence/2026-11-19-rsc02-s15-acceptance/`: dense `qwen27b-tp3-int8` equal projection and greedy hash, helper cohort equal hash, `flashnext-4gpu` 538 requests with 0 errors); S16 deleted the bash launcher, `lib/plan-check.sh`, `lib/sdk-check.sh`, the Python serving supervisor and the installer (#1131). Steps 1-6 above are met.
- Fixed since, all found on hardware and each with a test first: a stored policy-on manifest replays for an unchanged config (#1177); the runtime receipt accepts the boot placement (#1176); the heat guard waits for the first counted snapshot and names the failing check (#1174); a calibration run files its record under the key production looks up (#1169); calibration policy refusals (#1162); build tooling: per-source-path component trees (#1172) and a bounded retry of a signal-crashed compile with receipts (#1175).
- Known sharp edge: the stored-manifest replay key hashes every operator `--set`, so two starts that differ in any `--set` (for example a per-run histogram path) never replay and quietly plan afresh.
- Remaining: the RHP12 gate with at least four clean ABBA pairs per layout (one pair per cohort was run; today's dense TP3 no-regression run is separate evidence), release-installed configs (`config_sources()` from the verified prefix), and the RSC04 inventory leftovers.

## Change Log

- 2026-11-02T00:53:02.723581+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:48.446911+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T03:51:40.629364+00:00 (updated-by): Updated: section:title, section:steps, section:acceptance_criteria, section:notes
- chg_20261102_035356_serving-start-up-plan-one-str_4985
- 2026-11-02T03:54:04.045012+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T03:47:55.795874+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:47.730785+00:00 (updated-by): Updated: section:notes

## Update 2026-11-10 (GPT power manifest review, req_main_eb983900e21b)

Acceptance gate added: the Python power transaction (PWR09) and its receipt lifecycle are part of supervisor acceptance (apply before the first measuring or serving process, restore after verified shutdown). bc-serve starts no processes today and the shell launcher remains the supervisor (it also does readiness waits, the canary, MTP replay verification, cleanup); recommended: a thin Python bigcherry serve calling the C++ resolver/validator, not a second resolver. The launcher stays until RSC02 reaches equivalence (RSC04).

## RSC02 step 4 power/GFXOFF/host-check implementation inventory (2026-11-17)

These are source locations from the pre-retirement Python start-up path and the new C++ power slice. Paths below are repository-relative; line references describe the current slice and can shift at integration. This inventory is a migration checklist, **not** a claim that the C++ owner has already enabled v2 power at start-up.

| Behaviour / surface | Existing Python or shell source (file:line) | C++ source or remaining step |
| --- | --- | --- |
| Per-card power status, driver, OverDrive and cap writes, exact original values | `tools/bigcherry/hardware/gpu_power.py:21,43,87,167,207`; `tools/bigcherry/cli/power_cmds.py:75` | `power_transaction.cpp:245` produces typed commands; `power_linux.cpp:203,249` is the sole Linux sysfs adapter |
| `gpu-power apply`: layout BDF choice and user-visible refusals | `tools/bigcherry/cli/power_cmds.py:115,130`; `tools/lab/radiance/launch-flashnext-best.sh:412,432` | `power_transaction.cpp:195,374,433` binds strict-reader v2, preflights and starts transaction; merge with step-2 owner hook pending |
| `gpu-power restore`: require GPU users exited and verified lease | `tools/bigcherry/cli/power_cmds.py:158`; `tools/lab/radiance/launch-flashnext-best.sh:289,296,307` | `power_transaction.cpp:568,594` refuses unknown child exit and missing lease, preserves unresolved journals |
| PWR09 all-device preflight before first write, driver-reported ranges and original readback | `tools/bigcherry/hardware/power_transaction.py:126,191` | `power_transaction.cpp:374,433` validates complete selection before writing, and `power_transaction_acceptance_test.cpp` checks typed refusal |
| PWR09 durable prepare/applying/touched journal, reversed rollback, unresolved journal refusal | `tools/bigcherry/hardware/power_transaction.py:102,114,213,241`; `tools/bigcherry/hardware/power_receipt.py:21` | `power_records.cpp:461,466,480` handles journals/receipts with RAII descriptors, fsync and atomic rename; `power_transaction.cpp:433,513,568` handles touches and rollback |
| PWR09 restart after killed owner or failed restore; independent proof of child exit | `tools/bigcherry/hardware/power_transaction.py:257,268`; `tools/bigcherry/serving/supervisor.py:133,373` | `power_transaction.cpp:594` explicitly recovers under lease and verified child exit; `power_records.cpp:471` refuses symlinked/corrupt records |
| GPU power permissions and status-only admin commands (outside runtime owner scope) | `tools/bigcherry/cli/power_cmds.py:75,196,221`; `tools/bigcherry/hardware/gpu_power.py:231` | Host provisioning stays outside C++ runtime; no grant/revoke subprocess |
| PWR07 GPU capability probing (PCI BDF, amdgpu, hwmon, OD_RANGE, cap readback) | `tools/bigcherry/hardware/power_probe.py:313,396,405,485,515,533` | `power_linux.cpp:203` reads driver, OD_RANGE, cap/hwmon into typed Capability; `v2/s10_resolver.cpp:96` resolves power plan |
| PWR07 CPU energy and policies observational probe | `tools/bigcherry/hardware/power_probe.py:201,252` | `power_diagnostics.cpp:234` emits typed energy-access issue; no CPU power writes in v2 |
| Qualified named profiles, per-role evidence and fresh probes before 'apply' | `tools/bigcherry/hardware/power_profiles.py:50,104,117`; `tools/bigcherry/serving/power_plan.py:92,128` | `v2/s10_resolver.cpp:96` selects qualified profile; `power_transaction.cpp:195` never selects or overwrites policy |
| Manifest power section: digest, exact role/BDF order, apply vs observe, unsupported SCLK offset | `tools/bigcherry/serving/manifest_power.py:48,76` | `generated/v2/serving_types.h:528,1203,1223` typed contract; `power_transaction.cpp:195` binds after strict reader |
| v2 manifest power plan produced and accepted | `tools/bigcherry/serving/power_plan.py:62` (reference) | `manifest_writer.cpp:217,438,450,530` and `v2/s10_resolver.cpp:22,96`; live owner wiring belongs to integration |
| Manifest-bound receipts, phase observations, events and atomic persistence | `tools/bigcherry/hardware/power_receipt.py:21,50,60,78`; `tools/bigcherry/serving/supervisor.py:121,219,344` | `power_records.cpp:461,466` serializes durable receipt and journal; `power_transaction.cpp:433,568` records typed stages/observations |
| Lease verification bound to this process (expiry/ancestor PID/instance) | `tools/bigcherry/hardware/lease.py:19,26,40`; `tools/bigcherry/serving/supervisor.py:129` | `power_transaction.cpp:374,433,568,594` consumes owner-verified lease/exited predicates; no status-text parsing |
| PWR06 GFXOFF: GFX12 identification, 1 Hz no-process floor, loaded-idle sample, conservative verdict | `tools/bigcherry/cli/gfxoff_cmd.py:12,41,68; tools/bigcherry/hardware/gfxoff.py:87,109,133,150,204`; `tools/lab/radiance/launch-flashnext-best.sh:600,616` | `power_diagnostics.cpp:68,155,170` pairs typed windows and reads real sysfs, independent of power transaction; owner phase wiring pending |
| `host-check`: readable package energy, per-card configured power writability, typed WARN | `tools/bigcherry/cli/host_check.py:19,42,52`; launcher `launch-flashnext-best.sh:394,396` | `power_diagnostics.cpp:234` returns typed HostFinding with HostIssue enum, not strings |
| Python S12 supervisor ordering, crash cleanup and recovery | `tools/bigcherry/serving/supervisor.py:81,133,201,230,251,310,373` | Step-2 has an unmerged session PowerHook (step-2 worktree session.h and session.cpp). Integration must call `SessionPower::begin` before measuring/serving and `SessionPower::finish` only after verified exit; persist typed GFXOFF using `SessionPower::report_gfxoff`; no active hook in this slice |
| Cross-language golden fixtures and C++ fault-injection acceptance | `tools/tests/hardware/test_power_transaction.py:1,115`; `tools/tests/serving/test_supervisor.py:175,275,309,338,395`; `tools/tests/hardware/test_gfxoff.py` | `testdata/power_qualified_inputs.json` + `testdata/power_qualified_golden.json` generated by `tools/tests/hardware/generate_cpp_power_oracle.py`; `power_transaction_acceptance_test.cpp` asserts outcomes using C++ fakes, not Python internals |

**Integration boundary:** Keep Python and the shell launcher as the reference through RSC02 step 6/S16; do not delete or silently call them from `bc-serve`. The step-2 `PowerHook` exists in its own unmerged worktree and is not yet part of this slice. At the coordinated merge, adapt the owner hook to (1) authenticate v2 and select ordered BDFs, (2) persist a manifest-bound receipt, preflight and apply before *any* measuring/serving child, (3) capture GFXOFF windows without writing, and (4) restore only after verified exit while the owner still holds the lease. No hardware acceptance is claimed by card-free tests.

## RSC02 step 2 source inventory (2026-11-17)

This is the *helper-bearing* path from start to running. Line references are to origin/main before step 2.
Python and bash remain behavioural references until the S15/S16 acceptance and retirement gates.
The 2026-11-17 owner directive supersedes the earlier thin-Python recommendation:
the engine startup is a clean C++20 rewrite from RSC02/RSC11/RSC12 requirements
and the manifest contract. The Python resource/supervisor code supplies only
golden observable results and fault-injection scenarios, never C++ data shapes.
Resource admission has typed C++ inputs/results, with JSON only at the manifest
and parity-fixture boundaries.

| Responsibility | Bash/Python location (origin/main) | Native ownership / remaining work |
| --- | --- | --- |
| Launch options; release-tag/build exclusivity | launch-flashnext-best.sh:142-259 | main.cpp:119-200 has named-config serve/start/stop/status. Release-selection migration is RSC04/S16. |
| SDK/build checks, host diagnostics | launcher:384-396; lib/sdk-check.sh; Python host-check | sdk_check.cpp already runs in bc-serve. Diagnostic host-check integration is not step 2. |
| Resolve layout, BDF/HIP ordinals, per-role card visibility | launcher:344-369; radiance_layout.py, radiance_inventory.py | serving.cpp:1050-1119,1403-1425 is one C++ inventory; session.cpp:923-975 maps engine ranks; step 2 maps helper roles without a second inventory. |
| Model, kernels and helper build checks | launcher:384-393,466-483 | serving.cpp:1276-1300 derives kernels; session.cpp:948-965 verifies engine/build; step 2 checks helper binary/plugin and manifest identity. |
| Resource derivation (RSC03): host expert pool, KV, tail and per-helper budgets | launcher:448-466; radiance_launch.py:28-175; radiance_resources.py:126-220 | manifest_writer.cpp:486-525 publishes actual owner map; serving.cpp:1082-1120,1180-1220,1252-1265 computes BDF/worker-footprint helper budgets and tail; v2/s8_rules.cpp:57-67,197-250 has native host-ledger arithmetic. Step 2 moves Python derive-launch host pool/KV admission to C++ using the *same* owner map and inventory. |
| Resource inputs / host limits | radiance_launch.py:92-175; calibration.toml and radiance_resources.py | serving.cpp:971-983 provides MemAvailable/cgroup/pin; calibration.cpp provides measured target and margin; native code reads RAD v2 directory metadata only, counts unowned experts/embedding/staging/ngram and computes host pool, retained KV tier, helper budgets; refuse exceeded RAM/pin. |
| Cached measured manifest, miss and refit | lib/plan-check.sh:99-130,203-263 | manifest_writer.cpp:404-555 builds/validates strict manifest; session.cpp:976-1084 already handles engine-only miss exit 10 and derived-pool retry exit 11. Step 2 measures with helpers resident, no fallback. |
| Power apply/restore only after verified exits | launcher:286-307,412-436; supervisor.py:205-246,299-352 | **Step 4** supplies power_hook; step 2 declares/calls only the interface, no power writes or journal implementation. |
| Ordered helper start, ready and shared segment | launcher:100-101,495-523,563-566,583-591; supervisor.py:152-163,255-266 | manifest_writer.cpp:255-267 and v2/manifest.cpp helper_view carry roles. Step 2 starts helper0...N with fresh readiness logs, bounded gate, per-role visibility and shared-segment check, before engine. |
| RSS from measurement helpers (anonymous + shared once) | launcher:541-550,563-579; lib/plan-check.sh:246-263 | Step 2 reads /proc/PID/status RssAnon summed + RssShmem maximum, rounded up MiB, with helpers up during engine plan-memory. Reap all temporary children before serving. |
| Engine-only measurement and serving lifecycle | launcher:563-633; supervisor.py:257-289 | session.cpp:1003-1055 measures engine and handles resident report; session.cpp:719-760 already starts manifest-only engine, awaits /health and validates start canary. Step 2 adds helper lifecycle and shared manifest. |
| Child failure/stop safety | launcher:262-313,405-409; supervisor.py:292-355 | session.cpp:548-609,725-737 holds run lock, uses control socket, process groups, durable records and verified bounded termination. Step 2 extends unwinding to helpers and measurement. |
| Loaded-idle gfxoff, MTP replay, live heat/policy, lab calibration | launcher:595-618,635-735; calibration_watchdog.py; supervisor.py:248-252 | Runtime guards/lab modes are RSC02 **step 3**, not step-2 acceptance. Live policy manifest support in live_policy*.cpp and manifest_writer.cpp is not the same as a runtime guard. |
| Card-free acceptance / parity | tools/tests/serving/test_supervisor.py:175-275,371-392; campaign/test_rsc03_resources.py | session_fake_engine.cpp + session_test.cpp assert independently specified helper ordering, readiness and failure outcomes. Native ledger parity is checked against Python resource derivation for checked-in flashnext* and qwen27b* configs using recorded BDF/VRAM profiles from test_rsc11_inventory.py. |

**Python derive-launch details.** radiance_launch.py:180-202 accepts the layout name, model, bc-serve, runtime home, heat,
histogram, sessions, context and ngram placement; radiance_launch.py:28-175 asks
bc-serve for the authoritative BDF/helper inventory and actual unmeasured
owner map, reads only RAD v2 directory metadata, accounts model embedding,
unowned expert bytes, max expert staging size and RAM-resident ngram.
The pure radiance_resources.py:126-220 computes helper expert budgets as
selected card VRAM less arch reserve and actual worker footprint, host expert
pool including calibrated margin, retained host KV tier, and refuses
MemAvailable/cgroup or AMD pin overcommit. It prints HOST_POOL_MIB/HOST_KV_MIB
shell assignments plus a breakdown to stderr. The native start must use
these *values and refusal rules*, not spawn Python or consume shell syntax.
No resource fallback is permitted. Python stays in tests only until RSC11 S16.

**Explicit non-claims:** step 2 is not GPU hardware acceptance, power/lease
transactional completion, MTP/heat/lab guard completion or release/caller
retirement. Those remain steps 3, 4, S15 and S16 respectively.
- 2026-11-19T07:36:59.107423+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4515
- RV4516
- RV4518
- 2026-11-22T10:47:45.069205+00:00 (updated-by): Updated: section:steps
- RV4578
- 2026-11-25T01:03:10.136934+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4590
- 2026-11-25T01:36:31.503595+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:notes
- 2026-11-25T01:43:57.649464+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria, section:notes
- RV4611
- 2026-11-25T02:07:22.923091+00:00 (updated-by): Updated: section:notes
- RV4638
- 2026-11-25T02:43:21.540163+00:00 (updated-by): Updated: section:notes
- RV4655
- 2026-11-25T03:25:42.416856+00:00 (updated-by): Updated: section:notes
