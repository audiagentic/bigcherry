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

# bc-serve start/stop/status: the only start-up path, in C++, from a named config or a stored manifest

## Description

Replace `tools/lab/radiance/launch-flashnext-best.sh` (749 lines of bash plus 292 in `lib/`) with start, stop and status
commands in `bc-serve`, the C++20 serving component that already detects the host, resolves the configuration,
measures and caches the memory plan and writes the manifest. Owner direction 2026-11-14: one language for the start
path, radiance's (C++); the launcher is simple to read and run; it starts a named config (RSC12) or a stored manifest.

```
bc-serve start qwen27b/tp3-int8-556           # resolve (or reuse) the config's manifest, start, health, canary
bc-serve start --manifest FILE                 # rerun exactly a stored manifest; refused when its key no longer holds
bc-serve stop                                  # stop everything this start owns, then restore power
bc-serve status                                # config, manifest digest, processes, port, power state
bc-serve explain qwen27b/tp3-int8-556          # effective values and where each came from; starts nothing
```

What the bash launcher does today and the command keeps (inventory from origin/main, 2026-11-14): host site file and
SDK check; plan check with the measuring start (helpers up, engine `--plan-memory`) and the host-pool round; helpers
started in order and their resident memory read; engine started from the manifest only; `/health` wait; start canary
(deterministic completion, configurable prompt and expectation); MTP replay guard; GPU power apply and restore; gfxoff
floor; route-heat file and live heat recording; live placement policy report; lab calibration mode; cleanup of a
partial start; `--release TAG` (start from a verified release). Every one becomes a config field, a host-file field or a
step of `start`/`stop`, never a launcher option that rewrites settings.

## Steps

1. **Session owner, engine-only (dense configs, today's layouts).** `bc-serve serve|start|stop|status`: resolve, plan
   check and measuring start, engine from the manifest, `/health`, canary, control socket, ordered stop. Card-free tests
   ported from the Python supervisor's fakes first; then one dense layout on hardware.
2. **Helpers.** Ordered helper start and readiness, resource derivation in C++, measuring start with helpers up, any
   child's failure stops the set.
3. **Guards and lab modes.** MTP replay guard, live placement report, heat recording, lab calibration (refused by
   production configs).
4. **Power.** The manifest's power plan applied as one transaction by the owner, restored after verified exit (PWR09).
5. **Configs and stored manifests** from RSC12: `serve CONFIG` on the new files; `start --manifest FILE` replays a
   stored manifest without rewriting it (RSC12 step 5).
6. **Acceptance and retirement in RSC11's order:** the coordinated v2 switch (S13) stays one change; hardware and release
   acceptance (S15): same-session A/B against the bash launcher on one build, one dense and one helper config, RHP12
   gates, equal semantic execution projection (RSC12); retirement (S16): switch the callers (16 lab scripts, the release
   installer, tests) and delete `launch-flashnext-best.sh` and `lib/plan-check.sh` (RSC04).

### Step 6 inventory and hard gates (2026-11-18, origin/main 9549aa603)

The exhaustive caller/mode table is in [RSC04](../../completed/run-serving-configuration/RSC04.md#verified-inventory-2026-11-18-originmain-9549aa603). The older estimate of 16 lab scripts is **not** a verified executable count: this revision has 13 direct lab `.sh`/`.py` launch callers plus indirect wrappers, lab docs, launcher-sourced `lib/plan-check.sh`/`sdk-check.sh`, Python supervisor, campaign adapter, 40 `engines/radiance/reference/serving-settings-inventory.toml` launcher owners, and launcher-text test oracles.

The native session entry points are `bc-serve serve CONFIG --run-dir DIR
[--configs DIR --model-root DIR --work-root DIR --site FILE --engine-binary FILE
--power-profile NAME --allow-power-apply]`, `start CONFIG|--manifest FILE
--run-dir DIR`, `stop/status --run-dir DIR`, and `explain CONFIG` for
offline selection and diagnostics. The former lab
`serve --calibration-run --live-policy ...` route was removed from the
session owner; there is no supported calibration-producing start on main.
The standalone manifest-writer parser still recognizes a historical
`--calibration-run` flag, which is not a serving mode. Native replacement
work is in progress under PRs #1375 and #1384 and does not change this
current command inventory. Unsupported launcher-only forms are **not** to be introduced as backwards-compatible aliases; config fields or a typed lab interface must replace them and be tested with the real owner/writer/manifest call graph.

**Retirement order:** inventory first (this PR); card-free native gaps and tests next; an automated leased S15 same-build A/B on dense and helper layouts with equal RSC12 semantic execution projection, at least four clean ABBA pairs per RHP12 and all safety/release gates; only then S16 caller migration and deletion. Hardware parity is not shown by passing fake-engine CI. The release asset, installer and release workflow are owned by the concurrent RSP01 phase-2 lane and must not be modified by this change. Keep step 6 and RSC04 open until those proofs exist.

### S15 preparation: leased hardware gate, not yet satisfied

The bounded harness specification is
[the S15 acceptance runbook](../../../reference/engines/radiance/RSC02_S15_ACCEPTANCE.md).
The native `bc_s15_dry_run_test` runs with no cards, invokes the genuine
stored-manifest owner against a fake engine, exercises the pre-existing Bash
launcher fake-engine cases, and checks the strict-v2 semantic projection
(comparing role/BDF/model/plan/resource/handoff/argv/env and excluding only
explicit per-session transport and bookkeeping). It does **not** prove
hardware equality, power or throughput. RHP12's >=4 clean paired A/B
repetitions per dense and helper configuration and their retained-session
cohorts are required before S16. A remaining S14 integration dependency is
the serving-owner lifecycle adapter for the maintained
`bigcherry ab-benchmark --server-config` capture; that command currently
starts the engine directly, not both serving owners. Never substitute it for
launcher-to-owner hardware parity. RSP01 owns the release asset and workflow.

## Detailed Solution & Technical Design

**A session owner process (review 2026-11-14, P0).** `bc-serve serve CONFIG` is a foreground C++ process that owns
the whole serving session: it starts the children in their own process groups, reaps them (`waitpid`), watches health,
holds the power journal, and on any child's exit or on a stop request stops the rest in order and restores power only
after every child has exited. `bc-serve start CONFIG` is `serve` detached (`setsid`), returning once the canary passes;
`stop` and `status` talk to the owner over a control socket in the session's run directory (never by signalling a
recorded PID, which can be reused). The GPU lease must cover the
session, not the start: in the lab the lease wraps the foreground owner (`agent-lease acquire gpu ... -- bc-serve serve
CONFIG`), itself run detached by the shell (`setsid nohup ...`) when the caller should not wait. `bc-serve start`
(detached owner) takes no lease of its own and is for hosts with no lease system; a lease wrapped around `start` alone
ends when `start` returns, so lab scripts use the wrapped `serve` form, never `start`.

- Reuse what exists in C++: the SDK check (`sdk_check.cpp`, already called by `bc-serve`), the resolver, the plan cache,
  the manifest writer and reader, the host ledger and the power probes.
- The Python supervisor (`tools/bigcherry/serving/supervisor.py`, RSC11 S6/S11/S12) is the behavioural reference: its
  fake-process, lease and power-transaction tests are ported as the C++ owner's acceptance tests; it is not on the start
  path.
- Options that change settings (`--max-model-len`, `--kv-dtype`, `--budget`, ...) do not exist: a setting is a config
  value, a variant, or a `--set` experiment recorded in provenance. `--ngram-device` is not carried over (manifest
  starts refuse it today; it needs a typed manifest field first).
- Behaviour to keep, with the bash references (from the GPT review, req_rsc02-config-launcher_957d54878bcc):
  release mode (verified tagged bundle; `--release` and `--build` exclusive); SDK and ROCm environment refusal;
  a launch/stop lock (no concurrent measuring, plan storage or stop); stop order engine SIGINT then helpers, bounded
  waits then SIGKILL, never success while a process survives, power journal kept on a failed restore; binary and kernel
  library checks; BDF to HIP ordinal mapping, rank order, per-process GPU visibility; RSC03 resource derivation for
  helper starts (today `bigcherry resources derive-launch`, moved into C++); the measuring start (helpers up first,
  helper RssAnon + RssShmem recorded once, per-rank demand, exit 10 cache miss and exit 11 derived pool kept apart, no
  silent fallback); heat file and lab experiment hashed, histogram path and period, live policy fail-closed; ordered
  helper readiness from the fresh run, shared segment before the engine, `/health`, canary, MTP replay guard (depth > 1);
  power applied before measuring and serving, restored after verified exit; gfxoff floor and loaded-idle check; lab
  calibration (helper layouts only, two distinct prompt files, ABBA, strict ingestion); helper-free and helper layouts'
  different allowed overrides.

## Code Samples & Guidance



## Files



## Validation

- Card-free lifecycle tests with fake processes (ported from the Python supervisor's): ordered start; a helper or the
  engine failing during start and while serving; partial start cleaned up; repeated start/stop; stop of a dead set; a
  second start refused while one runs; stop through the control socket only; power restored only after every child
  exited, journal kept when restore fails; inherited environment ignored; `--manifest` refused when stale, naming why.
- Hardware (gpu lease held by the owner for the session): one dense config (qwen27b TP3) and one helper config
  (flashnext-4gpu) start, serve, survive a killed helper (set stopped, power restored), stop; same-session A/B against the
  bash launcher on one build passes the RHP12 gates.

## Effort & Risk



## Standards



## Acceptance Criteria

- `bc-serve start CONFIG` starts every configured model, dense and helper layouts alike; `bc-serve stop` stops it
  and restores the host's power state.
- `bc-serve start --manifest FILE` reruns a stored manifest exactly or refuses, naming the input that changed.
- No process receives configuration through flags other than `--manifest`/`--role`, or through environment variables.
- The bash launcher and `tools/lab/radiance/lib/plan-check.sh` are deleted; no caller references them.
- `bc-serve explain` (dry run) prints the manifest and the process plan without starting anything.

Native power lifecycle independent acceptance (transferred from superseded PWR09/PWR04): selected-BDF whole-layout preflight and exact original readback before first write; verified owner/lease and explicit qualified intent; durable journal and receipt, bounded apply/readback, reverse rollback, children reaped before restoration, exact prior values restored before lease release; failure and restore failure visible, no success on partial apply. Prove unresolved-journal protection across different run directories and physical devices; reconcile archived observe-only stale-journal semantics with current SessionPower::begin, no new mutation until authorized recovery. Privilege boundary prevents unauthorized sysfs writes; no promise of SIGKILL or expiry restoration without independent verified exit/lease recovery (WSL08). Native fresh YAML -> measurement -> measured start -> stop and stored replay tested with nonstock mixed TP plan; verify same policy, room recheck, fake fault/permission/missing lease/partial apply/child failure and live selected-device readback/restore. Preserve stock and zero-write observe; label admission/no-op separately from actual writes. Foreground leased serve first; detached apply not enabled until WSL08 proof. No runtime phase switching until quiescence/dwell/rollback separately proven. Existing bc_power_transaction_test, bc_manifest_power_test, session/stored/cache tests and installed release gate; no duplicate implementation.

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
