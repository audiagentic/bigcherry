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

# RSC11: serving completion programme (RSC01-RSC10 + PWR06-PWR09, one v2 cutover)

## Description

Source: GPT completion programme review 2026-11-10 (req_main_ac35b2590055), audited on origin/main 448cae58a; reviewed and adopted by bigcherry-1. Owner directive: complete RSC01-RSC10 and fold in PWR06-PWR09 as one programme. Not a rewrite: the C++ resolver (bc-serve), manifest-only engine/helper starts (#633 #651 #759), generated contract, strict reader, memory ledger and the PWR07 probes are retained.

## Decisions
1. v1 is closed and has now shipped (bc-radiance-1.17.0-r15 contains manifest-only launching): no power amendment in v1. Exactly ONE v2 contract, activated once (S13); preparation PRs add generator/tests/inactive v2 code; the final cutover switches bc-serve, radiance, r3_moe_worker, plugins, validation, launcher adapters and campaign generation together and rejects v1 (previous releases stay on their own tagged v1: release rollback, not a shim). The unmerged RAD49 diagnostic 'amendment 18' (db199c2eb on fix/rad49-tp-tail-kl) is not merged into v1; its field is assessed for v2 or left out.
2. v2 content: spec.intent.power {profile, role_overrides}; spec.plan.power {mode observe|apply, profile, profile_digest, devices[{role,bdf,settings}], cpu{}}; host pool / helper budgets derived (RSC03), the five handoff fields derived or calibrated (RSC09, fields stay), host_ledger_margin.mib refit with provenance (RSC10, no second margin field), typed native config without R3_* (RSC06/07), manifest BDFs authoritative (RSC05). Receipts carry observations (not the manifest). Serving profile and power profile are distinct CLI options (--profile / --power-profile).
3. Python supervisor `bigcherry serve start|stop|status|explain` is long-lived owner of the session: lease for the serving lifetime, ordered helpers, readiness, canary, MTP guard, plan measurement, one bounded retry, power transaction, restore only after verified child exit. It calls the C++ resolver/validator; no second resolver. The shell stop_all must not restore power while a process survives (current behaviour is unsafe).

OWNER DECISION 2026-11-14 (replaces decision 3): the supervisor is `bc-serve start|stop|status|explain` in C++, not a
Python `bigcherry serve`; one language for the start path. Inputs become readable per-model configs with named variants
(RSC12). Scope: RSC02 (rescoped), RSC12 (new).

## Gaps by item (code-based)
RSC01 partial (core delivered; client integration, remove duplicate paths) | RSC02 not started as the command | RSC03 partial (9,216 host pool fallback, 12,288 fallback+calibration, 22,528 in layouts.toml recipes) | RSC04 not started (release/install.py still emits the shell launcher) | RSC05 partial (Python layout_cmds._resolve uses environment inventory) | RSC06 partial (R3_* reads left in moe_remote_hook.cpp 130-161/1277-1307, libr3/r3_moe_settings.h, moe_mailbox_ops.hip 909-912; rad0009 patch setenv bridge; manifest_argv_test asserts it) | RSC07 partial (same bridge) | RSC08 partial (contract delivered; one-release cutover open) | RSC09 not started (five literals) | RSC10 not started as recovery path (structured host-room refusal + one retry).

## Slices (small PRs; lane owner; dependency)
S0 freeze baseline (lead) | S1 engine-side env removal: typed settings, hostile-R3_* negative test (engine) after S0 | S2 one device inventory: Python serving decisions via C++ BDF inventory (python) after S0 | S3 GPU probe + receipt types (power; probes landed: #810, #812) | S4 transactional power core PWR09 after S3 | S5 structured host-room refusal, engine half of RSC10 (engine) after S1 interface | S6 `bigcherry serve` skeleton with fakes (python) after S0 | S7 RSC03 derivations prepared for v2 (resolver) after S2 | S8 RSC09 five handoff rules/calibrations (resolver) after S7 design | S9 v2 registry/generator scaffold, inactive (contract), parallel | S10 v2 resolver/writer: named profile -> resolved power plan, inactive after S3 S7 S8 S9 | S11 plan measurement + RSC10 retry in Python after S5 S6 | S12 power lifecycle in supervisor after S4 S6 S11 | S13 single active v2 cutover after S1 S2 S7-S12 | S14 campaign + benchmark receipts after S2/S13 | S15 hardware and release acceptance, RHP12 cohorts, GPU lease after S13 S14 | S16 retire launcher and legacy surfaces (RSC04) after S15 | S17 close and reconcile RSC01-10, PWR06-09.
Single integrator for serving.cpp, manifest_writer.cpp and the generated registry (S7, S8, S9, S10, S13 conflict there).

## Gates (RHP12 definition, ABBA at least 4 clean pairs)
contract integrity (one fingerprint, v1/malformed/mixed rejected) | configuration isolation (hostile R3_* inert) | layout identity | memory admission (at most one retry) | power portability and safety (fault-injection) | serving lifecycle | numerical and performance (stock power profile first, power arms after) | release usability | benchmark evidence (manifest digest + power plan + observed state per cell).

## GPU lease policy
Almost everything card-free; first leased session only after v2 simulator, fake supervision, fake power transaction and component tests pass; then one known layout, then materially different layouts (4-card head/tail, 2xXTX, TP engine + helpers, engine-only).

## Steps

See the slice list in the Description (S0-S17); each slice is a small PR with the gate stated there.

## Detailed Solution & Technical Design

Retain the C++ resolver, strict reader, memory ledger and probes; finish integration and the single v2 cutover.

## Code Samples & Guidance


## Files


## Validation

Programme complete when a freshly installed release can: select a model, layout, serving policy and named power profile through `bigcherry serve`; generate one strictly validated v2 manifest with per-role hardware assignments and provenance; start engine and helpers from it with no R3_* configuration reads; measure and admit the memory plan with at most one bounded refit; apply the manifest power plan transactionally or observe without writes; serve, pass the canary and guards and write a receipt linked to the manifest; shut down with verified power restoration before the lease is released; pass the supported-layout RHP12 cohorts; and no serving, benchmark or release-install caller depends on the Bash launcher.

## Effort & Risk

Risks in priority order: partial/repeated manifest migrations; internal R3_* reads giving split configuration; power writes without all-device atomicity and safe stop; wrong-device selection from two inventories; host OOM / stale budgets; silent throughput regression; supervisor/cutover gaps leaving servers running; hardware portability.

## Standards

No legacy / no backward compatibility: v2 replaces v1; no translators or aliases.

## Acceptance Criteria

All items in Validation; RSC01-RSC10 and PWR06-PWR09 states updated from code evidence at S17.

## Notes

**Unpublished P1 prototype disposition (2026-11-13):** The original local `rad0014_plan_memory_v2_inputs` sidecar commit `006b7ce14` is preserved at GitHub `archive/rsc11-s13-p1-engine-native-20261113` and is **superseded, not merged**. Current `rad0011` already emits the authoritative post-rank-split rank-demand JSON consumed by native P1; the old three-field sidecar lacks `claimable_bytes` and `evidence` and has a different schema. No production or GPU testing is claimed for `rad0014`. See [unpublished branch reconciliation](../../evidence/2026-11-13-unpublished-branch-reconciliation/README.md). Retain the active RSC11 S11/S13 admission and hardware gates.


Lanes: engine (S1,S5), resolver/contract (S7,S8,S10; S9), power (S3,S4,S12), python serving/tooling (S2,S6,S11,S14), integration lead (S0,S13,S15,S16,S17). Supersedes the piecemeal RSC02/RSC04 sequencing.

## Change Log

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
