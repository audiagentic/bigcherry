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

Lanes: engine (S1,S5), resolver/contract (S7,S8,S10; S9), power (S3,S4,S12), python serving/tooling (S2,S6,S11,S14), integration lead (S0,S13,S15,S16,S17). Supersedes the piecemeal RSC02/RSC04 sequencing.

## Change Log

## RSC11 S7: offline v2 resource derivation (2026-11-10)

The preparatory GPU-free Python ledger is `tools/bigcherry/engine_adapters/radiance_resources.py`. A caller obtains `radiance_inventory.Layout` from bc-serve's BDF inventory and calls `Layout.derive_resources(model, targets, host, worker_footprints_bytes=...)`. The result carries a host pool, host KV tier, per-helper expert budgets and a byte-exact explanation. Inputs are mandatory: no guessed model footprint, ownership plan, or worker footprint. Re-resolve on device/owner-map/model/context/RAM change.

The shipped `engines/radiance/layouts.toml` now omits all literal helper `budget-mib` settings, allowing the existing v1 bc-serve worker-footprint-based budget derivation to take effect. The v1 manifest schema, C++ resolver, manifest writer and generated registry have not been changed.

**Required v2 handoff, for S10/S13:** bc-serve must publish the engine-measured host embedding, unowned expert, staging, n-gram, checkpoint and other resident bytes; the actual owner-map digest; host-pool margin calibration; KV bytes per token and the context/retention target; live MemAvailable, effective cgroup headroom and pin availability; and each selected helper BDF's complete non-expert worker memory (lanes, pointer banks, slot map, incoming slot) from `worker_footprint.h`. The output must carry provenance and be invalidated/recomputed with inventory, owner map, model or targets. The v2 writer should use those values to populate the existing resource budget concepts, enforce RAM/pin admission and compare with engine `--plan-memory`; fail closed if any required measurement is missing. No v2 contract fields are introduced by S7.

**Unresolved until the single v2 cutover:** `serving.cpp` still holds the v1 host-pool/host-KV defaults and engine-only recipe-specific pool sizes remain in layouts.toml. The v1 Bash helper launcher now calls the GPU-free Python launch adapter on every start: bc-serve selects actual BDF roles, helper footprints and owner map; the RAD directory and current host memory facts feed the shared resource ledger. The adapter prints a breakdown and injects computed host-pool and host-KV MiB without restoring the old constants. The C++ v1 defaults remain for non-launcher callers until the single S13 cutover, when model/plan sizes and KV geometry must be carried through a typed v2 resource interface. The v1 manifest schema is unchanged; S16 removes the Bash surface.
