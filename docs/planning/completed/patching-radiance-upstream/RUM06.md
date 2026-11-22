---
id: RUM06
order: 6
plan: patching-radiance-upstream
state: completed
created-at: '2026-11-18T01:28:27.514165+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: L
---

# Port engine-scoped remote MoE and mixed-TP collective lifetimes

## Description

Finish the post-release remote-MoE engine and TP-group lifetime cutover on the released Radiance 2.2 / ABI44 base. Replace the process-global rad0002 callback, layer-map, burst and policy state and the TLS-selected execution owner, plus libr3moe's DSO-global HookState/HookLifetime, with one opaque engine-owned generation context and explicit bind/detach. Preserve the existing exact admission ledger, step tickets, helper session, shared TP rendezvous, kernel rows and collective fences; no second scheduler, parser, registry or transport. The shipped IQ8R 27B 6,6,4 and Flash-Next compositions stay unchanged except for behaviour a lifecycle test proves necessary.

## Steps

1. Inventory the real owner/load sequence (registry plugin discovery precedes the engine's TP-group claim) and every caller of abi/bc_moe_remote.h, rad0002 and the moe_remote_hook direct APIs. The sole owner is an admitted engine generation, distinct from TP group, helper session, placement epoch and StepTicket.
2. Make S1 and S2 one atomic source/ABI boundary, or bring DSO constructor removal and the exact already-loaded plugin-handle factory bind into S1. Discover immutable kernel rows at registry time; after the engine claim bind exactly one remote generation with a size/version-checked, context-aware callback table and typed immutable policy. Delete g_plugin, g_mu, g_layer_maps, g_burst_active and TLS-as-owner dispatch rather than bridging them. Helper-off graphs stay unaffected; a second helper-on engine is refused before the first is disturbed.
3. Move libr3moe HookState/HookLifetime into the explicit generation lifetime; tie mailbox_device and owner_generation upload/ready to the same context, world, group and epoch. Validate missing, stale, wrong-group and wrong-engine calls before any HIP or callback side effect; remove constructor registration and DSO-global transport state.
4. Reuse admission_ledger, engine_admission_gate, engine_rank_coordinator, mailbox completion and the rad0017 shared group and poisoning. Clean stop: stop admission, cancel only an unused unbound tail, complete bound parents and ranks, wait for helper/GPU/rank proof, join workers, run native per-instance fini, detach and free; it must not interrupt a healthy ledger. Fault stop poisons and unblocks peers, then uses one ordered release tail.
5. Preserve post-publication ArInst pointer lifetime, group isolation, mixed-rank declared collectives, zero-width and tail spans and the IQ8R-derived scale-group validation. Failure-inject after each allocation, publish and bind; verify no double free, host/VRAM leak, stale map or cross-generation reuse.
6. Green a bounded host lifecycle suite (sequential and concurrent engine claims, static row discovery, direct APIs, bound issue-ahead stop, fault wake, partial init, exactly-once), one same-SDK component build and a new-lifecycle-only helper-on mixed-TP canary on the installed candidate. Reuse the RUM09 release and quality matrix as the baseline; rerun performance or all configs only if changed code demonstrably affects them. Record net code and target deletion and off-path cost.

## Detailed Solution & Technical Design

Actual extension callbacks include admit, bind, cancel, before_pass, host_done, rank_done, poll and shutdown. Preserve their admission ledger rules; do not replace them with a new task scheduler. Parent identity, placement epoch and helper session identity must remain distinct. A TLS ticket can carry per-rank current work, but it cannot select process-global policy for a different engine or survive a reused worker thread without clearing.

Upstream g_model is now indexed by rad_build_slot(ctx)/rad_slot(c); collective rendezvous uses a tp_group key. Engine rank 0 is not globally unique. Bind a narrow local EngineExecutionState reference to the context at creation; resolve that reference outside the MoE inner loop. Keep data structures allocation-free for normal route/combine calls.

Upstream AQL and kernel changes adjust system acquire, scalar-cache invalidation, doorbell flush and wake stores. Preserve the target's documented chain_fenced contract when reusing two-shot, duplex and once paths (rad0021/rad0024/rad0032/rad0033). Source-level fence removal is not an acceptable cleanup. Reuse the shared collective group with one scratch owner and release it after all users drain.

Failure handling is part of a working transition: helper death must name a failed helper/session, not let an absent remote result be read or misreported as generic OOM. Existing numerical-quality policy remains in force; do not add replication or a new fallback path to dodge missing owners.

ABI43 review findings: `moe_remote_hook.cpp` has process-global `HookState` and DSO-static registration. `RTLD_LOCAL` does not create an owner per Engine. The remote device and owner-generation APIs bypass the `BcMoeRemotePlugin` callback surface and must use the same bound owner. Clean shutdown cannot call the existing interrupt callback: ledger polling refuses after interruption, and `Engine::serve` currently aborts the barrier before joining. Clean stop must drain bound work; fault stop must interrupt to wake blocked peers. In candidate `libr4d/r4d_ar_entry.hip`, `ArInst::build` holds allocations in locals until late publication; failure can delete the instance without normal release and leak host/VRAM allocations. Deallocation must wait until no peer can dereference the shared address.

## Code Samples & Guidance

Proposed state ownership sketch:
```text
Engine admitted generation
  owns RemoteExecutionState
    owns helper session + immutable PlacementView + admission ledger
    references upstream TP group and bound kernel handles
Rank step
  carries parent ticket + generation; never re-resolves config
Shutdown
  admissions closed -> admitted work drained -> callbacks unregistered -> storage freed
```

Target source landmarks: upstream libr4d/r4d_ar_entry.hip rv_for(group), rv_barrier(rv,world); local BcMoeRemotePlugin callbacks in rad0002; existing admission_ledger.h and engine_rank_coordinator.h. These are extension points to reconcile, not a request for another callback framework.

### Deep scan implementation refinement

Trace bc_moe_remote_register/g_plugin -> admit/bind -> rank TLS -> before_pass -> route/combine -> host_done/rank_done -> shutdown. Reuse admission_ledger/engine_admission_gate/engine_rank_coordinator, bind engine/generation/group once and remove function-static policy caches. Reuse mailbox epoch; don't introduce a second ticket or epoch service. Preserve target fence/acquire/wake ordering.

Source evidence and removal conditions: docs/reviews/radiance-upstream-2026-11-18/CONSOLIDATION.md. Use patch-footprint.json to inspect every overlapping patch before changing an attachment site.

## Files

- engines/radiance/patches/rad0002_moe_remote_handoff/patch.py
- engines/radiance/patches/rad0017_vtp14_n3_shared_rendezvous/patch.py
- engines/radiance/patches/rad0018_vtp14_n2_tp_ownership/patch.py
- engines/radiance/patches/rad0019_vtp14_n2_tp_ownership_fp8/patch.py
- engines/radiance/patches/rad0020_vtp14_tp3_pipelined_prefill/patch.py
- engines/radiance/patches/rad0025_multiseq_pipelined_prefill/patch.py
- engines/radiance/components/moe-remote/admission_ledger.h
- engines/radiance/components/moe-remote/engine_admission_gate.h
- engines/radiance/components/moe-remote/engine_rank_coordinator.h

## Validation

Extend the existing ledger and coordinator fixtures for cancel-unused, partial-init failure, reused rank/group, drain-before-free and helper death. Build touched plugins once. Focused gates: engine-generation bind/unbind across repeated engine lifetimes; clean stop with a bound issue-ahead parent; fault wake, timeout and partial-registration failure; allocation failure after each ArInst allocation with zero leaked host/VRAM bytes; cross-DSO same-group rendezvous and different-group isolation; zero-width and uneven rank declaration order; exactly-once bridge call counts. One representative mixed-TP helper-on canary on the installed candidate is part of this item (RUM09 completed the release roster, so the wider hardware route is not repeated). No all-layout sweep or prolonged stress unless a specific lifecycle failure demands it.

## Effort & Risk

L: the main hazards are reused ranks/groups, stale callbacks and unfinished reservations during free. Reuse proven ledgers and make teardown order explicit; don't redesign transport algorithms.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

The native engine, not a process global or a rank number, exclusively owns one generation of remote callbacks, policy, maps, burst state, transport and direct device/owner operations. Registry discovery exposes stable rows before bind; every runtime action refuses a wrong or stale group, engine or generation. No DSO constructor registration, global helper policy, competing callback path or TLS-selected transport survives in the production composition. Admission, cancel, bind, host_done and rank_done are exactly once. A clean stop drains bound current and ahead work without fault interruption; an error stop wakes peers and frees shared allocations only after peer unwind. Repeated engine and TP-group lifetimes, second helper-on engine refusal, faults and partial initialization pass focused host and component tests, and at least one modified-lifetime mixed-TP helper-on start, canary and stop passes on the installed candidate. The shipped IQ8R TP3 6,6,4 roster is not re-qualified from scratch without a code-dependent reason.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM02, RUM03, RUM05. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM02/RUM03/RUM05. Coordinate RCS14 topology without absorbing unrelated backend/kernel redesign. Reuse the current protocol epoch rather than introducing another generation format.

Astra source review ASTRA-CONSOLIDATION.md: initial scope is the admitted single-engine generation, not a general multi-engine factory. Preserve independent helper model-byte ownership verification. Reuse existing parent tickets and mailbox epochs. One existing partial-start/drain fixture plus route/combine activation in RUM09 is sufficient unless a real failure requires expansion.

2026-11-19 Sol implementation refinement: start with the pre-publication ArInst allocation leak in ABI43 composed `libr4d/r4d_ar_entry.hip` (`build()` locals 424–470, `acquire()` failure 558–559). Make each acquired address rank-locally owned immediately and reuse the ordinary release tail only before publishing `rv.scratch/flags/seq`; do not free published pointers until peers unwind. A host-only fake-HIP test can call the real exported init path and inject failure at each allocation, asserting exact-once matching frees and no peer publication; verify the libr3 rewrite preserves the fix. Separately inspect `r4d_ar_oneshot_2rank_exact.hip` allocation cleanup on memset/sync failure. Keep post-publication unwind, engine-owned callback context, clean-stop drain, and helper lifecycle as separate required RUM06 work; this small slice does not complete the item.

Checkpoint cbdd21f5b frees rank-local collective buffers on pre-publication initialization failure, including GPU memset and stream-sync refusal. The exact 2.2 host fault fixture passed 23 cases; ordered 33-patch apply and libr3 rewrite verification passed. The existing historical N3 tests stop at rad0046's required-state gate. This checkpoint leaves coordinated post-publication unwind, engine-scoped remote callbacks and clean-stop drain open.

Checkpoint c5abd3b47 poisons and wakes failed rendezvous and holds published owner buffers until joined rank workers and device contexts have drained. Four rendezvous cases, exact ordered 33-patch replay, libr3 rewrite and host/HIP syntax checks passed. Live mixed-TP, clean stop and remote callback ownership remain open.

Checkpoint bedf51edb consolidates rank attachment rollback, clears the result pointer on refusal and destroys completion events on their owning device during shutdown. Exact ABI44/ROCm syntax and four focused host fixtures passed. This preserves the frozen canary SDK; engine-scoped remote lifecycle, clean-stop drain and hardware proof remain open.

2026-11-20 owner re-scope (RUM09): the bump must carry every config main serves, so this item is now a bump blocker: the Qwen 27B TP3 configs (gfx1100, gfx1100, gfx1201 engine ranks) and the Flash-Next tp2xtx configs must start and qualify on 2.2. The TP patches rad0016-rad0033 are already in serving-core on the candidate; what remains is the mixed-rank code-object, ordinal-reversal, group-rendezvous and span tests and a TP3 27B start/canary/paired comparison on the 2.2 build.

2026-11-20 progress: TP3 27B on the 2.2 build gets through admission and device selection to kernel resolution (mixed-arch rank-devices fixed in rad0009, 64bae9124). The remaining start blocker is RUM11 (E4M3 page16 store rows, per-rank head counts 6/1, 6/1, 12/2). After RUM11: TP3 start/canary/stop, then the mixed-rank code-object, ordinal, rendezvous and span tests, then the paired comparison and KLD for qwen27b-tp3-int8-556 and its long and bf16i8 variants, and flashnext-tp2xtx-r9700.

2026-11-21: qwen27b-2xtx's declare refusal is not a TP split problem. Radiance 2.2 retires the MXFP4 weight format (arch/common/rad_fp8.h:1174-1194 linear_retired / linear_refuse_retired, called from rad_fp8.h:1354), and the refusal names the int8 IQ8R recipes as the replacement. It affects the three configs on the MXFP4 container: qwen27b-2xtx, qwen27b-tp3 and qwen27b-tp3-capacity. A GPT lane drafted rad0055_qwen35_plain_mxfp4 (branch agent/rum06-qwen27b-tp2, 7082337c9), which removes the retirement check. Owner decision 2026-11-21: convert, don't patch back. The three configs move to the int8 (w8a8) Qwen3.8-27B container that the qwen27b-tp3-int8* configs already use, each with its own start/canary/stop, paired comparison and KLD on the new weights. rad0055 isn't merged; its branch is kept as an archive tag for the evidence. Their TP ownership splits must be re-validated against the w8a8 container (same model dimensions, so the specs should still parse).

2026-11-21 owner decision, which supersedes the int8-w8a8 note above: all eight qwen27b configs move to upstream's IQ8R container StillDeadcode/qwen3.8-27b-iq8r-dflash (revision c127a34140e4, 29 GiB, qwen35_w8a8 plugin, GPTQ int8 with u8 scale per 256 under a bf16 row scale, fwht128, DFlash2 drafter in IQ4R codes). Reasons: the MXFP4 container is retired in 2.2, and our w8a8-df2 container's DFlash2 drafter is MXFP4 (q38-27b-w8a8-df2.recipe); IQ8R matches w8a8 quality (KL 0.00039-0.00046, top-1 about 99.2%) with about 9% faster 8192-token prefill per upstream docs/IQR.md (group 256 vs 128). One model for every 27B config. Each config gets start/canary/stop, the paired comparison where it is measured production, and KLD/top-1 against qwen3.8-27b-bf16. kv-cache-dtype per config is to be confirmed against IQ8R's rows (upstream pairs it with i8).

2026-11-21 IQ8R on our TP3, measured on the 2.2 build of f5c82c33d. w8a8 scored against a bf16-weights reference (qwen3.8-27b-bf16, corpus rad-kld-corpus-4, 8240 positions, 5,5,6 split, fp8 KV) gives mean KL 0.00139, median 0.00049, p99 0.0116, top-1 96.8%, PPL ratio 0.9998. IQ8R refuses: (a) at the 5,5,6 and 7,7,2 splits, declare invalid argument at rad_block_gdn_fp8.h:387 (ssm_out column shard: lv 15 or 21 heads x 128 is not a multiple of IQ8R's 256-wide scale group); (b) at 4,4,8, rank 2 overruns 'h.s8' in op rmsnorm_had_quant_i8g {n=5120 group=256}, a buffer our int8 pipelined prefill sizes for group 128. Owner decision: port our TP patches to IQ8R in this bump. Scope: (1) make the int8 pipelined-prefill and TP patches (rad0016-rad0033, chiefly rad0030 int8 pipelined prefill, rad0025-0027, rad0018/19 ownership) derive buffer shapes from the container's scale group, not group 128; (2) TP ownership specs must keep each rank's V and FFN shards a multiple of the scale group, as a validated rule with a refusal naming both values, not a per-config constant; (3) the 5,5,6 and 7,7,2 layouts cannot exist under IQ8R, so re-derive the nearest valid splits by the VTP36 method (KV per card, streams, TTFT) and replace those configs' splits, with evidence; (4) KLD/top-1 of IQ8R vs bf16 on every 27B config. The w8a8 numbers above are the comparison bar.

2026-11-21 RUM06 IQ8R TP3 host-only patch checkpoint (`agent/rum06-iq8r-tp`; not hardware-qualified): source pin `b6b8751cccb0e834a13ae9e5afca824ee9a487aa`. The plain-norm pipelined issue in `rad0030` now slices `h.s8` by `ActFP8::g8` (probed from the container), or the E4M3 scale by its own fixed format group. The optional fused MLP pair slices both `cq/cs` scale operands by each pair's `cg()`. `rad0019` now probes the int8 weight's `linear_w8h_group(enc)` and checks *before activation or linear declarations* that both the V shard's width and start and the FFN shard's width and start are divisible by the probed group; its error gives rank, V/FFN widths/starts, and the actual group. The `DenseFP8::span` row/column split uses the format's group and scale-view extents (full [N,K] for IQ8R two-level scale, reduced K/group for a one-level scale), retaining 16-row tile alignment. The existing `rad0025`–`rad0027` staging uses `cq()/cs()`, not a fixed group width; E4M3-only `RAD_FP8_BLOCK`, 128-wide FWHT, and actual kernel-specific 128 groups were intentionally not changed.

IQ8R group=256 valid candidate ownerships, for bigcherry-1's **measured** VTP36-style KV capacity/streams/TTFT re-selection (no config promoted here):

| Replaces | Candidate `--tp-ownership` | IQ8R KV tokens per card | Status |
|---|---|---|---|
| `lk=5,5,6;lv=15,15,18` (invalid 1920-column V ranks) | `kv=1,1,2;ffn=5120,4864,7424;lk=4,4,8;lv=12,12,24` | **579,744** at fp8 KV with 8x8192, checkpoint4096 | **Existing** `iq8r-448.log:111` placement receipt (startup/weight load; step failed from now-patched `h.s8` slice); not a new GPU run |
| `lk=7,7,2;lv=21,21,6` (invalid 2688-column V ranks) | `kv=1,1,2;ffn=6656,6656,4096;lk=6,6,4;lv=18,18,12` | **~380k–400k estimated**, 16-token rounding; **unverified** | Analytic estimate: `iq8r-448.log:62-64` reports 6998/6756/10920 MiB shard weights and 4823/4823/9646 MiB KV pools. The 256-row FFN difference is 242 MiB (~0.945 MiB/row); shifting 4/4/8 to 6/6/4 adds ~1.45/1.69 GiB FFN to the XTX ranks, plus delta-head weights (~0.67 GiB/rank from the remaining logged weight delta), leaving roughly 3.3–3.4 GiB KV pool on the limiting XTX. Scaling the log's KV page and delta-net-state cost yields a preliminary ~0.38–0.40M-token ceiling; obtain a **fresh planner receipt** before treating this as capacity. |

The earlier VTP36 throughput/TTFT rankings on W8A8/MXFP4 do not carry over automatically to IQ8R. `4,4,8` is also a valid fallback for the former `7,7,2` profile if it wins the measured tradeoff. The new host C++ fixture executes the exact C++ admission check for group=256 failing `5,5,6` and `7,7,2`, and passing all ranks of `4,4,8` and `6,6,4`; checks FFN and start offsets, group=128 backward behavior, and `DenseFP8` scale views. Complete serving-prefix patch apply/idempotence, the focused Python suites and libr3 native rewrite `--check` pass. GPU startup, per-config KV admission receipts, IQ8R KLD/top-1, TTFT, streams and stop/drain are still for bigcherry-1; RUM06 is not closed.

2026-11-21 bigcherry-1: IQ8R TP port 45fd1eb4a reviewed and merged into node/radiance-latest-bump (06bacf8ac). patch-rebase-check --all 38/38 clean; focused tests pass. Engine build queued from that tree, then IQ8R KLD vs the bf16 ref at 4,4,8 and 6,6,4 under the gpu lease (task scratch runner run-iq8r-tp.sh, rum06-27b-kld).

2026-11-21 bigcherry-1: IQ8R on TP3 with the port (build bdd48feb from 06bacf8ac), scored against the bf16-weights reference (rad-kld-corpus-4, 8240 positions, fp8 KV, num-speculative-tokens 0). 4,4,8 (kv=1,1,2;ffn=5120,4864,7424;lk=4,4,8;lv=12,12,24): KLD mean 0.00123, median 0.00043, p99 0.0105, top-1 97.51%, PPL x1.0005. 6,6,4 (kv=1,1,2;ffn=6656,6656,4096;lk=6,6,4;lv=18,18,12): KLD mean 0.00128, p99 0.0111, top-1 97.27%, PPL x1.0005. Both beat the w8a8 bar (0.00139, top-1 96.77%), and both start and run on all three ranks. Remaining: switch the qwen27b configs to IQ8R and pick the split per config by measured KV capacity, streams and TTFT (VTP36 method), then the 27B starts.

2026-11-21 bigcherry-1, choosing the IQ8R TP3 split (build a329c651 at 45247293a, gpu lease, 2 A-B-A-B rounds on the qwen27b-tp3-int8-556 settings; stream-bench 1/4/8 streams, 4096-token prompts, 256 tokens). 6,6,4 vs 4,4,8: decode step at 1 stream 22.4 vs 27.5 ms (-18%), 71.7 vs 65.4 tok/s; 4 streams 73.5 vs 73.8; 8 streams 91.2 vs 88.1 tok/s; TTFT on a 4.2k prompt 1.93 vs 1.86 s (+3%). Capacity config (262144 ctx): 2.8 vs 2.3 full-length sessions. Decision: 6,6,4 for every TP3 config. qwen27b-tp3, -int8 and -int8-556 become identical, so they are consolidated into qwen27b-tp3, and the long variants are renamed qwen27b-tp3-long[-bf16|-bf16i8] (branch agent/rum06-27b-iq8r-configs).

CURRENT STATUS (main 60479d39d, 2026-11-22; assessed by an independent read-only review against code, see PR). The 2.2 pin and ABI44 composition are released; all 16 shipped native configs have a documented installed start/canary/stop (RUM09 Gate E); qwen27b-tp3 uses the IQ8R container with kv=1,1,2;ffn=6656,6656,4096;lk=6,6,4;lv=18,18,12. Met on main: ABI/rank layout, shipped IQ8R canary, baseline rendezvous and fences (rad0017), pre-publication ArInst cleanup and rendezvous poisoning (checkpoints below). Partial: exact-once ledger terminal states, fault wake, failure cleanup. Implemented on PR #1272 (engine-owned opaque generation replaces the process-global registry; clean stop; failure-atomic factory; lifetime borrow; owner-bound mailbox init; derived deadlines), with host tests, the sanctioned lane build and card smoke; the plan item closes when #1272 merges and the final-build smoke and identity check pass. Not proven: bound-ahead stop/death on real hardware, and an installed release candidate. RUM08 does not depend on this item. Completed predecessors RUM02, RUM03, RUM05 are evidence, not open owners. Note: rad0055 now means the step-phase trace (merged #1269), not the plain-MXFP4 experiment named in an older note.

HISTORY (kept as evidence; the earlier MXFP4 and w8a8 conversion proposals were superseded by the IQ8R decision and are not restated).

Imported from the reviewed design (docs/reviews/radiance-upstream-2026-11-18/README.md; MANIFEST-AND-EXTENSIONS.md; ASTRA-CONSOLIDATION.md: initial scope is the admitted single-engine generation, not a multi-engine factory; preserve independent helper model-byte ownership verification; reuse parent tickets and mailbox epochs).

ABI43/ABI44 lifetime checkpoints (host-only, exact 2.2 source): cbdd21f5b frees rank-local collective buffers on pre-publication initialization failure including GPU memset and stream-sync refusal (23 host fault cases, ordered 33-patch apply and libr3 rewrite check passed); c5abd3b47 poisons and wakes a failed rendezvous and holds published owner buffers until joined rank workers and device contexts drained (4 cases); bedf51edb consolidates rank attachment rollback, clears the result pointer on refusal and destroys completion events on their owning device. Post-publication unwind, engine-scoped callbacks and clean-stop drain remained open after these.

2026-11-20 owner re-scope (RUM09): the bump carries every config main serves, so the Qwen 27B TP3 configs and the Flash-Next tp2xtx configs had to start and qualify on 2.2 (mixed-arch rank-devices fixed in rad0009, 64bae9124; E4M3 page16 rows by RUM11).

2026-11-21 IQ8R decision: Radiance 2.2 retires the MXFP4 weight format (arch/common/rad_fp8.h linear_retired), so all eight qwen27b configs moved to upstream's IQ8R container (StillDeadcode/qwen3.8-27b-iq8r-dflash, revision c127a34140e4, qwen35_w8a8 plugin, u8 scale per 256 under a bf16 row scale). IQ8R's 256-wide scale group refused the 5,5,6 and 7,7,2 splits (ssm_out column shard not a multiple of 256) and overran h.s8 at 4,4,8 in the int8 pipelined prefill (sized for group 128). Port: rad0030 slices h.s8 by the container's group; rad0019 probes linear_w8h_group and refuses, naming rank, widths, starts and group, before any declaration; DenseFP8::span uses the format's group and scale-view extents; rad0025-rad0027 already use cq()/cs() (port 45fd1eb4a, merged 06bacf8ac; patch-rebase-check 38/38 clean).

IQ8R TP3 measured (rad-kld-corpus-4, 8240 positions, fp8 KV, num-speculative-tokens 0, bf16-weights reference qwen3.8-27b-bf16; w8a8 bar: mean KL 0.00139, top-1 96.77%): 4,4,8 mean KL 0.00123, p99 0.0105, top-1 97.51%, PPL x1.0005; 6,6,4 mean KL 0.00128, p99 0.0111, top-1 97.27%, PPL x1.0005. Split choice (build a329c651, two A-B-A-B rounds, stream-bench 1/4/8 streams, 4096-token prompts, 256 tokens): 6,6,4 vs 4,4,8 decode step at 1 stream 22.4 vs 27.5 ms (-18%), 4 streams 73.5 vs 73.8 tok/s, 8 streams 91.2 vs 88.1 tok/s, TTFT on a 4.2k prompt 1.93 vs 1.86 s (+3%), capacity config 2.8 vs 2.3 full-length sessions. Decision: 6,6,4 for every TP3 config; qwen27b-tp3, -int8 and -int8-556 consolidated into qwen27b-tp3, long variants renamed qwen27b-tp3-long[-bf16|-bf16i8].

COMPLETION (2026-11-22, PR #1272 merged; supersedes the CURRENT STATUS paragraph above where they differ). The native engine now owns one opaque remote-MoE generation: process-global registry, DSO constructor registration, TLS-selected owner and libr3moe DSO-global hook state are deleted; the explicit factory bind, slot-keyed lookup, serial handles with a lifetime borrow, owner-bound mailbox init, failure-atomic factory, derived deadlines (registration = startup protocol deadline, layer-map consume and stop budget = readiness/quiesce deadlines) and the clean-stop/fault-stop split are in. Three independent read-only reviews drove the fixes (lock held across a callback, borrow vs close ordering, group-only device init, factory failure rollback, a cancel classified as a fault, success after close began); the final review had no open P0/P1 after b1069c6a0. Host evidence: moe_remote_generation, hook_identity, hook_retire, generation_close, layer_map_wait, helper_registration, admission_ledger and rad0017 failure-injection tests under ASan/UBSan; patch and release suites pass. Card evidence on the sanctioned lane build of the final head (helper on, short gpu leases): clean start/canary/stop with exit 0 and no interrupt or fault line on flashnext-4gpu and qwen27b-tp3 (mixed gfx1100/gfx1100/gfx1201, 6,6,4); a helper killed mid-run takes the fault path (serve exit 1, cause named, no hang) and the same config restarts cleanly; a second helper-on engine is refused (its measuring helper runs out of GPU memory; the plugin-level busy refusal is proven by host tests only); greedy 96-token output is byte-identical to the installed 2020.0.1 on qwen27b-tp3 and on flashnext-tp2xtx-r9700 (previous builds of the same branch; the final build repeated qwen27b-tp3). The four mixed-TP checks (code object, ordinal, rendezvous, span) are covered as live behaviour on those two configs, not as separately named tests. NOT proven: bound-ahead stop or death on real hardware, and an installed release candidate of this change (the release bundle that carries it is the first installed proof). Off-path cost: no per-step lock or allocation added.

INSTALLED-CANDIDATE PROOF (2026-11-22): the releases that carry the change, bc-radiance 2020.1.0 and 2020.1.1, were downloaded from the release (archive, sha256, receipt), installed with bc-release install into an isolated root with the ROCm SDK, and verify and status reported verified. Under short gpu leases each installed prefix started, answered the canary and stopped cleanly (bc-serve stop and serve exit 0, no interrupt or fault line in the serve and engine logs) on flashnext-4gpu (helper on, four cards) and qwen27b-tp3 (mixed gfx1100, gfx1100, gfx1201, 6,6,4). This closes the 'installed candidate' gap named in the completion note above. Still not proven on hardware: a bound-ahead stop or death.

## Change Log

- 2026-11-18T01:28:27.514165+00:00 (created-by): Created by codex
- 2026-11-18T01:39:45.710628+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:17.981906+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:58.746185+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:19:52.290526+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:23:21.830497+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:26:23.178450+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4303
- 2026-11-19T02:21:54.505617+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:validation, section:acceptance_criteria
- 2026-11-19T09:12:22.140456+00:00 (updated-by): Updated: section:notes
- 2026-11-19T23:03:33.981692+00:00 (updated-by): Updated: section:notes
- 2026-11-19T23:20:26.906081+00:00 (updated-by): Updated: section:notes
- 2026-11-20T01:23:16.000046+00:00 (updated-by): Updated: section:notes
- 2026-11-20T08:50:11.218145+00:00 (updated-by): Updated: section:notes
- 2026-11-20T12:44:44.499463+00:00 (updated-by): Updated: section:notes
- 2026-11-21T00:44:35.690825+00:00 (updated-by): Updated: section:notes
- 2026-11-21T00:53:10.311864+00:00 (updated-by): Updated: section:notes
- 2026-11-21T01:38:17.936858+00:00 (updated-by): Updated: section:notes
- 2026-11-21T02:18:21.184326+00:00 (updated-by): Updated: section:notes
- 2026-11-21T02:35:33.763913+00:00 (updated-by): Updated: section:notes
- 2026-11-21T03:07:59.089816+00:00 (updated-by): Updated: section:notes
- 2026-11-21T20:17:02.760616+00:00 (state-transition): State: pending → in_progress
- 2026-11-22T01:51:51.539077+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-22T07:30:48.856632+00:00 (updated-by): Updated: section:notes
- 2026-11-22T07:31:40.203101+00:00 (state-transition): State: in_progress → completed
- 2026-11-22T08:39:32.179429+00:00 (updated-by): Updated: section:notes
