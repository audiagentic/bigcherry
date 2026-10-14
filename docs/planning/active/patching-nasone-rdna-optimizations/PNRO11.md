---
id: PNRO11
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-09T10:52:56.925530+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# DFlash/DSpark tensor-split shared lm_head: retire target-wide replication

## Description

### Authoritative disposition — 2026-10-14 (llama.cpp b11552)

**Done for the original target-wide lm_head replication proposal. No new patch, queue or GPU claim.** The proposed `src/llama-model.cpp` change from `GGML_BACKEND_SPLIT_AXIS_1` to `MIRRORED` would affect **every consumer of the target model**, including native MTP, to solve a draft-only access problem. BigCherry **1286_draft_local_shared_tensors** already provides the narrower, one-time, draft-device copy for the observed **single-device DFlash/DSpark draft + tensor-split target** route. QFP43/1286 own that implementation and qualification. Do not add a second model-load flag, global mirrored head, selector, allocator or speculative scheduler here. Earlier implementation steps, bias-mirroring edits and the proposed `1264_pnro11_lmhead_replication_tp` package are superseded; historical text is retained in Git history.

### Two distinct failure classes (do not conflate)

| Actual execution route | Source behavior | Disposition/owner |
| --- | --- | --- |
| Target `-sm tensor`, **single-device** DFlash/DSpark draft lacking its own head | Draft borrows `ctx_other->output` from a Meta buffer; draft backend cannot consume it directly. BigCherry 1286 copies Meta-backed `tok_embd/output/output_s` once to draft-device-owned storage and binds the draft graph to those copies. | Existing 1286/QFP43; target output remains vocab-sharded. Require memory and numerical qualification; patch remains `untested`. |
| **Tensor-split draft** running DFlash2's in-graph selector | Target or draft output is vocab-sharded; `MUL_MAT` yields axis-0 logits; `ggml_top_k` reaches Meta `handle_per_row`, which asserts on axis 0. 1286 explicitly excludes a Meta draft device. | Unsupported by 1286. Do not mirror the *target* head globally. A separate selector mechanism needs independent admission under QFP43. |
| Layer-split target, draft on a foreign device | Borrowed head is not Meta-backed; 1286 does not copy it, and the draft scheduler may lack the foreign backend. | Separate cross-context backend/device-identity issue (upstream #26636); QFP43 owns any disposition. |
| Draft owns its output head, or ordinary MTP/no draft | No borrowed Meta head to copy; normal target sharding is required. | No PNRO11 changes; negative controls. |

**Exact pinned source boundary:** llama.cpp b11552 `src/llama-model.cpp` chooses `AXIS_1` for ordinary `pattern_output_weight` and `AXIS_0` for `pattern_output_bias`. `ggml/src/ggml-backend-meta.cpp::handle_mul_mat` maps a vocab-sharded weight with mirrored activations to axis-0 logits; `handle_per_row` rejects axis 0. `src/models/dflash.cpp::build_dflash2_selector` invokes `ggml_top_k(ctx0, res->t_logits, top_k)`; the DFlash graph borrows `ctx_other` output only when draft output is absent. The original plan's pre-target-load capability probe and bias-mirroring edits are unnecessary for the existing draft-local route.

**1286 lifetime and cost:** `engines/llamacpp/patches/1286_draft_local_shared_tensors/patch.py` gates on `LLM_ARCH_DFLASH`, a non-Meta draft device and Meta-backed borrowed tensors; `llama_context` owns `bc_other_ctx/bc_other_buf`. A one-time `ggml_backend_tensor_get` gathers bytes to host and `ggml_backend_tensor_set` uploads the draft-local copy; optional `BIGCHERRY_DFLASH_INPUT_TRACE` hashes source/destination and first three feature handoffs. No per-token head transfer or extra target-rank replication is authorized. This is a **source/lifetime contract**, not proof that every quantization, bias, VRAM or device-topology lane passes.

### First-party measured evidence and opportunity

QFP43's 2026-10-05 dual-XTX 27B target + 6900 XT draft measurements: built-in MTP depth 5 **78.4–78.7 tok/s**, DFlash2 Q8 block 7 **74.0 tok/s**, DFlash2 Q4 block 4 **74.2 tok/s**; prompt **1315 vs 1157/1168 tok/s**. QFP43's separate post-1340 arena-fix evidence reports DFlash2 **70.4 tok/s** (block 4) and **75.6 tok/s** (block 7), with recovered acceptance. These are distinct historical runs, not a new matched PNRO11 A/B or evidence that global mirroring helps. The Flash-Next DSpark draft additionally fails the **Qwen4Exp hyper-connection feature-width** contract (10240 raw versus 2560 expected); head placement does not fix that separate QFP43 issue. No BigCherry end-to-end PNRO11 gain exists.

### Upstream and alternative-engine decision

- [llama.cpp issue #28777](https://github.com/ggml-org/llama.cpp/issues/28777), updated **2026-10-09**, identifies the exact `TOP_K`/axis-0 cause. Its **external NVIDIA V100** global-head-mirror control incurred approximately **+455 MiB per target rank** (Q4_0 example) and **+9.8%/+7.0% decode ms/step** at 8K/96K, including a regression to native MTP. These are not AMD measurements; they disqualify assuming a global mirror is a free draft-only fix.
- [llama.cpp PR #27858](https://github.com/ggml-org/llama.cpp/pull/27858), **open/unmerged** at `3286b033c5dc8bcce9958f724d4e4a93ae589646`, instead keeps target vocab sharding and moves DFlash2's selector to host on a tensor-split **draft**. Its changed `common/speculative.cpp` gathers full logits/nextn hidden states and loads selector tables from GGUF; `src/models/dflash.cpp` skips two GPU selector tables and suppresses in-graph `TOP_K` on TP. This is a bounded **reference**, not a qualified BigCherry port: host transfer, CPU top-k/lattice arithmetic, quantized table parity, tie ordering, per-request lifetime and end-to-end cost remain unmeasured here.
- [llama.cpp PR #26636](https://github.com/ggml-org/llama.cpp/pull/26636) remains open for adding borrowed-model device backends; it addresses a different foreign-device scheduler failure, not axis-0 `TOP_K`.
- vLLM's [DSpark speculator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py) keeps Markov selection in its own engine and is not a drop-in GGML Meta split handler. Do not extrapolate ROCm/TP speedups.

### Implementation and terminal gates (owned by QFP43, not a PNRO11 patch)

1. **Current single-device draft route:** retain 1286 only when exact borrowed tensor, Meta source, draft-device ownership and memory admission are proven. Check all copied bytes, `output_s` consistency, once-per-context lifetime, no per-token copies, mixed/long-context and repeated-request correctness. Compare logits/greedy/KLD, acceptance, prompt/decode and VRAM against existing MTP and a same-draft single-device reference. The patch remains untested until its own contract is met.
2. **If a tensor-split DFlash2 draft becomes necessary:** first reproduce the axis-0 `TOP_K` boundary on the exact pin. Reject unsupported topology **before graph execution** with a clear error rather than asserting or silently falling back after work begins. Reuse upstream #27858 as an *offline correctness discriminator*: compare candidate IDs, unary/transition scores and tie/NaN behavior against the native in-graph selector on a one-device control, including quantized GGUF selector tables, repeated requests and long contexts. Account for full-logit D2H bytes, host CPU time and graph submission. Do not introduce generic distributed TOP_K or target-wide mirroring as a first experiment.
3. **Performance admission:** no implementation/benchmark queue unless an eligible real draft exists and a matched current-pin screen shows an exposed end-to-end opportunity of **at least 3% versus the best working native MTP**, without losing prompt throughput or acceptance. If admitted, require >=4 independent sessions per relevant AMD architecture, >=10 paired ABBA rounds, positive CI95 lower bound on end-to-end gain, full numerical/candidate correctness, no >1% negative-control regression, and VRAM fit. Otherwise terminal **NO_PORT / KEEP_MTP**.

**Validation this audit:** 12/12 source assertions against pinned llama.cpp, 1286 and QFP43; 6/6 upstream PR-head assertions; 9/9 disposable host route/accounting assertions; Python compilation passed. No HIP compilation, GPU run, repository pytest or new benchmark. Recent independent work is Radiance mailbox/helper, patch-prefix and release; no PNRO11/1286/QFP43 changes in the preceding 12 hours. BCOP192 records this disposition.

## Steps

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

## Effort & Risk

## Standards

## Acceptance Criteria

## Notes

## Change Log

- 2026-10-14 (state): Marked done; target-wide replication proposal retired in favor of narrower 1286/QFP43 draft-local route.

## Ledger-events
