---
id: QFP12
order: 0
plan: patching-qwen-flash-next
state: deprecated
created-at: '2026-10-03T17:03:36.365732+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: S
---

# 1306 split the Qwen4Exp shared expert (mirrored upstream: full copy computed on every GPU)

## Authoritative b11552 disposition (2026-10-12)

**Closed as not adopted for the current production profile.** Keep patch 1306 evaluated, default-off, and outside production. The earlier open-ended Steps and Validation below are historical, not new queued work. No implementation, patch or benchmark definition changes are authorized by this audit.

### Implementation, graph, and lifetime

- \`engines/llamacpp/patches/1306_qwen4exp_shexp_split/patch.py\` inserts a Qwen4Exp-only \`BIGCHERRY_SHEXP_SPLIT=1\` opt-in before the output-head rule in \`src/llama-model.cpp::llama_meta_device_get_split_state\`. It places shared up/gate on AXIS_1 and down on AXIS_0, anchored to \`ffn_down_shexp.weight\`; the separate scalar gate \`ffn_gate_inp_shexp\` remains MIRRORED. Native b11552 and inspected current upstream master only apply these three shared-expert rules to DSV4, not Qwen4Exp.
- \`src/models/qwen4exp.cpp::graph::build_layer_ffn\` (1146–1193) builds routed \`moe_out\`, shared up/gate/GLU/down, sigmoid of the mirrored scalar gate, \`MUL(ffn_shexp, shared_gate)\`, then \`ADD(moe_out, ffn_shexp)\`. A sharded down projection can yield a PARTIAL result. Exact mathematical equivalence is \`sum(A_r) + g*sum(B_r) = sum(A_r + g*B_r)\`; floating-point summation order and MTP acceptance need independent proof.
- \`ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_get_split_state\` (552–612) and \`ggml_backend_meta_graph_compute::get_i_delayed\` (~2050–2260) are the actual synchronization authority. The graph may delay/merge two PARTIAL reductions only when branch use counts are one, shape/type match, the second branch is immediately followed by the exact ADD, and per-rank COMPUTE flags agree. Extra views/consumers or a graph rewrite can defeat the coalescing. **“No extra collective” is not guaranteed by 1306's tensor placement; it is an unverified runtime assertion.** A single Meta reduction must be observed per layer before any revival.
- FFN slice granularity is \`lcm(ggml_blck_size(type),128)\`, with per-layer rotation of the model's tensor-split vector; real physical bytes and work need not be exactly one-third. The patch uses the existing Meta buffer/graph/collective lifetime and depends on \`1303_attn_kv_tensor_split\`; no separate placement registry, scheduler, cache, allocator, collective or host staging is warranted. Check graph recapture/reallocation and zero-contributor semantics before claiming correctness.
- Manifest \`patch.toml\` records \`state="evaluated"\`, whereas the legacy Python module declares \`STATE="untested"\`. The manifest is the documented lifecycle state; the loader effect of this mismatch was not exercised. Do not silently equate it to a production-ready package.

### First-party evidence (2026-10-04)

Profile-v2, same-binary, \`BIGCHERRY_SHEXP_SPLIT\` on/off screens:
- ~24K: split 47.2 ms/step, mirrored controls 46.0 and 46.6 (split ~1.3–2.6% slower); MTP accepted/drafted 169/254 versus 175/240.
- ~80K: split 53.6 ms/step, controls 53.3 and 53.4 (~0.4–0.6% slower); MTP accepted/drafted 166/266 versus 170/255.
- VRAM change ~-185 MiB **per XTX** and -135 MiB R9700; summed 505 MiB is not 505 MiB of usable headroom on any single rank. Activation marker observed. These quick screens are not a matched-draft-stream logits/KLD or statistically qualified A/B; changed acceptance is observed but not causally attributed.

2026-10-12 cheapest discriminator: 15/15 source-static assertions passed across patch, b11552 graph, Meta delay predicates and manifest. A disposable NumPy F32 three-rank host fixture compared two reduction/grouping orders for 10,000 seeded cases: 4,958 had different F32 bits, maximum absolute delta 0.00048828125. This proves possible numerical sensitivity, not a GPU correctness defect. **No new BigCherry performance measurement, repository pytest, HIP/C++ compilation, GPU run or graph capture occurred.**

### External mechanism and consolidation

- Current llama.cpp master still has the DSV4-only shared-expert split; no Qwen4Exp upstream absorption was observed in the inspected source.
- vLLM \`vllm/model_executor/models/qwen3_next.py::_should_replicate_misaligned_shared_expert\` checks TP divisibility and quant group alignment, deliberately replicating in some EP/sequence-parallel cases. Use this as an **admission/semantic reference only**, not an AMD gain or drop-in GGUF patch.
- QFP12 owns the 1306 terminal disposition. QFP09 owns cross-rank skew, QFP13 owns launch counts, RNX10 owns separately attributable shared-expert scalar-tail fusion, and QFP07 owns attention placement. Do not touch active QFP49/1360/1361 collective/stream experiments, PA49 pin reconciliation or their queued lanes.

### Bounded reopen / terminal criteria

No further hardware reservation for 1306. Reopen only if **new independent** production evidence shows (a) saved per-device VRAM changes a failed maximum-context admission, or (b) attributed shared-expert critical-path cost permits >=3% E2E theoretical improvement, and the active QFP49 work is inactive >=12 hours. Cheapest first gate: existing Meta debug/rocprof receipts prove unchanged AllReduce count, exact physical slices, no added transfer, graph/ubatch lifetime and rank arrival. If that fails, terminal: retain evaluated/default-off.

If it passes, compare same pin/GGUF/quant/KV/flags with 2×gfx1100 and mixed gfx1100+gfx1201 no-P2P, contexts 24K/80K/160K, MTP off/on and verify widths 2–5, multi-request same-process and graph replay. Match draft streams; require full-vocabulary logits/KLD, greedy/acceptance parity, zero-contributor correctness and real bytes/call accounting. Promotion only with four independent sessions, >=10 paired interleaved ABBA rounds, CI95-low >=3% E2E gain, no >1% control regression and no context-capacity loss. Otherwise close permanently without another implementation.

Source references: https://github.com/ggml-org/llama.cpp/blob/b11552/src/llama-model.cpp ; https://github.com/ggml-org/llama.cpp/blob/b11552/src/models/qwen4exp.cpp ; https://github.com/ggml-org/llama.cpp/blob/b11552/ggml/src/ggml-backend-meta.cpp ; https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py .

### Superseded historical plan



## Description

GPT review req_07434b19f4534cb3 found that under -sm tensor upstream gives ffn_{up,gate,down}_shexp split axes only for DSV4; on Qwen4Exp they fall through to MIRRORED, so every GPU stores and computes the whole Q8_0 shared expert in every layer. Patch 1306 (BIGCHERRY_SHEXP_SPLIT=1) applies the DSV4 layout (up/gate axis 1, down axis 0, anchored to ffn_down_shexp.weight): each GPU computes ~1/3, its partial output joins the routed experts' partial sum before the same AllReduce (no new collective), and shared-expert VRAM per GPU drops to ~1/3. Same review: 1305's shexp rules were a no-op for this reason (documented), and the once-only log flags in 1303/1305 were a data race (fixed with std::atomic).

## Steps

1. Screen on profile v2 (240K f16) at ~24K and ~80K: SHEXP_SPLIT=1 vs 0, same binary. 2. Check acceptance/greedy (summation order changes). 3. Measure freed VRAM per GPU and re-run the max-context search. 4. Re-measure AR arrival skew (ar-boundary.py).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ms/step ABBA; greedy/KLD vs mirrored; VRAM per GPU after load; activation via BIGCHERRY_PATCH_HIT patch=1306_shexp_split.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Related: RNX10 (shared-expert launch fusion), PRBE34/101 (shared expert on an aux stream - with the split, the per-GPU shared-expert work is smaller, changing that trade-off), QFP09 (per-class balance).

2026-10-04 screens (flashnext-v2-1306-d24k/-d80k, profile v2 240K f16, same binary): ~24K 47.2 vs 46.0/46.6 ms/step (+1.9%), acceptance 169/254 vs 175/240; ~80K 53.6 vs 53.3/53.4 (neutral), acceptance 166/266 vs 170/255. VRAM freed small: -185 MB per XTX, -135 MB R9700 (shared expert ~0.5 GB total). Activation proven (PATCH_HIT patch=1306_shexp_split). Verdict: neutral-to-negative; parked. The mirrored shared expert is small enough that the redundant compute is cheap, and splitting changes summation order (lower acceptance) and likely adds a per-layer combine around the sigmoid-gated shexp output. Not adopted.

## Change Log

- 2026-10-03T17:03:36.365732+00:00 (created-by): Created by agent
- 2026-10-03T17:21:20.454741+00:00 (updated-by): Updated: section:notes
- 2026-10-03T17:21:23.551814+00:00 (updated-by): Updated: priority='P3'
