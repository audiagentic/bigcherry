---
id: QFP43
order: 43
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-08T15:51:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# DFlash/DSpark drafter path: selector, prompt catch-up, adaptive budget and Flash-Next conversion

## Description

DFlash2/DSpark are already close to MTP decode throughput on the Qwen3.8-27B hardware profile, but they pay a
large prompt tax and fixed long blocks waste proposals. Establish the b11474 path before changing it, then make the
target-feature handoff and draft budget first-class mechanisms shared with the MTP work rather than parallel
implementations.

Measured on 2026-10-05, Qwen3.8-27B Q8_0, dual 7900 XTX target, drafter on 6900 XT, 1665-token prompt, 128 greedy
tokens:

| arm | decode | acceptance | prompt |
|---|---:|---:|---:|
| no draft | 38.5 t/s | - | - |
| built-in MTP depth 5 | 78.4-78.7 t/s | 58% | 1315 t/s |
| DFlash2 Q8_0 block 7 | 74.0 t/s | 43.6% | 1157 t/s |
| DFlash2 Q4_K_M block 4 | 74.2 t/s | 59.3% | 1168 t/s |
| DFlash with 3-card target | - | - | 967 t/s |

A separate drafter-file comparison is pending and is evidence, not a prerequisite for this source/design item.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- QFP43 records source truth separately from inference.
- Exact current lab GGUF metadata proves whether DFlash2 selector/confidence features are active.
- Prompt-cost path is instrumented sufficiently to account for target-to-host, host wait/refusion and draft catch-up
  time.
- Deferred DFlash catch-up reuses/generalises 1348 lifecycle rather than introducing a second state machine.
- One speculative budget controller serves MTP, DFlash2 and DSpark.
- Tensor-split shared-head behaviour is explained by owned/shared tensor placement and has a clean one-time mirror
  design if required.
- Flash-Next conversion work extends the pinned converter rather than duplicating architecture support.
- Every later runtime patch is default off, has an explicit off switch and `BIGCHERRY_PATCH_HIT`.
- Every `patch.py` generated C/C++ string containing escapes is a raw Python string.

## Notes



## Hardware findings 2026-10-08 (Brutus), which reorder this plan

1. **Loading.** At b11474 a DFlash2 / DSpark draft aborts at context creation on a tensor-split target (`pre-allocated tensor (output.weight) in a buffer (Meta()) that cannot run the operation (NONE)`), on native llama.cpp too: upstream issue 27833, open. Our patch 1286_draft_local_shared_tensors fixes it (untested state, experiments `retest-1286` / `draft-local-shared`, composes on today's production set). A layer-split target with the draft on a different card than output.weight fails the same way and 1286 does not cover it.
2. **Acceptance collapsed.** Production set + 1286, Qwen3.8-27B Q8_0 dual-XTX tensor split, 1665-token prompt, 5 x 128 tokens: no draft 37.9 t/s; MTP depth 5 79.4 t/s at 58.3%, depth 4 80.2 t/s at 67.3%; DFlash2 Q8_0 block 4 / 7 on the 6900 XT 23.8 t/s at 1.7% / 20.7 at 0.7%; Q4_K_M 24.6 at 1.7% / 20.7 at 1.1%; BF16 20.7 at 1.7% / 16.3 at 0.6%; magnitudedev Q8_0 23.9 at 1.7% / 20.6 at 0.7%; DSpark Q8_0 block 6 21.9 at 2.0% (ours) and 23.3 at 2.5% (magnitudedev). All lossless. At b11402 the same files gave 44-59% (74 t/s). Every file behaves the same, so it is not a file problem: the draft receives wrong inputs or its outputs are mapped wrongly. Isolation runs queued (native vs ours on a single-device target; one production switch off per arm). **Nothing else in this plan matters until this is explained.**
3. **Files.** Our DFlash2 Q8_0 / Q4_K_M are byte-identical to the publisher's current GGUFs. BF16 and the 2026-10-06 magnitudedev re-publications are on the lab disk (hash-checked). Our DSpark file differs from the one public GGUF and is of unknown origin.
4. **Flash-Next drafter.** PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash converts with the pinned converter unchanged (`--target-model-dir` = Qwen/Qwen3.8-Flash-Next config + tokenizer): `/mnt/data/llm-models/qwen3.8-flash-next/gguf/dflash/Qwen3.8-Flash-Next-DSpark-PixelML-BF16.gguf`, 58 tensors, block 7, no confidence head. With 1286 it loads, then asserts `src/llama-context.cpp:2499: GGML_ASSERT(row_floats == model.hparams.n_embd)` in `extract_layer_inputs`: the Qwen4Exp layer-input tensor is the raw hyper-connection stream, 4 x 2560 = 10240 floats per token, and the extractor requires n_embd (2560). The checkpoint's card states the tap precisely: the five taps `[3, 15, 23, 35, 43]` are the **HC-contracted native-width (2560) residual from each tapped layer's own GatedResidual mix**, taken in vLLM at aux boundary ids tap+1 = `[4, 16, 24, 36, 44]`, i.e. `layers[i].attn_hyper_connection.mix / combine_and_mix(...)[1]`; concatenated `[T, 12800]` into `fc`. So the needed patch is a Qwen4Exp feature tap that hands the draft the contracted 2560-wide tensor at those boundaries instead of the raw stream (the hc-pre contraction already exists in the graph; 1311 / 1344 touch it). Lab launcher support is merged (`SPEC_TYPE=draft-dspark SPEC_PMIN=0 SPEC_N=7 DRAFT=<gguf>`).
5. **Cross-model rule learnt today.** Look-ahead (1321/1322) costs 24-30% decode where the draft shares the target's cards (27B built-in MTP); any drafter mechanism needs a second-model run before it is more than a profile switch.

## Source baseline

Pin: llama.cpp b11474.

Verified source paths:

- `common/speculative.cpp`: DFlash/DSpark construction, target feature extraction, DFlash2 selector walk, DSpark
  confidence thresholding and draft-size clamp.
- `src/models/dflash.cpp`: DFlash encoder/decoder graphs, DFlash2 dynamic convolutions/selector, DSpark
  Markov/confidence heads, target `output.weight` sharing.
- `src/llama-context.cpp`: target layer-input extraction and host output buffers; public embedding getters synchronize.
- `src/llama-model.cpp`: target output-layer placement under layer/tensor split.
- `conversion/qwen.py` and `conversion/base.py`: DFlash/DSpark architecture registration, target-layer metadata and
  NVFP4 conversion support.
- `patches/1348_mtp_deferred_catchup`: promoted two-slot deferred MTP catch-up and cancellation/poison semantics.
- `patches/1255_nro06_adaptive_mtp_depth` and `patches/1268_prbe52_adaptive_mtp_wiring`: adaptive draft-depth
  controller/wiring; PR #43 restores the b11474 experiment.

External artifact checks:

- `magnitudedev/Qwen3.8-27B-DSpark-GGUF`: Q8_0, architecture `dflash`, converted from
  `RadixArk/Qwen3.8-27B-DSpark` with llama.cpp revision `8e7f22b67ef4667b4ddd50230771287f328cfb3f`.
- The requested `magnitudedev/Qwen3.8-27B-DFlash2-GGUF` endpoint was not retrievable from the current HF index.
  Do not infer its embedded GGUF metadata from the repository name. The source/mirror DFlash2 GGUFs from
  incoai/z-lab are retrievable and establish the intended DFlash2 path, but the actual Magnitude file remains a
  mandatory metadata-dump preflight.
- `PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash`: `Qwen3DSparkModel`, block size 7, five BF16 draft layers,
  target taps `[3,15,23,35,43]`, no confidence head/Markov rank, stripped token embedding and LM head.
- The requested `tcclaviger/Qwen3.8-Flash-Next-Dflash2` endpoint was not retrievable from the current HF index.
  Treat its config/tensor schema as unknown until downloaded/dumped; do not code a name-map from assumptions.

## Candidate 1 - DFlash2 selector path and metadata

### Verified

At b11474:

1. `common_speculative_impl_draft_dflash` reads `llama_model_dflash_selector_top_k(model_dft)`.
2. `selector_top_k > 0` sets `is_dflash2=true`; backend sampling is disabled for this path.
3. `src/models/dflash.cpp` detects `selector_hidden.weight`. If present, block/conv/rank/top-k metadata must all be
   valid or model load throws. It loads:
   - `selector_predecessor.weight`;
   - `selector_successor.weight`;
   - `selector_hidden.weight`;
   - per-layer DFlash2 attention/FFN dynamic-convolution tensors.
4. The draft graph computes top-k candidate ids, unary logits, selector hidden gates and predecessor/successor
   transition scores. It packs candidate ids plus the `top_k^2` transition lattice into the NextN output.
5. After draft decode, `common/speculative.cpp` calls `llama_get_embeddings_nextn(ctx_dft)` and CPU-walks one path
   through the packed lattice.
6. Therefore selector matmuls/convolutions execute with the draft graph on the draft backend/device; only extraction
   and the final path walk are host-side.
7. The pinned converter writes DFlash2 metadata from `dflash_config`: `block_size`, `conv_kernel_size`,
   `conv_group_size`, `selector_rank`, `selector_top_k`, and target layers (+1 from source model layer ids).
   The intended Qwen3.8 DFlash2 config is block 8, top-k 16, rank 256, kernel 2, target ids
   `[5,19,33,47,61]` -> GGUF extraction ids `[6,20,34,48,62]`.

The selector is silently off only when the GGUF lacks the selector tensors and resolves
`dflash.selector_top_k == 0`. A file containing `selector_hidden.weight` but missing/invalid selector metadata
should fail load, not silently take plain DFlash.

### Inference / test required

The measured acceptance drop from about 57-59% at block 4 to 43.6% at block 7 is not evidence that the selector turns
off: there is no block-length branch that disables it. Later positions naturally accumulate proposal/conditioning
error, and their marginal accepted length can be smaller than their added draft cost. PixelML's independent serving
data shows the same diminishing-return shape: accepted length 2.756/2.883/2.984 at blocks 4/5/7 while engine-pass
cost rises 52.93/55.00/58.88 ms.

Before selector changes, dump the exact lab GGUFs and record:

- `dflash.block_size`, `dflash.selector_top_k`, `dflash.selector_rank`;
- conv kernel/group;
- target layer ids;
- selector predecessor/successor/hidden tensor presence;
- sample-from-anchor, causal/SWA metadata;
- drafter device placement.

Add env-gated diagnostics in the later budget slice: one `PATCH_HIT` init line plus per-position accepted/confidence
histograms. If the selector is already active, do not build a replacement selector.

## Candidate 2 - target-feature prompt cost

### Verified

DFlash prompt catch-up currently performs, for every target prompt chunk:

1. target decode requests each configured target-layer input;
2. `llama_context::extract_layer_inputs()` queues `ggml_backend_tensor_get_async` for every requested layer into
   host output buffers;
3. `common/speculative.cpp` calls `llama_get_embeddings_layer_inp(ctx_tgt, layer)`; the public getter calls
   `ctx_tgt->synchronize()`;
4. target-layer rows are CPU-copied again into one interleaved `features_buf`;
5. that host embedding batch is submitted to the DFlash encoder/draft context to catch its KV state up.

For five taps this is target-device -> host staging, a target completion barrier, host refusion, then host -> draft
device submission per prompt chunk. This directly explains why DFlash/DSpark can cut target prompt throughput and why
the penalty grows when target topology adds another GPU.

### Design

Generalise 1348 instead of duplicating it.

Keep 1348's existing two-slot ownership, final-flush, cancellation, cache/context-shift poison and failure semantics,
but make the payload a target-feature snapshot rather than an MTP-only NextN row:

- payload descriptor = source context + requested layer ids + row width + token range;
- MTP adapter = one NextN feature stream;
- DFlash adapter = N target-layer input streams fused into one per-token feature slab;
- two reusable pinned-host staging slots, no hot-path allocation;
- enqueue extraction behind target graph completion without a host-wide getter synchronization;
- prepare/submit the next target chunk before waiting for the previous feature slot;
- then inject/catch up the draft while the next target chunk computes.

QFP42 owns the narrow asynchronous target-output staging primitive. QFP43 should consume/generalise that primitive,
not create a second scheduler API. If QFP43 lands first, implement only the common snapshot/cancellation abstraction
and leave event-backed staging to QFP42.

First performance slice: `feat/qfp43-dflash-feature-catchup`, default off, explicit off switch, activation
`BIGCHERRY_PATCH_HIT patch=<id> mechanism=dflash-deferred-features`. All generated C/C++ replacement text in
`patch.py` must use raw Python strings.

## Candidate 3 - confidence-driven draft budget

### Verified

- DSpark can expose a confidence head through `dflash.has_confidence_head`; `--spec-draft-p-min` already stops a
  block early using that confidence.
- DFlash2 already derives selector path probabilities/scores and applies `p_min` while walking the lattice.
- `--spec-draft-n-max` remains a fixed hard cap.
- 1255/1268 already implement an acceptance-driven MTP depth controller; PR #43 restores its b11474 wiring and fails
  closed for the known adaptive-depth/look-ahead conflict.

### Design

Do not add a DFlash-only adaptive controller. Refactor the 1255 controller into a generic speculative budget policy
after PR #43 is resolved:

- hard bound = user `--spec-draft-n-max`;
- floor = existing/user minimum;
- observation = proposed count + accepted count;
- optional current-block confidence vector:
  - DSpark confidence head;
  - DFlash2 selector path probability;
  - MTP has no confidence input initially and uses acceptance history only;
- `p_min` remains the intra-block immediate stop;
- controller selects the next verification step's cap using the same evidence and hysteresis.

The controller must not increase beyond trained block size, must reset on sequence reset/context restoration and must
remain disabled by default.

Second performance slice: `feat/qfp43-spec-budget-controller`, with explicit enable/off switch and
`BIGCHERRY_PATCH_HIT patch=<id> mechanism=spec-budget drafter=<mtp|dflash2|dspark>`.

Primary hardware comparison for DFlash2: fixed n=4, fixed n=7 and adaptive hard-cap 7. Record accepted length,
per-position acceptance, target verify time, draft pass time and net decode t/s.

## Candidate 4 - tensor-split target shared head

### Verified

At b11474 DFlash token embedding and `output.weight` are optional draft tensors. If absent, the DFlash graph obtains
the target tensor pointer through `ctx_other`. `llama_context` requires `ctx_other` when a DFlash draft lacks
either shared tensor. The target's output tensor is placed using the target model's output-layer buffer/device policy;
there is no load-time copy into the draft device.

Therefore a stripped, single-device draft can directly reference a target tensor whose backing buffer belongs to a
tensor-split/Meta topology. That is a real cross-context/backend dependency. A tensor-split probe succeeding does not
prove the dependency is absent: the draft may contain its own head, the output may have landed on an accessible
backend, or that exact backend layout may support the read.

### Design

Add an immutable shared-target-tensor binding helper:

- if draft owns `tok_embd`/`output`, use them unchanged;
- if a required target tensor is already directly readable by the draft backend, use it;
- otherwise allocate a same-type mirror on the draft device once at context/model setup and copy once;
- bind the draft graph to the mirror; never copy the head per token/block;
- include both token embedding and output projection in the helper, not an output-only special case;
- log mirrored tensor, source/destination backend and bytes.

Default off for the experimental patch; prove stripped-head + tensor-split failure/control before promotion.

## Candidate 5 - Flash-Next drafter conversion

### Verified

The pinned converter already understands the relevant model classes:

- `DFlashDraftModel`, `DFlash2DraftModel` -> `DFlashModel`;
- `Qwen3DSparkModel`, `DSparkDraftModel`, `DSparkSpeculator`, etc. -> `DSparkModel`.

It already writes DFlash block/conv/selector/target-layer metadata and maps DFlash2 candidate-selector tensors.
`conversion/base.py` also detects/re-packs supported ModelOpt/compressed-tensors NVFP4 into GGUF NVFP4.

PixelML's Flash-Next drafter itself is BF16, despite the target name containing NVFP4. It strips
`embed_tokens.weight` and `lm_head.weight`; no NVFP4 dequantisation is required for those draft weights.
A BF16/F16 GGUF conversion followed by normal `llama-quantize ... Q8_0` is the preferred path.

### Design / feasibility

A separate full converter is not justified. Build a thin follow-up tool slice only after the checkpoint preflight:

`tools/convert_dflash_drafter.py`

Responsibilities:

1. validate accepted architecture/model class and target pairing;
2. validate required target taps, block size and DFlash2 selector metadata/tensors;
3. call the pinned upstream converter with `--target-model-dir`;
4. emit BF16/F16 and optionally run Q8_0 quantisation;
5. dump/assert GGUF metadata before declaring success;
6. preserve stripped embedding/head semantics so runtime sharing/mirroring is tested explicitly.

Tiny synthetic test: generate a minimal checkpoint/config for `Qwen3DSparkModel` and `DFlash2DraftModel`, convert it,
then assert target-layer (+1) metadata, selector keys/tensor names, block size, confidence-head flag and stripped-head
behaviour.

If the tcclaviger checkpoint proves to be true NVFP4 draft weights rather than BF16, inspect its quantisation schema
before coding. b11474 can repack supported NVFP4 to GGUF NVFP4; it does not needlessly dequant a pure NVFP4 model to
F16/Q8_0. Add an explicit dequant path only if its schema cannot be consumed on ROCm or Q8_0 is required for the
drafter; otherwise retain NVFP4.

The unresolved risk for Flash-Next is not class-name dispatch but semantic compatibility of its five HyperConnection
taps with llama.cpp's Qwen4Exp target layer-input boundary. Validate the expected 2560-wide contracted residuals
before trusting a converted file.

## Ranking

| rank | candidate | expected value on current hardware | risk |
|---:|---|---|---|
| 1 | deferred/general target-feature catch-up | high prompt/TTFT gain; directly targets measured 12-27% loss | medium |
| 2 | unified confidence/acceptance draft budget | medium-high decode gain; removes long-block waste | medium |
| 3 | selector activation + metadata diagnostics | potentially large if artifact wrong; otherwise diagnostic only | low |
| 4 | shared target tensor mirror | compatibility/topology gain; little direct speed when current layout works | medium |
| 5 | Flash-Next conversion wrapper | strategically high but checkpoint/tap validation still uncertain | medium-high |

## First two patch slices

### A. `feat/qfp43-dflash-feature-catchup`

- generalise 1348's deferred snapshot lifecycle for DFlash/DSpark target features;
- consume QFP42 async staging API when available;
- no target semantics/KV/logit change;
- default off; explicit off path returns to exact b11474/promoted behaviour;
- `PATCH_HIT`;
- raw Python strings for every generated C/C++ fragment containing escapes;
- mechanics/idempotence/lint plus hardware ABBA.

Hardware: current 1665-token profile plus 8K/24K prompt points, 2-card and 3-card target layout. Record prompt t/s,
TTFT, feature-stage wait/copy, draft catch-up, decode t/s, acceptance and target greedy identity.

### B. `feat/qfp43-spec-budget-controller`

- refactor/reuse 1255 controller after PR #43 disposition;
- DFlash2 selector and DSpark confidence feed one generic budget policy;
- `--spec-draft-n-max` remains hard cap;
- default off + explicit off path;
- init/activation marker states drafter kind and available confidence source;
- raw Python strings for generated C/C++.

Hardware: fixed 4 vs fixed 7 vs adaptive cap 7, same prompts/seeds/layout; record per-position acceptance/confidence,
accepted length, draft/verify time and output t/s.

## Validation order

1. Dump the exact lab GGUF metadata/tensors before patching. Fail the investigation if the intended selector or target
   taps are missing instead of compensating in runtime.
2. Add selector/device/feature-copy timing diagnostics and reproduce the current Q8 block-7 and Q4 block-4 arms.
3. Implement feature catch-up slice and qualify prompt throughput/identity.
4. Resolve/rebase on PR #43, then implement the shared draft-budget slice.
5. Test stripped-head vs self-contained draft across layer-split and tensor-split targets; only then build the mirror
   slice if it is needed.
6. Inspect/download the exact Flash-Next checkpoints; build converter wrapper/tests only for schemas verified from
   their files.

## References

- llama.cpp b11474.
- `patches/1348_mtp_deferred_catchup`.
- `patches/1255_nro06_adaptive_mtp_depth`.
- `patches/1268_prbe52_adaptive_mtp_wiring`, PR #43.
- https://huggingface.co/magnitudedev/Qwen3.8-27B-DSpark-GGUF
- https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF
- https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash
- requested but not retrievable during this review:
  https://huggingface.co/magnitudedev/Qwen3.8-27B-DFlash2-GGUF and
  https://huggingface.co/tcclaviger/Qwen3.8-Flash-Next-Dflash2

## Change Log

- 2026-10-08T15:51:00+11:00 (agent): Created from b11474 source audit and current drafter artifact review.

## Ledger-events

- chg_20261008_071201_several-experimental-patches-r_2753
- 2026-10-08T07:12:22.172301+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-08T08:45:41.372284+00:00 (updated-by): Updated: section:notes
