---
id: QFP40
order: 40
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:56.573121+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Audits from the external list: lazy-mode embedding table, MTP decode fused copies, gathered QSA decode

## Description

Three cheap checks from the external optimisation list. Two already map directly to upstream/BigCherry mechanisms; the third is substantially covered by a validated BigCherry patch. Do not create a new package unless an audit proves a remaining mechanism-level gap.

External gains were measured on a different 4-GPU system and are hypotheses only.

## Steps

1. Residency audit: compare production `AUTO` with `--lazy-mode off`; capture loader log, host RSS/page residency, major faults/read I/O and decode timing.
2. Copy audit: run one decode kernel census with 1308 on; map the remaining top copy launches to graph nodes. If no dominant same-shape sequence remains, close.
3. QSA audit: at 24K and ~98K compare 1334 on/off, then only if long-context decode still scales materially compare 1295+1334 against 1334.
4. Record one finding per audit. Create no new mechanism from QFP40 unless a missing owner is proven.

## Detailed Solution & Technical Design

### A. Residency: configuration-only

No code change. Qualification states:

- A: current production `--lazy-mode auto`;
- B: `--lazy-mode off`.

`OFF` causes the 26.8 GiB PLE tensor to be loaded into ordinary host backing rather than retained as a lazy file mapping. The tensor remains CPU-side in our placement; no GPU topology or Meta split assumption is involved.

Do not conflate “host tensor” with “resident RAM”: a lazy file mapping is a host tensor but can fault pages from the model file. Evidence must therefore include loader line `lazy read enabled`/absence plus process I/O/page-fault counters during steady decode.

If `OFF` wins and host memory is acceptable, make it a deployment/profile setting. No `BIGCHERRY_*` flag is justified because upstream already owns the switch.

### B. Copy audit

Keep `BIGCHERRY_ROLLBACK_NO_CONT=1`. Census the post-1308 decode graph/kernel stream and group copies by source node/name/shape. Only optimize a sequence after proving:

- source/destination shapes and strides;
- copy is semantically exact;
- source lifetime permits direct/fused handling;
- the change does not cross Meta-device or recurrent-state ordering boundaries.

Do not invent a generic “fuse copies” pass. If the remaining copies are the same rollback family, add an edit to patch 1308 behind its existing flag. gfx1030/gfx1100/gfx1201 use the same graph rule; any backend-specific kernel change would require a separate item.

### C. QSA decode

1334 remains production baseline. Its sparse path operates below Meta splitting: each attention-holding device compacts/gathers its own local K/V representation according to the graph's mask. It must not assume which card has attention heads.

1295's explicit gather remains an experimental fallback for small batches only. If re-tested, retain its existing dispatch:

- Qwen4Exp QSA only;
- `n_tokens <= 8`;
- contiguous K/V view;
- cache cells >= `BIGCHERRY_QSA_GATHER_MIN`;
- normal masked/sparse path otherwise.

No new flag or new package is needed. Do not combine 1295 and 1334 by changing arithmetic/fusion until ABBA proves 1334 alone leaves a decode bottleneck.

Architecture note: 1334's validated sparse WMMA path targets gfx1100/gfx1201. gfx1030 does not need a new QSA path for our topology because the sidecar MTP GPU is not an attention rank in the target Meta split; nevertheless no code should hard-code that topology. Unsupported architectures fall through existing FA dispatch.

## Code Samples & Guidance

### A

No patch anchors. Existing CLI anchor is `common/arg.cpp` option `{"-lzm", "--lazy-mode"}`. Use `LLAMA_ARG_LAZY_MODE=off` or the CLI option for ABBA.

### B

Existing owner/anchor in 1308:

`ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst)`

under `[TAG_RECURRENT_ROLLBACK_SPLITS]`.

Flag remains `BIGCHERRY_ROLLBACK_NO_CONT`. Any further edit to that exact mechanism belongs in patch 1308.

### C

Existing owners:

- 1295: `src/models/qwen4exp.cpp::graph::build_attn_qsa`, `bc_qsa_gather_enabled()`;
- 1334: `ggml/src/ggml-cuda/fattn*.cu/.cuh` sparse selection/compaction sites.

Flags remain `BIGCHERRY_QSA_GATHER`, `BIGCHERRY_QSA_GATHER_MIN`, and `BIGCHERRY_FA_SPARSE`.

Use existing `BIGCHERRY_PATCH_HIT` markers. QFP40 should not add another diagnostic patch merely to prove activation.

## Files

Planning outcome should normally update **no patch files**.

- A: deployment/profile configuration only; no patch package.
- B: if the census finds another rollback-copy improvement, update `patches/1308_qwen4exp_rollback_copy_no_cont` and its existing test/docs.
- C: re-use `patches/1295_qsa_gather_decode` and `patches/1334_hip_sparse_flash_attn`; do not create a third QSA package for the same mechanism.
- ABBA: `tools/lab/flash-next/queue-env-ab.sh`.

If a census reveals a genuinely different copy mechanism, open a new plan item before code.

## Validation

### A. Lazy-mode

Fully separated ABBA, production model/config:

A: `--lazy-mode auto`
B: `--lazy-mode off`

Capture loader evidence, host RSS, major faults/read bytes during warm and steady decode, 8K/24K/~98K decode t/s, and startup time. Arithmetic is unchanged: require greedy identity.

### B. Copies

- Existing 1308 mechanics test/patch-lint remain the base.
- Kernel census with `BIGCHERRY_ROLLBACK_NO_CONT=1`.
- If patch 1308 is extended: exact activation marker/census delta, patch-lint, and separated ABBA.
- Direct-copy/fusion changes must be exact; require greedy identity.

### C. QSA

Baseline production 1334. At 24K and ~98K:

A: 1334 enabled, 1295 disabled.
B only if needed: 1334 + 1295.

Capture QSA attention ms/step, total decode t/s, VRAM headroom and activation markers. Because sparse/gathered FA changes reduction order, use the already established explicit equivalence standard: no worse than validated sparse-FA fidelity against the CPU-f32 reference; do not require bit identity between dense/sparse/gathered paths.

No multi-session contract campaign for any audit.

## Effort & Risk

Effort: S for all three audits.

Risk:
- A: low correctness, medium host-memory/startup cost.
- B: low if no code is added; medium if copy lifetimes/strides are changed.
- C: medium due memory headroom and floating-point reduction-order differences.

Expected gain on our topology:
- A: potentially medium decode gain if current AUTO causes repeated PLE page faults/I/O; zero if pages are already resident.
- B: most of the known gain is already captured by 1308; likely low residual.
- C: potentially medium at ~98K only if 1334 leaves decode QSA scaling; otherwise zero and close.

## Standards

- Prefer existing upstream configuration over a duplicate BigCherry flag.
- Existing mechanism owner is extended in-place; no duplicate patch packages.
- No fixed GPU count, rank or attention placement.
- No q4 KV.
- No compatibility shim.
- External performance numbers are hypotheses.
- Audit first; code only after a measured mechanism gap.

## Acceptance Criteria

- A: determine from code + runtime evidence whether the 26.8 GiB PLE is lazy/file-backed in production; benchmark `--lazy-mode off`; close with configuration recommendation.
- B: identify the post-1308 copy census; either prove no dominant remaining same-shape copy or name the exact source owner.
- C: determine whether 1334 removes long-context QSA decode scaling; close 1295 as superseded if it does.
- Any ABBA uses complete process separation and activation evidence.
- Exact mechanisms require greedy identity; sparse-attention comparisons use the explicit validated fidelity standard.
- No new patch package is created without a demonstrated missing mechanism.

## Notes

Ordering: fourth overall after QFP32, QFP31, QFP33. These are cheap audits and may close work immediately. Within QFP40: lazy-mode first (zero code), copy census second, QSA comparison third.

2026-10-07 finding A (per-layer embedding table resident, --lazy-mode off): REJECTED on Brutus. ABBA lazyoff-ab on build b-metamem-rr98, A = default lazy mapping, B = LLAMA_ARG_LAZY_MODE=off: prefill 8K 904/1077 vs 434/629 t/s, 24K 1065/1072 vs one failed load/823; decode 82.5/84.6 vs 77.8/79.9 and 74.9/75.3 vs 70.2; greedy text identical where it ran. Host has 91 GB RAM for an 87 GB model file, so a resident 26.8 GB copy competes with the page cache (not confirmed from memory counters). The default stays. Parts B (MTP decode copies) and C (gathered QSA decode, 1295) are still open.

## What we already have

### A. Per-layer embedding table residency

This is already an upstream b11402 feature/configuration, not missing code.

- `src/models/qwen4exp.cpp::llama_model_qwen4exp::load_arch_tensors` creates `LLM_TENSOR_PER_LAYER_TOKEN_EMBD` with `TENSOR_READ_LAZY`.
- `src/llama-model-loader.cpp::llama_model_loader::lazy_read::add`
  - `LLAMA_LAZY_MODE_OFF`: keep the tensor non-lazy;
  - `AUTO`: lazily maps marked tensors larger than 4 GiB;
  - `ON`: lazily maps all marked tensors.
- The production PLE table is ~26.8 GiB, so with the default `AUTO` it qualifies for lazy mapping whenever mmap is supported.
- `src/llama-model-loader.h::lazy_read::buft` forces lazy tensors to the CPU backend; comments explicitly state lazy tensors are gathered on host.
- `src/llama-model-loader.cpp::init_mappings` maps lazy ranges even if ordinary mmap load mode is not selected.
- `common/arg.cpp` already exposes `-lzm/--lazy-mode {on,auto,off}` and `LLAMA_ARG_LAZY_MODE`.

Finding: **fully covered by upstream configuration**. Do not patch. The production question is whether we are running `AUTO` (therefore file-backed/on-demand) or `OFF` (fully loaded host tensor). Measure `--lazy-mode off` directly and close this sub-item.

### B. MTP/recurrent decode copies

Substantially covered by validated patch 1308.

- `patches/1308_qwen4exp_rollback_copy_no_cont` changes the Qwen4Exp recurrent rollback snapshot from `ggml_cpy(ggml_cont(tail), dst)` to direct `ggml_cpy(tail, dst)`.
- Owner: `src/models/qwen4exp.cpp`, tag `[TAG_RECURRENT_ROLLBACK_SPLITS]`.
- It removes one copy kernel per rollback slot; its census attributed ~89/108 runtime copy launches per generated token to this sequence.
- Hardware validation already measured ~3-3.6% decode improvement and greedy identity.
- Flag: `BIGCHERRY_ROLLBACK_NO_CONT`, currently on by default.

Finding: the external “leaner conv-state rollback” is **already covered**. Only the broader “fused same-shape copies” wording needs a census check. If remaining dominant same-shape copies belong to the same rollback sequence, improve **inside patch 1308**; do not create a new patch package. If remaining copies come from a different owner, create a separate plan item rather than broadening QFP40.

### C. Gathered QSA decode

We already have two implementations of the underlying sparse-attention idea.

- `patches/1295_qsa_gather_decode` (`BIGCHERRY_QSA_GATHER=1`, threshold `BIGCHERRY_QSA_GATHER_MIN`)
  - for batches <=8, explicitly gathers selected QSA K/V cells and attends over the gathered compact cache;
  - evaluated at +7% decode around 80K and larger gain at 160K;
  - not bit-identical to dense masked FA, though it measured closer to CPU-f32 than the dense HIP path;
  - carries extra gathered tensors/padding and showed headroom pressure at the largest tier.
- `patches/1334_hip_sparse_flash_attn` (`BIGCHERRY_FA_SPARSE`)
  - validated and promoted;
  - enables upstream sparse FA on HIP RDNA WMMA;
  - compacts the QSA mask to sparse cell indices and gathers only referenced K/V cells inside the attention kernel;
  - validated on gfx1100/gfx1201 and gave large long-prefill wins with accepted fidelity.
- b11402 Qwen4Exp supplies the sparse QSA mask/cell bound; 1334 is therefore the lower-overhead production mechanism for the same “do not scan masked KV” objective.

Finding: **do not promote 1295 as-is**. First measure whether 1334 already removes long-context QSA decode scaling at ~98K. If yes, close the gathered-decode sub-item as superseded. If no, compare 1295+1334 vs 1334 alone at the affected tier; only reopen 1295 if it adds a repeatable decode gain without unacceptable memory/fidelity cost.

## Change Log

- 2026-10-07T00:39:56.573121+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 and existing 1295/1308/1334; identified lazy-mode as upstream configuration, rollback copy as already validated, and 1334 as the production QSA baseline.
- 2026-10-07T06:35:50.784320+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261007_133405_flash-next-decode-at-long-cont_5826
- 2026-10-07T13:34:09.463673+00:00 (updated-by): Updated: section:ledger-events
