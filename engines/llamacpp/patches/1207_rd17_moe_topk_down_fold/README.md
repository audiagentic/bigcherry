# 1207_rd17_moe_topk_down_fold: fold the MoE topk-weights MUL into the down projection (RD17)

Patch id: `1207_rd17_moe_topk_down_fold`. Original plan item `RD17` is
superseded by `PRBE14` (docs/planning/active/patching-rdna-boost-experiments/PRBE14.md,
capability-rebaseline-v3-2026-09) -- PRBE14 is the authoritative tracking
item now. No Experiment Contract is bound yet -- this patch has no
`experiment-contract` field in `patch.toml` and no bespoke correctness
producer under `validation/`, unlike RD04/RD08/RD58/RD73. It has no
`validation.toml` at all yet (`apply`/`build` run via the generic default
adapter; there is no `activation` check wired -- see below).

**Composition, not a soft warning**: `CONFLICTS = ("1205_rd12_paired_mmvq_dual_output",)`
in `patch.py` is real and enforced (`patchset.resolve_exact()` raises if
both are explicitly selected). RD17's `common.cuh`/`ggml-cuda.cu` anchors
occupy the exact same insertion slots RD12 (1205) uses -- verified failing
identically on two different pinned revisions (not a rebase regression).
Selecting RD17 alone (as below) is unaffected by this; it only matters if a
future recipe ever tries to combine 1205 and 1207, which none does today
and PRBE14 explicitly requires composition-conflict validation (PKC02)
before that could change.

## Scope

No `validation-architectures` declared in `patch.toml` yet. Backend: HIP.

## What it does

The MoE output is normally scaled per token by the topk softmax routing
weights in a separate broadcast `MUL` kernel run after the down projection.
This folds that scale into the down-projection matmul's own epilogue: the
mmvq kernel multiplies each result row by the destination channel's
(token's) weight directly, via a new `x_scale_channel_dst` fusion flag that
indexes the scale array by `channel_dst` instead of the existing
NVFP4-only per-expert `channel_x` indexing, with the NVFP4-only scale-
fusion assertion relaxed to allow this second case. Detection pattern:
`[MUL_MAT_ID, MUL]` where the `MUL`'s second operand is a contiguous
per-channel F32 vector matching the matmul's row count. Ported from
stew675-rdna-boosts fork commit `5e545b7da` (https://github.com/stew675/llama.cpp);
not merged into ggml-org/llama.cpp master. The fork's own claim (not yet
independently verified): removes 40 kernel launches per decode token on a
qwen3.5-MoE-class model, with bit-identical perplexity.

**Real evaluation of this patch requires a MoE model** -- the detection
pattern only fires on a `MUL_MAT_ID`-based MoE down projection, which a
dense model never emits. `config/models.toml`'s
`tierM-qwen35b-a3b-moe-mtp` (Qwen3.6-35B-A3B) is this project's only
currently-registered MoE-family model and is the right choice for
activation/performance evidence here.

## Real hardware evidence (2026-09-12, Brutus, dual gfx1100)

Added an activation-trace marker this session (a real authoring gap this
patch had, unlike RD12/1205) -- `GGML_LOG_WARN` (not `_INFO`, per this
project's own HI90/1231 real-hardware finding that INFO-level ggml logs
are filtered below `llama-server`'s `-lv 4`), once-per-process
`atomic_flag`-guarded, `BIGCHERRY_PATCH_TRACE`-gated, following RD12's
exact pattern.

Verified the fork's own claim against its real source: commit
`5e545b7da` on `stew675/llama.cpp` states only "Decode on qwen35moe drops
40 kernels per token. PPL is bit-identical." -- no end-to-end throughput
claim, and no PR/discussion exists beyond this commit message (checked via
`gh api`/web search). Our measurement below neither confirms nor
contradicts a claim the fork never made; the throughput finding is new.

Built baseline (no patches) and subject (this patch alone), isolated
`bigcherry-native` compositions, real Qwen3.6-35B-A3B-Revised-q8_0 (this
project's only registered MoE model), `-sm tensor`, dual XTX.

**Activation**: launched with `BIGCHERRY_PATCH_TRACE=1`, sent a real
`/completion` request -- `BIGCHERRY_PATCH_HIT patch=1207_rd17
path=moe_topk_down_fold` appeared exactly once in the live server log at
normal verbosity. The fusion genuinely fires during real MoE decode.

**Decode-path correctness (GPT-reviewed correction, req_7add510830424329):
an initial batched `llama-perplexity` PPL comparison across 12 wikitext-2
chunks came back bit-identical (7.0962 +/- 0.31960 both, every per-chunk
value matching) -- but this patch's own code comment documents that
multi-token/batched prefill (`mm_node->ne[2]==1` guard) correctly falls
through UNFUSED, so that comparison, while real, does not exercise the
fused code path at all and is not correctness evidence for it.** Ran the
correct test instead: identical deterministic `/completion` request
(temp=0, seed=42, 64 tokens) against baseline and subject servers, subject
run with the trace marker confirmed firing. Output was byte-identical
between baseline and subject -- this is real correctness evidence for the
actual fused decode path, not the unfused batched path.

**Performance**: a first-pass 5-rep `llama-bench` aggregate looked like a
noisy -2.4% (elevated variance in the subject run); per this project's own
established methodology (RD33/1241 found a similar noisy single-pass
signal that reversed under proper interleaving), did not stop there. Ran a
6-round INTERLEAVED paired A/B (alternating base/subject, 1 rep/round,
same binaries/flags):

| round | base tg128 | subj tg128 | delta |
|---|---:|---:|---:|
| 1 | 80.74 | 81.29 | +0.68% |
| 2 | 82.59 | 81.20 | -1.68% |
| 3 | 82.52 | 81.33 | -1.44% |
| 4 | 82.55 | 81.20 | -1.64% |
| 5 | 82.62 | 81.21 | -1.71% |
| 6 | 82.24 | 80.49 | -2.13% |

Paired mean delta -1.32%, SD 1.00% (n=6); 5/6 rounds negative (round 1 is
the outlier, consistent with this project's documented round-1 warm-up
noise pattern). Paired t-test 95% CI approximately [-2.38%, -0.26%].

## GPT-reviewed disposition (req_7add510830424329)

Performance leg closed negative: no further rounds needed to justify
not-promoting -- the burden is demonstrating a win, and this evidence
clearly fails that bar. Root-cause analysis is optional follow-up, not
required to close. Activation and the actual fused decode path's
correctness are now both real and proven. **Do not mark this
"correctness-proven" in the PRBE14 sense**, though -- PRBE14 additionally
requires fused-vs-unfused routing/scale-case coverage, false-positive
fallback behavior, graph-capture interaction, NVFP4-preservation, and the
1205/1207 composition-conflict disposition (PKC02), none of which this
session's evidence covers.

## Known limitations

Not `deferred-hardware` -- real hardware evidence now exists. Remaining
PRBE14-owned gates (listed above) are still open; this session's evidence
closes activation proof, fused-decode-path correctness, and the
performance question (negative), not PRBE14's full scope.

## Evidence

Real hardware evidence recorded above (2026-09-12). No formal
`artifacts/patch-validation/` bundle yet (no bound Experiment Contract);
ad-hoc but real measurements, as documented.
