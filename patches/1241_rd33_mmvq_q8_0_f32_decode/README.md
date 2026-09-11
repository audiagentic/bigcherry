# 1241 (RD33 / AMD-MMV-001): dense Q8_0 decode without activation quantization

## Scope

Adds a defaulted `f32_act` template parameter to `mul_mat_vec_q` and a new
`vec_dot_q8_0_f32` device helper that dequantizes the Q8_0 weight block
directly and dot-products it against the original F32 activation (F32
accumulation), skipping the Q8_1 activation-quantization stage entirely.
Gated narrowly: dense (non-MoE), Q8_0, `ncols_dst == 1`, gfx1100 exactly, and
only when nothing has been forced onto another candidate.

## Why

`ggml_cuda_mul_mat_vec_q()` unconditionally calls `quantize_row_q8_1_cuda(...)`
to convert the F32 activation to `block_q8_1` as a separate GPU kernel launch
before every matvec, with no bypass for `ne11==1` (single-column decode) --
pure overhead when the matvec itself is bandwidth-bound and the weight is the
only operand whose quantization matters.

## Upstream / provenance

Local design (bigcherry-original, not a verbatim port -- no matching upstream
implementation was ever found; the backlog's AMD-MMV-001 entry cites only a
discussion, not a specific commit). Designed and verified against the real
pinned source via `dev-gpt-agent` review (session `ses_fec603fc33ee4089`).
Deliberately does not carry a fabricated external-source citation.

## Real hardware evidence (2026-08-27, dual gfx1100)

This is the most thoroughly investigated patch in the RD-series to date --
summarized here from its plan item (`RD33`)'s extensive notes:

1. **Build**: clean compile, isolated `rd33-only` experiment lane
   (`config/recipes.toml`).
2. **Correctness**: `test-backend-ops` MUL_MAT, `type_a=q8_0`, ROCm0
   (gfx1100): 48/48 passed against CPU-reference tolerance (not
   bit-identical -- accumulation order genuinely changes from int32 dp4a to
   F32 fmaf, as expected). Includes explicit `n=1` shapes.
3. **Path-execution confirmed**: temporary instrumented debug build showed
   16/48 relevant Q8_0 MUL_MAT cases actually took the new `f32_act` path
   (not a silent no-op), all 16 correct; the other 32 correctly stayed on
   the old path.
4. **Performance, round 1** (dense Qwen3.8-27B-Q8_0 MTP, dual gfx1100
   `-sm tensor`, seeded/deterministic): nominal deltas all negative
   (tg128 -3.7%, tg512 -1.1%, pp4096 -1.5%), but with 5-6x this hardware's
   established noise floor -- flagged as unexplained, not accepted at face
   value.
5. **Root-cause diagnostic** (rocprofv3 kernel-trace/stats): VGPR=24,
   SGPR=128, Scratch=0 -- **identical** on baseline and RD33 builds across
   all 14 real kernel launches. Falsified the register-pressure/occupancy
   hypothesis. RD33's total kernel duration was marginally *lower* than
   baseline in this isolated microbenchmark -- opposite direction from the
   noisy round-1 server result.
6. **Performance, round 2**: repeated the full server-level A/B with tight
   noise on both sides this time -- RD33 essentially matched baseline
   (tg128 +0.3%, tg512 +0.6%, both trivial/within noise). Round 1's
   negative signal was ambient system noise affecting both arms equally,
   not anything RD33 caused.
7. **Follow-up finding (2026-09-04)**: a real rocprofv3 decode capture on
   the actual production configuration (632,620 dispatches, MTP
   `spec_draft_n_max=5`) found the dominant MMVQ shape on that workload is
   `ncols_dst == 6` (the MTP speculative-verify batch width), not 1 --
   summing every `ncols==1` shape gives only ~7% of dispatches. **RD33's
   `ncols_dst == 1` eligibility gate essentially never fired on the
   production workload's hot path**, fully explaining the null result
   without needing "the quantize stage is cheap" as an explanation.

## Final verdict (converged, 2026-09-04)

**Correct, but no measurable performance benefit on the workload it was
gated for.** Not promoted -- no benefit to justify carrying the extra code
path at its current `ncols_dst == 1` gate. A genuinely useful negative
result: this specific "remove a pipeline stage" idea did not move the
needle at the shape it was tested against, and the reason (wrong dominant
shape) is understood, not hand-waved.

## Known limitations / concrete reopening path

- A specific, cheap-to-test follow-up is on record but not yet done:
  widen the eligibility gate from `ncols_dst == 1` to cover
  `ncols_dst == 6` (the actual MTP-width hot path) or `1..8` generally --
  the `vec_dot_q8_0_f32` helper and templated `f32_act` plumbing already
  exist and are correctness-validated for the single-column case; this
  would be a gate change plus new template instantiations, not a redesign.
  Its own arithmetic-equivalence-for-ncols>1 assumption needs re-checking
  against `test-backend-ops` before any new timing work.
- No `validation.toml` adapter exists (`origin = "local"`, no bound
  Experiment Contract).
- `state` intentionally stays `"untested"` per the investigation's own
  disposition -- not "validated" (it's correctness-proven but
  performance-null at its current gate) and not "rejected" (the mechanism
  isn't wrong, just gated on the wrong shape for this workload).
