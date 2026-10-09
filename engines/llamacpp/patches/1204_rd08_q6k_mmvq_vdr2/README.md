# 1204_rd08_q6k_mmvq_vdr2: Q6_K mmvq VDR=2 decode kernel (RD08)

Patch id: `1204_rd08_q6k_mmvq_vdr2`. Plan item: `RD08`. Bound Experiment
Contract: `RD08-Q6K-MMVQ-VDR2` (`config/experiment-contracts.toml`).

## Scope

Target architectures: gfx1100, gfx1201, gfx1030 (contract `scope.architectures`).
Backend: HIP. Weight type: Q6_K. Prerequisites: none.

Adds a VDR=2 Q6_K `vec_dot` entry point that processes both 8-element
chunks of a Q6_K dot product in one call (4 `dp4a` ops instead of 2,
halving loop iterations for the same row) and switches
`get_vec_dot_q_cuda`'s Q6_K case to it. Also makes `GGML_CUDA_OP_TIMING`
disable CUDA graph capture instead of aborting, and adds the decode-shaped
perf test cases the correctness producer below uses. Ported from
stew675-rdna-boosts fork commit `4591cc980`
(https://github.com/stew675/llama.cpp); not merged into ggml-org/llama.cpp
master. Decode is DRAM-bound, so this targets a modest tg128 gain, not a
compute-bound win — the fork's own claim is that the VDR=2 kernel is
bit-identical to the VDR=1 kernel it replaces, which is exactly what this
patch's bound correctness check proves.

## How to invoke validation

Real, authoritative full-qualification path (VA14):

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1204_rd08_q6k_mmvq_vdr2 \
  --model <tierM-gptoss20b-q6k.gguf> \
  --hip-path <production-rocm> --amdgpu-targets gfx1100 \
  --manifest <hip-autotune-manifest.json> \
  --workdir <fresh-workdir> --build-root <build-root> \
  --worktree-root <worktree-root> --bench-repetitions 3 \
  --run-rd08-contract
```

`--run-rd08-contract` executes RD08's real positive(decode)/control(prefill)
lane pairs (`tools/bigcherry/experiment/execution.py`), the real
bit-identical correctness producer (below), and a real subject-hit/
control-miss trigger probe, then composes them via
`evaluate_promotion_gate()` into `eligible_for_validated_state`. See
`docs/reference/testing/PATCH_VALIDATION.md`'s "Real contract-execution
architecture (VA14)" section for the full mechanism.

`--run-rd08-lanes` alone is diagnostic-only (lane execution + evidence,
never feeds eligibility) — use it only to inspect measured performance
without running the full (slower) correctness producer.

## Correctness producer

`validation/rd08_correctness.py` (`materialize_rd08_variants()`,
`require_rd08_correctness_evidence()`) is patch-specific: it materializes
its own isolated VDR2-subject/VDR1-control worktrees (two checked semantic
reversions of exactly this patch's routing edits — see
`apply_vdr1_control()`), builds `test-backend-ops` for both, and proves
bit-identical output across the 5 real decode shapes this patch's own
perf-case edit adds × 3 deterministic seeds (backend1_digest equality, not
NMSE agreement). This is a real source-level A/B of VDR=2 vs VDR=1, not a
dispatch-candidate comparison.

## Control vs. subject

Validation-domain control/subject follow the framework's normal
`control_src`/`patched_src` composition (this patch present vs. absent).
The correctness producer additionally uses its own VDR2/VDR1 source-level
A/B (above) — do not confuse the two: the validation-domain control never
has this patch at all, while the correctness producer's "control" has the
patch's other edits but reverts only the two VDR routing lines.

## Real full contract-qualification run at current pin (2026-09-13) -- real correctness-gate FAIL, investigated

Ran `run_rd08_contract_qualification()` directly for real on Brutus
against the current active pin (`b10901`/`28ff0958291c`), single gfx1100,
`tierM-gptoss20b-q6k`, 6 paired decode/prefill rounds, real trigger probe.

**Trigger proof: real PASS.** `subject_hit=true`/`control_hit=false`
(via `--verbose`-gated `GGML_LOG_WARN` marker), positive lane checked and
triggered.

**Performance: inconclusive at 6 rounds.** `target_kernel_gain_pct =
+0.30%`, but `ci95_low = -0.26%` (interval crosses zero) --
`min_paired_rounds` for this contract is not yet migrated to the
interval policy (still `point_estimate_v1`/legacy-waiver, per
`config/experiment-contract-legacy.toml`'s `RD08-Q6K-MMVQ-VDR2` entry),
so this is a real, honest data point but not by itself a promotion
blocker under the contract's own current (weaker) policy.

**Correctness: real FAIL on the `bit_identical` gate -- investigated,
NOT concluded to be a patch defect.** All 15 rows (5 shapes x 3 seeds)
failed the exact-digest equality check, but by a uniform, tiny margin:
subject vs. control `err` values differ by only ~1e-7 to ~1e-6
*relative* magnitude in every single row (e.g. `ffn` seed=1:
`2.353963937227954e-05` vs `2.3539629204239105e-05`), all far below the
gate's own `0.0005` tolerance, and every row's own internal
backend1-vs-backend2 comparison independently reports `status=ok`. This
uniform, sub-threshold pattern across every shape and seed -- rather
than an isolated failure or a magnitude comparable to the tolerance --
is inconsistent with a real VDR=2-vs-VDR=1 numerical divergence and
consistent with ordinary GPU non-associative floating-point reduction
non-determinism between separate process launches, which the
1222/1223 (HI67) deterministic-seed patches (already correctly included
in `RD08_PATCH_STACK`) control the RNG *inputs* for but cannot control
the GPU's internal warp/reduction execution order for. **Not
independently confirmed** (a same-binary-run-twice control was not
performed this session due to time) -- this is a strong, evidence-based
hypothesis, not a proven root cause. Filed as PRBE103 to run that
confirming control and, if confirmed, reconsider whether an exact-digest
`bit_identical` check is achievable at all for GPU kernels via this
methodology, or whether it needs a tolerance-based redesign.

**No promotion attempted or claimed** -- `evaluate_promotion_gate()`
correctly returned `status=fail` on the correctness gate; this is not
overridden or reinterpreted here.

## PRBE103 resolution (2026-09-13) -- nondeterminism hypothesis DISPROVED, real correctness FAIL confirmed

PRBE103's same-binary-run-twice control (GPT-designed and reviewed,
`req_d0be05f943d64d86` / `req_aabd4a5c2b274763` / `req_35aec570a6bb4038`)
ran the identical VDR=1 control binary 4 independent times for every one
of the 15 (shape, seed) cases, using the real campaign's own
environment-preservation runner (`{**os.environ, **env}`, matching
`validation_campaign.py`'s pattern -- a first attempt without it produced
spurious nonzero exit codes and was correctly discarded). **Result:
all 60 executions exited cleanly (`status=ok`), and every single
(shape, seed) case produced a bit-identical `backend1_digest` across all
4 executions.**

**PRBE103 CLOSED -- same-binary nondeterminism hypothesis disproved.**
The identical VDR=1 binary is deterministic across repeated real
executions on this hardware. Therefore the exact-digest divergence
observed between the VDR=1 control and VDR=2 subject in the contract-
qualification run above is deterministic and attributable to the real
implementation difference between the two kernels, not process-launch or
GPU floating-point nondeterminism. Cross-process exact digest comparison
is a valid, stable oracle for this workload; no tolerance-based
replacement is justified by nondeterminism.

## Final disposition (GPT-approved, `req_35aec570a6bb4038`)

**Correctness: confirmed real FAIL, not a methodology artifact.** The
VDR=2 implementation is NOT bit-identical to the VDR=1 implementation it
replaces, contrary to the upstream fork's own stated bit-identical claim
and this patch's bound `bit_identical = "required"` correctness
obligation. The observed numerical divergence is small (~1e-7 relative),
and both implementations independently remain within
`test-backend-ops`' own backend-reference numerical threshold -- this is
a real, deterministic loss of EXACT equivalence, not a demonstrated
material numerical-accuracy regression under a separate tolerance-based
metric. The Experiment Contract's `bit_identical` requirement is not
weakened post hoc to convert this FAIL into a PASS -- that would require
a separate, deliberate contract-policy review (contract authority, not a
validation-producer decision), followed by fresh qualification under any
revised contract. No validation promotion is permitted from this
evidence.

**Lifecycle: `state` stays `"untested"` (GPT-corrected, `req_d76911c6fc814809`).**
An initial recommendation to reject this patch (briefly applied to
`patch.toml`, then reverted the same session) was too strong: a prior
investigation (`config/external-sources.toml`'s RD08 note, dated
2026-09-11, `req_3c98f154389148bb`) had already hit this exact same
correctness-gate failure, ruled out RD25 as the cause, and explicitly
framed the real open question as a **contract-design review**, not a
proven defect -- VDR=2 intentionally changes lane assignment and
warp-reduction structure versus VDR=1, so whether exact bit-identity is
even the scientifically appropriate acceptance bar for this class of
kernel was left deliberately unresolved. PRBE103 answers a narrower
question (the divergence is deterministic, not GPU/process noise) and
does NOT resolve that pre-existing, still-open contract-design question.

**Current, accurate disposition: current `bit_identical` contract check
FAILS, confirmed deterministic; patch remains `untested` pending
independent contract-design adjudication.** No evidence presently
establishes a material numerical-accuracy regression, and no lifecycle
rejection is justified until the normative correctness requirement
itself is resolved. Do not weaken the contract merely to rescue RD08 --
any contract-design review must justify the intended correctness
semantics independently of this patch's outcome, per this project's own
lifecycle rules (contract authority, not a validation-producer
decision). If that review concludes exact bit-identity is the correct
required bar, RD08 should then become `rejected`. If it concludes the
contract encoded an inappropriate requirement for this kernel class, the
Experiment Contract should be deliberately revised and qualification
rerun under the new contract before any promotion.

## PRBE104 resolved: contract revised, real re-evaluation (2026-09-13)

GPT made the actual contract-design decision (`req_8163325eb9c545ea`):
revised `config/experiment-contracts.toml`'s `RD08-Q6K-MMVQ-VDR2`
correctness requirement from `bit_identical` to `backend_reference`
(NMSE within `test-backend-ops`' existing 0.0005 threshold) -- RD08
deliberately changes Q6_K accumulation grouping/lane structure, and
floating-point addition is non-associative, so exact output bytes were
never the scientifically appropriate bar for this kernel class. The
bit-identical digest-equality result is preserved as a non-gating
diagnostic fact (the fork's own bit-identical claim is genuinely
disproven, per PRBE103) but is no longer the promotion gate. Editing the
contract changed its hash, which voided the legacy point-estimate
waiver (VA24 policy) -- migrated to `ci95_threshold_bound_v1` with
`min_paired_rounds=10` in the same change, removing the stale waiver
entry, per the RD21/RD39-42/RD73 precedent.

**Re-evaluated using the already-gathered PRBE103 evidence (no new
correctness hardware run needed)**: all 15 rows' real subject `err`
values (~2.3e-5 to ~2.6e-5) are ~19-21x below the 0.0005 threshold --
**correctness gate now PASSES.**

**Performance gate: real FAIL, honestly evaluated.** Extended the
original 6 paired rounds to the required 10 (4 additional real rounds,
current pin). Full 10-round result: `target_kernel_gain_pct` point
estimate +0.261%, but **95% CI lower bound -0.062% -- crosses zero,
below the required 0.3% threshold.** `max_control_regression_pct` CI
upper bound (0.040%) comfortably passes its own 1% bound. **RD08's real
measured gain is small enough that 10 real paired rounds cannot
statistically distinguish it from zero** -- this is a genuine,
evidence-based performance FAIL under the interval policy, not a
methodology artifact.

**Overall: correctness now passes; performance does not. RD08 is not
promotable to `validated`** (both required gates must pass), but for an
honest, now-fully-resolved reason -- not the previously open
contract-design ambiguity. `state` stays `untested`. PRBE104 closed.

RD08's own contract (`scope.architectures = ["gfx1100", "gfx1201",
"gfx1030"]`) already declares all three architectures in scope, so per
`docs/reference/testing/STANDARDIZED_PATCH_VALIDATION_CRITERIA.md`
extended the trigger/activation evidence (previously gfx1100-only) to
gfx1201 and gfx1030. Real methodology note: an initial attempt without
`--verbose` got 0 hits on BOTH subject and control (unlike the earlier
`GGML_LOG_WARN` fix's own claim that WARN bypasses the `--verbose` gate
-- empirically, on this `llama-bench` build, `--verbose` was still
required for the marker to appear at all). Adding `--verbose` gave a
clean, real result:

- **gfx1201**: `subject_hit=1`, `control_hit=0`.
- **gfx1030**: `subject_hit=1`, `control_hit=0`.

**RD08's trigger/activation leg is now real and clean on all three
contract-declared architectures.** This does not change the open
correctness-gate/contract-design question above (PRBE104) -- the patch's
real, deterministic `bit_identical` FAIL stands independent of this
trigger result, which only confirms the code path genuinely executes.

## Real three-arm baseline comparison (2026-09-13, standardized criteria)

Per the standardized criteria's A/B/C requirement (A=stock upstream,
B=BigCherry baseline without RD08, C=BigCherry+RD08), gpt-oss-20B,
gfx1100:

- **Decode (RD08's own domain, `tg128`)**: A=177.49, B=178.23,
  C=179.09 -- all within noise of each other; no baseline concern, no
  material B-vs-C effect distinguishable from noise at this sample size.
- **Prefill (`pp512`, not RD08's domain)**: A=5173.25, B=4154.34,
  C=4215.66 -- **B and C both show PRBE107's already-documented and
  already-fixed regression** (these builds predate the `0300_mmq_forced_j`
  fix). Not a new finding -- corroborates PRBE107's root cause applies
  to this build too, and confirms the fix (already committed) will
  benefit RD08's own prefill baseline once rebuilt.

## Known limitations

None declared — this patch is not `deferred-hardware`. See the
2026-09-13 contract-qualification section above for a real, open,
not-yet-root-caused correctness-gate finding (PRBE103).

## Evidence

Runtime artifacts (build logs, raw benchmark output, correctness rows)
land under `artifacts/patch-validation/1204_rd08_q6k_mmvq_vdr2/<campaign-identity>/`,
outside this tracked patch directory. The compact, tracked record is
`evidence/validation.json`.
