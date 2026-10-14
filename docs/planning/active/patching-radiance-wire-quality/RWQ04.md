---
id: RWQ04
order: 4
plan: patching-radiance-wire-quality
state: pending
created-at: '2026-10-15T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Test WHT6 residual outliers, improved scales and controlled clipping

## Description

If RWQ02 WHT8 either fails the quality target or loses too much of the bandwidth advantage, study whether WHT6's error is caused by a few transformed coefficients per group. The current WHT6 scale is BF16_RNE(amax/31) over a 64-value Hadamard group, with q in [-32,31]. Rotation already spreads channel outliers, so extra outlier machinery is a hypothesis, not assumed gain.

## Steps

1. Gather histograms from real prefill activation samples (RWQ01), by layer/family and block: max/rms coefficient, quantisation residuals, fraction of saturated codes, KL sensitivity; do not use hidden-state dumps containing prompt text in tracked evidence.
2. Offline ablations against independent exact reference: baseline scale amax/31; candidate BF16-rounded scale after percentile clipping (quantify clipping loss); candidate power-of-two exponent; candidate one residual correction for largest absolute reconstructed-error coefficient, with deterministic minimum-index tie break; optionally two corrections.
3. Design logical WHT6+R1 residual packet: base 6-bit codes + BF16 scale = 50 bytes/64, plus one coefficient index (1 byte) and BF16 correction (2 bytes), i.e. 53 bytes/64 before alignment/metadata. Preserve rotated-domain rank contributions and fold correction before inverse WHT. Be conservative with BF16 residual underflow/overflow, or use FP16 and verify its dynamic range.
4. Compare at matched bytes with WHT8 and with a seven-bit uniform candidate if warranted; stage fallback exact for unrepresentable scales.
5. Only implement a HIP kernel where offline and real sampled activation evidence demonstrate numerical improvement; run device checker, then RWQ05 full model validation.

## Detailed Solution & Technical Design

The receiver reconstructs q*scale + residual at the chosen transformed coefficient. A residual is computed from the sender's actual quantised local code and rounded transmitted scale, not the unrounded ideal scale. Sender retains its packed local contribution; receiver reads peer packed + correction before sum/inverse transform. A single correction per group adds 3 bytes logically but may incur padding and extra uncoalesced memory accesses. Implement complete bounds checks and deterministic order. Avoid stochastic rounding on rank-local values, which would make cross-rank equivalence fragile unless a shared deterministic seed/stream was defined and validated. For percentile clipping, compare per-group mean-square and max error; do not assume lower mean-square means lower model KL.

## Code Samples & Guidance

Synthetic inputs and BF16 tie vectors, plus real activation histograms recorded only in compact aggregate form; candidate layout: [base codes (48B)][scale BF16 (2B)][index uint8][correction BF16 (2B)] for each group conceptually. Actual GPU representation may use SoA planes for coalescing; no compatibility guarantee from this diagram.

## Files

tools/lab/radiance/wire-quality/codec_mock.py (extend only when useful); sample generation/analysis code under same lab topic; possible new WHT6+R1 libr3 native source/rewrite; component.toml experimental setting; per-group error evidence in docs/evidence/RWQ04.

## Validation

Unit tests: group permutation, tie-breaking, residual quantisation error, max-negative code, sparse/all-zero, extreme-scale, packet bounds, exact duplicate-rank contribution and mixed-group-mode decoding. HIP tests: byte-layout/poison and per-layer real activation histograms. Full-model gate is RWQ05, not synthetic rel-L2.

## Effort & Risk

Medium-high: heavy-tailed activation payloads may not be the cause of logits KL; extra residual memory access can eliminate bandwidth gains. Maintain default off and stop early if outlier diagnostics do not support this direction.

## Standards

- Target Radiance 1.3.0 pinned at 89cee7ce414486c717aa2085c99e1f522e4c0ea6; no unpinned edits.
- Engine edits are explicit Radiance patch packages; gfx1100 kernels and settings belong to the libr3 component. Keep opt-ins OFF unless independently promoted with quality evidence.
- Use engine-specific recipes, source/build identities, runtime profiles containing only settings different from declared defaults, and existing bigcherry ab-benchmark/engine_quality evidence facilities. No second campaign or loader.
- Follow docs/reference/tooling/TOOLING.md, docs/reference/engines/RADIANCE_RUNBOOK.md, patch mechanics, relevant device checker, and main-branch code review requirements.
- Changes must preserve both-rank numerical agreement, graph replay and no-P2P mapped-pinned-host protocol; fail closed on missing kernel support or disagreement.

## Acceptance Criteria

- Real-data diagnostics establish or falsify outlier hypothesis.
- Each candidate has exact wire byte budget, offline tensor-error results, GPU validation where implemented, and KLD comparison where viable.
- Candidate rejected cleanly if no quality-throughput Pareto improvement; no production switch by assumption.

## Notes

Runs only after RWQ01 attribution. RWQ02 WHT8 and RWQ03 protected-site policies are cheaper first steps; this item is optional/research until they fail to meet the gate.

## Change Log

- 2026-10-15T00:00:00+00:00 (created-by): Initial plan; unimplemented GPU work remains pending.
