# RHP22 — four-GPU live-placement KL quality disposition

**Decision (2026-11-22): FAIL the existing four-GPU forced-churn numerical acceptance arm. Do not waive or relabel this as measurement noise.** This closes only RHP22's *decision* requirement; it does not accept or finish the live-placement policy, RHP16 calibration, RHP06 transaction correctness, RHP20 layout coverage, or the RUM06 runtime-lifetime gate. No new test run, GPU use, production configuration change or compiler/kernel change was performed here.

## Existing reproducible evidence

The source is the previously recorded RHP16/RHP22 result under `runs/policy-kld` (`out-aa`), summarized in the canonical RHP22 plan's November 11 note and RHP16's November 8/10 notes. Raw run directories were not present in this review checkout: these are **historical plan-record measurements**, not newly reproduced receipt files. The control and identical `policy-off-repeat` were bit-identical for every measured quantity, so a noise-based waiver is not supported.

| Metric | Static control | Forced-churn arm | Change | Existing limit | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Mean KL | 0.0660 | 0.0738 | +0.0078 | Existing mean gate | Within gate |
| KL p99 | 0.7036 | 0.9377 | **+0.2341** | **+0.2000** | **FAIL** |
| KL p99.9 | 3.0029 | 3.4331 | +0.4302 | +0.5000 | Within gate |
| Top-1 agreement | 88.54% | 87.94% | −0.60 percentage points | Existing top-1 gate | Within gate |

The four-GPU arm evaluates **8,240 positions** against the same historical rad116 BF16 reference; the forced transactions completed during engine startup **before** the short 15–17 s quality scoring pass. Thus this is a numerical result for **one swapped static final placement**, not a demonstration of an ownership transition during token scoring. The mechanism is consistent with RHP16's known gfx1030/RDNA2 tail accuracy sensitivity when hot layer-0 experts sit on the 6900 XT tail; consistency is not proof of why every high-KL position differs.

A separate `flashnext-2xtx-rdna2-tail` forced-move arm (`runs/rhp20-tail-kld`) reported p99 +0.112 with 132 swaps **during** scoring, under a different layout/source provenance. That is useful negative evidence against a universal migration-fails claim but **cannot waive** this distinct four-GPU result or establish a passing warmed four-GPU layout. The tail-combine path in RHP20 did not engage in its KL run.

## Single-owner follow-on gate (not another plan)

RHP16 owns policy calibration, movement choice and the numerical gate. RHP06 owns transaction correctness and mid-scoring move integrity. RHP20 owns the layout-specific cross-ISA acceptance matrix. Do not implement a second policy, KL scorer, staging helper, or model cache for RHP22.

For `flashnext-4gpu`:

1. With existing serving/calibration commands, first **run the ordinary calibrated live policy until its placement settles**; save the exact current-source model/container, engine/helper build, calibration key, per-rank device identity, owner-map generation, final expert map and movement/heat log. **No guessed wait or synthetic placement**, and do not use the forced-churn-only layout as a proxy.
2. **Freeze that exact settled final map** using existing placement/manifest and scorer mechanisms, and take a matched, same-build, same-reference static control-versus-frozen-policy KL run. Capture mean, p99, p99.9, top-1, source/format/identity, and the precise independent reference. Missing matching freeze/replay support is a technical blocker for RHP16; it is not approval to write another scorer.
3. Separately qualify **moves during scoring** under existing RHP06/RHP20 owner protocol; test transaction phase refusal, old/new generation, wait/drain, failed move recovery, and numerical quality for every supported cross-ISA format. The historical startup-only forced-churn result is not that proof.
4. Promote a live-policy quality verdict **only if** the frozen ordinary policy and mid-scoring migration gates both pass their existing limits on the intended current-pin layout, plus other normal performance and operational checks. If either fails, keep the affected layout unqualified and narrow its live-policy eligibility rather than waiving a deterministic high-tail error.

An ON calibration/manifest status is **not** a quality pass. Existing historical production-enabled status is recorded as fact; this decision **does not silently change** the running system or alter the current policy calibration record. It forbids citing the forced-churn arm as accepted quality or generalizing its conclusion to a different deployment.

## Source pointers

- `docs/planning/completed/patching-radiance-heat-placement/RHP22.md` — original history, duplicate-control note, exact fail.
- `docs/planning/active/patching-radiance-heat-placement/RHP16.md` — policy derivation, existing movement/accuracy evidence, next warm placement gate.
- `docs/planning/active/patching-radiance-heat-placement/RHP06.md` — live movement correctness.
- `docs/planning/active/patching-radiance-heat-placement/RHP20.md` — specific layout and cross-ISA receipts.
- `docs/reference/engines/radiance/LIVE_PLACEMENT.md` — operator-visible quality decision and acceptance boundary.

Only canonical Planning MCP transitions may declare RHP22's decision delivered; no plan quality or GPU acceptance is inferred from this document.
