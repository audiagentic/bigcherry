---
id: RSC19
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-27T12:56:15.096909+00:00'
breadth: ''
skill: ''
created-by: agent
work: M
priority: P1
---

# Component-owned configuration enhancements

## Description

Native YAML and upstream parser remain authoritative. Extend owned serving components and existing tuning/evidence owners; no new configuration layer or upstream patch for native settings.

## Steps

1. Seal managed environment; reconcile authored tuning projection. 2. Report mixed-step target, interleave mode/share, room/long-prefill/preemption controls, parsed defaults and resolved bring-up values. 3. Validate capability, pipeline, wire, KV and shard constraints. 4. Reuse exact kernel/provider/geometry/device/topology-qualified tuning; VTP60 and kernel composition own runtime implementation. Add a narrow engine seam only where native extension points are absent. 5. Audit duplicate variants with RSC12; qualify changes using existing quality and latency gates.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Matched native and owner builds; parser/receipt mutation tests, missing versus automatic settings, inherited environment, conflicting overrides, stale tune-cache identity, all-profile explain and representative GPU parity. No duplicate GPU campaign.

## Effort & Risk



## Standards



## Acceptance Criteria

One native YAML; no hidden overrides; authored, parsed and effective settings distinguished; all projected fields reconciled; no new patch for owned functionality; no unqualified production defaults.

## Notes

First slice: four native tuning options projected, existing projection reconciled, inherited RADIANCE_* isolated. Owner compiled; test execution blocked by ctest allowlist. Full native integration and remaining fields pending.

Owner build: bc_serving_test and bc_session_test linked. Matched Radiance host build: bc_native_fleet_config_test linked (222 build actions). Profile roster/variant/generator suite: 6 passed, 49 subtests passed. CTest execution denied by remote allowlist; C++ runtime/mutation and installed GPU gates remain pending. Changes confined to owned component and docs; no YAML production values or patch files changed. Reserved environment namespace now includes RADIANCE_*; use non-reserved names for API key references.

Second slice: native inspector v3 projects parse-time scheduler values and authorship (interleave steps/share, mixed-step target, room gate, long-prefill threshold, preemption wait). Distinguishes omitted, zero, auto and auto:1 without upstream patches. Strict owner validation and parser/mutation tests added. Effective bring-up receipt/manifest reconciliation and capability qualification remain pending; runtime test execution requires authorized runner.

CTest is authorized. Initial native parser and all four session groups passed; serving mutation gate exposed strict JSON rejecting floating-point shares. Fixed with round-trip decimal text, bounded numeric validation, strict-parser regression and valid auto-share admission tests; bc-serve explain now surfaces parsed scheduler. Rebuilt both owners: native test 1/1 passed, serving/session tests 5/5 passed (58.90s). Native /diagnostics already exposes configuration option/value/source and bring-up decisions; next slice should reuse it for effective-value reconciliation without a new engine patch. GPU/capability gates remain pending.

Coordinate with VTP59/VTP60: use existing MRC06 lab adapter capture_tuning_receipt against native /diagnostics, no second YAML parser. Reconcile direct manifest fields and additional explicit arm expectations, require matching pre/post receipts, retain through existing native lab ingestion. 26 targeted Python tests pass. Live exact 2.3 candidate (prior owner, new lab adapter) TP3 IQ8R FP8 KV 100/150ms targets: 4K/512 aggregate 179.990/186.543, max TTFT 15.782/14.604s, p95 gap 161.933/41.349ms; 512/128 aggregate 281.492/269.749, TTFT 2.139/2.166s, p95 gap 166.390/174.782ms. Four completed reports, zero preemptions, clean stops, identical prompt hashes/counts vs prior ABBA, both native receipts verified. Evidence docs/evidence/RSC19-VTP59-latency/summary.json. No production promotion; matched enhanced owner build, automatic expected settings from inspector, cross-field capability and numerical quality gates remain pending.

Fresh and replay owner session selection now records the matched parsed scheduler and sealed YAML SHA-256. Existing lab adapter automatically derives stable scheduler expectations, refuses mismatched snapshot/manifest/explicit intent, preserves omitted interleave as bring-up-resolved. Old owner needs explicit expectations. Added managed parser and bind refusal for proceed-with-incompatible-configuration; native declaration remains capability authority (FP8/packed KV, f16 state, per-rank kernel coverage, lossy wire, FP8 attention and shard geometry). Strict JSON safe-integer gate already refuses oversized wire threshold; do not duplicate parser. Python 31 passed, owner CTest 5/5 passed, native CTest 2/2 passed. No upstream patch or production YAML change. Next canonical candidate build, all-profile explain and automatic receipt GPU proof; do not promote until quality/latency gates.

Canonical Radiance 2.3 stock build plan 927a24bf85c2ac29d3dbe0b779063323 succeeded; 18/18 staged native config explain preflights passed. Matched enhanced owner TP3 IQ8R FP8 KV 75/50/75ms fresh starts, automatic scheduler reconciliation with empty explicit expectations, both receipts verified for all three runs. 4K/512 aggregate 188.774/180.289/180.855 t/s; max TTFT 14.574/15.768/15.763s; p95 event gaps 41.509/161.219/162.438ms. 512/128 aggregate 279.875/270.253/279.928 t/s; max TTFT 1.979/2.157/1.980s; p95 gaps 40.065/165.666/40.445ms. Six reports match previous prompt hashes/token counts, zero preemptions, clean owner stops. Long workload scheduling regimes persist; 75ms short result repeats, not a universal winner. Evidence docs/evidence/RSC19-VTP59-matched-owner/{build-preflight,summary}.json. Next mixed arrivals/interleave/phase tracing and independent quality then TP2/TP3/other model qualification. No production promotion.

Qwen FlashNext true TP4 native lab arm (2 gfx1100, gfx1201, gfx1030) on matched 2.3 build 927a24bf85c2ac29d3dbe0b779063323 refused before GPU launch: architecture gfx1030 cannot run the engine (archs.toml roles). Existing flashnext-4gpu is TP1+3 helpers. No role override; libr2 only builds helper MoE/activation quant units, no full engine coverage. Future TP4 requires full gfx1030 provider coverage, per-rank graph/collective and numerical qualification, then explicit role admission. Four-card control at FP8 KV, 8192 context bound, 8 streams, batch2048, no prefix cache, interleave/mixed target off completed 4096/512 workload: aggregate 173.412 t/s, max TTFT 11.417s, p95 event gap 66.229ms, zero preemptions, 8x512 outputs. Job daemon lost record before short workload/final receipt; owner stopped cleanly, GPU free. Not a qualified control or TP4 performance result. Evidence docs/evidence/RSC19-FN-topology/summary.json.

Owner direction: retain existing FlashNext four-card TP1+3-helper layout; concentrate tuning on dense Qwen 27B TP3 (two gfx1100 plus gfx1201). No true TP4 expansion required for this campaign. Matched owner 2.3 build 927a24bf85c2ac29d3dbe0b779063323, 75ms target, FP8 KV, 8 streams, pipeline min256, batch2048. Auto:0.25/auto:1/room-on (interleave off) long aggregate 160.398/132.717/184.784 t/s, worst TTFT 18.927/26.169/14.808s, p95 gap 133.871/34.020/41.409ms; short aggregate 262.525/255.018/278.946 t/s, worst TTFT 2.387/2.807/2.154s, p95 gap 40.201/30.592/169.665ms. Six reports automatic paired receipts passed, zero preemptions, clean stops. Single repetition per arm; auto interleave poor aggregate tradeoff; room gate mixed results. Archive docs/evidence/RSC19-VTP59-admission/summary.json. Continue arrival/admission controls, mixed workloads, quality and provider/geometry qualification on TP3. No production promotion.

## Change Log

- 2026-11-27T12:56:15.096909+00:00 (created-by): Created by agent
- 2026-11-27T12:58:57.211030+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:14:50.590396+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:22:28.116046+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:39:37.473752+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:49:43.098266+00:00 (updated-by): Updated: section:notes
- 2026-11-27T14:09:47.084434+00:00 (updated-by): Updated: section:notes
- 2026-11-27T14:27:55.436958+00:00 (updated-by): Updated: section:notes
- 2026-11-27T14:47:04.187572+00:00 (updated-by): Updated: section:notes
