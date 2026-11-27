# Radiance 2.3 baseline-first scope (MEN24 / RV4754)

Owner decision 2026-11-28: complete baseline first, integrate new features only if cheaper/safer during the bump. This scope narrows FAST-GATES, not existing safety/quality requirements. Source 3b13fd32663ef52252a7f0be255523480aa22ffe; prior canonical 69cccadc518f5282d312b0d84a08bc01 passed 71 ABI45 DSO checks, 93 component tests, CPU smoke, 18 YAML preflight and five model/topology smokes. Those receipts do not prove the later owner or installed release. Enhanced owner build 927a24bf85c2ac29d3dbe0b779063323 verified 18 preflights and paired scheduler receipts. Branch integration: f0d991032, 99a9bcc1e, d583b9fbd, e6fc3a05d; 117 tests/24 subtests passed. MEN29 is completed on main. No production defaults promoted.

| Owner | Baseline requirement | After baseline |
|---|---|---|
| MEN24/16/15, RSP01, PA49 | Exact canonical lane, reuse compatible proofs, CPU smoke, formal acceptance/reaudit/package/rollback | General cross-engine bump automation/time targets |
| MEN17, MRC08, RHP12 | Match candidate/reference runtime, existing model/variant coverage, independent quality and measured noise; preserve portable evidence | General export retries and historical record migration |
| RSC12/19 | Existing sealed YAML/manifest, parser, effective intent and cache safety; validated owned enhancements now | Nonessential variant cleanup, broader tuning registry |
| VTP59/60, VTP42, ENG03/04 | Existing concurrency regression, correct phase rates, matched TP3 and affected cache/cohort tests; reuse actual governor | Full policy search, new simulators, broad format and depth matrix |
| RSC16/17/18 | Legacy per-rank capacity and admission parity; only proven regression fixes | New shared L/S/C contract, exact allocation/retention receipts, new-mode scheduler qualification |
| RCS26, RUM15 | Affected owner/row selection, IQ8R group256 precision, ABI and existing provider/geometry correctness | General family catalogue, native format/trace patches, unrelated provider migration |
| PWR08, RSC01/02/05/06/11, RUM04, VTP33 | Preserve existing supported behavior; no repeat of completed projects | Unrelated power/fleet changes |
| XDP11, ENG07, PWR12 | No new converter, asymmetric format or power qualification required | Existing independent plans |

Required sequence: 1. freeze/reconcile exact source and approved roster. 2. Run affected offline tests and use canonical lane only when inputs match; verify metadata, runtime hashes and source. 3. Make acceptance fail closed on absent independent references, wrong lane and missing proof; don't promote legacy diagnostic captures. 4. Compare old/candidate cohorts and quality, preserve supported TP3 long/capacity formats and four-card FN; reuse receipts only when identities match. 5. Complete formal bump receipts, install/rollback and release. Full shared capacity follows; run its original RSC17->18->16 gates, not new plan items.

Unqualified diagnostic traces and native KV tool experiments stay isolated with evidence, not production recipes. Legacy KV uses native geometry, exact page boundaries, actual carve and front-ready; no added S*L change or 512K shared-mode claim. Governor admission safety and existing long policy remain in baseline; optimize TP3 only where regressions require it. No arbitrary tuning promotion from single screens.
