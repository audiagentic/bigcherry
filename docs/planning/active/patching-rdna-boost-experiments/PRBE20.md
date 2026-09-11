---
id: PRBE20
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:49.394058+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Make decode and speculative-verify batches bit-identical (attention + mmvq + CPU cluster)

## Description

Complete the coordinated decode/speculative-verify bit-identity cluster. Standalone subset exists; attention and post-fix MMVQ portions remain composition-gated.

## Steps

- Treat all five source commits as one determinism property: flash attention, non-flash attention, CPU, RDNA4 MMVQ/SSM and RDNA3 MMVQ.
- Keep standalone materialized subset explicit; port fattn portions only after PRBE02/03/06 retained and in the required family order.
- Port RDNA4/RDNA3 portions from branch-tip post-fix state under PRBE19; do not create half-deterministic composition.
- Run decode vs speculative-verify on identical inputs and compare byte identity across reachable gfx1100/RDNA4 paths and CPU cluster.
- Preserve native controls and record any determinism cost; no performance promotion is implied.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

attention fattn sources; ggml-cuda.cu; mmvq.cu; CPU sgemm; PRBE19 post-fix regions; patch 1210; exact decode/verify fixtures and campaign evidence.

## Validation

All five commit identities; dependency/order checks; byte-identical outputs; native control; gfx1100 and reachable RDNA4/RDNA3; graph/capture where applicable; no unintended performance regression.

## Effort & Risk



## Standards

Coordinated determinism change; no half-cluster acceptance; branch-tip post-fix provenance.

## Acceptance Criteria

Full cluster, not only standalone subset, produces byte-identical decode/verify outputs under declared paths; composition dependencies are satisfied; partial ports are not promoted.

## Notes

Supersedes: RD26
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd26



REAL HARDWARE CORRECTNESS CHECK RUN (2026-09-11/12), first-ever real execution of the materialized 2/5-commit subset (patch 1210_rd26_bitidentical_decode_verify_standalone): authored patches/1210_rd26_bitidentical_decode_verify_standalone/validation/rd26_correctness.py (reusing the shared tools/bigcherry/experiment/perplexity.py primitive, its fifth real caller) plus run_rd26_ppl_check() in validation_campaign.py.

HONEST SCOPE LIMIT, stated in the producer's own docstring and NOT to be conflated with PRBE20's real acceptance criteria: this check does NOT prove the actual cross-batch-size determinism claim (decode n_q=1 vs speculative-verify n_q up to 8 producing bit-identical logits against EACH OTHER) -- that needs a materially different test structure (a real within-binary cross-batch comparison), not implemented here. What it proves: the two ported kernel-routing hunks (MMVF batch threshold in ggml-cuda.cu, sgemm batch gate in llamafile/sgemm.cpp) do not regress ordinary single-token decode output.

Ran for real on Brutus: gfx1201, tierA-qwen4b-q6k, real wikitext2 corpus. Result: PASS, exact PPL match (10.4463 both subject and control, delta=0.0) -- no regression from the two ported hunks on ordinary decode.

PRBE20's full acceptance criteria (the real cross-batch bit-identity property across the FULL five-commit cluster, including the composition-gated flash-attn and RDNA4/RDNA3 hunks not yet ported) remain unmet -- this closes one real, narrow data point (no ordinary-decode regression from the standalone subset), not the item's actual scope. Patch state remains "untested".

## Change Log

- 2026-09-09T10:54:49.394058+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:02.133012+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.219018+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.935699+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:51:52.755045+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025218_rdna-successors-prbe2022-now_5714
- 2026-09-10T02:52:18.396789+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T16:37:09.158298+00:00 (updated-by): Updated: section:notes
- chg_20260911_163714_real-hardware-test-confirms-no_2191
- 2026-09-11T16:37:14.220103+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_212645_documented-two-more-optimizati_8645
- 2026-09-11T21:26:45.055158+00:00 (updated-by): Updated: section:ledger-events
