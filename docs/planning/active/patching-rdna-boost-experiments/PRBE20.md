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

Complete the coordinated decode/speculative-verify bit-identity cluster as one active plan. PRBE20 is the actionable successor to closed RD26; its internal Wave 1/Wave 2 structure preserves the historical five-commit evidence without splitting the acceptance property.

## Steps

1. Preserve all five source identities: 93510434f, b2655d381, d152888fc, 10b83d6b2 and 6cdf5aff9.
2. Wave 1: fattn/non-flash attention and CPU changes, sourced through the current owners of the required 1202 then 1203 preimages.
3. Wave 2: RDNA4/RDNA3 MMVQ/SSM regions sourced under PRBE19 post-fix semantics.
4. Keep the standalone materialized subset explicit; it is not full cross-batch determinism. Do not introduce a half-cluster acceptance.
5. Compare decode n_q=1 with speculative-verify n_q up to 8 on identical inputs using raw outputs/logits, native controls and graph/capture coverage; record determinism cost and causal performance.

## Detailed Solution & Technical Design

The acceptance property is cross-batch bit identity across the full five-commit cluster, not merely PPL equality or ordinary decode no-regression. Translate historical 1202/1203 artifacts to their current PRBE owners; PRBE06 is not a fattn prerequisite. PRBE19 is a source-state constraint for Wave 2, not a prerequisite patch. Keep native and partial-subset evidence clearly scoped.

## Code Samples & Guidance



## Files

fattn sources; ggml-cuda.cu; mmvq.cu; CPU sgemm; current owners of 1202/1203 preimages; PRBE19 post-fix regions; patch 1210 standalone evidence; exact decode/verify fixtures and campaign evidence.

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

Supersedes: RD26 (closed historical predecessor). Preserve all five immutable commit IDs as provenance. Existing 2/5 PPL/no-regression evidence remains narrow and does not satisfy PRBE20 acceptance. Do not split PRBE20; use internal Wave 1/Wave 2 sections.

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
- 2026-09-12T09:52:40.373896+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.254140+00:00 (updated-by): Updated: section:ledger-events
