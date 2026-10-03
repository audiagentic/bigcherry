---
id: PRBE01
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:31.014682+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Implement final native-BF16 flash-attention logical experiment

## Description

Port and qualify the folded native-BF16 flash-attention logical path from the reviewed commit series. Existing isolated evidence is directionally promising (clean decode gain and no measured prompt cost) but correctness and cross-architecture repeat remain before promotion.

## Steps

1. Port only the reviewed logical BF16 path as one precision experiment; preserve F32 accumulation and native-BF16 guards.
2. Keep unsupported hardware on the existing fallback and preserve no-F16-regression behavior.
3. Separate cleanup/operand-order evidence from the implementation arm.
4. Run the FLASH_ATTN_EXT backend correctness matrix, BF16 same-path checks, and exact output/precision checks without claiming F16 bit identity.
5. Run long/deep-context numerical-quality checks and repeat on gfx1100 and gfx1201.
6. Reproduce isolated decode/prompt characterization with balanced interleaved controls, then gate promotion on correctness, precision, quality, fallback, cross-architecture, and performance evidence.

## Detailed Solution & Technical Design

Treat this as a precision-path experiment, not six cherry-picks. Validate architecture guards, unsupported fallback, F32 accumulation, BF16 operand semantics, graph capture, and numerical stability. Existing evidence: materialized patch 1202; isolated gfx1100 decode was directionally positive and prompt deltas were noise-adjacent/indistinguishable at pp>=1024. This is not a promotion claim.

## Code Samples & Guidance



## Files

Future vendor HIP flash-attention sources; patches/1202_rd04_bf16_flash_attn_tile; correctness fixtures and backend matrix; balanced E2E evidence for gfx1100/gfx1201

## Validation

Build and apply/idempotence: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1202_rd04_bf16_flash_attn_tile`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1202_rd04_bf16_flash_attn_tile --source bigcherry-tuning`; package pytest offline. CORRECTED (source-verified): 1202's 35 source anchors match uniquely at b11126, but the patch has NO BIGCHERRY_PATCH_HIT/BIGCHERRY_PATCH_TRACE marker while validation.toml's activation-probe check requires one -- add a BIGCHERRY_PATCH_TRACE-gated GGML_LOG_WARN marker at the actual native-BF16 branch inside `ggml_cuda_flash_attn_ext_tile(...)` (ggml/src/ggml-cuda/fattn-tile.cu, the real dispatch switch on K->ne[0] -- NOT a `..._tile_case_type` function, which does not exist in the source) when the BF16 K/V case dispatches; require subject-hit/control-miss evidence per validation.toml's existing (currently BLOCKED) marker-probe check. The correctness producer is NOT test-backend-ops -o FLASH_ATTN_EXT and NOT the old `--overlay --arch` campaign invocation -- per PA35/PA36 and this patch's own validation.toml, correctness is a real whole-model PPL comparison (backend_reference + ppl_equality) driven by the package-local producer, invoked as `--validation-producer 1202_rd04_bf16_flash_attn_tile/rd04`. Hardware (Brutus, gfx1100 AND gfx1201 both required): run the package-local producer per validation.toml (apply/build/performance/controls/correctness checks), reproducing isolated decode/prompt characterization with balanced interleaved controls; promotion requires correctness+precision+quality+fallback+cross-architecture+performance evidence together, not isolated throughput alone.

## Effort & Risk



## Standards

Correctness before performance; explicit precision contract; fallback preservation; no claim of F16 bit identity.

## Acceptance Criteria

All RD04 requirements are either passed with evidence or explicitly dispositioned: F32 accumulation/native-BF16 guards, unsupported fallback, FLASH_ATTN_EXT correctness matrix, BF16 same-path checks, long/deep-context quality, gfx1100/gfx1201 coverage, unsupported fallback, no F16 regression, and balanced performance. No promotion on isolated throughput alone.

## Notes

Supersedes: RD04
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd04

Supersedes: RD04
Inherited semantic scope: preserve every actionable RD04 gate; historical evidence remains on completed predecessor.
Migration: capability-rebaseline-v3-2026-09

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1202_rd04_bf16_flash_attn_tile exists, state=untested. Item's own detailed_solution already records isolated gfx1100 evidence as directionally positive (decode gain, prompt deltas noise-adjacent at pp>=1024) but explicitly not a promotion claim -- cross-architecture (gfx1201) repeat and the full correctness/quality gate matrix are still outstanding. No upstream b11126 native-BF16 flash-attention path found matching this scope (b11126 ggml-cuda flash-attn is FP16/F32-path generic, not this project's folded BF16 logical-path experiment). Disposition: validate/qualify existing patch; no GPT design needed (patch already implements the described precision-path experiment).

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified 1202's 35 anchors are unique at b11126 but confirmed (grep) it has no activation marker although validation.toml requires one for its marker-probe check -- added a required BIGCHERRY_PATCH_TRACE-gated WARN marker at the real dispatch function `ggml_cuda_flash_attn_ext_tile` in fattn-tile.cu (not the nonexistent `..._tile_case_type`). Removed the stale test-backend-ops/old campaign validation language in favor of the actual package-local producer workflow (`--validation-producer 1202_rd04_bf16_flash_attn_tile/rd04`) already declared in validation.toml.

## Change Log

- 2026-09-09T10:53:31.014682+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:07.152011+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.129465+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.797954+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:10:06.643778+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_021313_the-semantic-audit-is-now-trac_4827
- 2026-09-10T02:13:13.308366+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:45.540247+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:40:44.032183+00:00 (updated-by): Updated: section:validation, section:notes
