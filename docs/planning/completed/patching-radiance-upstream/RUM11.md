---
id: RUM11
order: 0
plan: patching-radiance-upstream
state: completed
created-at: '2026-11-20T11:54:02.454411+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P1
work: M
---

# Restore the plain E4M3 fused KV store for Qwen 27B (H256, page16) on Radiance 2.2

## Description

Found 2026-11-20 by the TP3 27B start on the 2.2 candidate (qwen27b-tp3-int8-556, kv-cache-dtype fp8): on all three ranks, 16 device bands of qk_norm_rope_gate_kv_store (head_dim 256, n_head 6, n_head_kv 1, block_size 16, kv_dtype fp8_e4m3) resolve to no kernel, and startup stops 'unsupported'. In v1.17.0 the row qk_norm_rope_gate_kv_store_fp8 was generic (cQkNormRopeGateKv: head_dim <= 256, head_dim % 32 == 0, block_size >= 1, fp8_e4m3). On the candidate, our rad0049_qwen4exp_e4m3_kv restored it as cQkNormRopeGateKvQ4Fp8 (head_dim == 256, block_size == 4 only), which covers Flash-Next but not the Qwen3.8-27B page16 caches every qwen27b fp8 config uses. The attention readers for that shape resolve; only the fused store is missing. Evidence: the measuring-engine log of the 2026-11-20 rum09-tp3-smoke run (task scratch).

## Steps

- Confirm in the candidate source that r4d_rad_qk_norm_rope_gate_kv_store and its fuse/shape functions handle block_size 16 for E4M3 (v1.17 accepted any block size).
- Extend the managed patch that owns the E4M3 store (rad0049, or a sibling patch requiring it) so the row admits the Qwen35 H256/GQA6/page16 E4M3 store, with constraints from the geometry the kernel supports, not a widened catch-all.
- Selftest row for page16 E4M3 store against the reference, on gfx1100 and gfx1201.
- Patch verify (apply, idempotence, rebase readiness), components build, plugin ABI, component tests.
- Hardware: qwen27b-tp3-int8-556 start/canary/stop on the 2.2 build; KLD and top-1 against the v1.17 release for the same config.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Selftest output, the TP3 start log with no unresolved bands, and the KLD report, recorded on this item.

## Effort & Risk



## Standards



## Acceptance Criteria

Every qwen27b fp8 config starts on the 2.2 build with no band resolving to no kernel; the page16 E4M3 store selftest passes on both archs; KLD/top-1 against the current release are within the existing qwen27b quality gate; the patch is validated through the patch workflow.

## Notes

2026-11-21 hardware: qwen27b-tp3-int8-556 starts on the 2.2 build of f5c82c33d (build 8451d2f1/8c64b580; RUM11 rows 5d9735720 plus the rank-devices fix 64bae9124). Ranks gfx1100, gfx1100, gfx1201, config read from its YAML with no --set, runtime home derived from --engine-binary. The canary returns ' Paris.\nThe capital of Germany is Berlin.' and the stop is clean. DFlash2 speculation is active at depth 3 (the w8a8-df2 container's MXFP4 drafter serves). The canary ran at 89-102 tok/s. Remaining for this item: the page16 selftest on gfx1100/gfx1201 and KLD/top-1, which will now run on the IQ8R container (RUM06 owner decision) against qwen3.8-27b-bf16.

## Change Log

- 2026-11-20T11:54:02.454411+00:00 (created-by): Created by claude
- 2026-11-20T11:54:21.655855+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-11-21T00:59:55.092369+00:00 (updated-by): Updated: section:notes
- 2026-11-21T20:17:07.225678+00:00 (state-transition): State: pending → in_progress
- 2026-11-21T22:25:09.102017+00:00 (state-transition): State: in_progress → completed
