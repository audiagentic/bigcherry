# 1281_moe_mul_mat_id_range

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md. The
patch is a neutral enabler: it adds a variant of MUL_MAT_ID whose weight tensor holds only the experts
[id_base, id_base + n) and which yields an exact zero for every other selected expert. Nothing builds that variant
unless 1283 is switched on (`BIGCHERRY_MOE_EP=1`), so the ordinary op and every existing graph are unchanged.

## Evidence

- Reference test `tests/test-mul-mat-id-range.cpp` (`tools/lab/flash-next/mmid-range-test.sh`), build
  `b-moerange-b11402s` with the range-aware fused launches: CPU 432 / 432 cases bit-identical to the ordinary op on
  hand-translated ids, with an exact +0 on every lane that is not held. GPU on every device against the CPU
  reference, per batch band (tokens 1 / 2-4 / 8-9 / 33 / 300): 720 / 1440 / 1152 / 432 / 432 cases pass, 0 fail;
  288 cases in each of the last three bands are declined by the backend policy (`supports_op` false) and fall back
  to the CPU as intended. The control arm (ordinary op, same inputs) passes the same bands.
- Unused is unchanged: on the production topology the build that carries 1281 + 1283 gives the same greedy text
  (md5 f80ea4df) and the same speed as the production build without them (see 1283's README).
- Used end to end: with 1283 on, the fused gate + up + GLU launches and the MMQ / MMVQ range paths carry the whole
  MoE block; fidelity against the row split is inside the stack's envelope (top-1 21/24, TV mean 0.113 and 0.078)
  and prefill and decode are equal to the row split. Without the fused form the decode step was 14.1 against
  13.5 ms, so the fusion is part of the patch, not an option.
- Mechanics: `tools/tests/patch/test_1281_moe_mul_mat_id_range.py` (apply, idempotent, fail closed, both fusion
  argument structs carry the range) and patch-lint.

## Native llama.cpp comparison

Native llama.cpp b11402 has no range form of MUL_MAT_ID. Nothing builds the variant unless 1283 is switched on, and
the build that carries it is equal to the BigCherry production build in text and speed, whose standing against
native llama.cpp is recorded with the patches that produce it (see 1334's README). There is therefore no gain or
loss against native from this patch by itself.

## What is not claimed

- Duplicate expert ids within one token are not supported on the MMQ path (as for the ordinary op: one row per
  token and expert).
- The scale / bias fused forms stay unfused for range nodes.
- The repacked-weights CPU path has the translation but no dedicated test case yet.
- The op-level test does not exercise the fused launches; those are covered only end to end.
