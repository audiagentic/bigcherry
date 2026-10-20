---
id: RAD15
order: 15
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-20T12:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Engine-side MoE decode rows: the worker GEMVs as kernel-plugin rows for gfx1100 (libr3) and gfx1201 (bc-moe-gfx12)

## Description

The fast FP8 and W4 decode GEMVs (`lane_moe_gemv_h2`, `lane_moe_gemv_w4h2`; RAD14) run only inside the moe-remote worker. The engine's own MoE decode still uses libr4d's one grouped-GEMM row per format, `moe_gemm_fp8a8` / `moe_gemm_w4a8` (16/32-row WMMA tiles, NOTUNE). Radiance resolves every op by scanning the `--kernels` plugins in order: the first plugin with a matching row wins, and an op's schema is fixed by the first plugin that declares it (core/plugin/registry.h, select.cpp at 7a7707a). So the GEMVs can serve the engine with NO engine patch, as extra `moe_gemm_q` rows restricted to small decode steps, with libr4d serving everything else.

Two targets:
- gfx1100: rows in libr3, already a plugin that overrides libr4d rows. Speeds every XTX engine: Qwen3.8-27B on two XTX, the 2xXTX Flash-Next profile, and an XTX-primary plus R9700-worker layout. Measured on one XTX for a 16-row 35B request: grouped GEMM 260 us, packed-f16 GEMV 149 us.
- gfx1201: a new component `bc-moe-gfx12` listed before libr4d, on native `v_dot4_f32_fp8_fp8`. On the R9700, libr4d's grouped GEMM is 135 us against our f16 GEMV's 115 us on captured requests, and synthetic 16-row decode is near the DRAM floor (GPT: about 30% headroom at 16 rows; 103 us against a ~72 us floor at 4 rows), so this row is go/no-go on measurement.

## Steps

1. Fix `tools/lab/radiance/kbench_step.py` attribution: join each graph instance to its exact selected row and geometry; exclude vision and prefill; count MTP separately.
2. R9700 baseline: engine-alone `moe_gemm_q` gate/up and down under rocprofv3 on live decode (prefix cache off), 1 and 4 streams. Confirm whether gfx1201 has FP8 WMMA (ISA and libr4d disassembly).
3. gfx1100 rows in libr3: `moe_gemm_q` rows for fp8a8 and w4nl64a8h, schema-identical to libr4d's, constrained to small decode steps; the GEMVs adapted to the op's operands (sorted/off, gate/up part pitch, scratch). `--debug-selection` must show the row for decode and libr4d for prefill.
4. A/B with rad-kbench (`--kernels libr3:libr4d:libref --op moe_gemm_q --all-rows`) and on captured requests; then XTX engine stream A/B (27B and Flash-Next 2xXTX).
5. gfx1201 row (`bc-moe-gfx12`, dot4) only if step 2 shows at least 20% headroom.
6. Topology A/B: XTX tp2 engine with the new libr3 rows plus the R9700 as worker, against the current R9700 engine plus XTX workers.
7. Qualification: fix the broken W4 CPU reference in the worker harness first; then CPU reference, KL/top-1, MTP acceptance at 1/2/4/6 streams, and at least 10% end-to-end decode before default-on.

## Validation

`--debug-selection` shows the new row for decode-sized `moe_gemm_q` and libr4d for prefill; rad-kbench time against libr4d at the model's shapes; captured-request replay on identical inputs; KL unchanged against the same engine on libr4d rows; MTP acceptance unchanged; matched stream A/B at least 10% decode.

## Notes

No engine patch: plugin rows only (the plugin hierarchy is absolute, first match wins). Risks: the schema must match libr4d's exactly or the plugin is refused at load; the W4 packed layout is libr4d-private (verify byte identity); selection is by host-side geometry only (no device-to-host sync for routing statistics). Related: RAD14 (worker GEMVs), RCS05 (backend seam), RCS10 (calibrated crossovers).

## Change Log

- 2026-10-20T12:00:00+00:00 (created-by): Created by agent
