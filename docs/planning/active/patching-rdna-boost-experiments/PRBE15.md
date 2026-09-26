---
id: PRBE15
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:30.186606+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate IMRoPE plus BF16 SET_ROWS fusion extension

## Description

TODO, rescoped. Patch 1004 (the item's stated dependency) is `rejected`: its SUMMARY.md records the rms_norm+mul+rope(+view+set_rows) fusion it claimed to add was already ancestral at this project's pinned base (verified already-applied at b10362 and b10502) -- it was a silent no-op, not a real capability gate. That base fusion IS present natively in b11126 (ggml/src/ggml-cuda/ggml-cuda.cu:3226-3261: rms_norm+mul+rope and rope+view+set_rows fusion patterns, gated by ggml_cuda_should_fuse_rope_set_rows at line 2666). Verified against real source: that gate function explicitly excludes two things PRBE15 wants to add -- (1) line ~2692-2694 `if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) return false;` blocks IMRoPE-mode ropes (IMRoPE support itself already exists in ggml/src/ggml-cuda/rope.cu:221/256/469/608 via `is_imrope`, just not wired into the fusion gate); (2) line ~2678 `if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) return false;` blocks BF16 SET_ROWS. So PRBE15's real target is extending the NATIVE upstream fusion gate, not "patch 1004" (which adds nothing). Relevance and design done directly against verified b11126 source (GPT gateway returned repeated retry-safe queue rejections, VAL-AGW-025, on this item's batch attempt; proceeded with the same rigor using confirmed anchors instead of resubmitting further).

## Steps

1. Confirm at implementation time (re-grep, since this project's pin may move) that ggml_cuda_should_fuse_rope_set_rows in ggml/src/ggml-cuda/ggml-cuda.cu still has the exact mode and dtype guards cited above.
2. BEFORE widening the fusion gate's dtype check to accept BF16: verified via grep that ggml_cuda_op_rope_impl()'s dst_type dispatch in rope.cu (lines ~632-690 and ~924-931) only branches on GGML_TYPE_F32/GGML_TYPE_F16 for dst_type -- there is no BF16 branch. Add explicit BF16 destination handling (using nv_bfloat16, matching the existing F16 branch pattern) in ggml_cuda_op_rope_impl() before touching the fusion gate's dtype guard, or the widened gate will dispatch into a path that does not exist and will hit the function's fallback GGML_ABORT.
3. IMRoPE enters rope_multi_cuda()/rope_multi (rope.cu, is_imrope branch) -- verify at implementation time whether that signature accepts row_indices/set_rows_stride the way rope_norm/rope_neox do for the fused SET_ROWS case; if not, extend rope_multi_cuda/rope_multi to accept and apply them, or explicitly scope this item to exclude IMRoPE+SET_ROWS fusion until that plumbing exists.
4. Extend the mode check to also accept GGML_ROPE_TYPE_IMROPE alongside NORMAL/NEOX only once step 3 is resolved; extend the dtype check to also accept GGML_TYPE_BF16 for set_rows->type only once step 2 is resolved.
5. Add a BIGCHERRY_PATCH_TRACE-gated GGML_LOG_WARN activation marker in ggml_cuda_op_rope_impl() (or the fusion call site) that fires only when set_rows != nullptr and the new IMRoPE/BF16 path actually launches -- this item currently specifies no marker at all.
6. Add correctness fixtures: IMRoPE-mode fused vs unfused, BF16 SET_ROWS fused vs unfused, plus negative fixtures (mode/dtype still excluded outside the widened set) -- test each new mode against fusion-disabled output.
7. Run graph capture/replay with the newly-accepted patterns; run false-positive fallback checks; then balanced timing only after correctness passes, comparing B (native, unextended gate) vs B+PRBE15 (extended gate).

## Detailed Solution & Technical Design

This is a small, surgical native-fusion-gate extension, not a fork port and not built on patch 1004 (which is dead weight -- do not resurrect it as a dependency). The change lives entirely in ggml_cuda_should_fuse_rope_set_rows's two guard conditions plus verifying/patching the SET_ROWS write path for BF16. No new kernel is required for IMRoPE (rope.cu already branches on is_imrope at runtime); BF16 SET_ROWS needs its own audit since the guard rejects it today specifically because the write path may assume F32/F16.

## Code Samples & Guidance

Anchor (verified in b11126, ggml/src/ggml-cuda/ggml-cuda.cu, inside ggml_cuda_should_fuse_rope_set_rows):
```
if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
    return false;
}
...
const int mode = ((const int32_t *) rope->op_params)[2];
if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
    return false;
}
```
Proposed replacement (BigCherry patch package patches/<order>_rd18_imrope_bf16_set_rows_fusion/patch.py using `from bigcherry.patcher import Edit, FilePatch`):
```python
Edit(
    id="rd18-dtype-gate",
    anchor=r"if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {\n        return false;\n    }",
    rationale="accept BF16 SET_ROWS output for the rope+view+set_rows fusion",
    mode="replace",
    text="if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16 && set_rows->type != GGML_TYPE_BF16) {\n        return false;\n    }",
)
Edit(
    id="rd18-mode-gate",
    anchor=r"if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {\n        return false;\n    }",
    rationale="accept IMRoPE-mode ropes for the rope+view+set_rows fusion; is_imrope is already a runtime kernel branch in rope.cu",
    mode="replace",
    text="if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_IMROPE) {\n        return false;\n    }",
)
```
patch.toml sketch: schema=1, id="12xx_rd18_imrope_bf16_set_rows_fusion", state="untested", kind="enhancement", backend="hip", experiment-contract="RD18-IMROPE-BF16-SETROWS-FUSION", requires=[], conflicts=[], validation-architectures=["gfx1100","gfx1201","gfx1030"].

## Files

ggml/src/ggml-cuda/ggml-cuda.cu (fusion gate, ~lines 2666-2696 and the fusion-dispatch call sites ~3244-3266); ggml/src/ggml-cuda/rope.cu (is_imrope kernel path, no change expected, verify only); ggml/src/ggml-cuda/ggml-cuda.cu's set_rows op handler for BF16 write-path audit; new patches/<order>_rd18_imrope_bf16_set_rows_fusion/{patch.toml,patch.py}; tests/test-backend-ops.cpp IMRoPE+BF16 SET_ROWS fused/unfused cases.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay <new-id> --source bigcherry-tuning`, plus a test-backend-ops case pairing ROPE(mode=IMROPE)+VIEW+SET_ROWS(BF16) fused vs unfused output equality. Hardware (Brutus, not run here): gfx1100/gfx1201/gfx1030, a real IMRoPE-using model (e.g. Qwen3-VL family) for activation + correctness, B vs B+PRBE15 causal timing on the incremental arm only.

## Effort & Risk

M effort, low-medium risk -- two guard conditions plus a BF16 write-path audit; main risk is the BF16 SET_ROWS write path silently truncating/misinterpreting data if not actually dtype-generic underneath.

## Standards

No hidden baseline promotion; causal dependency arm; exact fusion pattern.

## Acceptance Criteria

The exact extension is correct and captured; only the incremental arm meets the performance/evidence gate; 1004 remains explicitly unpromoted unless independently accepted.

## Notes

Supersedes: RD18
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd18

2026-09-24 relevance at b11126: TODO, rescoped off dead patch 1004 onto the native fusion gate at ggml-cuda.cu:2666-2696 (verified: mode guard excludes IMRoPE, dtype guard excludes BF16). GPT gateway returned VAL-AGW-025 retry-safe rejections on submission for this item; proceeded with direct source-verified design instead of a GPT round.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: verified via grep that ggml_cuda_op_rope_impl() in rope.cu only dispatches F32/F16 dst_type (no BF16 branch) -- widening the fusion gate alone would route BF16 SET_ROWS into a nonexistent/aborting path. Added required BF16 destination handling in ggml_cuda_op_rope_impl() and an IMRoPE row_indices/set_rows_stride plumbing check for rope_multi_cuda/rope_multi as prerequisites before widening the gate, plus a required activation marker (none was specified before).

## Change Log

- 2026-09-09T10:54:30.186606+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:39.895272+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.196499+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.902384+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:49:05.374902+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024925_rdna-successors-prbe1416-now_9529
- 2026-09-10T02:49:25.519390+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:28:47.844482+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:39:34.219151+00:00 (updated-by): Updated: section:steps, section:notes
