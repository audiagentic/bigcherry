---
id: PRBE54
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:14.838615+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-KV-001: Dedicated F16 dequant path for Q4/Q5 KV

## Description

TODO. Dedicated F16 dequant for Q4_0/Q4_1/Q5_0/Q5_1 KV cache types. Relevance at b11126: confirmed still generic/scalar -- ggml/src/ggml-cuda/convert.cu's templated `dequantize_block<qk,qr,dequantize_kernel_t,dst_t>` (lines 8-40) is a per-thread scalar-callback template (calls `dequantize_kernel(vx, ib, iqs, v)` then writes v.x/v.y individually via `ggml_cuda_cast`), used for the general quant types. Only Q8_0 has a dedicated vectorized half2 fast path: `dequantize_block_q8_0_f16` (lines 43-80) uses `half2 * y2` + `__hmul2` SIMD multiply. No equivalent half2-vectorized function exists for Q4_0/Q4_1/Q5_0/Q5_1 at this pin -- item's premise holds, not upstream-absorbed.

TODO, corrected scope. GPT-corrected premise: Q4_0 and Q4_1 already have dedicated (non-generic-template) dequant kernels at b11126 -- `dequantize_block_q4_0`/`dequantize_block_q4_1` (ggml/src/ggml-cuda/convert.cu:85,113), NOT the generic templated `dequantize_block` this item originally assumed for them. Only Q5_0/Q5_1 remain on generic contiguous dequant (`dequantize_block_q5_K` at :163 is a different, already-specialized K-quant, not the Q5_0/Q5_1 legacy types this item targets -- verify Q5_0/Q5_1 specifically are NOT specialized, separately from Q5_K). More importantly, `ggml_get_to_fp16_cuda()` (convert.cu:547) is a GLOBAL type->function table with no call-site/use-context parameter -- replacing its Q4/Q5 entries cannot be scoped to "KV-only" without affecting every other caller (e.g. weight dequant) of the same function.

## Steps

1. Confirm no Q4/Q5-specific vectorized function exists: `git -C work/upstream/llama.cpp.git grep -n 'dequantize_block_q4\|dequantize_block_q5' b11126 -- ggml/src/ggml-cuda/convert.cu` (expected: no hits beyond the generic template instantiation lines -- verify before implementing).
2. Read the full `dequantize_block` template (convert.cu lines 8-40) and the per-type `dequantize_kernel_t` scalar callbacks (likely in ggml/src/ggml-cuda/dequantize.cuh) for Q4_0/Q4_1/Q5_0/Q5_1 to confirm they only produce one float2 pair at a time with no explicit half2 arithmetic.
3. Model the new vectorized path on `dequantize_block_q8_0_f16`'s pattern: a `dequantize_block_q4_0_f16`/`_q4_1_f16`/`_q5_0_f16`/`_q5_1_f16` (or a shared templated variant parameterized on qk/qr/nibble-unpack) that loads a block once and does 2-wide `half2`/`__hmul2` conversion instead of scalar `dequantize_kernel` calls, output type F16 (since KV cache F16 dequant is the item's target, not arbitrary dst_t).
4. Wire the new kernel(s) into the same lookup/dispatch table `dequantize_block_q8_0_f16` is selected from (grep `dequantize_block_q8_0_f16` call sites in convert.cu/convert.cuh and ggml_get_to_fp16_cuda to find the type-to-function table) -- add Q4_0/Q4_1/Q5_0/Q5_1 entries pointing at the new functions, gated so this only replaces the KV-cache dequant path (to_fp16_cuda_t used by fattn-common.cuh's K/V staging conversion, ggml/src/ggml-cuda/fattn-common.cuh:1033/1041/1067/1076) and does not change the generic dequant path used elsewhere for the same types (e.g. weight dequant), unless benchmarking shows the generic path is also safe to replace.
5. Preserve the existing generic `dequantize_block` template as fallback for any type/shape combination the new function does not cover (e.g. odd needs_check tail handling -- copy `dequantize_block_q8_0_f16`'s `need_check` template-bool pattern).
6. Add test-backend-ops CPY/GET_ROWS-equivalent correctness cases (exact numerical match against the existing scalar path) for each of the 4 types, plus a KV-shaped MUL_MAT/FLASH_ATTN_EXT case exercising the dequant through the real FA K/V staging path.
7. Benchmark dequant kernel time + PP/TG + memory vs Q8_0/F16 controls at the context lengths in the item description; promote per-type only where AMD (gfx1100/gfx1201) shows a repeatable win.

1. Confirm precisely which of Q4_0/Q4_1/Q5_0/Q5_1 lack a half2-vectorized fast path: `git -C work/upstream/llama.cpp.git grep -n 'dequantize_block_q4_0\|dequantize_block_q4_1\|dequantize_block_q5_0\|dequantize_block_q5_1' b11126 -- ggml/src/ggml-cuda/convert.cu` and read each found function body for half2/__hmul2 usage (Q4_0/Q4_1 exist per this pass's grep but their vectorization level is unconfirmed; Q5_0/Q5_1 need the same check).
2. Do NOT modify ggml_get_to_fp16_cuda()'s global table. Instead add a FA-specific selector `ggml_get_to_fp16_fattn_cuda(ggml_type type)` that defaults to calling `ggml_get_to_fp16_cuda(type)` for every type except the Q4/Q5 KV types being optimized, where it returns the new half2-vectorized variant.
3. Call `ggml_get_to_fp16_fattn_cuda()` only at the K/V conversion call sites in ggml/src/ggml-cuda/fattn-common.cuh (real anchors: lines ~1033/1041/1067/1076 per prior plan draft -- verify exact line numbers at b11126 before finalizing) -- every other caller of ggml_get_to_fp16_cuda() (weight dequant, etc.) is untouched.
4. Model any new vectorized function on `dequantize_block_q8_0_f16`'s pattern (half2/__hmul2, block-once load) for whichever of Q4_0/Q4_1/Q5_0/Q5_1 are confirmed (step 1) to still be scalar.
5. Add test-backend-ops correctness cases (exact numerical match against the existing path) for each newly-vectorized type, plus a KV-shaped FLASH_ATTN_EXT case exercising the new FA-specific selector.
6. Benchmark dequant kernel time + PP/TG + memory vs Q8_0/F16 controls at the context lengths in the item description; promote per-type only where AMD (gfx1100/gfx1201) shows a repeatable win.

## Detailed Solution & Technical Design

New half2-vectorized dequant kernels for Q4_0/Q4_1/Q5_0/Q5_1, following the existing Q8_0-specific precedent (`dequantize_block_q8_0_f16`) rather than modifying the generic templated `dequantize_block`. Hooked into the same to_fp16_cuda_t dispatch table used by FA's K/V F16 staging conversion (fattn-common.cuh), so it only affects the KV dequant path this item targets. Exact/tolerant numerical semantics preserved: same dequant formula, just vectorized execution.

## Code Samples & Guidance

Target anchors (verify exact text before writing Edit()): ggml/src/ggml-cuda/convert.cu lines 8-40 (`dequantize_block` generic template, insertion point for new functions nearby) and lines 43-80 (`dequantize_block_q8_0_f16`, the pattern to replicate). New patch patches/<order>_prbe54_kv_dequant_half2/patch.toml:
```toml
schema = 1
id = "<order>_prbe54_kv_dequant_half2"
order = <next available>
state = "untested"
kind = "enhancement"
origin = "local"
backend = "hip"
plan-ids = ["PRBE54"]
requires = []
conflicts = []
requires-options = []
forbids-options = []
subsystems = ["kv-cache", "dequant"]
hardware = ["amd"]
validation-architectures = ["gfx1100", "gfx1201"]
backends = ["hip"]
```
patch.py skeleton:
```python
from bigcherry.patcher import Edit, FilePatch

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/convert.cu",
        description="PRBE54 vectorized half2 F16 dequant for Q4_0/Q4_1/Q5_0/Q5_1 KV",
        edits=(
            Edit(
                id="prbe54-vectorized-dequant-fns",
                anchor=r"<TODO-VERIFY: exact text after dequantize_block_q8_0_f16, from step 2/3 reading>",
                rationale="add half2-vectorized dequant functions for Q4/Q5 KV types, modeled on the existing Q8_0 fast path",
                mode="insert_after",
                text=r"<TODO-VERIFY: new dequantize_block_q{4,5}_{0,1}_f16 functions>",
                guard=r"dequantize_block_q4_0_f16",
            ),
            Edit(
                id="prbe54-dispatch-table",
                anchor=r"<TODO-VERIFY: to_fp16_cuda_t / ggml_get_to_fp16_cuda dispatch table entries>",
                rationale="route KV-cache Q4/Q5 F16 conversion through the new vectorized functions",
                mode="replace",
                text=r"<TODO-VERIFY: updated dispatch table>",
                guard=r"BIGCHERRY_PATCH_HIT patch=prbe54",
            ),
        ),
    ),
]
```

## Files

ggml/src/ggml-cuda/convert.cu (and convert.cuh if the dispatch table lives there); ggml/src/ggml-cuda/dequantize.cuh (reference only, scalar callbacks); tests/test-backend-ops.cpp; patches/<order>_prbe54_kv_dequant_half2/{patch.toml,patch.py,SUMMARY.md}

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay <id> --source bigcherry-tuning`, exact/tolerant dequant unit tests per type. Hardware (Brutus, not run here): Q4_0/Q4_1/Q5_0/Q5_1 long-context Qwen vs Q8_0/F16/decode controls at 8K/32K/64K/128K and q_rows 128/512/2048, dequant kernel time + PP/TG + memory, via `python -m bigcherry.patch.validation_campaign`.

## Effort & Risk

M / medium -- new kernel code in a hot KV path; correctness risk mitigated by exact-numerical-match requirement against the existing scalar reference.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only a KV type with output/dequant correctness and repeatable AMD PP/TG benefit versus Q8_0/F16/decode controls; retain generic path otherwise.

## Notes

Supersedes: RD64
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd64

2026-09-24 relevance at b11126: confirmed Q4/Q5 KV dequant still uses generic scalar `dequantize_block` template (convert.cu:8-40); only Q8_0 has a dedicated half2-vectorized fast path (`dequantize_block_q8_0_f16`, convert.cu:43-80). GPT design request req_bf8959fb36d248e4 (batched with PRBE53, submitted, still running as of this pass -- PRBE53 was dispositioned directly from source evidence without waiting; this PRBE54 plan was likewise written directly from source evidence, GPT response to be cross-checked opportunistically).

2026-09-24 GPT review req_2b65d50ebe9547fd applied: NOT-READY -- corrected premise (Q4_0/Q4_1 already have dedicated dequant_block_q4_0/_q4_1 kernels, convert.cu:85/113, verified); scoped fix to a new FA-specific ggml_get_to_fp16_fattn_cuda() selector rather than editing the global ggml_get_to_fp16_cuda() table which has no call-site context.

## Change Log

- 2026-09-09T10:57:14.838615+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:28.184233+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.372555+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.168472+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:11:58.576713+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031217_repaired-four-more-active-succ_7909
- 2026-09-10T03:12:17.962856+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:02.800907+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:42:19.256026+00:00 (updated-by): Updated: section:description, section:steps
- 2026-09-24T04:42:24.254176+00:00 (updated-by): Updated: section:notes
