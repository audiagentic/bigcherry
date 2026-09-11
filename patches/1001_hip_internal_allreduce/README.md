# 1001: enable the internal (non-RCCL) AllReduce on HIP

## Scope

Removes the `GGML_USE_HIP` compile-out guard on `allreduce.cu`'s
pinned-host-memory AllReduce, substitutes `__builtin_amdgcn_s_sleep(4)` for
CUDA's `__nanosleep(100)` in the cross-GPU spin-wait, and maps the four HIP
host-mapped pinned-memory alloc APIs the implementation needs.
`GGML_CUDA_ALLREDUCE=internal` selects this path over RCCL at runtime.

## Why / evidence

Full real-hardware evidence (correctness, 3-arm decode performance,
prefill-regression warning, production recommendation) is already recorded
in `SUMMARY.md` -- do not duplicate it here. Summary: **+17.33% decode**
vs this project's own RCCL path at exact precision, but **-32% to -34%
prefill regression** -- this is why patch 0840 (GP03, validated) exists,
to dispatch per-call between this path and RCCL rather than picking one
globally.

## Upstream / provenance

Cherry-picked from open upstream PR
https://github.com/ggml-org/llama.cpp/pull/27825.

## Known limitations

**Gap found 2026-09-11 (process audit), closed same day (real re-sweep)**:
this patch is tagged `optimization` and carries `state = "validated"`,
with unusually thorough real-hardware A/B/C evidence (see SUMMARY.md) --
but every prior comparison arm was BigCherry-internal
(`GGML_CUDA_ALLREDUCE=nccl` was still BigCherry's own RCCL path, not
unmodified upstream llama.cpp). Re-ran a genuine 3-arm sweep same day.

**Real 3-arm sweep (2026-09-11, Brutus, dual gfx1100 `HIP_VISIBLE_DEVICES=0,1`,
`-sm tensor`)**: `llama-bench`, `tierB-qwen9b-q6k` (Qwen3.5-9B-Q6_K, this
family's cross-architecture reference lane), decode only (`-n 128 -p 0`),
`-r 5`, isolated scratch clone at pin `28ff0958291ce3465fabd7bd679d4b0edd742bd9`
(controller commit `f91db8ce`), same isolated-materialization methodology
as RD19's re-sweep (see `1200_rd19_single_gpu_meta_bypass/README.md`):

| arm | tg128 (t/s) | stddev |
|---|---:|---:|
| native llama.cpp (unmodified, default RCCL) | 102.61 | 3.07 |
| BigCherry baseline (default RCCL, 1001 excluded) | 96.52 | 1.15 |
| **BigCherry + 1001** (`internal`, `AR_BF16_THRESHOLD=0`) | **113.68** | **1.04** |

**Two real findings, not one**:
1. This patch's own claim holds up strongly against a genuine external
   baseline: **+17.8% over BigCherry's own RCCL baseline, +10.8% over
   native llama.cpp** -- closely matches (and slightly exceeds) the
   historical +17.33%-vs-BC-RCCL claim in SUMMARY.md.
2. A secondary, unexpected finding: BigCherry's own RCCL-default baseline
   measured **~5.9% slower than native llama.cpp** in this exact config
   (96.52 vs 102.61 t/s) -- though native's stddev (3.07) is high enough
   relative to the gap that this should be treated as a real signal worth
   investigating, not a settled fact from 5 reps alone. Not a defect in
   this patch (1001 doesn't touch the RCCL path at all); flagged as a
   separate, real observation about BigCherry's framework overhead on the
   default provider in this workload, worth its own investigation
   elsewhere rather than folded into this patch's own evidence.
