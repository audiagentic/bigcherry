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

**Gap found 2026-09-11 (process audit)**: this patch is tagged
`optimization` and carries `state = "validated"`, with unusually thorough
real-hardware A/B/C evidence (see SUMMARY.md) -- but every comparison arm
is BigCherry-internal (`GGML_CUDA_ALLREDUCE=nccl` is still BigCherry's own
RCCL path, not unmodified upstream llama.cpp). None of this patch's
evidence compares against native/vanilla llama.cpp. Per this project's new
rule (`docs/reference/patches/PATCH_AUTHORING.md`'s "`optimization` carries
a real validation obligation"), flagged for a decision (re-sweep including
a native-llama.cpp arm vs. accept the historical BigCherry-internal
evidence as sufficient, since 0840/GP03's own real evidence already
includes a native-llama.cpp RCCL arm and this patch is 0840's direct
dependency) rather than silently demoted.
