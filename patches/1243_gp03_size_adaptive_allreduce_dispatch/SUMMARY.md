# 1243_gp03_size_adaptive_allreduce_dispatch: size-adaptive per-call AllReduce provider dispatch

**Status:** untested
**Group:** upstream-fixes
**Plan item:** GP03

Deliberately not "validated" despite the real hardware evidence below --
see the safety gap section near the end of this document.

## What it does

Requires 1001_hip_internal_allreduce. Adds a per-call size-adaptive override
on top of the existing `GGML_HIP_DISPATCH` reduce-plan mechanism:

- `ggml_backend_cuda_comm_init_internal()` brings up a secondary NCCL/RCCL
  communicator set alongside the internal `ar_pipeline`, best-effort, so a
  per-call "rccl" decision has real communicators to use. The internal path
  stays the committed default (`comm_ctx->try_allreduce`); a failure here
  does not affect it.
- `ggml_backend_cuda_comm_reduce_plan(nbytes)` (previously argument-less,
  always "auto" unless overridden) now routes reductions at or above
  `GGML_HIP_REDUCE_RCCL_THRESHOLD` bytes (default 256 KiB) to "rccl", and
  everything smaller stays "auto" (internal). `GGML_HIP_REDUCE_PLAN` keeps
  working as an explicit override, taking priority.
- The call site in `ggml_backend_cuda_comm_allreduce_tensor()` computes the
  real tensor byte count and passes it through.

## Why

Patch 1001's own validated numbers show the internal path is a real decode
win (+6.9% tg) but a severe prefill regression (-32% to -34% pp) versus
RCCL -- both effects of the same single committed-at-init provider choice.
This patch keeps the win and removes the regression by choosing the
provider per call instead of once per session.

## Validation evidence (Brutus, 2x RX 7900 XTX / gfx1100, 2026-09-02)

Real llama-bench runs, `HIP_VISIBLE_DEVICES=0,1`, `GGML_CUDA_ALLREDUCE=internal`,
`GGML_CUDA_AR_BF16_THRESHOLD=0`, `-sm tensor -fa on -b 2048 -ub 512`, r=3,
against patch 1001's own SUMMARY.md baselines:

| metric  | RCCL baseline | internal-only baseline | this patch (dispatch) |
|---------|--------------:|------------------------:|-----------------------:|
| pp512   |       1504.67 |          998.21 (-33.7%) | 1502.49 (matches RCCL) |
| pp2048  |       1447.58 |          980.64 (-32.3%) | 1451.17 (matches RCCL) |
| pp4096  |       1431.77 |          972.08 (-32.1%) | 1435.27 (matches RCCL) |
| tg128   |         33.69 |           36.01  (+6.9%) | 37.23 (+10.5%, decode win retained and improved) |

Also validated on the RCCL-viable heterogeneous 2-GPU subset (both devices
on CPU-direct PCIe root ports, no device-3 involvement): {0,2} (XTX+R9700,
gfx1100+gfx1201) pp512=1205.82/tg32=29.58; {1,2} pp512=1283.78/tg32=30.46 --
both clean, dispatch worked as designed, no crash.

## Root-cause note from development (not a defect in the patch itself)

During development, the secondary NCCL init appeared to silently never take
effect (`comms.size()==0` despite `backends.size()==2`). Root cause: the
source tree under test was missing 1001 entirely, so
`ggml_cuda_ar_pipeline_init()` always hit its CUDA-only stub (`nullptr` on
HIP) -- `ret->ar_pipeline` was never truthy, so this patch's added code
(guarded by `if (ret->ar_pipeline)`) never executed. Once 1001 was applied,
the mechanism worked immediately with no further changes. Recorded here
because it is the reason this patch declares an explicit `requires` on
1001 in `patch.toml` rather than assuming build-order handles it.

## WARNING: known unresolved safety gap -- device 3

Tested directly against a topology including physical device 3 (RX 6900XT,
this box's chipset-routed GPU, permanently lacking PCIe AtomicOps
completion capability per HI138): the secondary `ncclCommInitAll()` this
patch adds reports spurious success (`rc=0`, `comms` populated), because
communicator init alone does not exercise the atomics path required by a
real collective. This patch's own admissibility check
(`comms.size() == backends.size()`) therefore also passes, and the process
hard-aborts inside `ggml_backend_cuda_comm_allreduce_nccl` on the first
real pp-sized reduction -- a crash, not a fallback, and strictly worse than
not having this patch at all on any device-3-inclusive topology.

Patch 1225 (`1225_hi85_nccl_heterogeneous_arch_guard`) does not protect
against this: it only guards the original `ggml_backend_cuda_comm_init_nccl()`
call site, not the new secondary init this patch adds inside
`ggml_backend_cuda_comm_init_internal()`.

**This patch must not be promoted into any default patch-set** until either
GP02 (docs/planning/active/gpu-collectives/GP02.md) lands and this patch is
updated to consult the same guard before its secondary `ncclCommInitAll()`
call, or an equivalent topology check is inlined here directly. Full
evidence: docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md (2026-09-02
addendum).

## Upstream / provenance

Project-original enhancement on top of 1001's upstream backport, not itself
sourced from an upstream PR.
