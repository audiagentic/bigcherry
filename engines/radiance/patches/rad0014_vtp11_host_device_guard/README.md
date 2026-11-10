# RAD0014 — VTP11 BF16 host/device activation-domain guard

**State: untested.** This is an experimental, separately selectable source
patch. It is not in the production serving-core patch set and is **not**
evidence of a working IQ6R calibration, a safe GPU GDN implementation, or
an end-to-end quality result.

## Reproducer and diagnosis

VTP10 R9700 TP1, original Qwen3.8-27B BF16, `--placement layer_offload`,
repeatedly faults in `rad_store_f32` from
`avx_gdn_conv_prep_v2._omp_fn` while creating calibration Hessians.
`OMP_NUM_THREADS=1` also faults. Removing libavx produces zero Hessians
and an extended futex wait, not a solution.

The v1.17.0 source permits an operation whose weights are all placed on
`Site::Host` to choose its CPU kernel
(`core/runtime/ctx.cpp:406-435`). The BF16 GDN architecture declares
its `gdn_in`, `gdn_q/k/v` and `gdn_g/beta` buffers using `decl_b`,
whose default domain is `RAD_DOMAIN_DEVICE`
(`arch/common/rad_arch.h:398`). Its `rw` registrations name those
buffers. `Ctx::bind_buffers` binds such buffers into a device allocation;
`Ctx::issue_body` passes their pointers to the selected kernel. The
host GDN writer has no device->host staging step. This is a *static,
source-backed path to an invalid host write* consistent with the debugger
trace. The faulting pointer's provenance has not yet been captured
dynamically.

## Patch design

- At architecture declare, capture whether any op's declared read or
  write buffer lives in `RAD_DOMAIN_DEVICE` in `OpInfo`. The CPU-only
  backend is not constrained, because its device allocation is host
  memory.
- The placement planner refuses a host-site candidate for such an op
  **when a GPU kernel exists for every band**. Its weights therefore
  take the existing host-pinned + device-staged path if they cannot fit
  in VRAM. The offline `weight_footprint` uses the same criterion.
- If an op has no device implementation, do not pretend that staged
  weights can make it work: retain its original host-only classification
  and reject the unsafe GPU plan with a clear bind-time error.
- Explicit `set_op_domain(HOST)` and an eventual host issue are
  fail-closed against a device-domain buffer. These are guard rails for
  unusual calls and incomplete op declarations, not hidden fallbacks.
- No pinned upstream files, kernels, production flags or canonical
  binaries are changed. GDN on gfx1201 still needs a numerical
  self-test: its row is marked *unvalidated* in the pinned source.

## Card-free evidence (2026-11-10)

- 10/10 exact managed edits applied to five pinned v1.17.0 files, and
  10/10 idempotent on reapplication.
- GCC 14 `-std=c++23 -fsyntax-only` passed each changed source unit:
  `core/build/rad_builder.cpp`, `core/place/planner.cpp`,
  `core/runtime/ctx.cpp`, and `core/runtime/issue.cpp`. Isolated
  source copy: `task-scoped scratch source`.
- `tools/tests/patch/test_vtp11_host_device_guard.py` checks unique
  guards, atomic fail-closed drift behavior, idempotence and key
  staging/domain conditions. This is a mechanics test, not an engine
  execution test.

## Outstanding acceptance

1. Build the experimental `radiance-vtp11-bf16` recipe without modifying
   existing serving binaries; run CPU/backend and planner regression tests.
2. Under **our own exclusive GPU lease**, run bounded R9700 TP1
   `--calib-layers 0:1` on the same original BF16 checkpoint and the
   existing two-document smoke corpus; capture kernel selection and
   placement/VRAM receipts, including the staged site.
3. Confirm both documents finish and RADGRAM2 Hessian schema, dimensions
   and finite statistics are valid. If the device GDN row gives wrong
   results or stalls, investigate it rather than enabling unvalidated
   fallback by default.
4. Separately gate representative 64-layer calibration, BF16 numeric
   reference and a complete IQ6R model-quality/performance comparison.

The small smoke corpus is solely for validating the calibration pipeline,
never a valid quantisation-quality calibration dataset.
