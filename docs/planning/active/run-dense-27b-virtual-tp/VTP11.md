---
id: VTP11
order: 10
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Original BF16 calibration: GDN kernel routing and host-buffer ownership

## Motivation and evidence (2026-11-10, card-free audit)

VTP10 established a repeatable `SIGSEGV` in `rad_store_f32` called by
`avx_gdn_conv_prep_v2._omp_fn` during TP1 R9700 original-BF16
calibration. VTP10's `--kernels libr4d,libref` alternative did not pass:
zero documents and zero Hessians after >210 seconds, with a long
futex wait and low GPU utilization. Neither result qualifies IQ6R.
The original full-shard BF16 fingerprint is already recorded in VTP06/VTP10.

The pinned Radiance source snapshot has these important facts:

- `libr4d/r4d_rows.cpp` `cGdnConvPrep` constrains convolution width 4,
  key/value head widths 128 and chunk 64 (around line 4616).
- Original Qwen3.8-27B `config.json` declares width 4, key/value head
  widths 128, 16 key heads and 48 value heads; `arch/qwen35_bf16/qwen35_bf16.cpp`
  sets chunk 64. Thus declared GDN geometry is compatible with the device
  row; the reason it ran on AVX is **not established**.
- `libr4d/r4d_rows.cpp` marks the device `gdn_conv_prep_w4_h128_bf16`
  entry unvalidated (around line 7870). That entry checks operand layouts,
  dtypes and strides in
  `libr4d/r4d_gdn_conv_w4_h128_bf16.hip` (around lines 524-602).
- `arch/common/rad_block_gdn.h` issues `gdn_conv_prep` with positional
  operands including q/k/v/gate/beta outputs (around lines 318-336).
  `libavx/avx_gdn.cpp` writes these outputs in several parallel phases
  (around lines 300-444). The captured crash does **not** identify which
  tensor pointer, stride or address space is invalid.
- `core/engine_bringup.cpp` describes legitimate host fallback for some
  device misses (near lines 68-74), but this does not by itself prove
  that host fallback and live device-output ownership are safe in calibration.

These are static-source findings only; no kernel fix, new binary, or
hardware result is claimed.

## Scope and design

Find a *minimal*, fail-closed explanation and correction for the BF16
GDN calibration failure. Keep pinned upstream untouched; apply any
necessary engine change only through the managed BigCherry patch route,
with explicit ownership and numerical tests. Do not bypass failure by
globally disabling AVX, changing serving defaults, or silently falling back
from a device miss.

1. **Selection trace (card-free first):** locate the path binding the
   GDN op's declared bands to the selected device/host row, plus
   placement and copy transitions. Record the selected row's exact
   constraints and the rejected candidates, not just the plugin name.
   Add a focused test or diagnostic for the Qwen27 GDN geometry.
2. **Crash contract:** instrument the narrow GDN boundary, in an isolated
   debug build, to report for each of the 16 operands: presence, dtype,
   rank, shape, strides, memory domain/owner, allocation bounds and
   intended read/write role. Do not dereference an unvalidated host
   address. Compare the AVX output pointer in the failing call against
   its arena allocation and lifetime; establish the invalid store's
   provenance before changing memory routing.
3. **GPU correctness gate:** independently qualify the existing device
   `gdn_conv_prep` row (currently explicitly unvalidated) against the
   BF16 reference at matching width/head/chunk geometry, including
   startup/replayed cache states and variable sequence lengths. A
   successful compile or kernel entry lookup does not qualify it.
4. **Scheduler/hang gate:** investigate the zero-progress, no-AVX run
   separately. Identify the exact pending work, owning thread and
   dependency, host/GPU buffer transitions and whether any event waits
   can be unsatisfied. Require bounded timeouts with receipts; do not
   infer success from absence of a crash.
5. **Minimal Hessian smoke:** under a verified *exclusive* GPU lease and
   with GPUs/processes idle, use the original fingerprinted BF16 source,
   a known 2-document smoke corpus and `--calib-layers 0:1`. Pass only
   if both documents complete, nonempty RADGRAM2 output has valid
   schema/shape, memory bounds hold, and the reference forward path
   is numerically acceptable. This is plumbing proof, NOT calibrated Q6.
6. **Scale gate:** only after step 5, plan representative corpus, bounded
   64-layer Hessian ranges and conversion, then BF16-vs-IQ6R
   logits/KLD/perplexity and full-engine automatic dispatch trace. Preserve
   Q8 serving as the measured baseline (VTP10).

## Safety / experiment ordering

- Card-free selection/ownership inspection and small tests first.
- No lease available/owned => no GPU runs or live process manipulation.
- Use one existing worktree `exp/xtx27-prefill-opt`; no new worktree or
  upstream/vendor edits. Scratch results under the existing VTP work root,
  with commands, binary/model fingerprints, exit codes and per-stage logs.
- Avoid concurrent changes to the production engine, running service or
  another agent's lease. Only interrupt a specifically owned test process.
- Do not treat the tiny smoke corpus as representative IQ6R calibration.

## Validation and acceptance

Pending. Acceptance requires (a) an exact, trace-backed cause of the
AVX invalid store, (b) correct and bounded GDN operation or an explicit
fail-closed refusal of unsupported placement, (c) completion and schema
validation of layer-0 calibration, (d) a documented reason for the prior
`libref`-only hang, (e) card-free regression tests, and (f) a separate
quality-gated plan for representative 64-layer calibration.

**Status:** diagnosis staged; no GDN patch and no Hessian smoke PASS.


## Experimental host/device domain guard — 2026-11-10

Pinned Radiance source confirms a **credible static crash path**: Ctx::bind_ops
selects host execution for operations whose weights were placed on Site::Host
while the Qwen3.8 BF16 GDN architecture uses the default RAD_DOMAIN_DEVICE
for its convolution inputs and q/k/v/gate outputs. Ctx::bind_buffers places
those in a device-only arena. The AVX writer receives the kernel output
pointers unchanged and can fault. The precise crashing pointer still needs a
run-time trace; this is a source-backed diagnosis, not a full fault proof.

An **opt-in** managed patch rad0014_vtp11_host_device_guard now carries each
op's declared device-buffer ownership to placement. Where a GPU row exists,
weights offloaded from VRAM are staged for GPU execution rather than driving
AVX kernels against GPU-only output memory. Host launch and explicit-host
remapping have fail-closed guards. A CPU-only backend retains its prior behavior.
The patch is packaged only in experimental radiance-vtp11-bf16 and is
**not added to production serving-core**.

Card-free evidence: 10/10 anchored edits applied to a scratch pinned-source
copy; all idempotent; 3 dedicated Python tests pass (including malformed
upstream anchor refusal), Ruff passes, and GCC 14 syntax checks passed for all
four changed C++ translation units. Full integration, GPU GDN numerical
qualification and layer-0 RADGRAM2 calibration remain **PENDING**.

First isolated build attempt stopped at BigCherry's dirty-repository
materialization gate (no C++ compilation); commit changes and retry from a
clean worktree. Logs under BC_WORK_ROOT/build-logs/vtp11-host-guard-isolated/.
Do not count that build attempt as evidence of a functional calibration path.
