---
id: VTP11
order: 10
plan: run-dense-27b-virtual-tp
state: superseded
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Original BF16 calibration: GDN kernel routing and host-buffer ownership

> Archived 2026-11-12 from branch `exp/xtx27-prefill-opt` (git tag `archive/exp-xtx27-prefill-opt`), which never merged. Its lab programs (tools/lab/radiance/xtx27_*), tests and early patch drafts are in that tag. Superseded by VTP14 (native Radiance TP3, completed 2026-11-12), which kept these programs as reference oracles only.

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
   if both documents complete, nonempty packed-F32 hessian_upper safetensors output has valid
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

**Status:** isolated fix and build passed; first layer-0 Hessian smoke PASS on R9700 TP1 and dual-XTX TP2. Representative corpus, independent GDN oracle and model quality remain open.


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
qualification and layer-0 Hessian calibration were **PENDING at that stage**.

First isolated build attempt stopped at BigCherry's dirty-repository
materialization gate (no C++ compilation); commit changes and retry from a
clean worktree. Logs under BC_WORK_ROOT/build-logs/vtp11-host-guard-isolated/.
Do not count that build attempt as evidence of a functional calibration path.


## 2026-11-10 — first completed BF16 calibration, both GPU configurations

**Result: layer-0 Hessian pipeline passes, independent numerical oracle pending.**

Experimental source radiance-vtp11-bf16 with managed
rad0014_vtp11_host_device_guard built successfully in isolation: all
materialize/configure/build stages succeeded. Build plan ID:
8dd8360cb45d3c86998a9026dcfad293; logs:
BC_WORK_ROOT/build-logs/vtp11-host-guard-isolated-v2/.
The patch was committed at 0acea3809; no production default changed.

Using an exclusive GPU lease and the original fully fingerprinted BF16
checkpoint (VTP06), both configurations ran the **same** 2-document,
185-token micro-corpus (SHA-256
3907585b5cb21b71baaeae584c730c577674732d3259f40dee809903667718c9),
--calib-layers 0:1, exact TP, layer_offload, context 512, batch 128,
BF16 KV and no speculation:

| Hardware | TP | Host pool | Exit | Hessians | Elapsed incl. load |
| --- | ---: | ---: | ---: | ---: | ---: |
| R9700 gfx1201 | 1 | 32768 MiB | 0 | 4 | 95.49 s |
| RX 7900 XTX gfx1100 x2 | 2 | 16384 MiB | 0 | 4 | 78.07 s |

Run-specific argv, selected GPU device, logs, SHA-256s and validation
receipts:
BC_WORK_ROOT/xtx27-prefill/vtp11-bf16-calibration-smoke-20261110-v1/
BC_WORK_ROOT/xtx27-prefill/vtp11-bf16-calibration-tp2-20261110-v1/

The engine wrote four Hessians at rank 0 in each run; dual-XTX rank 1
reported zero files, avoiding writer races. A full F32 element scan of
**every** matrix in both runs passed: valid packed-upper safetensors
schema and offsets, dimensions 5120/6144/17408/5120, exactly 185 rows,
no NaN/Inf and no negative diagonals.

Full-matrix same-corpus comparisons, TP2 dual XTX vs TP1 R9700:

| Layer-0 input | Relative L2 difference | Cosine |
| --- | ---: | ---: |
| input (GDN in-projection) | 1.0133e-7 | 1.000000 |
| linear_attn.out_proj.input | 3.0171e-5 | 0.9999999995 |
| post_attention.input (MLP) | 0.0022303 | 0.99999755 |
| mlp.down_proj.input | 0.0034094 | 0.99999419 |

Comparison receipt:
BC_WORK_ROOT/xtx27-prefill/vtp11-bf16-calibration-compare-20261110-v1.json.
These establish **cross-backend consistency**, not an independent
numerical oracle: the GDN GPU gdn_conv_prep_w4_h128_bf16 row remains
upstream-marked UNVALIDATED. The precise AVX crash pointer was not
recorded dynamically; source-level host/device ownership mismatch
plus same-run before/after recovery support, but do not independently
prove, pointer provenance. The libref-only hang remains unresolved.

**Do not use the 185-row outputs to quantise a production IQ6R model.**
They prove calibration plumbing, not representative 27B quality. VTP06
owns the diverse corpus, bounded 64-layer runs, conversion and BF16
quality comparison. The GPU lease was released; no KFD PIDs remained.


## Independent GDN oracle — benchmark coverage correction (2026-11-10)

The isolated build includes upstream \`rad-kbench\`, whose \`--op
gdn_conv_prep --kernels libr4d:libref --ref libref\` is a suitable
correctness ladder **only if checked GPU cases are nonzero**.
The first R9700 test using existing original-model-derived W8A8
RAD2 geometry completed with exit 0 but **0 checked / 28 skipped**.
The precise skip was \`dt_bias is a weight and the declaration bound
none to it\`. This is a **failed coverage gate**, NOT numerical success.
Attempting to pass the original HF safetensors *directory* directly
to rad-kbench refused \`mmap: No such device\` because it requires RAD2.

The root cause is in the benchmark: it advances its fixed-weight
cursor for absent optional schema weights. GDN's optional
convolution bias is *between* convolution weight and A_log;
the model declares only w/A_log/dt_bias. Upstream's op weight-position
map is generated sequentially, so using it unmodified alone would
not solve this missing-middle-slot geometry. Experimental
rad0015_vtp11_kbench_optional_weights counts the **present** fixed
weights before drawing, rejects ambiguous counts, and advances
only on present fixed-weight slots. It is a test-harness-only patch,
separate from production and the already passing rad0014 runtime.

Three focused tests, pinned-source patch/idempotence check, Ruff and
GCC 14 syntax passed. A numerical oracle result is **pending** an
isolated benchmark build and nonzero checked GPU cases, on both
gfx1201 and gfx1100. Do not interpret exit code alone as a pass.


## 2026-11-10 — three-card GDN cold-prefill oracle qualification

**Independent per-op seeded reference gate: PASS for tested prefill geometries.**
After rad0015 fixed kbench's absent optional convolution-bias cursor,
an opt-in isolated source radiance-vtp11-gdn-kbench built successfully.
The production serving recipe remains untouched. The new build plan ID
is a231db79c1d59b4e550585ccaf460281, build status succeeded,
with logs under
BC_WORK_ROOT/build-logs/vtp11-gdn-kbench-build-v1/.
The test-only patch is committed as 5600a10ed. Its three unit tests,
pinned-source application/idempotence, Ruff and GCC 14 C++23 syntax
all passed.

Using the existing W8A8 RAD2 container's real Qwen3.8-27B GDN geometry
(the GDN operator is BF16 h128, w4, chunk64), seeded kbench inputs,
--op gdn_conv_prep, --ref libref, and libraries
libr4d:libref for gfx1201 or libr3:libref for gfx1100:

| Gate | R9700 gfx1201 | XTX0 gfx1100 | XTX1 gfx1100 |
| --- | ---: | ---: | ---: |
| M=1–128, 8 cases per M, seed 20261110 | 64/64 PASS | 64/64 PASS | 64/64 PASS |
| M=1–512, 8 cases per M, seed 20261111, tolerance 0.001 | 80/80 PASS | 80/80 PASS | 80/80 PASS |
| Unchecked GDN cases | 0 | 0 | 0 |
| Failed or diverged GDN comparisons | 0 | 0 | 0 |
| Guarded-allocation out-of-bounds writes | 0 | 0 | 0 |
| Strict-sweep worst relative L2 | 0.0003186 | 0.0003186 | 0.0003186 |

**M=512** spans eight 64-token chunks and the exact h128/w4/chunk64
operator row was selected on each card. The 0.001 strict-sweep tolerance
is tighter than kbench's standard GDN relative-L2 allowance (0.02).
The third sweep tests a distinct seed, not a copied answer. It does
not cover an initialized convolution-history cache (has_init=1),
speculative rollback, complete BF16 graph outputs, or full-model logits.

All raw per-GPU argv/receipt/log/report artifacts for the tight
sweep are below
BC_WORK_ROOT/xtx27-prefill/vtp11-gdn-kbench-20261110-v5-m512-strict/.
A machine-checked aggregate is at validation.json there, recording
**240 actual GPU-vs-libref cases**, all passed with 0 skipped.
The earlier 0-checked/28-skipped benchmark is kept as negative evidence;
rad-kbench exit code 0 alone is never admission.

A separate proper-BF16 .rad container or an original-weight-loaded
operator oracle is needed to exclude any W8A8 artifact-dependent
differences. The cold-prefill numerical gate is **qualified for the
tested seeded fixtures**, not universally for all warm/decode states.
Keep both patches in isolated experimental source until the remaining
gates and integration/quality tests pass.

The GPU lease was explicitly released and no KFD processes remained.

## Change Log

- 2026-11-12T22:13:46.872064+00:00 (state-transition): State: in_progress → superseded
