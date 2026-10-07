# Release patch set

Selection: --source bigcherry

- **bigcherry revision:** 8823f3a367f3344e42f67732ecd73347255d0512
- **llama.cpp revision:** b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea
- **source:** bigcherry
- **target:** b11474

48 patch(es) included.

---

# 0100_cmake_options: CMake options for HIP replay/serving build (HI02, PA27)

**Status:** validated
**Plan item:** none

## What it does

Adds GGML_HIP_DISPATCH_REPLAY and related build options to ggml/CMakeLists.txt
(with configure-time validation) and turns them into HIP-backend compile
definitions plus the production dispatch/replay source list in
ggml/src/ggml-hip/CMakeLists.txt. Sets the `_BC_HIP_SERVING_BUILD_PLUMBING`
non-cache marker that `0110_campaign_tune_record_build` reads to fail closed
if its own options are activated without this package applied.

PA27 narrowed this package to only what a replay/serving build needs; the
tune/record-only options, tuner source, and campaign-only diagnostics moved
to `0110_campaign_tune_record_build`. There is no SQLite link edit anywhere
in the split (the declaration-only, no-consumer GGML_HIP_AUTOTUNE_SQLITE
option this package used to carry was dead code and was removed).

## Why

Measured dispatch needs its own build switches, and an illegal build
combination must fail at configure time rather than produce a silently
incomplete or inert build: GGML_HIP_DISPATCH_REPLAY=ON with GGML_HIP=OFF, and
dispatch combined with GGML_CUDA_FORCE_MMQ/GGML_CUDA_FORCE_CUBLAS (which
would hide candidate families from measurement).

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI02).

---

# 0200_dispatch_hook: Route the dense matmul selector through measured dispatch (HI04)

**Status:** validated
**Plan item:** none

## What it does

Inserts a single guarded hook (ggml_hip_dispatch_mul_mat) at the top of upstream's ggml_cuda_mul_mat entry points; the hook returns false whenever it declines, so upstream's own ladder runs untouched. Also exposes the previously-static cuBLAS entry point so the BLAS candidate can reach it.

## Why

Upstream's selector decides and launches in one motion, so there is nothing to measure, store, or replay. A minimal, appended hook keeps the diff tiny and durable across releases while guaranteeing the native fallback is upstream's real code, not a reimplementation.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI04).

---

# 0300_mmq_forced_j: MMQ forced-J variant dispatch (HI06)

**Status:** validated
**Plan item:** none

## What it does

Splits upstream's mul_mat_q_switch_J into a scan (mul_mat_q_compute_J_best, lifted unchanged) and a launcher (mul_mat_q_launch_forced_J) that takes J as an explicit parameter, so a forced value can override the scan's answer while the native path stays identical.

## Why

The tuner needs to select and measure a specific MMQ tile width J instead of only ever seeing upstream's own scanned choice; separating the scan from the switch is the least invasive way to do that since launch_mul_mat_q already templates on J.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI06).

---

# 0400_mmvf_forced_block: MMVF forced block-size and accumulator-mode dispatch (HI07)

**Status:** validated
**Plan item:** none

## What it does

Threads a forced block-size/accumulator-mode value down through appended, defaulted parameters from ggml_cuda_mul_mat_vec_f to its launcher, touching only the call chain a forced value actually travels (replace_all edits with asserted match counts).

## Why

An earlier thread-local-override design was rejected because production replay builds would pay a per-launch read on the hottest path for a value that's always zero in production; an explicit parameter keeps the native path byte-identical to upstream.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI07).

---

# 0500_mmf_forced_nwarps: MMF forced-nwarps dispatch (HI08)

**Status:** validated
**Plan item:** none

## What it does

Same explicit-appended-defaulted-parameter shape as HI07, applied to MMF's three dispatchers (which share an identical signature/call tail); shared-memory sizes are recomputed from the forced nwarps immediately after the scan so allocation stays correct.

## Why

Needed so the tuner can force and measure a specific MMF nwarps value while leaving the native path byte-identical to upstream, without under-allocating shared memory for a forced value larger than native's choice.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI08).

---

# 0600_mmvq_geometry: Explicit MMVQ geometry variants (HI09 part 1)

**Status:** validated
**Plan item:** none

## What it does

Adds two defaulted template parameters (nwarps_explicit, rows_per_block_explicit) to the MMVQ kernel template; zero means derive geometry as upstream does (native instantiations unchanged), non-zero compiles a new geometry instance. Bounds are static_assert-checked in-kernel as a backstop.

## Why

MMVQ derives its geometry from calc_nwarps/calc_rows_per_block at compile time, so an alternative geometry needs genuinely new compiled code rather than a runtime switch, unlike MMQ/MMVF/MMF.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI09).

---

# 0650_mmvq_native_variant: Route a forced MMVQ geometry to its compiled instance (HI09 part 2)

**Status:** validated
**Plan item:** none

## What it does

Threads a forced-geometry struct down the existing native chain (ggml_cuda_mul_mat_vec_q -> mul_mat_vec_q_switch_ncols_dst) to the point where quantization/strides are already computed, diverging only at the launch call via ggml_hip_mmvq_find_instance.

## Why

Makes the geometry variants compiled by patch 0600 actually reachable, without duplicating upstream's quantization/stride logic (which would drift silently on every release). Refuses an unmatched geometry, MUL_MAT_ID width>1, and leaves fusion to the resolved instance rather than the forced path.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI09).

---

# 0700_coverage_counters: Family-entry instrumentation and coverage counters (HI13)

**Status:** validated
**Plan item:** none

## What it does

Adds counters at every real family entry point (not just the dense selector) to measure what fraction of matmul launches actually reach measured dispatch, since the graph optimizer calls MMVQ/MMVF directly for fused patterns, bypassing the dense selector.

HI168: counter calls and their reentrancy probes are compiled only with
`GGML_HIP_DISPATCH_DIAGNOSTICS`. Production retains family dispatch collection
without diagnostic counting. The upgrade edit also guards previously applied
hooks. This removes known instrumentation work; throughput parity still requires
a controlled hardware comparison.

## Why

Without this number, a tuning run's coverage of real model work is unknown, and 'we tuned the model' is an unverified assumption; test-backend-ops cannot produce this figure since it bypasses the graph optimizer entirely.

## Upstream / provenance

Local design, not in the original plan; added to answer a coverage question no other patch answers (HI13).

---

# 0860_allreduce_provider_cli

**Status:** validated
**Plan item:** PGC04

## What it does

Adds `--allreduce <auto|ccl|host|adaptive|p2p|root3|butterfly>` and `--allreduce-wire <native|q8>` so the multi-GPU AllReduce implementation is chosen by CLI argument instead of the GGML_CUDA_ALLREDUCE / GGML_CUDA_AR_WIRE environment variables.

## Why

Several AllReduce implementations (stock NCCL/RCCL, host, butterfly, plus 0840/1244/1252 providers) need to be selectable per run without rebuilding. Unknown or unsupported provider/wire combinations are a startup error; `auto` never silently selects p2p.

## Upstream

BigCherry-original base framework; no promotion gate.

---

# 0910_feature_sets

**Status:** validated
**Plan item:** QFP18/QFP23

Kind: framework (no behaviour change unless `BIGCHERRY_FEATURES` is set).

## What it does

- **Runtime profiles from config files.** Model- and scope-specific runtime settings live in `profile/*.ini`, not in
  patch code. The canonical copy is the source overlay's `src/profile/`; the build copies the folder next to the
  binaries (`bin/profile/`, part of the runtime bundle hash) and installs it with them. The loader finds it relative to
  `libggml-base`; `BIGCHERRY_PROFILES=<folder|file>` overrides.
- `BIGCHERRY_FEATURES=<profile>[,<profile>...]` applies profiles. A profile lists `NAME = VALUE` flags and may include
  other profiles (`@name`). Explicit environment variables win. Application is all-or-nothing: unknown/duplicate
  profiles, include cycles, conflicting assignments and malformed lines apply nothing; a failed set rolls back.
- `BIGCHERRY_FEATURES=auto` picks the profile by model architecture: when the first model loads
  (`llama_model_create`, before its hyperparameters and tensors), the profile whose `arch =` list contains the GGUF
  architecture is applied. Flags read before model load keep their values.
- `BIGCHERRY_FEATURES=help` prints the loaded profiles and every runtime flag documented by the patches in this build
  (name, values, default, owning patch, description). The library never exits; llama-server maps help to exit 0 and
  errors to exit 2.
- `ggml_bigcherry_features_init()` is idempotent and runs before any flag is read: `ggml_init`, `get_reg()` before
  the backend registry is constructed, `ggml_backend_load_all_from_path` before backends are dlopened, llama-server's
  `main`, and a GCC/Clang load-time constructor. Works on MSVC through the explicit hooks.

## Profiles shipped (src/profile/)

| File | Profiles |
|---|---|
| `base.ini` | `hip-q81` (HIP MMVQ Q8_1 activation path, any quantized model), `sched-async` (scheduler, multi-backend runs) |
| `flashnext.ini` | `flashnext` (Qwen3.8 Flash-Next on 2x XTX + R9700 + 6900 drafter: `@hip-q81 @sched-async` + placement) |

Tuned per-model values go into the model's profile file with the evidence run in a comment.

## Documenting a flag (patch authors)

A patch that reads a runtime flag documents it in its own patch.py and declares `requires = ["0910_feature_sets"]`:

```python
from bigcherry.patcher import EnvDoc
ENV_DOCS = (
    EnvDoc("BIGCHERRY_ACT_Q81", "0|1", "0", "activation ops write the Q8_1 activation directly"),
)
```

The patch loader turns `ENV_DOCS` into rows of this patch's help table, so the help reflects the actual build.
`patch-lint` validates the profile files (grammar, duplicates, unknown includes, cycles, conflicts) and rejects
flags that no patch documents.

---

# 1006_rdna4_mmq_q6k_codegen_fix: Upstream backport, RDNA4 MMQ codegen fix for Q6_K only

**Status:** validated
**Plan item:** PA35

## What it does

Cherry-picks the Q6_K half of unmerged upstream PR #25940 into the MFMA/WMMA
MMQ vec_dot: adds an explicit float cast before a scale multiply to change
ROCm's codegen. This is a split of the rejected patch
`1000_rdna4_mmq_q2k_q6k_fix`, keeping only the edit whose real-hardware
measurement was a genuine gain.

## Why

PA35 step-1 real hardware evidence (2026-09-16, gfx1201/RDNA4, exact-shape
backend-ops paired A/B, control=serving-core without 1000 vs
subject=serving-core+1000, 5 rounds, correctness PASS): Q6_K measured
1.365x [CI95 1.362x-1.367x], a real, statistically significant gain (below
the PR's own claimed 1.90x, but real). The same measurement found Q2_K
measured 0.959x [CI95 0.954x-0.963x], a real regression -- see
`patches/1000_rdna4_mmq_q2k_q6k_fix/SUMMARY.md`'s DEMOTION section for the
full record and the GPT design-review (dev-gpt-agent, req_24ceb64100cb41dd)
that recommended this split.

## Upstream / provenance

Cherry-picked from open upstream PR
https://github.com/ggml-org/llama.cpp/pull/25940. Excludes the PR's Q2_K
change (real regression on this project's hardware) and its second change
(a hand-written RDNA4 native-select heuristic), since this project's own
tuner already measures candidates head-to-head per shape.

## Lifecycle note

`state = "validated"` (2026-09-29). The 1.365x figure above (measured
against the combined 1000 composition) was preserved as directional
evidence only and never used as promotion evidence for this patch's own
composition identity. A fresh Q6-only hardware campaign (4 sessions,
gfx1201, `tierA-qwen4b-q6k`) was run instead -- see `README.md`'s
"Promotion evidence" section for the full 3-arm result (BigCherry vs
native llama.cpp vs BigCherry+patch): a real, consistent ~+18% prefill
gain with no decode regression, contract `RDNA4-MMQ-Q6K-CODEGEN`
status=pass. Not yet added to `config/recipes.toml`'s
`[patch-set.validated-enhancements]` -- that recipe-selection decision is
separate from this lifecycle promotion.

---

# 1007_meta_subgraph_realloc_fix

**Status:** validated
**Plan item:** QFP22

Kind: upstream correctness fix (meta backend), no flag.

`ggml_backend_meta_graph_compute` resets its graph context when a graph raises the node or subgraph high-water mark.
The reset frees every per-subgraph cgraph, but only the current graph's `n_subgraphs` were re-created (with the
current graph's node capacity). A later graph with more subgraphs than that one, but not more than `max_subgraphs`,
skipped the reallocation and used freed cgraph pointers (SIGSEGV in `ggml_backend_meta_graph_compute`).

The fix re-creates `max_subgraphs` cgraphs with `max_nnodes` capacity, which is what the context is sized for.

## Evidence

- b-chunk7 (1332, chunk 256, no MTP, 24.5K prompt): segfault at `mov %eax,0x4(%r13)` = `cgraph_ij->n_nodes = ...`
  with subgraph 96 of 97 on a 6810-node dense 4-token graph, after reserve (7139/6814 nodes) and a 7197-node chunked
  prefill graph (`qfp22-chunk7-diag2/gdb2`). Same binary without chunking does not crash.
- Hardware confirmation: b-chunk8 (same recipe + 1007) runs the same case to completion, rc=0, 39.1 t/s decode vs
  38.4 t/s for chunk 0 (`qfp22-chunk8-nomtp`, 2026-10-05). Promotion record: README.md.

---

# 1200_rd19_single_gpu_meta_bypass: Skip the Meta device wrapper when tensor-splitting a single GPU (RD19)

**Status:** validated
**Plan item:** RD19

## What it does

Uses the plain device instead of the Meta wrapper in llama_prepare_model_devices when n_devices == 1 (both the explicit-device-list and default-selection branches), leaving the multi-GPU Meta path untouched.

## Why

With one device, -s tensor still wraps the graph in a Meta device that splits it into extra subgraphs and compute calls even though no splitting is possible, adding launch overhead and clearing the Q8_1 quantize cache between subgraphs; the fork reports +1.4-1.8% tg64 from removing this.

## Upstream / provenance

Ported verbatim from stew675-rdna-boosts fork commit 3c48ecd63 (https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master.

---

# 1225_hi85_nccl_heterogeneous_arch_guard: fail closed when a NCCL/RCCL participant lacks PCIe AtomicOps capability (HI85/GP02)

**Status:** validated
**Plan item:** GP02

## What it does

Adds `ggml_backend_cuda_comm_rccl_admission_ok()` -- a shared, ordinal-independent
admission check queried once per device via the real HIP runtime attribute
`hipDeviceAttributeHostNativeAtomicSupported` -- and calls it before
`ncclCommInitAll` in `ggml_backend_cuda_comm_init_nccl()`, aborting with a
clear, named `GGML_ABORT` (not the uncatchable HIP SIGABRT the crash would
otherwise produce) when any participating device lacks PCIe AtomicOps
completion capability. The admission function is defined once, before every
`comm_init_*` function, specifically so other independent
`ncclCommInitAll()` entry points (patch 0840's hybrid dispatch) can call the
same check.

## Why

GP02 rewrite (2026-09-02): the original version of this guard used raw
GPU-architecture inequality as a proxy for "will this crash RCCL" -- a cheap
heuristic adopted because no better runtime signal was known at the time.
That proxy is now confirmed wrong in a way that actively blocks real work:
it would reject `{0,2}`/`{1,2}` (XTX+R9700), a pair HI138/GP06 spent real
hardware evidence confirming is RCCL-safe, purely because the architectures
differ -- unrelated to the actual failure mechanism.

Real fix: `hipDeviceAttributeHostNativeAtomicSupported` exposes the actual
fact directly, per device, via the standard HIP API. Verified on real
hardware (2026-09-02, standalone HIP probe binary): returns `1` for devices
0/1/2 (all CPU-direct PCIe root ports) and `0` for device 3 (chipset-routed)
on Brutus -- with no device ordinal hardcoded anywhere in the check itself,
consistent with this project's own topology-identity rules (no persistent
identity may be bound to a HIP ordinal/PCI BDF/hostname).

## Real hardware validation of the admission-check mechanism (2026-09-02)

The `hipDeviceAttributeHostNativeAtomicSupported` signal itself (not yet
this patch's full compiled guard) was verified directly against real
hardware via a standalone HIP probe program:

```
device 0 (AMD Radeon RX 7900 XTX): hipDeviceAttributeHostNativeAtomicSupported=1
device 1 (AMD Radeon RX 7900 XTX): hipDeviceAttributeHostNativeAtomicSupported=1
device 2 (AMD Radeon Graphics):    hipDeviceAttributeHostNativeAtomicSupported=1
device 3 (AMD Radeon RX 6900 XT):  hipDeviceAttributeHostNativeAtomicSupported=0
```

Exactly matches HI138's lspci/dmesg-based finding, with a portable runtime
API instead of shelling out to `lspci`/parsing `dmesg`.

## Upstream / provenance

Local design, based on real-hardware findings
(docs/planning/active/hip-autotune/HI85.md, HI138.md). GP02
(docs/planning/active/gpu-collectives/GP02.md) rewrote the trigger from
architecture-inequality to a real per-device PCIe-atomics capability check,
and made the admission logic shared/reusable across every
`ncclCommInitAll()` call site rather than specific to this guard's original
location.

---

# 0840_hybrid_allreduce_dispatch: size-adaptive internal/RCCL AllReduce provider dispatch

**Status:** validated
**Plan item:** GP03/PGC09

## What it does

Adds `--allreduce adaptive`: a fourth provider that brings up both
RCCL and the internal AllReduce pipeline (patch 1001) simultaneously, then
picks per call based on `ggml_nbytes(tensors[0])` against the internal
pipeline's own real copy-engine threshold -- below it, tries internal
first (falling through to RCCL on failure); at or above it, goes straight
to RCCL. `comm_ctx->provider_name` is set per call right before each
sub-provider runs, so the existing 0830 telemetry seam attributes every
call correctly (`effective_provider`: "internal" or "rccl") with no
separate labeling change needed. Init also forces the internal pipeline
to exact F32 (`ggml_cuda_ar_pipeline_force_exact_f32`) rather than
trusting `GGML_CUDA_AR_BF16_THRESHOLD`'s own default (1, BF16 for every
nonzero reduction) -- hybrid's internal side must never silently degrade
to the same lossy wire encoding 1001's evidence shows is a net loss.

## Why

Patch 1001 (validated) is a large win for decode (+17.33% TPS,
MTP completion-bench) but a severe regression for prefill (-32% to -34%,
real llama-bench pp512/pp2048/pp4096) -- the `--allreduce` selector can only
pick one provider for a whole server session, so neither `internal` nor
`rccl` alone is safe to ship as a blanket default. Real HI155-1 telemetry
(0830's new `reduction_bytes` field) captured on real traffic found a
clean, 10x, zero-overlap separation: MTP decode tops out at 1,044,480
bytes, pp2048 prefill is a flat 10,485,760 bytes -- and decode's max sits
almost exactly at `allreduce.cu`'s own default copy-engine threshold
(1,048,576 bytes), meaning decode never reaches the large-message strategy
that the prefill regression is entirely attributable to. Dispatching on
the pipeline's own real threshold (not a second, independently-tunable
constant) keeps this policy from ever routing a call into exactly the
regime the regression evidence implicates.

## Fixes applied 2026-09-02 (GP03 consolidation, gpt-dev-agent review)

This patch is the consolidation target superseding `1243_gp03_size_adaptive_allreduce_dispatch`
(same problem, independently implemented, retired once this patch passes
validation). Four real bugs found by gpt-dev-agent adversarial review, all
fixed:

1. **Missing `requires` on 0830.** `comm_ctx->provider_name` is an
   0830-provided field; `patch.toml` only declared `requires =
   ["1001_hip_internal_allreduce"]`. Fixed: `requires =
   ["0830_split_reduce_telemetry", "1001_hip_internal_allreduce"]`.
2. **Large calls fell to META when RCCL was unavailable, instead of
   internal.** The original dispatcher only tried internal when the
   reduction was below the copy-engine threshold; at/above it, if RCCL
   also wasn't available (NCCL init failed, virtual devices), the call
   fell straight through to `return false` (META) even though internal
   WAS available and is strictly better than META for any size. Fixed:
   internal is now always the last-resort fallback before META, not only
   the below-threshold path.
3. **`GGML_CUDA_AR_COPY_THRESHOLD=0` sentinel inverted.** The internal
   pipeline treats `copy_threshold=0` as "never use the copy-engine, the
   chunked kernel handles every size." The dispatcher's naive `bytes <
   threshold` comparison made that sentinel mean the opposite (never
   route to internal) -- an operator explicitly forcing internal-always
   got the reverse of what they asked for. Fixed with an explicit
   `below_copy_threshold` check treating `threshold==0` as always-eligible.
4. **Explicit-override telemetry mislabel.** `GGML_HIP_REDUCE_PLAN=rccl`
   bypasses this patch's own per-call dispatcher entirely via 0830's
   shared `try_reduce_plan()` rccl branch, which called
   `ggml_backend_cuda_comm_allreduce_nccl()` directly without updating
   `comm_ctx->provider_name` -- in hybrid mode that field was last set at
   init, so an explicit-rccl-forced call could genuinely run RCCL while
   telemetry still reported "internal". Fixed by setting
   `provider_name = "rccl"` in that shared branch immediately before the
   call.

**RESOLVED (2026-09-02, GP02)**: this patch now `requires` the rewritten
`1225_hi85_nccl_heterogeneous_arch_guard` and calls its shared
`ggml_backend_cuda_comm_rccl_admission_ok()` before its own secondary
`ncclCommInitAll()` -- the same real, per-device, ordinal-independent
PCIe-atomics check (`hipDeviceAttributeHostNativeAtomicSupported`) that
protects the original RCCL init path. Verified on real hardware: a
device-3-inclusive topology (`{0,3}`) now declines RCCL cleanly (no crash)
and falls back to the internal pipeline (unaffected by PCIe atomics, since
it doesn't use RCCL) -- real working inference, not just a safe abort.
`{0,2}`/`{1,2}` remain correctly admitted through RCCL (confirms the new
predicate, unlike the old architecture-inequality version, does not
falsely reject them). See GP02's plan notes for the full validation.

## Real hardware validation (2026-09-02) -- CONSOLIDATION COMPLETE

Built from a clean, zero-BigCherry-patches vendor/llama.cpp checkout (pin
b10705, 2578138397d7) in an isolated scratch clone, patches applied and
verified to apply cleanly + idempotently against the true pinned source
(0100_cmake_options, 0830, 1001, this patch). Compared against 5 arms on
{0,1} (2x RX 7900 XTX): native llama.cpp RCCL (zero BigCherry patches --
confirmed the --allreduce selector itself is genuine unpatched
upstream code, a legitimate baseline), native META (`-sm tensor`,
butterfly), layer-split (`-sm layer`), this patch's `hybrid` provider, and
`internal`-only.

**pp/tg synthetic** (llama-bench, `-b 2048 -ub 512`, r=3): hybrid matches
or slightly exceeds native RCCL at pp512 (1478 vs 1465) and tg128 (37.25
vs 34.62), roughly ties native RCCL at pp2048/pp4096. Layer-split beats
BOTH native RCCL and hybrid at pp2048/pp4096 by 10-19% -- confirmed NOT a
defect in this patch (native RCCL has the identical shortfall against
layer-split at the same sizes; this is a real tensor-split-prefill
characteristic of the hardware, present in upstream).

**Real MTP completion-bench** (the metric that resolves the above --
production flags, real `mtp-27b-v1` corpus, 48 requests/arm,
`predicted_tps` mean):

| arm | predicted_tps | vs layer-split | vs native RCCL |
|---|---|---|---|
| layer-split | 44.99 | -- | -31% |
| native-meta | 53.04 | +18% | -19% |
| native-rccl (baseline) | 65.50 | +46% | -- |
| **hybrid (this patch)** | **68.03** | **+51%** | **+3.9%** |
| internal-only | 68.02 | +51% | +3.9% |

Draft acceptance (0.487-0.491) and mean accepted length (2.93-2.95) are
consistent across every arm -- no correctness/behavioral regression in
speculative decoding from any provider choice. Real MTP serving is
decode-dominated (many small verification-reduction calls, not large
prefill-sized ones), which is why layer-split -- the pp-synthetic winner
at large sizes -- is actually the WORST arm under the workload that
matters. Under real MTP serving, this patch beats every baseline including
layer-split by the largest margin of any arm, and beats native RCCL by
3.9% -- a genuine, validated improvement, not a wash.

Full evidence: `artifacts/gp03-validation/` on Brutus (bc-native, bc-hybrid
scratch builds; `mtp-results/*.completion.jsonl` + server/bench logs for
all 5 arms). See GP03's plan notes for the complete writeup.

**2026-09-16 (PA35 step 4) raw-artifact recheck:** `artifacts/gp03-validation/`
was searched for across Brutus's current checkout (`~/bc-pa-work`) and every
other known BigCherry working directory/scratch clone on the host
(`~/bc-gp11-fusion`, `~/rd73-*`, `~/hi158-syntax`, `~/hi168-*`,
`~/pha07-work`, `~/perf-sweep-20260911`, `/mnt/vault/development/bc-branch`,
plus a broad `find` for `*gp03*` under `/home/audumla` and `/mnt/vault`) and
was not found anywhere. The raw per-arm JSONL/log files referenced above are
therefore currently non-reinspectable -- honestly marked as such here rather
than silently reconstructed. The recorded aggregate result (the table above,
the 68.03/65.50 predicted_tps figures, and the draft-acceptance/mean-accepted-
length consistency claim) is preserved as-is; it is a summary claim, not raw
data, and is not itself contradicted by the missing raw artifacts. If the
underlying `mtp-results/*.completion.jsonl` are needed for re-inspection in
the future, they must be regenerated by rerunning GP03's real hardware
comparison, not reconstructed from this prose.

**Consolidation outcome**: this patch supersedes and replaces
`1243_gp03_size_adaptive_allreduce_dispatch` (deleted 2026-09-02) -- same
problem, independently implemented, retired once this patch reproduced and
exceeded its real hardware numbers under a proper baseline comparison.
This is now the sole implementation for size-adaptive AllReduce dispatch.

## Upstream / provenance

Local, BigCherry-authored (not an upstream port) -- plan item GP03,
opened following gpt-dev-agent's explicit design guidance (dev-gpt-agent
gateway session `ses_5307d9c58ec645cb`) after the real prefill-regression
and reduction-byte-histogram evidence in
`patches/1001_hip_internal_allreduce/SUMMARY.md` and project memory.
Requires 0830 (and 1225); the internal pipeline this dispatches into was
patch 1001 until pin b11126, where upstream absorbed it verbatim (1001
superseded, dependency dropped). 0830 is (the
`provider_name`/telemetry field this patch reads and writes). Real hardware
consolidation validated 2026-09-02 (see above) -- remaining before
STATE=validated / default patch-set promotion: GP02's admission predicate
must land and be consulted by this patch's secondary `ncclCommInitAll()`.

---

# 1235_rd09_q81_activation_cache_foundation: Per-graph Q8_1 activation-quantization cache, foundation only (RD09 stage 1)

**Status:** validated
**Plan item:** PRBE05

## What it does

Adds a generation-scoped Q8_1 quantization cache implementation (stable-slab allocator, find/reserve/publish API, GGML_HIP_Q8_1_CACHE_MODE env gate) as pure additions to three vendor files, and gives ggml_backend_cuda_context an opaque pointer to own one instance; no caller references it yet.

## Why

Reusing one Q8_1 quantization of an activation across every MMVQ consumer that needs it within a graph avoids re-quantizing per node, but this foundation stage deliberately adds no caller so it can be reviewed and tested with zero behavioral risk before stage 2 wires it into ggml_cuda_mul_mat_vec_q().

## Upstream / provenance

Reimplemented (not verbatim-ported) from stew675-rdna-boosts fork commit 299f6eaf7 (https://github.com/stew675/llama.cpp), with two required adaptations found in design review: a cache key that includes view offset (the fork's key collides on same-root different-offset views), and never-relocated slabs instead of a growing arena (a relocating arena would corrupt baked-in pointers in captured HIP graphs).

---

# 1237_rd30_moe_mmq_compact_grid: Compact the MoE MMQ launch grid (RD30, AMD-MOE-001)

**Status:** validated
**Plan item:** RD30

## What it does

Adds a prep kernel (mmq_build_moe_block_map) that flattens (expert, expert-local-tile) pairs into one linear grid dimension sized to the real total tile count, instead of upstream's rectangular grid sized from the worst-case expert width times n_expert; falls back to the exact legacy grid whenever the compact map would exceed grid or shared-memory limits. Gated to gfx1100 exactly.

## Why

Upstream's non-stream-K MMQ launch gives every MoE expert the same worst-case tile-column count regardless of its real routed-token occupancy, launching far more blocks than needed on real production models (confirmed via rocprofv3 on Qwen3.6-35B-A3B, n_expert=256); AMD's own PR #63 reports a modest +1.9-5.4% prefill gain from compacting this.

## Upstream / provenance

Concept ported from AMD-Ecosystem/llama.cpp PR #63 (fork-only, https://github.com/AMD-Ecosystem/llama.cpp), redesigned against the real b10502 source rather than ported verbatim. Extensive informal real-hardware evidence gathered, but STATE stays untested pending the project's own formal HI83-governed validation campaign.

---

# 1241_rd33_mmvq_q8_0_f32_decode: Dense Q8_0 decode without activation quantization (RD33, AMD-MMV-001)

**Status:** validated
**Plan item:** RD33

## What it does

Adds a defaulted f32_act template parameter to mul_mat_vec_q and a new vec_dot_q8_0_f32 device helper that dequantizes the Q8_0 weight block directly and dot-products it against the original F32 activation (F32 accumulation), skipping the Q8_1 activation-quantization stage entirely; gated to dense (non-MoE), Q8_0, ncols_dst==1, gfx1100, and only when nothing has been forced.

## Why

ggml_cuda_mul_mat_vec_q unconditionally quantizes the F32 activation to block_q8_1 before every MMVQ call, including plain single-token decode where there is no batching to amortize that extra kernel launch and pool allocation against, and the weight is already the only operand whose quantization matters for a bandwidth-bound matvec.

## Upstream / provenance

Local design, part of this project's own rdna-boosts experiment work (RD33), designed and verified against the real pinned source via dev-gpt-agent review.

## Lifecycle note

`state = "validated"` (2026-09-30), scoped to gfx1100 (dual 7900 XTX, `-sm tensor`), dense Q8_0, `ncols_dst == 1` decode. Contract `RD33-MMVQ-Q8_0-F32-DECODE` status=pass over 4 sessions (`tierL-qwen27b-q8` positive tg128 +4.56..+4.67%, 10/10 pairs each; `tierA-qwen4b-q6k` non-firing control within the no-regression bound; CPU-reference `test-backend-ops` MUL_MAT q8_0 50/50; trace-marker activation at ncols=1 only). Final sign-off by dev-gpt-agent. See README.md "Promotion evidence".

Not claimed: other architectures or models, MTP verify batches (ncols>1 keep the stock path; acceptance parity 0.90101 vs control), prefill. The Q6_K control moved +1.4% in every session (build-layout effect, not attributable to this patch); the control-adjusted decode effect is about +3.2%. Not yet added to a recipe patch-set -- separate decision.

---

# 1253_nro04_gfx1100_bf16_chunked_gdn

**Status:** validated
**Plan item:** NRO04

## What it does

Adds the nasone fork's BF16/WMMA chunked GatedDeltaNet prefill kernels (gfx11 and gfx12, new files) and routes K == 1 prefill with S_v == 128 on RDNA3/RDNA4 to them by default, falling back to the sequential kernel if the driver rejects the launch. Opt out with `GGML_CUDA_GDN_CHUNKED_BF16=0`.

## Why

Sequential GDN recurrence dominates prefill on hybrid Qwen models; the chunked form uses tensor cores. Near-lossless, not bit-exact.

## Upstream

Port of nasone commit `4169fbbf50d24beb6d269a2350e7f780b85369e6` (block 02). The fp32 chunked kernel is not ported (RD50/1221 scope; conflicts with 1221).

---

# 1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2

**Status:** validated
**Plan item:** RD30

## What it does

Widens 1237's compact MoE MMQ launch-grid gate from gfx1100 only to also admit RDNA4 (gfx1201) and RDNA2 (gfx1030).

## Why

1237 measured +7.3..7.4% MoE prefill on gfx1100 with byte-identical output. The compaction is host-side launch geometry, so the same empty per-expert launches exist on the other cards; this package measures it there without touching 1237's own evidence.

## Upstream

Extension of 1237 (AMD-Ecosystem/llama.cpp PR #63 concept). Requires 1237.

---

# 1274_mmvq_kquant_f32_decode

**Status:** validated
**Plan item:** RD33

Extends the validated 1241 dense single-token F32-activation MMVQ path from Q8_0 to Q6_K on gfx1100 (Q4_K was measured and dropped: -1.7% decode on one XTX, flat on two; Q6_K +1.6%..+1.7% decode). It requires 1241 and reuses its `f32_act` kernel/launcher seam, so Q8_0 remains owned by 1241 while 1274 adds only the K-quant helpers and dispatch gate.

For eligible dense `ncols_dst == 1` calls, Q6_K keeps the native MMVQ weight-lane unpacking but dot directly against the original F32 activation. This removes the per-call Q8_1 activation allocation, quantization kernel, and quantized activation write/read. The tradeoff is extra per-weight-block unpack plus F32 FMAs. Q4_K/Q6_K are the first probes because their unpack is relatively direct; Q5_K is deferred because its additional high-bit plane increases unpack/register cost without increasing the fixed activation-quantization saving.

Validation required: compose after 1241, HIP build on gfx1100, Q6_K CPU-reference correctness, marker activation only for dense width-1 decode, forced-candidate/non-width-1 negative controls, and paired performance A/B before promotion.

---

# 1281_moe_mul_mat_id_range

**Status:** validated
**Plan item:** MET02

Kind: enhancement, a new ggml primitive. Nothing uses it yet; ordinary `ggml_mul_mat_id` is unchanged.

`ggml_mul_mat_id_range(ctx, as, b, ids, id_base)` is MUL_MAT_ID over a tensor that holds only the experts
`[id_base, id_base + as->ne[2])` of a larger set. `ids` stay global. A lane whose id is in the range is computed with
the local expert; a lane whose id is outside it is an exact +0 and reads no expert weight. A tier graph (MET03) and
whole-expert parallelism (MET04) are sums of such ops, one per device.

Phase A is the semantic primitive only: constructor and accessors, the CPU implementation, and a reference test
(`tests/test-mul-mat-id-range.cpp`). The HIP backend refuses the range variant in `supports_op`, so the scheduler
runs it on the CPU and no global id reaches a GPU kernel. Phase B (HIP translation at the existing grouping) and
phase C (compact dispatch) follow in MET02's order.

## Evidence

- Offline mechanics test: pending.
- Reference test on Brutus (CPU): pending.

---

# 1283_qwen4exp_expert_parallel

**Status:** validated
**Plan item:** MET04

Kind: enhancement, flag `BIGCHERRY_MOE_EP` (default 0). Requires 1281.

Whole-expert parallelism inside the tensor split. With the flag the routed expert weights are split across the
devices along the expert index, so each device holds whole experts and computes the selected ones it holds (range
MUL_MAT_ID of 1281, exact zeros for the others). The AllReduce is delayed through the block - gate, up, GLU, down -
and then by the existing delay to the block output: one AllReduce per MoE block, the same as today's row split.

It does not reduce collectives. What it changes is the kernel work per device (a few whole experts instead of a
slice of every selected expert) and placement freedom for quants whose experts do not all fit. The risks are load
imbalance (a step is as slow as the device holding most of a token's experts) and the fused expert kernels, which
must be range-aware before decode can benefit.

The delay is a strict pattern match with fallback to an immediate AllReduce; expert biases, the merged gate_up
tensor and scale MULs are not matched.

## Evidence

- Offline mechanics test: pending. Hardware: blocked on 1281 phase B (range op on the GPU).

---

# 1291_ar_cpu_root

**Status:** validated
**Plan item:** QFN01

## What it does

Adds `--allreduce cpu-root` (HIP only). Small f32 contiguous messages (<= `BIGCHERRY_AR_CPU_ROOT_MAX_BYTES`,
default 65536) use a CPU-root one-shot AllReduce: each rank's kernel copies its slice into pinned,
device-mapped host memory and publishes a per-rank epoch; a persistent CPU worker sums in fixed rank order
(exact f32) and publishes a result epoch; each rank's kernel waits for it and copies the result back, all
on the backend streams. Larger or other messages go to RCCL. Measured standalone on 2x 7900 XTX + R9700:
10 KB 13.8 us/call vs RCCL 33 us. Activation marker: `BIGCHERRY_PATCH_HIT patch=1291_ar_cpu_root path=small`.

## Hardware result (2026-10-03, flashnext-cpuroot-4)

Flash-Next UD-IQ4_XS, 2x 7900 XTX + R9700, `-sm tensor -ts 4,4,3 -ub 1024`, ABBA vs `auto` (RCCL), 3
requests per arm, greedy output identical in all 8 arms: no MTP decode 38.7/39.1 vs 36.5/36.6 t/s
(+6.4%); MTP3 (draft on the 6900) 70.5/70.4 vs 67.0/67.1 t/s (+5.1%), acceptance 74.6% vs 73.0%;
prefill unchanged (large messages stay on RCCL). Earlier builds produced garbage because ranks whose node
the meta backend left uncomputed must contribute zeros (the RCCL provider memsets them). Not yet
qualified: needs KLD and a balanced contract run.

---

# 1292_kpool_tail_truncate

**Status:** validated
**Plan item:** QFN01

Qwen4Exp indexer pool layout (`kpool_layout_update`) rebuilt every sequence from scratch after any edit.
MTP removes the rejected draft tail with `seq_rm [p, inf)` on almost every decode step, so each step walked
the whole cell map: O(n_ctx) host work per step. At ~80K cached context, `perf` on the server showed
`std::_Rb_tree_increment` and `vector::_M_assign_aux` (from that map) as the top libllama symbols while the
GPUs were idle ~75% of each step.

The patch truncates the layout at the stale position (cells before it are unchanged by the stale contract)
and keeps the pools that end before the cut; the append path re-reads the tail. Same layout as the full
rebuild. Sequences with shared cells still rebuild.

Validation: decode t/s at 80K cached context with MTP3, A/B against 1291 alone, greedy
output identical.

## Result (2026-10-03)

Speed: two ABBAs at ~80K cached context, MTP3 (flashnext-kpool-ab-1/2): 77.1/77.3 ms per step (1291 only)
vs 65.5/66.4 and 65.5/65.6 (with 1292), -14%; at 10K -2 to -3%. No effect without MTP (no tail edits).
Parity: greedy text without MTP byte-identical to the 1291+1294 baseline at 10K and 80K, all four arms
(flashnext-kpool-parity-2; 1294 makes long-context runs deterministic, which this check needed).

---

# 1294_topk_deterministic_ties

**Status:** validated
**Plan item:** RNX02

## What it does

The HIP parallel radix TOP_K picks tied columns at the cut by lowest column index (one extra block per row,
ballot/popcount prefix count) instead of atomic arrival order. `BIGCHERRY_TOPK_DETERMINISTIC=0` restores the
old path.

## Why

Qwen4Exp's QSA indexer scores are sums of ReLU, so many pooled KV blocks score exactly 0.0; past ~10K
context the top-k cut falls inside that tie and the attended blocks varied between runs. Greedy text
diverged across server starts at 32K and 80K with RCCL, cpu-root and host AllReduce alike.

Validation: tools/lab/flash-next/determinism.sh (cross-start greedy identity at 32K/80K, no MTP), decode and
prefill timing A/B, test-backend-ops TOP_K.

## Result (2026-10-03, flashnext-topk-ab-1, on 1291+1292)

Cross-start greedy identity, no MTP, cpu-root: 32K 317/317 chars identical (s1 vs s2), 80K 262/262 identical;
before 1294 the same probe diverged at character 45 (32K) and 0-1 (80K). Speed neutral: ~50 ms/MTP step at
10K and ~66 ms at 80K in all ABBA arms; with MTP at 10K three arms reached identical acceptance (351/478).

---

# 1295_qsa_gather_decode

**Status:** validated
**Plan item:** RNX02

## What it does

For batches of at most 8 tokens, Qwen4Exp QSA attention gathers each token's selected KV cells (padded to a
multiple of 256 with a masked sentinel) and runs flash attention over them, instead of masking the full
cache. Prefill keeps the masked path. On by default; `BIGCHERRY_QSA_GATHER=0` restores the masked path.

## Why

On HIP the QSA mask does not reduce work: decode attention per token grew 7.6x from 10K to 80K context
(target 0.44 -> 3.36 ms/token, draft 0.18 -> 1.41 ms/token summed over GPUs). Upstream's sparse attention is
NVIDIA-only. See RNX02 review RV4213. Compared against 1296 (HIP port of the upstream sparse kernel path).

Validation: decode ms/step at 10K/80K/160K, greedy parity without MTP at 10K and 80K against the masked path.

## Results (2026-10-03)

Accuracy: greedy text differs from the masked path, but vs an f32 CPU reference (FA off, no tensor split) the
gathered path is closer (max |dp| over 8 decoded tokens at 10K: gathered 0.056, masked FA 0.073; masked vs
gathered 0.023). The masked HIP tile path rescales its half-precision accumulators over every KV tile, so its
error grows with n_kv; a 2-GPU run (one KV head per rank) showed the same difference, ruling out the meta split.
v1 (no threshold): 10K +3% ms/step, 80K -5.7%, no-MTP decode at 80K +16%.
v2 (BIGCHERRY_QSA_GATHER_MIN=32768, cache view reshaped directly, sentinel rows clamped), full ABBA on the
deployment candidate (flashnext-gather-v2-ab): 10K identical (threshold); 80K 56.5/55.6 -> 53.8/53.7 ms/step
(-3.7%, +7% t/s); 160K (192K tier) 70.3 -> 61.7 on one pair (-12%), the other 1295 arm ran out of memory on the
R9700 at runtime (that tier has ~0.3 GB headroom). Scope: tiers with headroom (96K / 144K / 160K-q8_0 at 4,4,3);
in the 192K tier cap context near 184K or lighten the R9700 share before enabling.

---

# 1297_draft_vocab_trim

**Status:** validated
**Plan item:** QFN01

## What it does

Opt-in (`BIGCHERRY_DRAFT_VOCAB_N=N`): the Qwen4Exp MTP draft computes logits over output rows [0, N) plus all
control / user-defined / end-of-generation tokens, scattered into -inf full-vocabulary logits.

## Why

The MTP draft's own output.weight (Q8_0, 2560 x 248320, ~675 MB) is read for every draft token: ~31% of the
6900's draft kernel time per MTP step at 80K. The target verifies every draft token, so output cannot change;
only acceptance can drop.

Validation: decode ms/step and acceptance A/B for N = 16384 / 32768 / 65536 vs unset at 10K and 80K; greedy
parity is guaranteed by verification but checked without MTP-irrelevant changes (draft only).

## Result (2026-10-03)

Quick screens (~30K cached, ABA): N=32768 -10% ms/step but acceptance fell (163/273 vs 170-176), ~neutral
throughput; N=65536 -6% ms/step with acceptance in the base range, +8% t/s. Full ABBA (flashnext-trim-ab-4,
MTP3, f16 draft KV, 192K deployment config): 10K 50.0 / 51.0 -> 46.1 / 46.6 ms/step (-8%); 80K 58.6 / 57.6 ->
54.0 / 53.9 (-7%); complete separation, acceptance comparable. Deployment: BIGCHERRY_DRAFT_VOCAB_N=65536.
Run 2 crashed on MTP prompt-replay batches with no output rows (fixed: those keep the plain head).

---

# 1302_cuda_graph_oom_evict

**Status:** validated
**Plan item:** QFN03

## What it does

Graph instantiation (`cudaGraphInstantiate` / `hipGraphInstantiate`) that fails with out-of-memory no longer
aborts the server. The backend context destroys every other cached graph (each holds an executable instance in
device memory), synchronises, clears the error and instantiates again; only a second failure aborts. Found at
Flash-Next 192K: a ~164K-token fill OOMs in hipGraphInstantiate on the R9700 after KV and compute buffers fill
it (production build, flashnext-gather-ab-2/d131072). Evicted graphs are re-captured on their next use.
Activation evidence: `BIGCHERRY_PATCH_HIT patch=1302_graph_oom_evict` with the evicted count.

## Hardware result (2026-10-04 review)

Part of production profile v2 (2026-10-03/04): needed to instantiate HIP graphs at 240K f16 KV without OOM; greedy identical in every v2/v3 screen.

---

# 1303_attn_kv_tensor_split

**Status:** validated
**Plan item:** QFN03

## What it does

Under `-sm tensor`, `BIGCHERRY_ATTN_TS="a,b,c"` gives the full-attention family (q/k/v/qkv weights and biases,
q/k norms, sinks, gate, attn_output, KV cache) its own split vector, independent of `-ts`, so KV/attention
placement is decoupled from expert-weight placement and every GPU can be filled at long context.
`BIGCHERRY_ATTN_ROTATE=0` disables the per-layer rotation for that family (pin whole KV heads to chosen devices).
Recurrent (GatedDeltaNet) layers' attn_qkv/attn_gate and their state stay on `-ts` (anchored to ssm_out);
indexer tensors and caches stay mirrored as upstream. All attention-family members use the same vector and
rotation, preserving GQA grouping and the meta backend's split-state equality. Design: GPT req_0f390645741c4413.
Activation evidence: `BIGCHERRY_PATCH_HIT attn_ts=... attn_rotate=...`. Qwen4Exp only: any other architecture fails to
load with a clear error (Gemma 4 produced garbage silently with the split on, 2026-10-04 smoke).

## Hardware result (2026-10-04 review)

Production profile v2 (BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0): f16/f16 KV at 240K; profile ABBA +3% at ~10K, +7% at ~80K; greedy identical. Fails closed on non-qwen4exp (Gemma 4 smoke).

---

# 1307_q81_activation_cache_mmvq

**Status:** validated
**Plan item:** PRBE05/QFP13

## What it does

Stage 2 of RD09: with `GGML_HIP_Q8_1_CACHE_MODE=on`, `ggml_cuda_mul_mat_vec_q` looks up the Q8_1 quantization of
src1 in 1235's generation-scoped cache (keyed by the consumed node, not its view root, so in-place rewrites never alias) and skips the `quantize_q8_1` launch on a hit; a miss quantizes into a
stable cache slab and publishes it; any reserve failure falls back to the original pool allocation and quantizer.
A cache generation begins at every `ggml_backend_cuda_graph_compute`, and slab growth is blocked while a HIP graph
is captured, so captured graphs reference only never-relocated memory. Motivation (QFP13): decode is launch-gap
bound and issues ~183 quantize launches per generated token per GPU, one per MMVQ consumer, many re-quantizing the
same activation. Cache statistics (1235) report hits and launches saved.

## Hardware result (2026-10-04 review)

Profile v3 adoption ABBA (flashnext-v2-fusion-ab-3, with 1308-1310): ~10K 74.3 -> 76.5 t/s (+3%), ~80K 53.3 -> 55.2 (+3.6%), complete separation, greedy identical; quantize_q8_1 183 -> 45/token with the producers.

---

# 1308_qwen4exp_rollback_copy_no_cont

**Status:** validated
**Plan item:** QFP13

## What it does

With `BIGCHERRY_ROLLBACK_NO_CONT=1`, the Qwen4Exp recurrent conv-state builder writes each rollback snapshot
(n_rs_seq + 1 slots per recurrent layer, 4 under MTP3) as `ggml_cpy(tail, dst)` instead of
`ggml_cpy(ggml_cont(tail), dst)`: one copy kernel per slot instead of two. The decode kernel census attributes ~89
of ~108 runtime copy launches per generated token to this sequence, so this removes ~45 launches per token per
GPU (decode is launch-gap bound, QFP13). The copy is exact: outputs must be bit-identical.

## Hardware result (2026-10-04 review)

Profile v3 adoption ABBA (with 1307/1309/1310): +3% ~10K, +3.6% ~80K, greedy identical; kernels/token 1270 -> 1212 on its own.

---

# 1309_rms_norm_mul_q81

**Status:** validated
**Plan item:** PRBE06/QFP13

## What it does

With `BIGCHERRY_RMS_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on` (1307), a decode-shaped fused RMS_NORM+MUL (ncols >= 1024 and a multiple of 32,
<= 16 rows, contiguous output) runs `rms_norm_mul_q81_f32`, which writes the normal F32 result and, in the same
launch, the Q8_1 blocks of that result into a 1235 cache slab published under the exact key
`ggml_cuda_mul_mat_vec_q` builds for src1 == the MUL node. Its MMVQ consumer then hits the cache and skips the
standalone `quantize_q8_1` launch. Q8_1 layout and math match `quantize_row_q8_1_cuda` (padded rows, zero padding
blocks, amax/127, roundf, half2(d, sum)), so the result is bit-identical. PRBE06 Stage 0 measured 29.9 such
RMSNorm -> quantize -> MMVQ triples per generated token per GPU with 1307 + 1308 on. Any reservation failure (e.g.
during graph capture) or ineligible shape uses the unchanged kernel. Activation evidence:
`BIGCHERRY_PATCH_HIT patch=1309_rms_norm_mul_q81` under `BIGCHERRY_PATCH_TRACE`.

## Hardware result (2026-10-04 review)

Profile v3 adoption ABBA (with 1307/1308/1310): +3% ~10K, +3.6% ~80K, greedy identical.

---

# 1310_act_q81

**Status:** validated
**Plan item:** QFP13

## What it does

With `BIGCHERRY_ACT_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on` (1307), an F32 plain (`ggml_cuda_op_unary`) or gated
(`ggml_cuda_op_unary_gated`: SWIGLU/GEGLU/...) activation whose row length is a multiple of 32 (rows are written in MMVQ's padded layout,
GGML_PAD(ne0, 512) with zero padding blocks), with <= 64 rows (decode and MTP-verify shapes incl. routed experts, tokens x n_expert_used; prefill batches are excluded because they use MMQ, which never reads the cache) and a contiguous output, runs `bc_act_q81_kernel`: it writes the normal F32
result and, in the same launch, native-layout Q8_1 blocks into a 1235 cache slab published under the key
`ggml_cuda_mul_mat_vec_q` builds for src1 == this node. Its MMVQ consumer then skips the standalone `quantize_q8_1`
launch (QFP13 census: ~30 gated + ~30 plain activation -> quantize -> MMVQ chains per generated token per GPU).
The Q8_1 math matches `quantize_q8_1` exactly. Ineligible shapes or a failed reservation run the unchanged kernel.
Activation evidence: `BIGCHERRY_PATCH_HIT patch=1310_act_q81` under `BIGCHERRY_PATCH_TRACE`.

## Hardware result (2026-10-04 review)

Profile v3 adoption ABBA (with 1307-1309): +3% ~10K, +3.6% ~80K, greedy identical. 2026-10-04: flatten01 parameter added for 1312 (behaviour unchanged for 1310's own callers).

---

# 1311_hc_pre_q81

**Status:** validated
**Plan item:** QFP13

## What it does

With `BIGCHERRY_HC_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on` (1307), an eligible hyper-connection pre-mix
(`GGML_OP_DSV4_HC_PRE`: contiguous F32 output, n_embd a multiple of 32, <= 16 tokens) runs `dsv4_hc_pre_q81_f32`,
which writes the normal F32 result and, in the same launch, native-layout Q8_1 blocks (MMVQ's padded rows, zero
padding blocks, `quantize_q8_1` math) into a 1235 cache slab published under the key `ggml_cuda_mul_mat_vec_q`
builds for src1 == this node, so its MMVQ consumer skips the standalone quantize launch. Motivation: the largest
remaining class of Q8_1 misses after 1307-1310 (~2958 of ~8056 in `flashnext-v2-q81c-trace`). Ineligible shapes or
a failed reservation launch the unchanged kernel. Activation evidence: `BIGCHERRY_PATCH_HIT patch=1311_hc_pre_q81`.

## Hardware result (2026-10-04 review)

Adopted in profile v3 2026-10-04: quantize/token 74 -> 45, kernels/token 1138 -> 1116; screens neutral (~24K) to ~-2% ms/step (~80K); greedy identical.

---

# 1312_mul_q81

**Status:** validated
**Plan item:** QFP13

## What it does

Upstream fuses `UNARY(sigmoid/silu/softplus) -> MUL` pairs into `ggml_cuda_op_unary_mul` (one gated launch writing the
MUL node), which bypasses 1310's producers. With `BIGCHERRY_ACT_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on`, the F32
branch of that fused path now goes through 1310's `bc_act_q81_try<op, true>` with the MUL node as output: the same
`op(x) * g` result plus MMVQ-padded native Q8_1 published for its MMVQ consumer, in one launch. Targets the remaining
Flash-Next misses fed by a gate multiply: the full-attention output gate (`attn_gated` -> wo) and the GatedDeltaNet
gated output norm (`final_output` -> ssm_out). Bit-identical F32; Q8_1 matches `quantize_q8_1`. Requires 1310.
Activation evidence: `BIGCHERRY_Q81 publish-act node=...(attn_gated-N)` under `BIGCHERRY_Q81_TRACE`.
GDN `final_output` (per-head rows of 128 read through `reshape_3d(head_v_dim*heads, T)`) is published flattened to the
consumer's padded row (1310 `flatten01`), matched by 1307's flattened-reshape lookup.

## Hardware result (2026-10-04 review)

flashnext-v3-1312c (build A/B vs v3): ~24K 44.3 vs 44.8/44.9 ms/step, ~80K 51.0 vs 51.8/51.1; quantize/token 45 -> 30, kernels/token 1116 -> 1091; greedy identical. v4 candidate (multi-request ABBA pending).

Adopted in production profile v4 (2026-10-04, flashnext-v4-abba): v3 -> v4 +2.1% at ~8K and ~64K, complete separation, greedy identical across 8 arms per depth. State stays evaluated until a qualification package exists.

---

# 1313_scale_act_fuse

**Status:** validated
**Plan item:** QFP13

## What it does

With `BIGCHERRY_SCALE_ACT_FUSE=1`, `ggml_cuda_try_fuse` runs `SCALE -> UNARY(SILU|SIGMOID)` and
`SCALE -> UNARY(SILU|SIGMOID) -> SCALE` chains (F32, contiguous, single-use intermediates) as one `bc_scale_act_kernel`
launch computing the same expressions in the same order (bit-identical). Targets the Flash-Next hyper-connection
blocks: `silu(scale(w_down @ xn))` and `2 * sigmoid(scale(inject))`, ~98 scale launches per token per GPU. When the
chain ends at the activation and 1310's rules hold (`BIGCHERRY_ACT_Q81=1`, cache on, decode graph), the launch also
publishes the MMVQ Q8_1 activation like 1310. `UNARY -> MUL` is left to upstream's unary_mul fusion (1312).
Activation evidence: `BIGCHERRY_PATCH_HIT patch=1313_scale_act_fuse` under `BIGCHERRY_PATCH_TRACE`.

## Hardware result (2026-10-04 review)

flashnext-v3-1313b (env screen): ~24K 43.7 vs 45.2/44.5 ms/step, ~80K 50.1 vs 51.1/51.1; kernels/token 1116 -> 1024 (elementwise ~357 -> ~275); greedy identical; t/s within draft-acceptance noise (QFP15). v4 candidate (multi-request ABBA pending).

Adopted in production profile v4 (2026-10-04, flashnext-v4-abba): v3 -> v4 +2.1% at ~8K and ~64K, complete separation, greedy identical across 8 arms per depth. State stays evaluated until a qualification package exists.

---

# 1326_sched_async_host_inputs

**Status:** validated
**Plan item:** QFP16

## What it does

With `BIGCHERRY_SCHED_ASYNC_INPUTS=1`, ggml_backend_sched_compute_splits copies a split input whose source buffer is
host-resident (non-weights, contiguous) with ggml_backend_tensor_set_async on the split backend instead of
synchronizing the split backend and copying synchronously; the meta (-sm tensor) backend fans the copy out to every
device stream, and its set_tensor_async now falls back to the synchronous buffer path for split states it cannot
splice instead of aborting. 1325 measured ~2.6 ms per target verify round and ~0.4 ms per draft call in these copies.

## Hardware result (2026-10-04, flashnext-1326-d24k/d80k, env screen on one v4+1326 build)

~24K 44.3/44.5 -> 40.0 ms/step (-10%), 72.2/71.9 -> 78.0 t/s; ~80K 50.6/50.0 -> 46.7 ms/step (-7%), 58.8/60.2 -> 64.5 t/s; greedy identical at both depths; acceptance equal. Target submit 5.7 -> 3.0 ms/round; meta split input handling 2.56 -> 0.38 ms; draft split input 0.45-0.57 -> 0.014-0.016 ms/call. Profile v5 candidate (with the prefill patches 1237/1265/1253 if their screen is clean).

Adopted in production profile v5 (2026-10-04, flashnext-v5-abba): v4 -> v5 decode +10.8% at ~8K, +7.8% at ~64K; prefill +3.7% / +7.2%; complete separation; greedy identical across 8 arms per depth.

## 2026-10-04 lifetime fix (reviewer-gpt-agent req_3b37d17e61374a6a)

The fast path now copies each host input into scheduler-owned pageable staging (per input copy) before ggml_backend_tensor_set_async: the source may be a pinned host buffer (truly async DMA) that the caller rewrites on the next set_inputs (e.g. the next prefill ubatch). The results above were measured before this fix; re-measure (queue-v5b-abba.sh) before final adoption.

## 2026-10-04 v5c re-measure (staged) and size cap

v5c ABBA x2 with pageable staging: decode ~8K 79.1 -> 86.5 t/s (+9.4%), ~64K 57.1 -> 61.5 (+7.8%), greedy identical 8/8 per depth, but prefill -2.3% / -2% (the earlier +3.7/+7.2% prefill came from the unsafe zero-copy path). The fast path is now limited to inputs <= 4 MiB (decode KQ mask at 240K for a 4-token verify is ~2 MB; prefill masks ~10 MB take the upstream path), so prefill returns to v4 behaviour. A pinned staging ring with copy-slot events could recover the prefill gain safely (future work).


## 2026-10-06 async-semantics clarification

ROCm HIP documents that hipMemcpyAsync with non-pinned host memory is performed synchronously. The current scheduler-owned std::vector staging is pageable, so its production decode gain must not be described as true H2D overlap. Its value is that the source lifetime is made safe while the scheduler avoids the old per-input destination-backend synchronization; the HIP pageable transfer itself may block the host.

This also explains why replacing the unsafe direct pinned source with pageable staging retained the decode win but lost the prefill gain. QFP16 now owns one bounded residual gate: measure whether large prefill input handling remains >=1 ms or >=3% of prefill wall time. Only then consider extending this same 1326 owner with a bounded pinned staging ring whose slots are protected by completion events from every consuming destination device. Otherwise retain the current <=4 MiB path and close the residual.

---

# 1327_qsa_host_remap

**Status:** validated
**Plan item:** QFP13

## What it does

Pin 0504396 (upstream #29819) remaps dead QSA selection slots to private dump rows with ~12 small graph ops per QSA
layer; the kernel census after the bump showed +~54 kernels per token per XTX and decode -2.3% at ~8K. With
`BIGCHERRY_QSA_HOST_REMAP=1` the two terms that depend only on host data - `live_tail` (from the host input tail_idxs)
and `dump` (n_kv + slot) - are computed once per graph on the host in the kpool input's set_input and fed to every QSA
layer, removing six ops per layer. Values are bit-identical; the device live_pool and final remap are unchanged.

## Hardware result (2026-10-04, pin 0504396, env screen on one v5+1327 build)

~24K 41.4 vs 42.1/41.4 ms/step, ~80K 46.8 vs 47.8/47.1 (0..-1%); greedy identical at both depths. Census: kernels/token 1053 -> 1027 per XTX (elementwise 312 -> 292); the old pin was ~1000, so ~half of the #29819 cost is removed. The rest (device live_pool/get_rows/repeat/concat/final remap) needs a fused remap kernel. Small win, no regression: include in the next profile.

---

# 1332_qsa_token_chunk

**Status:** validated
**Plan item:** QFP17

## What it does

With `BIGCHERRY_QSA_CHUNK=<tokens>` (e.g. 256), `build_qsa_sel` returns the remapped selection indices instead of
the dense `[n_kv + n_sel, T]` mask, and `build_attn_qsa` builds the same mask (fill -inf, scatter zeros, out-of-place
add of the causal rows) per chunk of query tokens and runs QSA flash attention per chunk, concatenating the outputs.
The 1331 peak trace showed the ub1024 compute-buffer peak is the two QSA masks (484 + 480 MiB) next to the kq_mask
input (480 MiB); chunking keeps one chunk's pair live (~240 MiB at 256 of 1024). Off by default. Conflicts with 1330.

## Result (b11402, on top of 1334)

`-ub 1024` with `BIGCHERRY_QSA_CHUNK=256` against `-ub 512`, production config with MTP: prefill +4% at ~99K and
~202K tokens, decode time per step unchanged, next-token fidelity within the envelope of the other attention-path
changes. Evidence and the promotion rationale are in README.md.

---

# 1333_mixed_batch_on_demand

**Status:** validated
**Plan item:** QFP22

Kind: performance fix for an upstream regression, no flag.

Upstream #29622 (0bb496dbd) builds a third, "mixed token/embd" input branch into every graph of every architecture
that is not on a short exclusion list: three more graph inputs per ubatch (one as large as the embedding input), a
dup, a second token-embedding lookup and a `set_rows`. The branch is only selected for a ubatch that mixes token ids
and embedding rows.

The patch adds `ubatch.is_mixed()` to the condition in `build_inp_embd`, so token-only and embedding-only ubatches
get the pre-#29622 graph and a mixed ubatch gets upstream's graph. Graph reuse already distinguishes the two
(`llm_graph_params` compares `is_mixed()`). The batch allocator is untouched and still accepts mixed batches.

Trade-off: the mixed graph is not part of the worst-case reserve any more, so the first mixed ubatch of a context
makes the scheduler reallocate once.

## Evidence

Bisect on Brutus (2x 7900 XTX + R9700 tensor split, 6900 XT MTP drafter), recipe deploy-v6-plus-chunk on every build,
each point compared with b11402 in the same session, greedy text and draft acceptance identical throughout:

| Upstream point | Flash-Next 24K MTP decode (ms/step) | 27B prefill 10K / 32K (t/s) |
|---|---|---|
| 0504396 (old pin) | 41.1 - 41.8 | 1285 - 1292 / 1251 |
| 0eb6d9a81 (#29940) | 41.2 | 1292 - 1296 / 1246 - 1250 |
| 2ca15f540 (#29612) | 41.1 | 1290 - 1291 / 1247 |
| 0bb496dbd (#29622) | 44.6 | 1267 - 1276 / 1228 - 1230 |
| b11401 (a7fb71fab) | 42.5 | 1279 - 1280 / 1234 |
| b11402 (d89651a7b) | 42.3 - 43.5 | 1255 - 1282 / 1227 - 1236 |

First form of this patch (whole feature behind `BIGCHERRY_MIXED_BATCH`, default off; build b-mixoff on b11402)
against the old-pin build in one session: Flash-Next 41.1 ms/step vs 41.8 / 41.4; 27B prefill 10K 1290.7 / 1291.1 vs
1291.9 t/s. That form was replaced by the on-demand condition before promotion.

- Hardware confirmation of the on-demand form and the mixed-batch test: README.md.
- Which part of the branch costs the time has not been profiled (for Flash-Next the token embedding table is on the
  CPU, so the second lookup adds a CPU-side node to every graph).
- 27B prefill at 32K stays about 0.3% below the old pin with either form of the patch; that part of the b11402
  regression is not from #29622. It appears between upstream 0eb6d9a81 and 2ca15f540, i.e. with #29612 (CUDA
  swizzling refactor), by elimination (QFP22).

---

# 1334_hip_sparse_flash_attn

**Status:** validated
**Plan item:** QFP25/QFP17

Kind: enhancement, on by default; `BIGCHERRY_FA_SPARSE=0` is the off switch.

Enables upstream's sparse flash attention on RDNA3/RDNA4 WMMA. Upstream compacts the attention mask into one index
list per tile of queries and gathers only those K/V cells in the MMA kernel, but compiles the path out for HIP and
selects it on NVIDIA only. On AMD, Qwen4Exp QSA attention therefore reads the whole KV cache although each query can
see about 2048 cells.

The patch adds a HIP version of the index kernel (AMD wave ballot and popcount, correct for 32- or 64-lane waves), compiles the host side and
the two dispatch sites for HIP, accepts RDNA WMMA in the selection when the flag is set, and makes the RDNA tile-shape
choice pick ncols2 = 8 (the only shape with sparse kernels) when the sparse path would be taken.

## Evidence

Motivation (b11402, production config, rocprofv3 kernel trace of one uncached prefill): flash attention is 1.96 s of
23.5 s kernel time per XTX over 31.8K tokens and 18.9 s of 89.5 s over 99.3K tokens (32 -> 97 ms per 512-token
ubatch); the R9700 holds no attention and spends the equivalent time waiting in the all-reduce.

Results (b11402, Brutus, 2026-10-05/06; details and the promotion rationale in README.md):

- Prefill, ABBA, production config: ~99K tokens 862.4 / 872.1 -> 978.7 / 982.5 t/s (+13%); ~202K tokens
  658.2 / 663.5 -> 846.1 / 847.8 t/s (+28%). 24K MTP decode unchanged (41.5 / 41.6 vs 41.2 - 41.4 ms/step).
- Correctness: test-backend-ops sparse-mask flash attention matches the CPU backend on all four GPUs with the flag
  off and on, with the activation marker proving the 8x8 sparse kernel ran.
- Fidelity: not bit-identical to the dense path (different summation order feeding a discrete top-k selection);
  against a CPU f32 reference the sparse path is no further away than the dense path.

---

# 1339_meta_memory_report

**Status:** validated
**Plan item:** MSM01

Kind: diagnostic, flag `BIGCHERRY_META_MEM` (default 0).

Prints the real size of every tensor-split (meta) buffer on each device: the scheduler's compute arena, which is
allocated at the same size on every device, and the static buffers (weights, KV, indexer state) with the size of
each device's slices and the first tensor's name. The server otherwise logs one size per Meta buffer, which hides
what each card holds. Nothing is allocated differently.

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1339_meta_memory_report.py`. Hardware: pending.

---

# 1340_meta_per_device_arena

**Status:** validated  
**Plan item:** MSM02

Kind: enhancement, on by default; `BIGCHERRY_META_PER_DEVICE_ARENA=0` restores the common-size arena.

With the flag enabled, the Meta tensor-split compute buffer no longer gives every simple device the scheduler's common physical arena size. `ggml_backend_sched_reserve()` materialises the scheduler's worst-case measure graph once, then 1340 translates that graph to each simple backend. Each device owns one grow-only physical gallocr arena; graph-shape gallocr plans keep allocation metadata only and bind into that shared arena during compute.

Compute-time growth of the physical arena is not a normal path: it emits `GGML_LOG_ERROR` (`arena_grew`) and is counted. A new layout inside the reserved arena is normal and quiet (`arena_replan` under `BIGCHERRY_META_MEM=1`).

With `BIGCHERRY_META_MEM=1`, load/reserve reports:

`BIGCHERRY_META_MEM arena dev=<n> buft=<name> reserved_mib=<MiB> plans=<count> replans=<count>`

Static weight/KV allocations are unchanged. With the flag off, the upstream common-size Meta compute-buffer path is unchanged. Patch 1341 can additionally make subset-inactive mirrored inputs zero-sized before this reserve translation, so devices reserve only for tensors they actually own.

## Offline evidence

- `tools/tests/patch/test_1340_meta_per_device_arena.py` applies against pinned b11402 source via `git show`, composes the active Meta/Qwen patch stack, checks idempotence/fail-closed anchors, verifies guards only match post-edit output, verifies reserve-only growth/shared-owner binding, and locks the `[experiment.meta-memory]` selection.
- `tools/tests/patch/test_1341_meta_subset_mirrored.py` is run in the same gate.
- Required pre-push gate: `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --experiment meta-memory` must report zero failures.

## Hardware status

Pending owner build/runtime validation at ctx 245760 / ub512. Acceptance signal: final per-device `reserved_mib` is established during load and stays fixed through fill, `replans=0`, and output/fusion behaviour matches production.

---

# 1341_meta_subset_mirrored

**Status:** validated
**Plan item:** MSM03

Kind: enhancement, flag `BIGCHERRY_META_SUBSET_MIRROR` (default 0).

Adds `active_mask` to Meta split state (0 = legacy/all devices), propagates it across MIRRORED operations, and honors it in simple-tensor creation and Meta transfers. This first step seeds only Qwen4Exp `cache_idx_(k|v)_l*` from nonzero `BIGCHERRY_ATTN_TS` entries. KQ/kpool/QSA compute inputs are intentionally not seeded yet.

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1341_meta_subset_mirrored.py`.
- Hardware: see README.md (identical output, no speed change, 1,440 MiB less on the R9700 at ctx 245760).

---

# 1344_dsv4_hc_grid_index

**Status:** validated
**Plan item:** QFP35

## What it does

The Qwen4Exp hyper-connection PRE and POST kernels (`ggml-cuda/dsv4-hc.cu`) take their coordinates from the launch
grid: PRE runs on a 2-D grid (embedding index in blocks of 256, token), POST on a 3-D grid (embedding index,
destination stream, token). On by default; `BIGCHERRY_HC_GRID_INDEX=0` restores the flat kernels.

## Why

The flat kernels run on a 1-D grid and every thread recovers its coordinates with 64-bit `%` and `/` by run-time
dimensions (two in PRE, three in POST). AMD GPUs have no 64-bit integer divide, so each is an emulated loop executed
once per output element, in kernels that otherwise do a few multiply-adds. An external report on the same model family
measured 230 -> 79 us per call from this change; that number is a hypothesis here until measured.

## Scope

Indexing only. Each thread computes the expression the flat kernel computes for the same element, so the result is
bit-identical. The Q8_1 PRE path of 1311 is not touched. A grid dimension is limited to 65535; a larger batch uses
the flat kernels.

## Activation

`BIGCHERRY_PATCH_TRACE=1` prints `BIGCHERRY_PATCH_HIT patch=1344_dsv4_hc_grid_index path=pre|post` once each.

---

# 1345_moe_ids_multiwarp

**Status:** validated
**Plan item:** QFP36

## What it does

For batches of 128 tokens and more, the MoE routing helper (`ggml_cuda_launch_mm_ids_helper`, `ggml-cuda/mmid.cu`)
runs 8 warps per expert block where the native helper runs one. Each warp walks its own contiguous slice of the
tokens with the native per-warp code; the warps exchange their counts once and write their rows at the right
offsets. On by default; `BIGCHERRY_MOE_IDS_MULTIWARP=0` restores the native helper.

## Why

The native helper is one warp per expert walking every token in sequence (256 dependent iterations for 512 tokens at
top-10), for each of the 512 experts. In the Flash-Next prefill profile on the production build `mm_ids_helper<10>`
is about 3.8% of all kernel time on the three target cards (run `gate0-d24576`).

## Scope

The rows of an expert stay in ascending token order, so `ids_src1` (forward and inverse form), `ids_dst` and
`expert_bounds` are byte-identical to the native helper's; nothing downstream changes. Smaller batches and the generic
top-k path use the native helper. Works on the global ids and on
1281's translated local ids (an id of INT_MAX is in no expert's list, as in the native helper).

## Activation

`BIGCHERRY_PATCH_TRACE=1` prints `BIGCHERRY_PATCH_HIT patch=1345_moe_ids_multiwarp used= experts= tokens= warps=
inverse=` once.

---

# 1347_f32_thin_transposed_mmvf

**Status:** validated
**Plan item:** QFP34

## What it does

An F32 matmul whose weight has 2 to 8 rows and whose activation has more than 8 columns runs through the float vector
kernel (`mul_mat_vec_f`) with the roles swapped, instead of rocBLAS SGEMM: the activation matrix is the matrix, the
weight rows are the vectors. One launch computes the result transposed into a pool buffer and a small kernel writes it
into the destination. On by default; `BIGCHERRY_F32_THIN_MMVF=0` restores SGEMM.

## Why

Flash-Next's hyper-connection blocks project the 10240-wide state onto 4 values per token (`hc_attn_inject`,
`hc_ffn_inject`, F32 [10240, 4]), twice per layer: 96 calls per 512-token prefill chunk, each a GEMM with a 4-row
weight. SGEMM's tiled kernel is built for large square problems. In the prefill kernel profile of the production
build (run `gate0-d24576`) the SGEMM kernel serving these calls and the router is 12.7% of an XTX's kernel time, at
332 us per call. Upstream already does the role swap for a one-row weight; this is the same for the batch width the
vector kernel supports.

## Scope

Plain MUL_MAT only (not MUL_MAT_ID), F32 weight and activation, contiguous tensors, no batch dimensions. Wider
weights (the 512-row router, the 48-row SSM projections) stay on SGEMM. The sums are formed in another order than
SGEMM forms them: equal within float tolerance, not bit-identical.

## Activation

`BIGCHERRY_PATCH_TRACE=1` prints `BIGCHERRY_PATCH_HIT patch=1347_f32_thin_transposed_mmvf k= rows= cols=` once.

---
