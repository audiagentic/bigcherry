---
id: VTP07
order: 6
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Native IQ6R gfx1100 relayout blocker — initial Q6/Q8 hardware screen

## Actual hardware and provenance (2026-11-10)

The GPU lease was free ("no leases"), no KFD processes were present, and
all cards had negligible utilisation/no allocated VRAM. Isolated,
exclusive-lease runs on XTX0, XTX1 and R9700 used pinned prebuilt
`builds/f17f21fda44979e9abe84f13e96b8896/08b5dd81b4874390bf0cbf489db0d7df`
and the VTP05 model-shape harness on branch `exp/xtx27-prefill-opt`.
The three physical cards were enumerated as gfx1100/gfx1100/gfx1201;
HIP/ROCr device visibility was checked by harness receipt. No model
checkpoint, production engine, P2P path or numerical accuracy test was
run. Every successful case is **TIMING_OBTAINED_NUMERICAL_UNCHECKED**.

### N=output width, K=input width, M=rows=4

| Format / weights | GPU | Gate/up (N,K) | Gate/up kernel us | Down (N,K) | Down kernel us |
| --- | --- | --- | ---: | --- | ---: |
| W8A8 INT8, warm-only | XTX0 gfx1100 | 10240,5120 | 81.9 | 5120,5120 | 54.9 |
| W8A8 INT8, warm-only | XTX1 gfx1100 | 10240,5120 | 81.3 | 5120,5120 | 56.3 |
| W8A8 INT8, warm-only | R9700 gfx1201 | 14336,5120 | 78.7 | 5120,7168 | 36.2 |
| IQ6R V4, cold 256 MiB | XTX0 gfx1100 | 10240,5120 | **relayout refused** | — | not run |
| IQ6R V4, cold 256 MiB | XTX1 gfx1100 | 10240,5120 | **relayout refused** | — | not run |
| IQ6R V4, cold 256 MiB | R9700 gfx1201 | 14336,5120 | 108.2 | 5120,7168 | 59.6 |

Both XTX failures printed `gemm_iqr_nt_m64 relayout refused`.
The selftest returned exit 0, but our runner rejected it as
`FAILED_NO_SELECTED_KERNEL_TIMING`; a binary symbol/row-presence
check alone was insufficient to detect unsupported relayout. The
gfx1201 R9700 successfully executed the IQR row with valid N/K/M
header and 108.2 us. A 3-card native IQ6R virtual TP4 path is therefore
**not runnable using this pinned gfx11 plugin**. Root cause (missing
layout conversion, unsupported IQR-V4 form or an ABI/dispatch problem)
has not been isolated. Do not say the entire IQR format is inherently
unsupported on gfx11.

The W8A8 **ideal FFN-only two-projection barrier** at M4 is
max(81.9,81.3,78.7) + max(54.9,56.3,36.2) = **138.2 us**,
omitting collective, activation, GDN, attention, KV and all CPU launch
overhead. Summed physical per-card computation is 136.8/137.6/114.9
us (XTX0/XTX1/R9700). This is one short, warmed sample, 3 iterations
per row; not a stable whole-model inference estimate.

The R9700 IQ6R sum is 167.8 us, but the **256 MiB cold flush** makes
comparison with warm W8A8 invalid. The IQ6R-V4 synthetic trellis is
also not automatically the full calibrated mixed-bit IQ6R recipe.

### Receipts and campaign state

- First XTX0 IQ6R failure is preserved in
  `${BC_WORK_ROOT}/xtx27-prefill/quant-shape/campaigns/pilot-20261110-v2/runs/20261110T045645636613Z-iq6r-xtx0/receipt.json`.
- Other nine per-case diagnostics are under
  `${BC_WORK_ROOT}/xtx27-prefill/quant-shape/diagnostics-20261110/runs/`.
  Successful cases record exact file hashes, selected physical index,
  observed N/K/M, actual GPU timing and `numerical_oracle=NOT_TESTED`.
- The frozen pilot-v2 campaign remains **0/12 accepted** because its
  first case did not pass; the separate diagnostic receipts were not
  silently incorporated into its immutable sequential campaign. This
  is intentional. No counts were fabricated and no failure overwritten.
- After every case, exclusive lease was released; check confirmed
  no active leases and no KFD processes.

## Bounded follow-up design

1. **Investigate gfx11 IQR relayout card-free first** (layout/relayout
   registration, intermediate packed IQR storage, encoding descriptor
   and 6-bit synthetic V4 form). Compare the pinned libr3 compatibility
   implementation against the working libr4d path. Reuse the current
   worktree, do not edit pinned upstream. Add a *minimal exact-layout
   oracle* before any new timing. Check whether a managed native gfx11
   rewrite or a supported IQ6R variant is required; do not invent a
   fallback or silently benchmark W8 as Q6.
2. Run independent IQ6R oracle on both XTXs only after relayout is fixed,
   then rerun equivalent cold/warm model-shape kernels and a matching
   gfx1201 control. Stop if layout/quality cannot be made correct.
3. **W8A8 is currently the viable common kernel path** on all 3 GPUs
   at M4. Re-run balanced, longer-iteration forward/reverse paired
   samples, include M1/8 and both FFN projections and actual numerical
   libref/activation reference. The timing-only W8A8 row was warm
   even when `--cold` is accepted by upstream; use rad-kbench for a
   true cold measurement. No full 27B allocation before qualification.
4. Only after these screens compare weight splits, physical R9700
   combined-vs-two-shard execution and three-way host collective.
   Neither a fast kernel nor a theoretical 138.2-us floor authorises
   changing production dispatch or a virtual-GPU illusion.

## 2026-11-10 follow-up: concrete gfx11 root cause and bounded fix

**Root cause confirmed, superseding the earlier unknown-cause hypothesis.**
The gfx1100 `engines/radiance/components/libr3/CMakeLists.txt` declared
`r4d_rows.cpp`, `r4d_layout.cpp`, `r4d_fp8_layout.cpp`,
`r4d_shapes.cpp` and `r4d_fuse.cpp`, but **omitted
`r4d_iqr_layout.cpp`**. Pinned upstream `libr4d/CMakeLists.txt`
includes that source in `R4D_DECLARATION`. The omitted file is pure host
C++ and provides the core IQ6R layout / relayout / unrelayout hooks.

Proof from the original gfx11 plugin:

- Its generated `components/libr3/r3_stubs.S` contains labels
  `r4d_rad_layout_iqr`, `r4d_rad_relayout_iqr` and
  `r4d_rad_unrelayout_iqr` that return `RAD_E_UNSUPPORTED = -18`.
- `nm -D` places all three between `r3_stub_begin` and
  `r3_stub_end`. The IQR **GEMM launch** was real, so its row existed
  and an ELF string or row-presence preflight falsely accepted it.
- The pinned gfx1201 `libr4d.so` exports proper 0x526 / 0x118b /
  0x1262-byte implementations instead of the stub functions.
- The corrected host source passes standalone C++17 syntax compilation
  and the isolated gfx11 scratch build compiled
  `r4d_iqr_layout.cpp.o` with real global symbols for all three hooks.

**Minimal fix:** add `${r4d}/r4d_iqr_layout.cpp` to the libr3 host
`set(declaration ...)` alongside the other layouts. Do not edit
`vendor/radiance` or change kernel dispatch priority. A matching
card-free test at `tools/tests/lab/test_xtx27_iqr_build.py` checks the
complete host source list so another layout function cannot be
silently replaced by a -18 stub.

**Additional guard:** VTP05's `build_paths` checks `nm -D` dynamic
symbols and rejects the selected IQ6R plugin if any required hooks
fall inside `r3_stub_begin/r3_stub_end`, or if the symbols are absent.
The old gfx11 build fails that check; the existing gfx1201 build passes.
Synthetic tests cover stub/nonstub symbols and a failed nm call.
This prevents future *campaign preparation* from greenlighting a
broken IQR kernel.

**Validation:** 110 XTX27 card-free tests and Ruff passed. A
single-unit gfx11 `libr3` build was configured in scratch with
`-DR3_ONLY=r4d_gemm_iqr` and `-DR3_TARGETS=gfx1100`; host layout
object compiled successfully and IQR GPU code compilation is in
progress at the time of this note. Production engine binaries and
GPU data remain untouched. The old frozen pilot-v2 manifest is
invalidated by the new benchmark runner hash; recreate after a
complete same-source prebuilt binary/plugin pair is available.

**Remaining gates:** finish and inspect the isolated plugin's real
symbols and registration, run a correctly leased IQ6R relayout and
numerical check on **both** XTX cards, and only then collect new
timings against a matching gfx1201 control. A host-source build fix
does not by itself prove that the gfx11 GPU kernel executes correctly.
