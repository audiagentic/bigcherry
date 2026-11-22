# RUM13 — one native AQL exact collective lab

**2026-11-22.** Scope: shared/model-independent Radiance test harness only.
Implementation branch `agent/rum13-native-aql`, based on the current
unmerged LTX foundation branch so the old lab's source is available.
Not a production kernel or transport change. RUM13 is a planning
candidate pending the Planning MCP.

## Deletion and ownership

Relocated the single remaining supported harness from
`tools/lab/radiance/ltx-tp3/mixed-ar-gate/` to
`tools/lab/radiance/native-mixed-ar/`, changing three source/build
files and the lab README in-place and updating their tooling
dispositions and the LTX lab reference.

| Metric | Before (C07 + C08) | After RUM13 | Reduction |
| --- | ---: | ---: | ---: |
| CMake | 104 lines | 72 | 32 |
| Native C++ | 271 lines | 254 | 17 |
| Python bounded runner | 68 lines | 68 | 0 |
| Lab README | 89 lines | 78 | 11 |
| **Implementation source total** (CMake/C++/Python) | **443** | **394** | **49 lines** |
| **Lab package total** | **532** | **472** | **60 lines** |
| Executable build targets | 2 | 1 | 1 removed |
| HIP/AQL preprocessor alternative blocks | 7 | 0 | 7 removed |
| HIP-off standalone core dependency | 1 | 0 | 1 removed |
| Separately compiled core shared-slot implementation | 1 | 0 | 1 removed |

The surviving executable is `radiance_mixed_ar_gate`. It links the
exact HIP-enabled Radiance 2.2/ABI44 core **whole** including native
AQL dispatch, `hipLaunchKernel`, `__hipRegisterFatBinary` and
the native `rad_bc_shared_slot`. It preserves native gfx1100
`libr3.so` and gfx1201 `libr4d.so` kernel rows, their shared
rendezvous, BF16 ascending-rank exact sum, and the original
eleven × three rank numerical cases. No installed production
file or patch was changed. The earlier hard-coded media paths
were replaced with Radiance's existing `PkgConfig` discovery for
`libavformat`, `libavcodec`, `libavutil`, `libswscale`
and `libswresample`. A fresh native CMake configure with the
Radiance-matched CMake prefixes, build and CTest all passed.

## Host verification

- CMake configure and native C++ build: **pass** (one target).
- The same CMake configured with the previously used HIP-off
  Radiance core: **refused** with the expected exact-version/source/
  HIP-on error; no compatibility fallback.
- `nm -D` verifies four exported symbols: `hipLaunchKernel`,
  `__hipRegisterFatBinary`, `rad_bc_shared_slot` and
  `rad::bc_shared_slot_keys`.
- CTest `rum13_reject_duplicate_devices`: **1/1 passed**.
  Executing the binary with plugin paths that deliberately do
  not exist and `0,0,0` returns **1** with
  `duplicate physical GPU ordinal` *before* attempting any
  `dlopen` or setting up device streams.
- Existing LTX C01–C06 native CTests remained **5/5** on the
  separately owned source-backed reference build. They are host
  declarations, not issued GPU graph parity.

## GPU validation

Leased exclusive three-card hardware (gfx1100 + gfx1100 + gfx1201)
with native Radiance AQL and HIP-enabled 2.2 archive:
The final pkg-config-linked build was also qualified on the same
three physical GPUs: one-shot 33/33 exact in ~0.41 s and
two-shot 33/33 exact in ~0.41 s; this is whole-process elapsed time.

| Row | Native rank × case comparisons | Exit | Whole-run time |
| --- | ---: | ---: | ---: |
| `ar_oneshot_3rank_exact_host` | **33/33 bitwise BF16 exact** | 0 | 0.42 s |
| `ar_twoshot_3rank_exact_host` | **33/33 bitwise BF16 exact** | 0 | 0.41 s |

The two native shared-library participants rendezvoused in their real
shared slot. Case sizes cover 1, 4, 16 and 64 rows, 2048/4096-wide
inputs, both legal 8/8/16 and 12/12/8 head partitions, overlapping
output partial sums and in-/out-of-place exact output reductions.
The durations include process startup/teardown, **not native
per-operator latency**. The prior C08 entire-run results were
approximately 0.41 s each; this is too little evidence to conclude a
speedup/regression beyond test functional parity.

The known unsupported single-GPU three-rank placement is no longer
launched: a specific preflight rejection is tested on the CPU.
The earlier 40.02-second timeout and no-queue-progress warning are
preserved as diagnostic evidence in
`POST-MAIN-2026-11-22.md`. The GPU lease was automatically
released; `agent-lease status` reported `gpu: free` and
`rocm-smi --showpids` reported **no KFD PIDs**.

**Raw results, outside Git:** task work roots
`rum13-native-aql-proof/` and `rum13-native-aql-pkgcfg-proof/`, with one JSON+log pair per selected
native collective row and physical device map.

## Boundaries

RUM13 owns this single model-independent AQL test. RUM06 retains
production cross-DSO shared-slot rendezvous/lifecycle, RUM05 owns
device/admission, LTX02 owns actual checkpoint-backed `RadCtx`
C05 Q/K RMS and C06 self-attention TP1 parity. C08/RUM13 raw
collective numerical success must **not** close LTX02 or enable
production LTX TP3. The LTX upstream world-size >1 guard stays
closed until model-specific parity and safe serving acceptance.
