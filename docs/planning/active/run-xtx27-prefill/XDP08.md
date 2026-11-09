---
id: XDP08
order: 7
plan: run-xtx27-prefill
state: active
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Apply gfx11 metadata-reuse findings to existing INT8, MXFP4 and MoE kernels

## Goal and limits

Transfer XDP07's **measured mechanism** (avoiding repeated global
metadata loads and long-lived scale/address registers) to the existing
BigCherry gfx11 libr3 kernels, rather than replacing them with a
second standalone INT4 implementation. Keep the current worktree and
branch; do not edit pinned `vendor/radiance`. Production defaults and
output formats remain unchanged until paired correctness/performance
and real-model quality pass. RAD50 has priority over the GPUs; check
`agent-lease status` before hardware, never interrupt a held lease.

XDP07 evidence: same-card M512 N17408 K5120 native IU4 128x128
staged-scale W4A4 1,057 us versus 1,952 us before staging; 149→110
VGPR, 0 spills. The lift **does not apply automatically** to every
precision. It came from moving global per-group scales and sums
out of each output fold, and reducing register/address liveness.
No cross-format improvement has yet been measured.

## Read-only source audit and decisions

| Existing kernel | Present mechanism | Is XDP07 header staging missing? | Decision |
|---|---|---|---|
| Native INT8 `gemm_i8a8_tiled` | 128x128 tile, 8 waves, A double-buffered LDS, A scales in `s_as[2][128]`, W per-row f16 scales loaded by each weight wave as two `uint4` chunks per K128 | **Only W scales are unstaged**, with duplicated wave/lane loads | Controlled weight-scale-LDS ablation first |
| INT8 RB8/RB32 / IQR | RB-specific scale factors, bias correction and special fold; activations already staged | Not identical to per-row INT8 | **Do not change** RB/IQR in first experiment |
| MXFP4 tiled | weight slab already LDS, decode table `sMag[32]` LDS, activation ratios `sR` LDS, scale carries across K128 | Relevant quantities are **already staged**; rewrite widens W to bf16 for gfx11 | Profile table lookups/weight widening and register lifetimes, not duplicate scale staging |
| MoE gfx11 `moe_gemm_q` | `s_as` and `s_ws` already staged per tile/pass, prefetch and LDS-only synchronization | **No** simple missing scale buffer | Profile E4M3→BF16 widening, occupancy, output ownership, CPU/helper coordination before code changes |

The INT8 upstream body explicitly reports a **+4%** gain from staging
activation scales in `s_as`, while its best per-row INT8 tile was
**+15–19%** versus an RB32 variant in its own tested shapes. RAD48
already changed gfx11 to **native IU8 WMMA**, improving one M2048,
N34816,K5120 fixture from 25.8ms to 8.33ms (3.1x) with numerical
comparison. Those are prior measurements, not new XDP08 results.

## XDP08-A: optional native INT8 per-row weight-scale LDS staging

### Exact semantic contract

Only `I8 && RB == 0 && !ROWF` in upstream
`r4d_gemm_8a8_prefill.h`, not fp8 and not the RB8/RB32/IQR
paths. Keep `raw`, `out`, K128 folded fma, the existing
native IU8 activation fragment mapping, weight prefetch,
global W/A layouts, output `C`, scale order and all result
types **unchanged**. For each weight row `n` in the
128-row output block, old per-wave scale sources are

```text
Sw[((wt0+i)*nkb + kb)*16 + (lane>>4)*8 + t], t=0..7
```

The replacement has 128 cooperative lanes (`tid<128`)
load exactly one f16 scale each into a 128-element LDS array:

```text
s_ws[tid] = Sw[((n0/16 + tid/16)*nkb + kb)*16 + (tid%16)]
```

The existing `tile(kb,0)` `lds_barrier()` publishes these
values. After the K128's two 64-wide weight/activation tiles
and their barriers, the unchanged fold reads its eight weight
scales per wave from
`s_ws[(ww*kPTW+i)*16 + (lane>>4)*8]` as `uint4`.
Every source and destination index must be proven equal on
128 rows, all `kb<nkb`, and both lane halves. There must
be no race with the next K128 group's write: reads occur
after both tile barriers and before the next loop's stage.

### Runtime/compile safety

Use `R3_I8_SWS_LDS` as an **OFF-by-default CMake option**
on libr3. The native rewrite can insert guarded code, but the
default kernel is instruction-for-instruction unaffected by the
opt-in portion; `I8 && RB==0` prevents touching fp8 and
RB variants even when the option is enabled.
Fail rewrite drift with `r3_rewrite.py --check`.
An experiment must be built as a separate libr3 plugin
artifact, not by overwriting the currently serving plugin.

Expected possible gain: fewer repeated global half-scale reads,
cache sectors and scalar/vector address lifetimes. Potential
regressions: extra 256-byte LDS, bank conflicts, shifted global
load scheduling, or lower clock/occupancy. The old kernel may
already benefit from cache/broadcast and vectorized `uint4`
reads. **No performance claim** until the exact variant is built
and measured.

### Gates

1. **Card-free**: index-bijection simulation over all
   `ww=0..3, i=0..1, lane=0..31, t=0..7`, with multiple
   `nkb`, `kb`, `n0`; prove all 128 scale entries
   staged by one thread and no read-before-barrier.
2. **Rewrite**: full native libr3 rule drift check and
   no-vendor-edit proof. Build opt-in separately and inspect
   gfx1100 code object for IU8, VGPR/SGPR/LDS/spills.
3. **Numeric**: `rad-kbench` or selftest on INT8 dense and
   tail geometries versus the *same unchanged* known-good
   implementation, including BF16 output. Require exact
   equivalence to the previous scale/fma order or a
   documented bounded tolerance; independently check
   no regressions for fp8/RB8/RB32/IQR.
4. **Performance**: matched same-card three-plus independent
   rotated A/B runs for M=64,128,512,2048;
   N=34816 K=5120 (dense 27B) and at least one smaller shape.
   Separate GEMM from runtime A8 quantizer and all-reduce.
   Reject if worse or below noise; record median and variance.
5. **Serving**: original BF16-origin model quality (KLD,
   PPL/top-1) and exact TP2 prefill/decode + KV safety,
   cold/warm TTFT, before enabling production.
   No change to conversion or format policy based on
   this optional kernel optimization.

## XDP08-B: MXFP4 and MoE follow-ups (not blindly staged)

MXFP4: Profile global weights and decoded bf16 staging,
`sMag` lookup pressure, activation scratch `3x` size,
`sR` scale carry, VGPR and memory-store instructions.
Existing native rewrite has explicit one-fragment-at-a-time
activation reads and one-time gfx11→gfx12 accumulator
conversion. A new scale cache would duplicate those
mechanisms. Only implement a genuinely distinct,
profile-justified optimization.

MoE: The existing `s_as[kBpS*FM]` and
`s_ws[kSubT*FN]` are explicit, cooperative tile
staging. Prior Flash-Next profile highlighted a
`moe_gemm_q` bottleneck, but that runtime, expert
density and tensor-parallel layout differ from this
dense 27B. Confirm whether E4M3 widening, remote
helper routing, or CPU/all-reduce is dominant before
changing math/layout. Do not replace E4M3 with IU8:
the codebook is non-linear and native IU8 would
change quantization semantics.

## Progress/decision log

- 2026-11-09: Audited pinned libr4d sources and gfx11 native
  rewrite. The main discovery is **INT8 A-scale,
  MXFP4 ratio/table and MoE A/W scales are already
  staged**; only a narrow per-row INT8 W-scale
  source remains for a first controlled trial.
  No production rollout authorized.

## Implementation checkpoint and benchmark setup

The managed change is now in
`engines/radiance/components/libr3/native/r4d_gemm_8a8_prefill.h.rw`,
behind `-DR3_I8_SWS_LDS=ON` in libr3's
`CMakeLists.txt`. The option is **OFF** by default and
requires `R3_NATIVE=ON`. The implementation uses
a 16-byte-aligned, 128 × `uint16_t` LDS slab (256 bytes)
for the per-row weight scales. Threads 0..127 each
load exactly one scale, and the existing pair of
`tile(kb,half)` barriers publishes the values.
After both K64 tiles, a wave reads the same two
16-byte fragments it previously loaded from global,
with **no change** to `bscale`, the per-8-row output
scale products, f32 fma, accumulator conversions,
or BF16 output stores.

The source-only logical load reduction is from
8 waves × 32 lanes × 2 `uint4` fragments
(**8,192 B of requested, duplicate weight-scale data
per K128 workgroup**) to 128 unique 2-byte
loads (**256 B**, plus LDS consumers). This is
not an asserted 32× DRAM reduction. The original
loads duplicate addresses within 16 lanes and
across two M wave bands, so caching/broadcast
may already deduplicate the physical transactions.
Also the upstream old weight-scale prefetch
overlapped with tile computation; shifting the
consumer to after the tile may expose LDS
latency. Those risks are why the feature must
remain OFF until measured.

**Completed card-free gates:** all 128 row scale
addresses, all wave lanes and both 8-element
fragments are bijective under the upstream
`[N/16][K/128][16]` format, tested at K128,
K256, K5120 and nonzero N tile starts.
The pinned-vendor rewrite passed with
`{"failures":{}}`, with no vendor edits.
The original tile entry checks `N%128==0`,
so the cooperative W-scale staging does not
introduce a new partial-N case.
The native rewrite drift tests (12 cases)
and XDP card-free suite (66 cases total at
this checkpoint) pass. **GPU/ISA compile,
numeric and kernel performance are pending**
while RAD49 owns the lease.

### Reproducible isolated A/B recipe (when the GPU is free)

Use the *existing* `tools/lab/radiance/kbench_ab.py`
rather than inventing another tester. The
recorded per-op fixture is the existing
`q27-w8a8-tok2048.rkb` fixture; record its
SHA-256. Use one pinned radiance source,
ROCm toolchain and runtime prefix for
both variants. Build two **separate** libr3
plugin directories (never overwrite an
active `radiance_home`):

```sh
# From the existing worktree with correctly chosen, pinned
# RADIANCE_SRC and RADIANCE_PREFIX and BC_WORK_ROOT.
for mode in OFF ON; do
  cmake -S engines/radiance/components/libr3 \
    -B "${BC_WORK_ROOT}/xtx27-prefill/int8-stage/build-${mode}" \
    -G Ninja \
    -DRADIANCE_SRC="$RADIANCE_SRC" \
    -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX" \
    -DR3_TARGETS=gfx1100 -DR3_NATIVE=ON \
    -DR3_I8_SWS_LDS="$mode"
  # Run ninja only after checking the shared agent lease,
  # and with low priority (nice -n 15).
done
```

Then construct `sides.json` for the existing
A/B driver, with two separate `radiance_home`
copies, identical non-libr3 plugins, pinned
rad-kbench, libref and fixture, and only the
libr3 kernel differing. Run **checked**
`rad-kbench` INT8 cases before the
`kbench_ab.py run --op gemm_nt_q --repeat-a`
timing pass; its timing-only mode is
not a numerical test. Choose timeouts
based on an initial one-case probe.
Include M=64,128,512,2048 and the
N34816/K5120 dense 27B shape. For
a win, record source/binary SHA, old/new
native IU8 code objects, register and
LDS changes, and three independent
rotated runs at comparable clocks.
A compiler-only gain or small isolated
speedup does not mean actual serving
improves; KLD, full TP2 prefill/decode
and physical GPU memory are separate gates.

**Decision XDP08-D1:** do not retrofit
already-staged A-scale, MXFP4 ratio or
MoE per-tile scale quantities with
redundant LDS arrays. Test the one
missing INT8 W-scale reuse independently.
Keep OFF unless both isolated numerical
equivalence and repeatable same-card
speed-up are demonstrated.
