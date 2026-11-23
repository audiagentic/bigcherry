# VTP50 — K-parallel native token fragments (gfx1100)

## Stage 1 — native B, compile-time resource guard: PASS

The native token load reads one contiguous 16-K row per gfx11 lane into four
dwords and feeds the WMMA directly. The K-group accumulator, weight-fragment
conversion, wave reduction, fold, unroll rule, and prefetch selector are retained.
The separate weight-plane read remains unchanged.

The first ungated unit build compiled and exposed 17 instantiations with spill
or wave-residency loss. The final constexpr selector
(NT == 3u) || (NT <= SB / 128u) was built again; this selects NT1 and NT3
for both scale blocks and NT2 for SB256. SB128 NT2 and both NT4 forms use
the prior path. This rule depends solely on template parameters, never N, K,
M or model-specific shapes.

The final isolated kernel build completed with reduced CPU priority, one build
at a time. The full 48-instantiation code-object metadata comparison reports
no new spills, no increase in VGPR or private bytes, and no waves/SIMD loss;
all 278 other kernels in the device object retain identical resource records.
NT3 SB128 chained improves 240/3/16/6 to 182/0/0/8 for KW2 and KW4.
NT3 SB256 unchained improves 212/0/0/7 to 178/0/0/8 for KW2 and KW4.

Resource fields: VGPR / spilled VGPR / private bytes /
floor(1536 / VGPR) wave32 waves per SIMD. Comparisons use the same
previously validated pre-change implementation, with the gfx1100 objects
extracted from the two plugin artifacts and read using llvm-readobj --notes.
The source-generation difference is limited to this native-B rewrite;
rewrite drift checks and whitespace checks pass.

| KW | SB | NT | chain | before VGPR/spill/private/waves | after VGPR/spill/private/waves |
|---:|---:|---:|---|---|---|
| 1 | 128 | 1 | ch | 117/0/0/13 | 117/0/0/13 |
| 1 | 128 | 1 | un | 114/0/0/13 | 114/0/0/13 |
| 1 | 128 | 2 | ch | 177/0/0/8 | 177/0/0/8 |
| 1 | 128 | 2 | un | 174/0/0/8 | 174/0/0/8 |
| 1 | 128 | 3 | ch | 240/2/12/6 | 180/0/0/8 |
| 1 | 128 | 3 | un | 240/0/0/6 | 177/0/0/8 |
| 1 | 128 | 4 | ch | 179/0/0/8 | 179/0/0/8 |
| 1 | 128 | 4 | un | 176/0/0/8 | 176/0/0/8 |
| 1 | 256 | 1 | ch | 181/0/0/8 | 181/0/0/8 |
| 1 | 256 | 1 | un | 178/0/0/8 | 178/0/0/8 |
| 1 | 256 | 2 | ch | 154/0/0/9 | 154/0/0/9 |
| 1 | 256 | 2 | un | 151/0/0/10 | 151/0/0/10 |
| 1 | 256 | 3 | ch | 213/0/0/7 | 179/0/0/8 |
| 1 | 256 | 3 | un | 210/0/0/7 | 176/0/0/8 |
| 1 | 256 | 4 | ch | 173/0/0/8 | 173/0/0/8 |
| 1 | 256 | 4 | un | 170/0/0/9 | 170/0/0/9 |
| 2 | 128 | 1 | ch | 118/0/0/13 | 118/0/0/13 |
| 2 | 128 | 1 | un | 115/0/0/13 | 115/0/0/13 |
| 2 | 128 | 2 | ch | 178/0/0/8 | 178/0/0/8 |
| 2 | 128 | 2 | un | 175/0/0/8 | 175/0/0/8 |
| 2 | 128 | 3 | ch | 240/3/16/6 | 182/0/0/8 |
| 2 | 128 | 3 | un | 240/1/8/6 | 179/0/0/8 |
| 2 | 128 | 4 | ch | 180/0/0/8 | 180/0/0/8 |
| 2 | 128 | 4 | un | 177/0/0/8 | 177/0/0/8 |
| 2 | 256 | 1 | ch | 182/0/0/8 | 182/0/0/8 |
| 2 | 256 | 1 | un | 179/0/0/8 | 179/0/0/8 |
| 2 | 256 | 2 | ch | 155/0/0/9 | 155/0/0/9 |
| 2 | 256 | 2 | un | 152/0/0/10 | 152/0/0/10 |
| 2 | 256 | 3 | ch | 215/0/0/7 | 181/0/0/8 |
| 2 | 256 | 3 | un | 212/0/0/7 | 178/0/0/8 |
| 2 | 256 | 4 | ch | 174/0/0/8 | 174/0/0/8 |
| 2 | 256 | 4 | un | 171/0/0/8 | 171/0/0/8 |
| 4 | 128 | 1 | ch | 118/0/0/13 | 118/0/0/13 |
| 4 | 128 | 1 | un | 115/0/0/13 | 115/0/0/13 |
| 4 | 128 | 2 | ch | 178/0/0/8 | 178/0/0/8 |
| 4 | 128 | 2 | un | 175/0/0/8 | 175/0/0/8 |
| 4 | 128 | 3 | ch | 240/3/16/6 | 182/0/0/8 |
| 4 | 128 | 3 | un | 240/1/8/6 | 179/0/0/8 |
| 4 | 128 | 4 | ch | 180/0/0/8 | 180/0/0/8 |
| 4 | 128 | 4 | un | 177/0/0/8 | 177/0/0/8 |
| 4 | 256 | 1 | ch | 182/0/0/8 | 182/0/0/8 |
| 4 | 256 | 1 | un | 179/0/0/8 | 179/0/0/8 |
| 4 | 256 | 2 | ch | 155/0/0/9 | 155/0/0/9 |
| 4 | 256 | 2 | un | 152/0/0/10 | 152/0/0/10 |
| 4 | 256 | 3 | ch | 215/0/0/7 | 181/0/0/8 |
| 4 | 256 | 3 | un | 212/0/0/7 | 178/0/0/8 |
| 4 | 256 | 4 | ch | 174/0/0/8 | 174/0/0/8 |
| 4 | 256 | 4 | un | 171/0/0/8 | 171/0/0/8 |

## Stage 2 — exact BF16 byte identity: HARD STOP / FAIL

The first same-process bit-check, IQ8R unchained, ran under its own queued
GPU lease with identical deterministic inputs and scales, comparing the
validated pre-change full plugin with the stage-1 isolated-unit candidate.
All 576 M/shape checks executed and the harness returned nonzero:

**144 PASS, 0 SKIP, 432 FAIL.** M=1–48 failed on all nine shapes.
M=49–64 passed on all nine shapes (the NT4 fallback uses the original
token-fragment route). Every failing check reported equal=0, touched=1:
destination writes occurred, but the complete BF16 output bytes differed.
The failure covers KW2 as well as the standard long-K groups and ragged N.

Individual shape totals (all 16 passes per shape are M=49–64):

- SHAPE gate_up      N=13312 K=5120 PASS=16 SKIP=0
- SHAPE down         N=5120 K=6656 PASS=16 SKIP=0
- SHAPE delta_in     N=6144 K=5120 PASS=16 SKIP=0
- SHAPE delta_out    N=5120 K=2304 PASS=16 SKIP=0
- SHAPE attn_qgate   N=3072 K=5120 PASS=16 SKIP=0
- SHAPE attn_out     N=5120 K=1536 PASS=16 SKIP=0
- SHAPE ragged_N_16  N=5136 K=5120 PASS=16 SKIP=0
- SHAPE kw2_iq       N=5120 K=512 PASS=16 SKIP=0
- SHAPE kw2_i8       N=5120 K=256 PASS=16 SKIP=0

The numerical cause and first differing byte were **not isolated**; the
current native token fragment is **not bit-identical** and must not ship.
No argument based on eliminated exchange instructions or reduced VGPR
overrides this hard correctness gate.

The guard stopped the sweep at the first failing process. IQ8R chained,
I8A8 unchained, and I8A8 chained were NOT RUN. The 115-unit full libr3
build, targeted narrow-row/IQR r3_selftest cases, and cold replay were
NOT RUN, as required by the hard stop. Consequently no full-build or
selftest PASS is claimed. No serving run, ms/step A/B, or KLD ran.
The per-run GPU lease was released; the post-run device check showed
no active compute PIDs.

**Final disposition: NOT READY — bit identity failed.** The experimental
branch and resource-gate evidence are retained for review only. No push,
merge or shipping action was performed; no additional kernels or
rewrite files were edited. Further work needs a separately authorised
correction and full requalification starting at the resource gate.
