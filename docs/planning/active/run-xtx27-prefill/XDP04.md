---
id: XDP04
order: 3
plan: run-xtx27-prefill
state: active
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# gfx1100 native INT4×INT4 W4A4 proof and quantization quality gate

## Description

Prepare a bounded standalone gfx1100 native W4A4 experiment **on the
existing XDP worktree**, ahead of any Radiance engine or model change.
Owner direction (2026-11-09): use original BF16 weights, test native
4-bit XTX execution, and eventually choose quant formats by the
receiving GPU's execution and resident-KV constraints (XDP03).

Measured reason to investigate: the current W4A8I8 prefill kernel on
one XTX, N=17,408 K=5,120 M=512, had **3,226.6 us** minimum versus
**931.6 us** for MXFP4 tiled. Its 4-bit weights expand into signed
INT8 operands and use INT8 WMMA, not the XTX's native INT4 WMMA.
Prior register-only roof measured ~285.8 TOPS gfx11 IU4 against
~144.1 TOPS IU8; the roof does not include packing, scales, traffic,
quality or full-model kernels. Existing W4->MXFP4 mixed recipe saves
only ~70 MiB in planned container size (XDP03). No reason to broadly
convert the 27B until a better W4 path and numerical gate exist.

## Code and operation

The experiment contains exactly these new files:

- `tools/lab/radiance/xtx27_w4a4_probe.hip` — **isolated** HIP program,
  a deliberately untiled one-wave-per-16×16 proof, no Radiance plugin
  registration and no vendor code changes. Uses the real gfx1100
  `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` with W unsigned and
  A signed, on 16 packed 4-bit elements (2 x i32) each per wave lane.
  Processes each K=128 group as 8 WMMA steps, exact int32 arithmetic,
  then subtracts weight zero × signed activation sum and applies
  per-group floating scales. Produces f32 C[m,n] and verifies against
  a separate host integer oracle. The kernel measures its own elapsed
  time only *after* passing correctness.
- `tools/lab/radiance/xtx27_w4a4.py` — the single, auditable CLI:
  `--phase plan` (default; no subprocess), `--phase cpu` (card-free
  synthetic BF16-like quantization/accuracy), `--phase compile --run`
  (one gfx1100 standalone HIP file, outputs outside checkout) and
  `--phase gpu --run` (version-checked binary, `agent-lease acquire
  gpu -- ...`, automatically releases on process completion).
  `--suite` performs six fixed geometry/seed oracle probes under
  **one** lease; no additional scripts or permission changes.
- `tools/tests/lab/test_xtx27_w4a4.py` — card-free correctness,
  symmetric signed-int4 / unsigned-int4 pack and round trip,
  group-boundary and zero-point invariants, malformed input,
  deterministic normal/outlier synthetic error comparison, scratch
  and opt-in/lease/phase restrictions.

The input storage format is **not MXFP4**. Activation `A` is a
signed two's-complement INT4 plane, one scale per (activation row,128).
Weight `W` is unsigned INT4 (0..15), a zero point and scale per
(weight row,128), and a packed nibble stream. The exact target math is:

```
C[m,n] = sum_g ( dot_i32(A4[m,g], W4_u[n,g]) -
                 zero_w[n,g] * sum_i32(A4[m,g]) )
               * scale_a[m,g] * scale_w[n,g]
```

This corrects asymmetric weights without unpacking them to INT8 for
the WMMA. The GPU's group accumulation is exact int32; final group
scaling is floating arithmetic. The CPU test also compares rounded
W4A4 against W4A8 under the same W4 weight scale and original float
reference, including outliers. **These are synthetic numerical
screens, not model perplexity, KLD or accuracy measurements.**

### RDNA3 layout pitfall and verification

gfx11's accumulator layout is *interleaved*, not gfx12's contiguous
form: native result row = 2×register + floor(lane/16), column =
lane % 16. The first matrix operand is the output-row matrix, the
second is the output-column matrix. This probe uses W as the first
operand and A as the second, so native C is [weight row, activation
row] and must be transposed at the store into C[m,n]. A first smoke
compiled but failed all off-diagonal outputs when the transpose was
wrong; that was fixed **before** any benchmark. Correctness checks
all entries, not just the diagonal.

Architecture reference: AMD ROCm Optimization Guide, RDNA3 dense
WMMA builtins; independent BigCherry `rad32-wmma-exact.hip` fixture
for gfx11 output element mapping. No compatibility shim is needed for
this direct gfx11 experiment. It must not be called an optimized GEMM.

## Reproduction (one tester)

All examples run from the existing checkout. The default output is
`${BC_WORK_ROOT}/xtx27-prefill/w4a4`, outside Git. GPU lease is
acquired *inside* the last command; no manually held lease needed.

```sh
python3 tools/lab/radiance/xtx27_w4a4.py --phase plan --suite

python3 tools/lab/radiance/xtx27_w4a4.py --phase cpu \
  --m 16 --n 32 --k 256 --distribution normal
python3 tools/lab/radiance/xtx27_w4a4.py --phase cpu \
  --m 16 --n 32 --k 256 --distribution outlier

# Build only the isolated gfx1100 probe, no Radiance rebuild.
python3 tools/lab/radiance/xtx27_w4a4.py --phase compile --run

# When GPUs are idle, one lease wraps the whole six-case oracle suite.
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu --run --suite --device 0

# Or explicitly target one geometry and seed.
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu --run \
  --m 16 --n 32 --k 128 --iters 25 --seed 7 --device 0
```

The standalone GPU executable itself refuses a non-gfx1100 target,
and requires M/N multiples of 16 and K multiple of 128. The wrapper
requires `--run` and verifies source and compiled binary SHA-256
before touching a GPU. Lease conflict or oracle mismatch fails
closed and is never reported as a successful performance result.
Every run writes a log to scratch; the source and build manifest stay
in `${BC_WORK_ROOT}`, never the model store or tracked repository.

## Verified preflight evidence (2026-11-09)

The isolated `hipcc -O2 --offload-arch=gfx1100` compiler completed
without warnings. One direct native-WMMA smoke at M16 N32 K128
initially failed most off-diagonal outputs; the error was the C
transpose described above. After fixing it, all 512 cells passed with
exact agreement. The six-case suite subsequently passed with one
lease and auto-release (zero reported absolute error in every case):

| M | N | K | Groups | Untuned kernel-only average |
|---:|---:|---:|---:|---:|
| 16 | 16 | 128 | 1 | 7.158 µs |
| 16 | 32 | 256 | 2 | 7.050 µs |
| 32 | 64 | 512 | 4 | 8.048 µs |
| 64 | 128 | 1024 | 8 | 10.283 µs |
| 128 | 256 | 256 | 2 | 7.154 µs |
| 16 | 16 | 1024 | 8 | 10.043 µs |

These times are dominated by small problem sizes, launch and
unoptimized direct global reads. They are **not** a benchmark against
MXFP4 and must not be used to predict 27B prefill.

Synthetic model-agnostic BF16-like activation/weight screening at
M16 N32 K256 seed7 (W4 asymmetric per group128, compared to original
floating matrix multiply):

| Synthetic distribution | W4A4 relative L2 | W4A8 relative L2 | W4A4 top-1 rows | W4A8 top-1 rows |
|---|---:|---:|---:|---:|
| Gaussian | 0.147 | 0.104 | 100.0% | 87.5% |
| Outliers ×14 | **0.374** | 0.193 | **56.25%** | 87.5% |

Top-1 from one synthetic seed fluctuates; the relative L2 is the
meaningful warning. W4A4 has **substantial** additional activation
error, particularly with outliers. Until real model activation
histograms and end-to-end KL establish an acceptable result, this
architecture is **not quality qualified**, irrespective of its
native INT4 instruction throughput.

Provenance/logs:
`${BC_WORK_ROOT}/xtx27-prefill/w4a4/build.log`,
`w4a4-manifest.json`, and `gpu-suite.log`. No model weights were
converted, no serving engine was modified, and no GPU lease remains.

## Risks and exit criteria

1. **Correctness first.** GPU must pass all outputs of 16×16/16×32
   and larger multi-group K scenarios, multiple seeds, varied 4-bit
   signs and nonzero zeros. Add boundary and pressure tests before
   using it as a backend oracle for a production GEMM. Verify the
   actual emitted gfx11 instruction if a future compiler changes
   code generation.
2. **Quantized activation accuracy.** Run real Qwen3.8 BF16 activation
   capture for GDN/attention/FFN inputs. Evaluate row/group scales,
   outliers, KL, top-1, and decoded text against BF16 and current
   MXFP4. A native-WMMA fast kernel with poor W4A4 KLD is rejected.
3. **Real throughput.** Untiled one-wave/global-load proof is *not*
   a tuned GEMM. A production-worthy path needs tiled M×N×K,
   coalesced packed weights and activations, LDS/fragment staging,
   fusion of quantization and scale operations, reduced launch count,
   scratch/allocator budget, and ROCm ISA proof. Benchmark at M=1,
   64, 512, 2048 and the *actual* Qwen3.8 projection shapes.
   Compare end-to-end TTFT/decode with exact same model TP wire,
   no P2P, requested context and resident KV accounting.
4. **Integration isolation.** Only after (1)–(3) pass add a
   managed gfx11 `libr3` component/patch, not edits to
   `vendor/radiance`. Add kernel registry/layer resolver,
   producer A4+scale buffer declarations, TP2 consistency,
   per-layer quant selection, and fallback to proven MXFP4.
   No R9700 dense-stage execution is implemented by XDP04.
5. **Rejection gate.** Do not promote based on an isolated kernel
   time. Require numerical oracle, KL/quality and paired
   end-to-end prefill/decode with actual VRAM/KV headroom.

## Acceptance Criteria

Standalone native gfx1100 IU4 instruction compiles; card-free tests
pass; six-case GPU suite passes without borrowing someone else's GPU
lease. Document measured quality gap and optimizer risk. Production
integration remains a separate measured, reviewed work item.

## Change Log

- 2026-11-09: Compiled cleanly on gfx1100 and passed the single-case
  smoke and six-case multi-group GPU oracle (zero absolute error).
  Synthetic outlier A4 quantization exhibits 0.374 relative L2 vs
  W4A8 0.193; defer serving integration until model KL and a tuned
  actual-size kernel prove net benefit.
- 2026-11-09: Prepared standalone HIP native W4A4 WMMA with asymmetric
  group-128 correction, deterministic card-free A4/A8 vs W4 numeric
  evaluator, and one lease-guarded experiment CLI. Initial GPU
  off-diagonal failure exposed gfx11 accumulator transpose, fixed
  in isolated probe; first 16×32×128 oracle then passed exactly.
