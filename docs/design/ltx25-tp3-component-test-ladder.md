# LTX-2.5 IQ8R TP3: component-first test ladder

**Status 2026-11-20:** This is a lab-only scaffold for LT3-01–15. All current native tests are independent CTest executables and use synthetic arithmetic/protocol data. The separate Python inspectors optionally read the **actual** IQ8R container, with no GPU. Neither path builds a production LTX TP3 plugin; **the upstream `ltx2` TP>1 refusal remains in place**. Do not interpret green tests here as device activation or end-to-end parity.

## One gate per component

| Gate | What to run / owned harness | What green means now | Required next native / hardware gate |
|---|---|---|---|
| **C00 Container and IQ8R** | `test_ltx_iq8r_planes.py`, `test_ltx_iq8r_numeric.py`, `ltx_tp3_i8a8h_test` | Real RAD2 metadata and **1,306** IQ8R tensor plane extents; sampled real I8/U8/BF16 rotated rows satisfy double-precision H128 identity; reject malformed planes | Native RadFile reader; exact signed rounding and i8a8h activation quantizer vs reference; BF16/FP32 tolerances on gfx1100/gfx1201 |
| **C01 QKV / KV / gates / FFN and output projection** | `ltx_tp3_linear_test`, `test_ltx_iq8r_shards.py`; original `ltx_tp3_reference_test` | Uneven 32-head ownership, fused QKV and KV bank order, one output bias after three partial sums, K=256 alignment and negative cuts | Opt-in per-rank native LTX builder mapping via `rad_weight_shard_span_parts` / `LinearFP8::span_parts`; compare actual shard views with TP1 on card |
| **C02 Global Q/K RMS** | `ltx_tp3_rms_test`; original `ltx_tp3_reference_test` | Distinct Q/K token row counts; all-head denominator over 2048/4096 input widths, asymmetric 3-rank shares; zero, NaN/Inf and shape refusal | Real rank collective with f32 accumulation and deterministic ordering, BF16 gain/rounding vs TP1 before RoPE |
| **C03 Self/text/cross-modal attention** | `ltx_tp3_numerical_test` | Tiny f64 dense vs uneven TP3 simulated self/cross attention with softmax, per-head gate and output bias-once | Separate video/audio/text cross/A2V/V2A native block parity, including RoPE, attention masks, row counts and real checkpoint kernels |
| **C04 Guidance and collective order** | `ltx_tp3_guidance_test`; original `ltx_tp3_protocol_test` | Per-pass identical rank traces for plain/STG/ISO/combined, with STG block-28 and isolated-branch omissions; detects divergent rank event lists | Engine-graph-emitted operation tickets, rank mismatches, timeouts, abort/recovery and no-deadlock test for every guidance combination |
| **C05 Text encoder and connector handoff** | `ltx_tp3_handoff_test`; `test_ltx_iq8r_numeric.py` | Model confirms 8 video and 8 audio connector layers, IQ8R QKV/out/FFN shapes; simulated remote/same-card text placement, fixed layer-stack/conditioning sizes, epoch/shape/replay/cancel checks | Native Gemma4 **on gfx1030** (independent P1 kernel gate), retained-root/VRAM admission, real GPU transport, connector output parity |
| **C06 Resource identity and safe placement** | Original `ltx_tp3_reference_test`; `ltx_tp3_handoff_test` | Mock 2x gfx1100 + gfx1201 + optional gfx1030 unique identities, memory-overflow and reserved-capacity refusals | Actual BDF/device identity, measured budgets and startup admission; no assumption that 6900 can hold full Gemma working set |
| **C07 VAE / decoder / media ownership** | `ltx_tp3_output_test`; original `ltx_tp3_protocol_test` | Single leader for first-frame VAE, video decode, audio decode and upscaling; duplicate/stale/nonleader writers refused | Actual VCOND/VDECODE/ADECODE/VUPSCALE/Diffusion/VAE production outputs, timing, persistence and cancellation |
| **C08 Component-chain smoke** | `ltx_tp3_micro_pipeline_test` | One synthetic request carries text ticket, guidance trace, all-head RMS, 3-rank output reduction and a unique output owner across 16 mode combinations | Real native callback chain on **one device** (TP1 unchanged), then opt-in 3-rank single block and full 48-block DiT with rollback |

The current per-component tests intentionally have **no production call site**. Each gate needs a real engine implementation seam before being promoted from CPU algebra to device parity. The distinction between C03's *miniature attention* and C08's *miniature integration* is deliberate: neither implements the full LTX graph.

## Model-specific admission discovered by C00–C01

The actual RAD2 model contains **20 required IQ8R projection names/shapes per 48 DiT blocks**, i.e. 960 checked entries. Candidate 8/8/16 and 12/12/8 head splits preserve IQ8R 256-wide output-column groups for the tested DiT operators. Candidate **10/10/12 cannot serve four audio-side output projections per block by direct K slicing**, for **192 invalid projections** in total. The four types are `audio_attn1.out`, `audio_attn2.out`, `a2v.out` and `v2a.out`. The 10/10/12 audio FFN-down happens to remain group-aligned because its K is 8192; do not reject or accept an entire partition based on heads alone. For the currently leader-only connectors, the corresponding hypothetical audio K split would also be invalid, but **this is not a proposal to shard connectors**.

The real-file numerical oracle samples **123** DiT block-0 QKV/output rows plus **104** additional connector and cross-modal/FFN rows (selected blocks), checks fused banks, stored code/group/row scales, the unnormalised FWHT128 identity, and exactly-once real BF16 output bias in the block-0 projection tests. These are double-precision mathematical checks; they do not reproduce i8a8h kernel rounding, scale precombination or all GPU layouts.

## How to run each component independently

These are repository-relative examples for a normal interactive shell. When the remote shell has no working-directory argument, use the corresponding absolute paths in each tool call (do not store machine paths in this document). Set `ROOT` only in a **normal interactive shell**; the Remote Agent tool executes one executable per invocation and does not support `cd` or environment-variable prefixes.

Build (no GPU needed):

```sh
cmake -S tools/lab/radiance/ltx-tp3 -B ${BC_WORK_ROOT}/ltx-tp3-native-build -DBUILD_TESTING=ON
cmake --build ${BC_WORK_ROOT}/ltx-tp3-native-build --parallel 2
cmake --build ${BC_WORK_ROOT}/ltx-tp3-native-build --target test
```

Each executable is also an independent build/test target under `${BC_WORK_ROOT}/ltx-tp3-native-build/` and can be chosen with `ctest -R ltx_tp3_<component>_test` on a runner supporting that command. The CMake test target runs all ten.

Real-model read-only CPU check (not a pytest CI dependency):

```sh
python3 tools/tests/lab/test_ltx_iq8r_numeric.py ${BC_MODEL_ROOT}/ltx-2.5/ltx-2.5-iq8r.rad
```

The CI-safe Python tests include synthetic corruption and K-group negative controls; optional real-file pytest cases skip without `BC_LTX_IQ8R_MODEL`.

## Promotion sequence toward an actual E2E

1. **Native source applicability:** retain the frozen upstream ABI/version and exact recipe patch; add a dedicated opt-in LTX TP3 declaration, *not* an unconditional widening of the current TP1 path. Use existing row/column shard builder interfaces and native feature/admission reporting.
2. **One operator at a time:** promote C01 IQ8R maps and output reduction, then C02 all-head Q/K norm, then C03 all six attention kinds and FFN. For each, test real model weights on one GPU against TP1 first, then a two-rank smoke, then the intended mixed gfx1100/gfx1201 triplet under a GPU lease.
3. **Control plane:** promote C04 request/pass/epoch-specific collective ordering and cancellation; verify high-level and kernel traces match. Explicitly handle STG block 28, cross-modal isolation and >1 guidance evaluation per diffusion step.
4. **Roles and conditioning:** prove C05 leader-only connectors plus optional gfx1030 Gemma4 separately. Remote text is **optional**, never a prerequisite for first TP3 DiT. Validate C06 actual power/VRAM/identity facts and C07 non-DiT pass/output ownership.
5. **Integration ladder:** C08 synthetic chain → native **TP1 source/build regression** → one-rank real model TP1 output → 3-rank single-layer numerical parity → 48-layer T2VA at one short duration → guided T2VA/I2VA → audio+video decode, cancellation/restarts, and output ownership → longer jobs/performance and numerical quality. Record pass/fail/per-rank tracing and keep rollback to TP1.
6. **E2E acceptance:** only mark TP3 functional when a real model request runs all required native passes, three ranks agree and finish without deadlock, outputs are finite and aligned, quality/parity compared to TP1 meets an explicit tolerance, and a measured record includes real device, build, config, policy and feature-state provenance. CPU reference green alone never closes this.

All GPU work requires the shared `gpu` lease. The current worktree is lab-only; do not run card work when another actor holds the lease. Do not edit authoritative Planning/Ledger files without their MCPs; those services were not available during the earlier foundation work.

## Stronger C00/C01 proof: source-defined quantizer and actual model planes

The native CPU `ltx_tp3_i8a8h_test` independently models `libr4d/r4d_quant_act_fp8.hip`: two H128 transforms per 256 K group, common max/127 scale, nearest-even i8 codes and BF16-rounded activation. Legal TP3 cuts yield bit-identical codes and f32 scales to full-input quantization. A negative control distinguishes an incorrect separate scale per H128. This does not call the HIP kernel.

`test_ltx_iq8r_shards.py` reads 768 physical DiT output/FFN rows across 48 blocks. It checks 1,920 legal raw I8/U8/BF16 shard reconstructions and refuses 384 sampled bad row/split cases (two sampled rows for each of 192 10/10/12-invalid projections). It also checks 288 fused QKV/KV projection shapes and 756 actual bank-boundary source rows from blocks 0, 28 and 47. This tests logical source planes, not native packed GPU layouts.

Checkpoint bias contract: video `ff_down` is biasless in all 48 blocks. Audio `audio_ff_down` and six attention output projections have one BF16 bias each (336 tensors total). A present bias is added exactly once after reduction; absent bias stays absent. The current native `arch/ltx2/ltx2.cpp::Lin::declare` still uses `RAD_SHARD_NONE`; this needs a guarded builder migration and device qualification.

```sh
python3 tools/tests/lab/test_ltx_iq8r_shards.py "${BC_MODEL_ROOT}/ltx-2.5/ltx-2.5-iq8r.rad"
```
