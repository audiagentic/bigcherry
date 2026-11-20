# LTX-2.5 real-container IQ8R plane evidence

**Observed 2026-11-20, local model.** This is a bounded structural inspection, not a decoder, a read of quantized payloads, a model digest, or GPU parity evidence.

The remote shell denied the existing native rad-info executable because it is not on the shell's allowed-command list. The shell also rejected a leading environment assignment for pytest. We did not attempt to defeat those policies by hiding the native binary behind a subprocess, alias or renamed executable.

An equivalent, deliberately narrow **development-only** inspection now lives in tools/tests/lab/test_ltx_iq8r_planes.py. It uses Python's standard library only and accepts a positional model path:

    python3 tools/tests/lab/test_ltx_iq8r_planes.py "$BC_MODEL_ROOT"/ltx-2.5/ltx-2.5-iq8r.rad

This command was tested against the actual model on the development host. It neither runs rad-info nor allocates GPU resources. It loads small RAD2 structural tables, not the 37-GiB payload.

## Confirmed data

The container is 39,578,374,144 bytes, RAD2 format, architecture ltx2. Header tables declare 423 metadata rows, 6,114 directory entries, 8,726 planes and three encodings. Encoding 1 is exactly **affine/fwht128** with signed-I8 code elements [1x1], unsigned-U8 scale elements [1x256 K] and BF16 scale.1 elements [1xfull K]. The test found **1,306 IQ8R entries** and rejected none: every checked IQ8R entry has the expected three plane lengths, 256-byte plane alignment and all planes fall within the weight entry. This is a metadata/extent proof, not an arithmetic proof.

| Actual IQ8R tensor | Shape | Codes bytes | Scale bytes | Secondary scale bytes |
| --- | --- | ---: | ---: | ---: |
| dit.blk.0.attn1.qkv.weight | 12288 x 4096 | 50,331,648 | 196,608 | 24,576 |
| dit.blk.0.audio_attn1.qkv.weight | 6144 x 2048 | 12,582,912 | 49,152 | 12,288 |
| conn.video.blk.0.qkv.weight | 12288 x 4096 | 50,331,648 | 196,608 | 24,576 |
| te.blk.0.attn_q.weight | 4096 x 3840 | 15,728,640 | 61,440 | 8,192 |

Model metadata independently declares **48 DiT blocks, video 32 heads x 128, audio 32 heads x 64, Gemma 48 layers x 3840 hidden, and eight connector layers per modality**. There are 3,584 other BF16-encoded and 1,224 F32-encoded directory entries. The existing plan's asymmetrical 8/8/16 or 12/12/8 attention-head approach remains applicable, but the model's precise encoding must determine every scale/stride when real row/column-shard logic is added.

## Reproducibility and boundaries

The new test has a CI-safe sparse RAD2 fixture with one IQ8R tensor, plus a negative control that corrupts the U8 scale-plane length. It has an opt-in real-file check via BC_LTX_IQ8R_MODEL, and also supports positional execution so no shell variable injection is needed. pytest is test-only; no Python files were added to the installed Radiance engine.

The inspection **does not** decode a signed-I8 value, execute FWHT128, validate physical weights against the source checkpoint, confirm all RadFile::validate() constraints, hash the full model, or establish numerical parity. For production acceptance, validate actual decode/layout using the native model reader and mixed-card HIP kernels, with an explicit hardware lease. The latest CI may not have the gated model, so absence of the optional file must not cause CI failure.

The native rad-info restriction should be solved by the normal allowlist authorization process if that exact tool is required. The existence of this read-only inspector is not permission to bypass executable restrictions.


## Real-file numerical row oracle (2026-11-20)

The separate, CI-safe `tools/tests/lab/test_ltx_iq8r_numeric.py` test and positional CLI now exercise **actual signed-I8 codes, U8 per-256-K group scales, BF16 per-row secondary scales and H128 transforms** from the local IQ8R model, without starting the engine or any GPU process:

    python3 tools/tests/lab/test_ltx_iq8r_numeric.py "$BC_MODEL_ROOT"/ltx-2.5/ltx-2.5-iq8r.rad

Against the real file it checked **123 selected model rows** (video 63; audio 60), including Q/K/V bank boundary rows, output-projection rows, real BF16 output biases and three candidate head partitions. For each sampled weight row the CPU oracle independently computes `(H128 * decoded_rotated_weights) dot x` and `decoded_rotated_weights dot (H128 * x)`; for representable output-column splits the three per-rank dot products sum to the dense result. The test also shows that adding the **real, nonzero** output bias on each rank (three times) is wrong; the bias belongs after the sum once. This is a double-precision algebraic test against real quantized weights, **not** the HIP i8a8h rounding/activation-quantizer comparison.

**New concrete admissibility constraint:** the actual video output projection is 4096x4096 (32 heads x 128) and its 10/10/12 split gives 1280/1280/1536 K channels, all 256-aligned. The actual audio output projection is 2048x2048 (32 heads x 64); its 10/10/12 split gives 640/640/768 K channels, *not* 256-aligned. The 8/8/16 and 12/12/8 splits are safe for K-group slicing on **both** streams. A head split is not automatically a legal IQ8R output-projection column split; production admission must check both. A native C++ reference test now explicitly refuses audio 640/640/768.

The pure-Python reference reads selected rows only, checks each file section against known encoding metadata, and is test tooling rather than engine runtime. Its normal CI suite uses synthetic FWHT/codec/negative K-boundary fixtures and skips real-file access unless an opt-in path is supplied. It does **not** establish model SHA identity, a full numerical fixture for all rows, exact GPU BF16 rounding, rotary/attention output parity, liveness, or functional TP3 inference.


## Full 48-block DiT projection inventory

The real model has the same **20 IQ8R linear projection names and shapes in each of its 48 DiT blocks**: 960 source entries were checked for their expected encoder and row/column geometry. Of these, **eight output/FFN-down projections per block** require row-parallel K cuts (384 per split). Group-width checks yielded:

| Proposed head split | IQ8R K cuts rejected across all 48 blocks |
| --- | ---: |
| 8/8/16 | 0 |
| 12/12/8 | 0 |
| 10/10/12 | **192** (4 per block) |

The four affected names are **audio_attn1.out.weight**, **audio_attn2.out.weight**, **a2v.out.weight**, and **v2a.out.weight**. All have K=2048, and the 10/10/12 head split yields K spans of 640/640/768, which split the 256-column scale groups. By contrast, audio_ff_down.weight has K=8192 and proportional K spans of 2560/2560/3072, all group-aligned. For an actual production TP3 implementation, head-only admission and output K-sharding admission must therefore be separate per operator. These are geometry/plane rules only: kernel coverage, inter-rank reductions and activation quantization remain untested.

## Connector and cross-modal model sampling

The component-first ladder extended the read-only real-model numerical oracle to check **64 IQ8R projection entries** across the video/audio connector stacks (8 blocks each; QKV, output, FFN-up and FFN-down), plus each connector block norm/gate shape. It also checked 40 cross-modal/text-cross/FFN projection entries from DiT blocks 0, 27, 28 and 47. An additional **104 selected actual IQ8R weight rows** passed BF16/U8/I8 dequantization and the f64 FWHT128 matrix-product identity; total selected real-row checks are now **227** (123 earlier + 104 new). The connector is still intended to execute leader-only: the 8 rejected *hypothetical* audio connector 10/10/12 K cuts are advisory geometry, **not evidence that connectors are distributed**. This is not native codec/kernel parity.
