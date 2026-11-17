

### 2026-11-11 — VTP25/VTP27 original BF16 layer-0 FFN and GPU GDN→FFN hardware acceptance

**VTP25 physical-TP3 original checkpoint FFN GPU ACCEPTANCE:**
The new original layer-0 5120/4864/7424 BF16 SwiGLU FFN
ran on 2x gfx1100 RX 7900 XTX + 1x gfx1201 R9700.
It takes the original post-GDN CPU-reference residual,
applies source-hashed original post-attention weighted
RMSNorm, executes original BF16 gate/up/down weights,
three concurrent per-rank contributions, pinned-host
ordered F32 reduction and BF16 residual, then broadcasts
the output. Both M1 and M4 GPU runs have rc=0 and
independent checkpoint-derived CPU-oracle tolerance PASS.

| VTP25 case | Wall p50 | Wall p95 | Final BF16 exact | Largest absolute error |
| --- | ---: | ---: | ---: | ---: |
| M1 | 875.92 us | 1,005.88 us | 5,107 / 5,120 | 0.00390625 |
| M4 | 1,564.19 us | 1,898.33 us | 20,455 / 20,480 | 0.00390625 |

Immutable receipts:
- $BC_WORK_ROOT/xtx27-prefill/vtp25-original-ffn-gpu/runs/20261111T064141720549Z-4afdb501e8/summary.json
- $BC_WORK_ROOT/xtx27-prefill/vtp25-original-ffn-gpu/runs/20261111T064157636260Z-4afdb501e8/summary.json

**VTP27 first connected original-weight GPU GDN→FFN M1 ACCEPTANCE:**
Under ONE runner-managed exclusive GPU lease, the actual
VTP20-based GDN GPU checkpoint execution exported its
post-GDN BF16 residual. The independent VTP25-derived
FFN GPU process imported that exact GPU-produced file.
The imported GDN output matched its source-hashed CPU
reference 5,120 / 5,120 BF16 words bitwise, with both
SHA256 digests
7b51c63c34a341a3ee5cc1872a6ef6dfdab6fd4259fd2bf30803af76414.

The complete actual original checkpoint-derived GPU
layer-0 FFN residual, ready as an input hidden tensor
for layer 1, matched 5,107 / 5,120 BF16 words exactly.
The remaining 13 were within the current numerical
tolerance, maximum absolute difference 0.00390625.
**Not bit-exact complete layer-0 output**. Both internal
GPU stage subprocesses recorded rc=0, oracle_pass=true.
Immutable receipt:
- $BC_WORK_ROOT/xtx27-prefill/vtp27-layer0-gpu-gdn-ffn-bridge/runs/20261111T064222931275Z-864e59e3e5/summary.json

Actual GPU-produced file:
- $BC_WORK_ROOT/xtx27-prefill/vtp27-layer0-gpu-gdn-ffn-bridge/handoffs/20261111T064222611374Z-864e59e3e5/gpu-after-GDN-residual.bf16

Actual complete layer-0 GPU-derived hidden state:
- $BC_WORK_ROOT/xtx27-prefill/vtp27-layer0-gpu-gdn-ffn-bridge/handoffs/20261111T064222611374Z-864e59e3e5/gpu-after-layer0-FFN.bf16

**Exact limitation:** VTP27 handoff runs each model
operator as a separate 3-GPU process, serialised with a
host-file BF16 bridge, **NOT** resident GPU-to-GPU
state transfer or a native model runtime. First layer
input is deterministic synthetic BF16 hidden, not an
actual token embedding. VTP27 is only M1; prior VTP20
M4 post-GDN output had 109 BF16 differences, so blindly
reusing the precomputed M4 CPU normalized FFN input would
not represent the actual GPU hidden tensor.
No model text generation, no 64-layer execution, no
persistent 16-layer causal KV or 48-layer GDN history,
no measured VRAM admission, no end-to-end TP3 vs TP2
performance claim and no production activation.

All three GPU acceptance runs released their lease;
at the post-run check no lease or resident KFD process
remained. The tests do not alter GPU clock/power settings.
Next highest priority: layer-agnostic GPU operator runtime
within one process, with original embedding/head and
per-layer residual dispatch, physically resident weights
and recurrent/KV state; only then actual tokens/s and
resident context qualification. Keep previous exact
VTP19/GDN/attention results separate from model output.
