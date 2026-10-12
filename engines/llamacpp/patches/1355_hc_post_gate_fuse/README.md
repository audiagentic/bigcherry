# 1355_hc_post_gate_fuse

QFP35 residual fusion against llama.cpp b11474, composed after production patches 1313 and 1344.

Default: **on**. Control: `BIGCHERRY_HC_POST_GATE_FUSE=0`.

Build experiment `hc-post-gate-fuse`, then run one binary with full process separation:

```bash
BIGCHERRY_PATCH_TRACE=1 \
AB_ENV="BIGCHERRY_HC_POST_GATE_FUSE=0" \
tools/lab/flash-next/queue-env-ab.sh qfp35-hcpost <build-run-id> 8192 24576 98304
```

A = fused/on; B = composed production/off. The 2026-10-09 hardware screen showed no prefill separation and approximately +1% pooled decode with identical greedy text. This is experiment-only evidence, not production qualification. Require marker attribution, backend-op F32/logits parity, fixed-draft MTP acceptance, graph/replay and multi-request controls, and four independent paired decode sessions with positive CI95-low and no prefill regression. Do not claim a prefill win.

The fused kernel evaluates expf for each embedding-output thread even though the gate is invariant over embedding; compare generated ISA/SFU counts and paired HC_POST-plus-producer timing before attributing the gain or proposing a wave-broadcast change. See QFP35 and BCOP170. Because this is
intended to reproduce the same F32 expressions and HC reduction order, any greedy mismatch is a failure pending
fidelity diagnosis, not an accepted equivalence result.
