# 1355_hc_post_gate_fuse

QFP35 residual fusion against llama.cpp b11474, composed after production patches 1313 and 1344.

Default: **on**. Control: `BIGCHERRY_HC_POST_GATE_FUSE=0`.

Build experiment `hc-post-gate-fuse`, then run one binary with full process separation:

```bash
BIGCHERRY_PATCH_TRACE=1 \
AB_ENV="BIGCHERRY_HC_POST_GATE_FUSE=0" \
tools/lab/flash-next/queue-env-ab.sh qfp35-hcpost <build-run-id> 8192 24576 98304
```

A = fused/on; B = composed production/off. Require the patch-hit marker in A, complete prefill timing separation to
claim a win, unchanged decode control/acceptance, and identical greedy md5 A/B at all three depths. Because this is
intended to reproduce the same F32 expressions and HC reduction order, any greedy mismatch is a failure pending
fidelity diagnosis, not an accepted equivalence result.
