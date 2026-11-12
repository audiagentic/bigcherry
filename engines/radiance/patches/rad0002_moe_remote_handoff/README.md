# rad0002_moe_remote_handoff

## Summary

The Radiance v1.14.0 production hand-off patch delegates routed experts to GPU helpers.

## Outcome

Validated on 3 November 2026 for Qwen3.8-Flash-Next fp8-iq4r-moe, an R9700 engine and RX 7900 XTX helpers. The recorded quality and helper-activation checks passed for this profile.

## Detail

### Promotion record

Promotion evidence, profile-evidence tier, radiance pin v1.14.0, 2026-11-03. Profile-scoped: Qwen3.8-Flash-Next is the only model served with remote-MoE helper offload.

Model: Qwen3.8-Flash-Next fp8-iq4r-moe (.rad), R9700 engine with RX 7900 XTX expert helpers.

Mechanics: patch tests (tools/tests/patch, test_radiance_patch_packages) pass on origin/main 2f5fc39df; patch-lint clean; the lane build radiance-lab:stock:linux-radiance+moe-remote-handoff builds rad0002 with the moe-remote component (build ids under ${BC_WORK_ROOT}/runs/men26-kld/main.build.dir, rhp11-c2-ab7). Implementation digest (patch_validation_subject_digest) 663b0856fbc4da97703a059711349be1c5c3cd0373211e701f6e8864f215c947.
Activation: helper logs show the hand-off serving every capture: two XTX helpers served 5,764 requests and 1.76M-1.81M routes each (runs/men26-kld/capture-isolate/quality/helper*.log); in serving A/Bs (runs/rhp11-c2-ab7) every arm reports helper-served requests and 28 placement cycles per cycling arm.
Identity: radiance KL mode against a freshly recorded engine-local reference (TP2 exact over the two XTX, corpus rad-kld-corpus-4.jsonl, 8,240 positions), helper offload on the two XTX vs the R9700-alone control: KLD mean +0.003, median +0.0006, p99 -0.034, p99.9 +0.41 (limit 0.5), top-1 -0.36 points (limit 1.0), perplexity +1.2% (limit 5%): all tools/lab/radiance/helper-kld.json limits pass; three captures bit-identical (runs/men26-kld/capture, capture2, capture3, capture-isolate). Serving text identical to main in every arm of runs/rhp11-c2-ab7.
A/B: helper offload is the only way this model serves on this host (the engine card alone cannot hold it at serving context), so the comparison is against main with the same hand-off: runs/rhp11-c2-ab7 (9 interleaved arms): 1-stream decode 24.47 vs 24.43 ms/step (+0.14%), 6-stream 47.94 vs 48.16 ms (-0.46%), prefill TTFT 396 vs 395 ms (1 stream) and 5.62 vs 5.65 s (6 streams).

Known limitation, not part of this patch: the production tail layout with the RX 6900 XT (gfx1030 helper kernels) fails KLD p99.9 (+1.58); tracked as RAD46.
Plan items: RAD01, MEN26, RHP11.
