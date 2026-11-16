# rad0012_helper_kernel_config

## Summary

Gives the kernel library a mandatory, size-checked, install-once configuration interface, so the helper process hands it the settings it read from the serving manifest instead of the library reading the environment. The engine keeps its own environment read until RSC07.

## Outcome

Validated on the promotion evidence below.

## Detail

### Promotion record

Model: Qwen3.8-Flash-Next fp8-iq4r-moe (.rad), flashnext-4gpu (R9700 engine, two RX 7900 XTX and one RX 6900 XT helpers) and flashnext-2xtx
Mechanics: the patch's tests in tools/tests/patch and the moe-remote card-free helper-settings tests (unknown, missing and out-of-range key fail; a valid manifest round-trips) pass; patch-lint clean on the promotion PR.
Activation: helpers start as `--manifest M --role helperN` with no hand-set R3_* variable and become ready (releases/evidence/rad0009_serving_manifest-promotion.md, flashnext-4gpu from the manifest alone, build b58adaca5; and the flashnext-2xtx re-check on build 4d808005a1b9, where the helper arm placed 21.43 GiB of weights against 31.07 GiB for the engine alone because the manifest set R3_MOE_WORKER_SKIP_OWNED); a bad setting is refused before any HIP call.
Identity: helper-on KL against the exact two-XTX reference matches the env-launched helper: mean 0.06374 vs 0.06393 helper-off, median 0.01497 vs 0.01596, p99 0.7191 vs 0.7270, top-1 88.228% vs 88.556% (8240 positions); manifest-started text sha 5691231c identical on all 8 arms of runs/rsc08-s3-gate (the helper-offload hand-off carried by the lane build).
A/B: no serving-speed claim: configuration only; step time 25.10 ms (manifest) vs 25.16 ms (argv), 6-stream 49.36 vs 49.38 ms (ABBA x2, runs/rsc08-s3-gate); the p99.9 +0.94 on the 4-GPU gate is the RAD46 static tail present with and without the change.
