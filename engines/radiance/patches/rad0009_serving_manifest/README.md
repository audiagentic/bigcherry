# rad0009_serving_manifest

## Summary

Lets the engine start from a serving manifest v1: `engine --manifest PATH [--role engine] [-v|-vv]`, with no command-line converter in our tooling.

## Outcome

Validated (profile evidence, Qwen3.8-Flash-Next on the four-card layout, Radiance v1.17.0): the engine started from the measured serving manifest serves identical text and the same speed as the argv start. See SUMMARY.md for the design.

## Detail

Model: Qwen3.8-Flash-Next fp8-iq4r-moe (.rad), R9700 engine with two RX 7900 XTX expert helpers and an RX 6900 XT tail (flashnext-4gpu)
Mechanics: patch tests (tools/tests/patch) and the serving component's card-free tests (bc_v1_test, bc_serving_test, manifest_argv_test through the real main.cpp order of rad0011 then rad0009) pass on the final PR #651 tree; patch-lint clean; the lane build radiance:stock:linux-radiance builds both patches (build c6938079 / plan 8b4285bc) and the engine starts with --manifest.
Activation: every gate arm's startup report labels every option [manifest 8f0efb5b89c7] (the engine read its configuration from the manifest, rad0009), the cold measuring start from an unmeasured manifest wrote the plan in 64 s through --plan-memory (rad0011) and bc-serve --measured-plan stored it; plan check "admitted" from the manifest on every arm; a one-byte plan drift was refused before weights with BC_CFG_PLAN_MISMATCH (runs/rsc08-s3-gate, drift.result).
Identity: 1-stream text sha 5691231c on all 8 arms (manifest-started engine and main's argv engine, ABBA x2, 3 reps each); accept-bench on the manifest-started cold start against main's argv engine: identical text sha, accepted count and acceptance on all five RAD23 prompts (code 370/417, hello 323, list 382, random 381, story 292).
A/B: runs/rsc08-s3-gate, ABBA x2, temperature 0: 1-stream step 25.10 ms (manifest) vs 25.16 ms (argv) (-0.25%), 6-stream 49.36 vs 49.38 ms (-0.05%), TTFT 426 vs 427 ms (1-stream) and 5,719 vs 5,777 ms (6-stream); 24 of 24 six-stream soaks passed; /metrics answered on every arm.

### Promotion record

Model: Qwen3.8-Flash-Next fp8-iq4r-moe (.rad), R9700 engine alone (tp 1) and flashnext-4gpu (R9700 engine, two RX 7900 XTX and one RX 6900 XT helpers)
Mechanics: patch tests (tools/tests/patch) and the card-free manifest_argv_test pass on main after #711 and #714: the KL flags --kld-record/--kld-corpus/--kld-ref/--kld-out accompany --manifest --role engine and arm no plan check, and the manifest's hand-off settings (R3_MOE_WORKER, _SHM, _MAX_ROWS, _HISTOGRAM, _HISTOGRAM_PERIOD_S) are set in the engine environment before it initialises and beat a stale value; patch-lint clean.
Activation: every option in the KL run's startup report is labelled [manifest ...]; on flashnext-4gpu started from the manifest alone (main b58adaca5, no hand-set R3_MOE_WORKER* or histogram variable) the engine wrote the route histogram from spec.handoff.telemetry and the three helpers started with --manifest M --role helperN and became ready; 162 live swaps committed during one forced-churn KL pass, 0 aborted.
Identity: KL from the manifest equals KL from raw flags on the same binary to every digit (R9700, one corpus pass, 8240 positions, build 503069237359): KLD mean 0.063926, median 0.015956, p99 0.726998, p99.9 1.764050, top-1 88.556 %, PPL 8.8485, repeat arm identical; on flashnext-4gpu policy-off from the manifest alone equals the earlier env-set run (mean 0.0660, p99.9 3.0029, top-1 88.5437 %) and policy-on with no moves equals policy-off.
A/B: no serving-speed claim: the serving start path is unchanged (the plan check is armed for it as before, the earlier rad0009 A/B of manifest against argv, runs/rsc08-s3-gate, stands); the new paths are the KL-mode start and the hand-off environment, checked by the identity runs above; the 4-GPU gate's p99.9 +0.94 failure is the RAD46 static tail, present with and without the policy.
