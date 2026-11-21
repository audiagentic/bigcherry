# RUM09: calibrated four-GPU admission

Status: host tests, full stock build, and leased `flashnext-4gpu` startup/canary/stop passed (2026-11-21). This is an admission regression proof, not an end-to-end performance or release qualification.

## Failure and differential

The pre-fix `flashnext-4gpu` manifest had a single gfx1201 engine and three helpers. Its measured GPU VRAM claimable budget was **32,135,505,400 bytes**, but serving refused with `BC_CFG_ADMISSION_REFUSED`: derived live allowance **15,393,411,072** versus required **28,743,036,928** bytes.

The derived allowance was exactly static weights **5,800,083,456** + expert weights **2,902,323,200** + staging **1,133,551,616** + the refitted GPU KV pool **5,557,452,800** bytes. It was *not* the engine's physical VRAM or the manifest's claimable budget. The selected `kv_backing_floor.fraction_ppm=600000` calibration instead required **18,907,078,656** GPU KV bytes from a measured **31,511,797,760** byte KV ceiling. Admission applied that floor after the old refit had sized KV for only one full session. The admitted 12 GiB host KV tier is separate and does not erase the selected GPU backing requirement.

The passing `flashnext-2xtx-rdna2-tail-combine` configuration had the *same* single-engine VRAM plan but no selected backing-floor calibration; one full-session pool was sufficient. The passing `flashnext-tp2xtx-r9700` uses two engine GPUs with a smaller per-engine KV ceiling and no selected backing-floor calibration. These differentials isolate the late-calibration bug, not hardware exhaustion or timing.

## Owner change

Source commit: **`b1ca98631`** (`fix(radiance): fit calibrated GPU KV backing before admission`).

- `components/serving/v2/admission.{h,cpp}`: `refit_serving_pools` computes the same `kv_required` floor as admission, respects exact MiB pool rounding and the measured expert floor, and refuses when that combined floor really does not fit.
- `components/serving/bc_serving_manifest.{h,cpp}`: the measuring engine receives its already-selected typed KV backing calibration before constructing pools; the existing serving manifest binding is preserved.
- `patches/rad0011_plan_memory/patch.py`: the managed plan-memory integration uses that selected calibration when refitting. The `rad0009_serving_manifest` pinned owner-source digests were refreshed from the exact changed source bytes.
- `components/serving/v2/s5_refusal_test.cpp` and `components/serving/native_fleet_v3_engine_test.cpp`: regression assertions cover the fit, real refusal when ownership cannot fit, and measuring-side calibration binding.

No fallback configuration, compatibility shim, or fixed production VRAM allowance was introduced.

## Verification

1. Regression added before the implementation and initially failed to compile against the old refit API, as intended.
2. Nice-19 serving component build: `bc_s5_refusal_test` and `bc_v2_s8_test`, **2/2 passed**.
3. Managed patch C++ integration: `TestPlanMemoryPatch.test_plan_flags_compiled_and_run` and `TestServingManifestPatch.test_manifest_to_argv_logic_compiled_and_run`, **2/2 passed** with an executable task-local temp directory.
4. Full stock Radiance 2.2 build including the engine and all required GPU helper plugins **passed**; build plan `606d58e85c1fb289019e164e5aa53918`.
5. The actual `flashnext-4gpu` startup under a 12-minute GPU lease logged:

   ```text
   plan check: admitted: claimable 32135505400 >= required 28743036928 bytes (margin 3392468472);
   resolved KV 18907922432 >= kv_required 18907078656 bytes (margin 843776)
   ```

   The exact MiB GPU KV allocation was **18,032 MiB** (versus **5,300 MiB** pre-fix), and the resident expert floor remained **2,902,323,200** bytes.
6. The four-GPU session reached **`state=up`**, passed the service canary, and exited **0** after a clean stop. Prompt `The capital of France is`; 12-token completion: ` Paris. The capital of Germany is Berlin. The capital of`. Session events included `canary_passed` and `running`.
7. `git diff --check` passed. The branch was committed locally and not pushed.

Raw task-local run outputs are preserved under the `rum09-starts/out-calibrated/flashnext-4gpu` scratch directory, separate from the prior failure and passing-layout runs. This evidence qualifies the admission defect and startup canary only; it is not a numerical parity, latency regression, or release promotion claim.
