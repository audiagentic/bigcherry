# RCS13 / PR #946 — Run 12 production-faithful A/B/B/A and W4H2 trace

**Verdict: NOT CLEARED; do not merge #946.** Run 11's AQL exception is not reproduced under the known-good production Flash-Next configuration. All four production serving cells passed and stopped cleanly, and a separately instrumented candidate helper executed 6,488 real W4H2 kernel dispatches. However, **two-stream output hashes are not repeatable even within an arm** (and do not match across all arms). Strict output-identity clearance is therefore withheld; the small measured decode/prefill differences are not proof of improvement or a regression.

## Scope and source identity

- A = detached plain main source `7c0273976627c339fa8f50c54543da2c1419ac2f`; bundle plan `2f3a19bc78d0dcd2fcb881c6c1aa4358`.
- B = #946 head `17a16f5d7132efaa1c07dc858e111dfb7645ad10`; bundle plan `a0a99806f887aaecce91ac4369d4b17c`.
- Same upstream Radiance source slice `833e240cd588a59e91ba7960fcbbda9f`; same effective recipes, build toolchain, input model, engine executable and `libr4d` runtime hashes, and strict ABI 35/35. Exactly the seven RCS13 files differ (see Run 11's source/component ELF attestation). Fresh official complete bundles were **rehashed**, not mixed or rebuilt. Original build records and all source-matched binaries were retained.
- All starts used the actual `tools/lab/radiance/launch-flashnext-best.sh`, the layout `flashnext-tp2xtx-r9700` (two gfx1100 XTX ranks and one gfx1201 R9700 helper), repository `perf51.rank0.hot-first` heat, disk n-gram placement and strict v2 measured production manifests. Normal current-main plan-check is authoritative for both arms; first A/B cells required the measured host-pool remeasure, subsequent starts hit their source-specific plan caches. All cells received an exclusive bounded GPU lease, normal canary, independent short smoke, benchmark requests and normal launcher stop.

## H1 versus H2: why the original Run 11 failure was not a main defect verdict

Run 11 used a direct Python `ServerRunner` and a narrow manifest: 8,192 context, one session, 256-token chunks, 64 helper rows, 30,555 MiB helper budget and 32,301 MiB measured host pool. The known-good production layout uses 65,536 context, six sessions, 2,048-token chunks, 4,096 helper rows, 29,656 MiB helper budget and 33,219 MiB measured host pool. See [the full comparison](settings-comparison.md). Both cases used FP8 KV, host KV zero, disk n-gram, MTP depth three, a matching heat file and `small_rows=8`.

The current host has no configured GPU power settings for the selected devices. The initial normal launcher attempted `gpu-power auto`, refused an empty explicit request, and **stopped before GPU children**. The actual production-faithful comparison therefore used the launcher's documented `--no-gpu-power` option consistently in **all four cells**; this is a performance-comparability caveat against older power-calibrated runs, not an A/B-arm asymmetry.

Arm A's original failing seed-17 request (1,066 actual prompt tokens, 80 generated tokens) **passed under this production configuration**, along with seeds 31 and 43. The Run 11 initial AQL exception preceded the 16 KiB HIP allocation failure in `r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip:877-899`; the subsequent NOMEM cannot be interpreted as independent physical GPU exhaustion. This run identifies a **reproduction configuration mismatch**, not which individual knob caused Run 11's AQL fault. No earlier-main bisect was warranted after faithful A passed.

## Scored A/B/B/A (A0, B1, B2, A3)

Each of the four cells issued real decode requests with 1 and 2 concurrent streams for each of three seeds (17, 31, 43), using 600 input words (~1,066 model tokens at seed 17), `max_tokens=80`, temperature zero; prefill/TTFT requests used 2,048 input words (~3,400 actual tokens), `max_tokens=1`. Every input prompt SHA-256 was identical across arms and repeat cells. All 24 decode rounds and 12 prefill rounds passed (36 workload rounds, with 36 decode streams); all launcher canaries and independent smoke requests passed. MTP drafted and accepted counts came from the engine's request deltas, not launch arguments.

| Metric | A0 | B1 | B2 | A3 |
| --- | ---: | ---: | ---: | ---: |
| Engine decode ms/step (mean over 1- and 2-stream rounds) | 37.8490 | 37.5317 | 36.9564 | 37.7027 |
| Engine decode ms/step, 1 stream | 33.8491 | 33.0316 | 33.3815 | 33.3233 |
| Engine decode ms/step, 2 streams | 41.8488 | 42.0318 | 40.5314 | 42.0821 |
| Prefill TTFT, milliseconds (mean) | 1,410.49 | 1,402.07 | 1,440.93 | 1,415.27 |
| MTP accepted / drafted | 479 / 702 | 474 / 687 | 480 / 696 | 475 / 705 |
| Live helper W4H2 | enabled | enabled | enabled | enabled |
| Observed small-lane requests | 8,367 | 8,133 | 8,320 | see `evaluation.json` |

Pooled B against pooled A: **decode ms/step −1.408%** (a smaller number means faster); **prefill TTFT +0.610%** (a larger number means slower). The A-to-A and B-to-B spreads do not justify a speed claim.

**Hash gate:** all **3/3 single-stream decode** positions and **3/3 prefill** positions match across all four cells. **0/6 two-stream decode** positions match across all four. Also, the same-arm A0/A3 output hashes match in just **5/9** decode positions, and B1/B2 match in **6/9**; these instabilities occur with an unchanged binary and identical prompt hashes. Therefore the observed two-stream differences **cannot be attributed to the RCS13 candidate**; equally, the strict output-identity gate **cannot be waived**. No merge or approval on the strength of throughput or kernel activation alone.

The A0 scored run passed all requests and normal stop, then hit a scratch **evidence-copy-only** permission failure when recopying onto an already read-only archived manifest. A CPU-only audit verified each immutable request receipt, manifest SHA and clean GPU teardown, retained the original error in a recovery receipt and corrected the scratch copier to atomic replacement; no score was fabricated or GPU measurement re-run. B1, B2 and A3 passed without this bookkeeping repair.

## Real device execution and dynamic LDS

The candidate's helper startup flag alone was insufficient, so **after** A/B/B/A a *separate unscored* production serving start wrapped only the R9700 helper in `rocprofv3 --kernel-trace --stats`. It passed the canary, extra smoke and original seed-17 decode request, then stopped cleanly. ROCprof's actual device-dispatch stream contains:

| Executed gfx1201 kernel | Dispatches | Workgroup size | `LDS_Block_Size` profiler field |
| --- | ---: | ---: | ---: |
| `r3_moe_gpu::lane_moe_gemv_w4h2<true>` | 3,244 | 256 | 512 |
| `r3_moe_gpu::lane_moe_gemv_w4h2<false>` | 3,244 | 256 | 512 |

The source uses an aligned `extern __shared__` region in `moe_lane_gemm.h` and passes the checked `selected.dynamic_lds` bytes as the HIP shared-memory launch argument in `moe_remote_gpu.h:103-108`. The profiler's reported **`LDS_Block_Size` is reproduced verbatim** and is **not** equated with the HIP request size; it does not independently establish that byte quantity. Both names and dispatch counts were independently cross-checked against `kernel_stats.csv`. Full raw ROCprof CSV files remain in the local run evidence and were not added to Git because the sanitized summary and count gate are sufficient. This trace was **excluded from scored A/B/B/A times**.

## Recorded artifacts, reproduction and limits

- `replay.py`: standard-library-only, card-free checks of the sanitized four-cell manifest settings, prompt hashes, per-arm metrics, kernel dispatch counts and withheld verdict (`python3 docs/evidence/2026-11-18-rcs13-pr946-run12/replay.py`).
- `evaluation.json`: aggregate gates, all 4×3 seed/stream metrics, MTP counts, SHA-256 output comparisons, prompt hashes, production manifest-projected configuration, per-arm helper request counts.
- `kernel-trace-summary.json`: exact trace dispatch counts, sizes as reported, source pointer and interpretation limit.
- `settings-comparison.md`: documented Run11 versus production-setting differences and launcher/power admission deviation.
- Local raw run evidence: `rcs13/run12-serving/cell-{00-a,01-b,02-b,03-a}-no-power/` and the non-scored `cell-01-b-smoke-no-power-profile/` (request JSON, server/helper logs, measured v2 manifest, lease/gpu snapshots, kernel CSV). Original Run11 strict main/candidate bundle build attestation in `rcs13/run11-attest-{a,b}.json`.

For faithful reproduction, use the complete official source-matched A or B build through `launch-flashnext-best.sh start --build <build> --layout flashnext-tp2xtx-r9700 --heat tools/lab/radiance/heat/perf51.rank0.hot-first --ngram-placement disk --no-gpu-power`, let the *production launcher* own measured host-pool correction and normal canary, then run matched seeds in the same lease with the ordinary `stream_bench.run` benchmark. Never directly transplant a narrow 8,192-token Run11 manifest. In an environment with valid production power settings, restore the normal power-apply path for qualified speed comparisons.

**Next gate:** establish a deterministic concurrent-request output oracle (including same-arm control repeats and output-hash definition) before comparing the candidate for identical outputs, and separately quantify any speed changes beyond run-to-run spread. Earlier gfx1201 captured K=2560 replay byte identity (eight captures) remains historical evidence, not a substitute for concurrency parity or qualification of gfx1100/gfx1030. **No source commit to #946; no merge of #946.**
