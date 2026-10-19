# Qwen3.8-Flash-Next on radiance (Brutus)

Current-state page. Rule: every measurement or decision updates this page in the same change. Dated history and raw
tables: [engines/radiance/README.md](../../../../../engines/radiance/README.md) (the "lab log"). Plan items are in
`docs/planning/active/patching-radiance-rdna3/` (RAD*). Overview: [../README.md](../README.md).

Sources in this page: `RADn` = plan item notes (dated entries); `lab log` = engines/radiance/README.md section named;
`MEN11` = docs/planning/active/run-multi-engine/MEN11.md. Short hashes in parentheses are the completion hash or build
the item recorded. Run IDs are quoted where the source gives one; where it does not, the page says "run not recorded".

## Container and quantisation

Source: lab log, "Models and quantisation behind the measurements" (read with `rad-info`).

- Container `qwen3.8-next-flash-fp8-iq4r-moe.rad`, 113.6 GiB, under `/mnt/data/llm-models/radiance/StillDeadcode/` on
  Brutus. 512 experts, top 10, 48 layers plus an MTP layer.
- Routed experts `w4nl64a8h`: 4-bit codes into a 16-level nonlinear table, E4M3 scale per (row, 64 of K) x 2^-13,
  Hadamard-rotated with the activation; 49 x 512 x 2.42 MiB, about 58 GiB.
- N-gram embedding table fp8 E4M3, 47.7 GiB. Dense and attention int8 (group 128, bf16 scale, 3.4 GiB), fp8 E4M3
  (about 0.66 GiB), bf16 (2.4 GiB). Activations E4M3 with an f32 scale per 128 of K.
- KV cache `fp8` in every run (the owner's floor is 8-bit; f16 is preferred and has not been measured on radiance).
- The container's own MTP layer is fp8 e4m3 g128. Draft acceptance with radiance is 0.61-0.65 against llama.cpp's
  0.79 (lab log, "MTP against llama.cpp on the same workload", runs `fnm3`, `fnm0`, `fnmh3-1`, `fnmh3-0`, `fnr97`);
  neither the draft head's precision (rad0003) nor libr3 is the cause (same acceptance on the R9700 with libr4d).

## Reference quality against the full model

Source: owner-supplied, 2026-10-15 (lab log, "Reference quality against the full model"). The container against the
full model; "text+assistant" is the subset of tokens in text and assistant turns.

| Metric | All tokens | Text+assistant turns |
|---|---|---|
| Mean KL | 0.0789 | 0.0239 |
| Median KL | 0.0037 | 0.0029 |
| P99 KL | 1.65 | 0.32 |
| P99.9 KL | 5.33 | 1.03 |
| Top-1 agreement | 92.85% | 94.57% |

Layout drift is measured on a different footing (a 4-document corpus, 64-token steps, against the radiance two-XTX
exact run of the same container; see "Quality measurements"), so those KLD values are not directly comparable to this
table. The full metric set (median, P99, P99.9, top-1, text-only split) is not yet reported for the helper layout:
RAD08 extended the parser and gate (feat/rad08-kld-metrics, 4919eb61); reading the `--kld-out` JSON and a chat-format
corpus that yields the text+assistant column remain open.

## Current best layout and numbers

Layout (R9700 engine plus two XTX expert helpers; RAD02 pieces 1-5, RAD05, RAD06; engine patch rad0002, plugin
libr3moe, process `r3_moe_worker`):

- Engine on the R9700, `--tp 1 --kernels libr4d,libr3moe,libref`, radiance's own kernels.
- One `r3_moe_worker` per XTX (`R3_MOE_WORKER_COUNT=2`, devices "0 1"), `R3_MOE_WORKER_ENGINE_RANKS=1` (full-width
  experts), 22 GiB of experts each (`R3_MOE_WORKER_MIB=22528`), `R3_MOE_WORKER_SKIP_OWNED=1` (helper-owned experts are
  not placed on the R9700), ownership by route heat (`R3_MOE_WORKER_HEAT=<histogram>`), hottest experts to the
  helpers (`perf51/heat.rank0.hot-first`, RAD06 2026-10-17: +4% prefill against coldest-first, decode not worse), `R3_MOE_WORKER_MAX_ROWS=4096`, `R3_MOE_WORKER_SMALL_ROWS=8`.
- `--ngram-placement ram`, `--host-pool-mib 30720` (RAD02 used 30 GiB; `helper-ab.json` 30720), `--max-num-seqs 6`,
  `--max-num-batched-tokens 4096 --checkpoint-interval 4096`, `--kv-cache-dtype fp8`, `--num-speculative-tokens 3`.
- Harness form of exactly this: `tools/lab/radiance/helper-ab.json` (MEN11).

Radiance 1.7.0, harness (`helper-ab.json`, 2 balanced rounds, natives on; baselines in
`docs/reference/engines/radiance/baselines/`):

| Ownership | pp4096 tok/s | pp1024 tok/s | tg512 tok/s | tg2048 tok/s | KLD mean / P99 / top-1 | Runs |
|---|---|---|---|---|---|---|
| hottest experts to helpers, f16 WMMA MoE GEMM + half-staged activations + 16-byte lane I/O (current, `R3_MOE_FP16`, `R3_MOE_AHALF`, `R3_MOE_WORKER_VECIO`) | 3,309 / 3,314 | 3,314 / 3,325 | 137 / 138 | 144 / 144 | 0.0633 / 0.673 / 88.3% (FP16 build) | rad06-vecio e2e pairs, rad06-fp16kld-10181239 |
| hottest experts to helpers, f16 WMMA MoE GEMM only | 3,149 / 3,150 | 3,145 / 3,143 | 131 / 136 | 141 / 143 | 0.0633 / 0.673 / 88.3% | rad06-fp16-10181128, rad06-fp16kld-10181239 |
| hottest experts to helpers, bf16 WMMA | 2,134 / 2,140 | 2,144 / 2,143 | 136 / 130 | 143 / 140 | 0.0633 / 0.673 / 88.3% | rad06-heat-10172150, rad06-kld-hot-10172208 |
| coldest experts to helpers | 2,050 / 2,052 | 2,105 / 2,102 | 126 / 126 | 141 / 141 | 0.0642 / 0.684 / 88.7% | rad06-heat-10172150, port-native-kld-10171827 |

Helper large lane per request (hottest-first): routes 0.36, gate_up 2.10, quant 0.04, down 1.17, reduce 0.36 ms
(coldest-first: gate_up 2.52, down 1.41) - fuller 16-row tiles.

Earlier, radiance 1.3.0 (7,306-token prompt plus 256 greedy tokens, balanced A/B):

| Result | Prefill tok/s | Decode tok/s | Source |
|---|---|---|---|
| 22 GiB helpers, GEMV + native MoE, three runs | 2737 / 2762 / 2763 | 108.0 / 108.8 / ~108.5 (69036e14) | RAD06 2026-10-15 08:10 |
| ROWS=0 arm of the rows A/B | 2757.9 / 2764.0 / 2761.3 | 108.1 / 108.8 / 108.7 | RAD06 2026-10-15 (few-rows entry) |
| After reconciling onto main (one rad0002, one hook), identical completion hash | 2760 / 2761 | 108.4 / 108.9 | RAD09 acceptance, 2026-10-15 (main #236, 63cc952f) |

So the current figures are prefill ~2,755-2,764 tok/s, decode ~108.5 tok/s. Run IDs for these lab runs are not
recorded in the plan items (branch feat/rad03-gpu-lanes lab scripts, since retired).

Harness acceptance, MEN11 2026-10-17: `ab-benchmark --server-config tools/lab/radiance/helper-ab.json --pairs 4`,
Brutus, build c95d49e2, run `men11-fire-10171108`, bench config `mtp-dual`, 8 of 8 cells clean, both helpers ready and
stopped with exit 0 every cell. Arm `rows0` (the layout above): pp4096 1989-1991, pp1024 2035-2037, tg512 128.3-128.6,
tg2048 141.2-143.4 tok/s. These use the harness's own prompts, so they are not the same workload as the 7,306-token
lab figures above and are not to be compared to them directly.

How the layout got here (7,306-token prompt unless stated; "XTX pair" is the two-XTX tensor-parallel engine):

| Layout / step | Prefill tok/s | Decode tok/s | Source |
|---|---|---|---|
| radiance two XTX, first run, wht6, no stager | 178-245 | 28.3 | lab log, `fn2`, `fnp1` (2026-10-12) |
| two XTX, stager armed, 4K chunks, exact wire, n-gram in RAM, MTP 3 | 1,198-1,263 (`fnz*`) | 39.3 with MTP | lab log, `fnz1`-`fnz5` (2026-10-13) |
| two XTX, helper off (perf19 control) | 1185.5 / 1170.8 / 1191.1 / 1179.9 | 27.0-27.2 (349546fd) | RAD03 speed A/B, run `perf19` (2026-10-14) |
| two XTX + R9700 helper (CPU-free GPU lanes), perf19 "on" | 1257.6 / 1226.4 / 1257.4 / 1253.4 | 48.6 / 48.6 / 47.3 / 48.6 (7fdbdc2c) | RAD03, `perf19` |
| two XTX + R9700 helper, later state | ~1,480 | ~68 | RAD02 notes (2026-10-15), run not recorded |
| R9700 alone | 526 | 8.2 | RAD02 2026-10-15 01:20 |
| R9700 + one XTX helper, SKIP_OWNED | 992.7 / 1004.3 | 21.3 / 21.2 | RAD02 02:15 |
| R9700 + two XTX helpers (20 GiB) | 1881.5 / 1880.4 | 46.5 / 46.5 (6242c9b1) | RAD02 03:05 |
| + small lane 8 | 1783 / 1808 | 54.6 / 54.0 | RAD02 03:50 |
| + full-width helper | 1893 / 1943 | 52.8 / 53.1 (305c5aff) | RAD02 04:45 |
| + route-heat ownership | 2635.6 / 2638.6 | 59.8 / 59.8 (fbb7cfab) | RAD02 05:10 |
| + stage-once, 20 GiB helpers | 2765.6 / 2767.2 | 60.0 / 60.0 | RAD02 05:35 |
| 22 GiB helpers | 2529.4 / 2458.1 | 69.6 / 69.7 (bef41b47) | RAD02 05:35 |
| + decode GEMV (rad05, `R3_MOE_WORKER_GEMV`) | 2197 (warm-up) / 2524 | 95.7 / 95.5 | RAD05 07:20 |
| + native MoE inner loop in the helpers | 2324 (warm-up) / 2665 | 108.2 / 108.8 | RAD06 07:45 |
| + helper budget re-tune (22 GiB) | 2737 / 2762 / 2763 | 108.0 / 108.8 / ~108.5 | RAD06 08:10 |

References, not this layout: llama.cpp three cards, IQ4_XS GGUF: prefill 1,012 (3.2K) and 1,469 (6.2K), MTP decode
72.2 (3.3K) / 55.6 (6.8K) (lab log, `lcfn-*`; GGUF quantisation, not like-for-like). Radiance on two R9700: about
6,500 tok/s prefill at 26K and 15.3 ms a decode step (radiance `docs/MOE-W4.md`, quoted in the lab log; container not
recorded, treat as indicative). Prefill of the best layout is about 40% of that two-R9700 figure (RAD02 05:35).

MTP draft acceptance of the current best layout has not been compared (RAD02 notes: "different text; acceptance not
compared"); generated text differs from the XTX-pair run, as the noise-floor finding predicts.

## What is on and why

Engine patches (state from each SUMMARY.md; none is `validated` yet):

| Patch | State | In the current layout because |
|---|---|---|
| rad0001 drain mover before arena free | evaluated | the prefill stager (needs `--max-num-seqs 6`) hung at shutdown in 2 of 3 runs without it; 3 of 3 clean with it (runs `fnh1`-`fnh3` vs `fnp1`-`fnp3`). In source `radiance-lab`. |
| rad0002 MoE remote hand-off | untested | the seam through which helper-owned routes are masked locally and published to the helper; the whole helper layout depends on it. Enabled by `R3_MOE_WORKER`. Reconciled with the libr3moe hook on main (RAD09, #236). |
| rad0003 MTP full-head switch | untested | not used: `RAD_MTP_FULL_HEAD=1` did not change acceptance (0.63-0.65 vs 0.62-0.64). |

Component settings (`engines/radiance/components/libr3/component.toml`; default and status as declared there):

| Setting | Default | Status | Role here |
|---|---|---|---|
| `R3_MOE_WORKER` | unset | experimental | turns on the rad0002 GPU mailbox ops |
| `R3_MOE_WORKER_DEVICES` | unset | experimental | harness-only: one helper process per listed HIP index ("0 1") |
| `R3_MOE_WORKER_COUNT` | 1 | experimental | 2 = two XTX helpers behind a one-rank engine |
| `R3_MOE_WORKER_ENGINE_RANKS` | 2 | experimental | 1 = one engine (the R9700) asking for full-width experts |
| `R3_MOE_WORKER_MIB` | 28672 | experimental | helper expert budget; 22528 (22 GiB) in the best layout (20 GiB gives decode ~89 and no prefill gain, RAD06 08:10) |
| `R3_MOE_WORKER_MAX_ROWS` | 64 | experimental | 4096 so no step falls back to the engine path |
| `R3_MOE_WORKER_SMALL_ROWS` | 64 | experimental | 8: the pre-queued small lane sized for decode and MTP verify steps |
| `R3_MOE_WORKER_MOE_DOWN_TILE` | unset (production configs: 1) | experimental | narrow down tile on gfx1100: +0.6% prefill, bit-identical (RAD06 2026-10-18) |
| `R3_MOE_WORKER_HEAT` | unset | experimental | route-heat file; helpers own the experts it ranks first - production uses `heat.rank0.hot-first` (hottest first, +4% prefill over coldest-first, RAD06 2026-10-17; coldest-first was +36% over unordered, RAD02) |
| `R3_MOE_WORKER_GEMV` | 1 | experimental | decode MoE GEMV for the small lane; decode 69.6 to 95.7 tok/s (RAD05) |
| `R3_MOE_WORKER_HISTOGRAM` | unset | diagnostic | produces the heat file (profile run) |
| `R3_MOE_WORKER_FUSESTAGE` | 1 | experimental | the engine's activation staging for a decode-sized step (at most 16 rows) done by the publish kernel instead of a separate staging kernel in front of it: a launch and its gap off the path before every helper request; bit-identical data, same system fence; four balanced pairs: tg512 141.6 -> 144.2 (+1.8%), tg2048 151.1 -> 152.3 (+0.8%), better in all eight comparisons, prefill unchanged (RAD06 2026-10-18). Read per launch in the engine process; 0 keeps the separate kernel. |
| `R3_MOE_WORKER_FUSEROUTE` | 1 | experimental | the small (decode) lane's whole front end (status, the activation copy, the lane's expert tables and the routing prologue: four memsets and seven kernels, eleven launches) as one block, lane_front_small; bit-identical, the replay's request span -13% to -21%; decode tg512 132 -> 141.7 (+7%), tg2048 143.7 -> 151.4 (+5.4%), prefill unchanged (RAD06 2026-10-18). Folding the activation copy and tables into the routing block added about +0.8% over fusing the routing alone. Read per request; 0 keeps the separate kernels. |
| `R3_MOE_WORKER_TRACE` | off | diagnostic | every Nth finished helper request prints its critical-path stamps (engine publish, wait entered, completion seen; helper start and done, each on its own device clock); read with `tools/lab/radiance/heat-placement/trace_report.py`. The stamps are nine stores a request in the mapped slot (protocol version 12). |
| `R3_MOE_WORKER_VECIO` | 1 | experimental | the helper's activation copy and result write use 16-byte accesses to the engine's mapped host memory instead of 1- and 2-byte ones; bit-identical (4 real captures), lane kernels -31% (3,490 -> 2,410 us), prefill 3,160 -> 3,310 tok/s (+4.7%), decode +1.5-3% (RAD06 2026-10-18). Read per request; 0 keeps the scalar kernels. |
| `R3_MOE_FP16` | 1 | experimental | f16 WMMA with E4M3 widened to half in the helper's native MoE GEMM; bit-identical output, grouped GEMM 1.6-2.75x faster on four real captured requests (bit-identical output), prefill 2,130 -> 3,150 tok/s, decode unchanged (RAD06 2026-10-18). On by default in the helpers (main() sets it unless set); 0 turns it off. |
| `R3_MOE_AHALF` | 1 | experimental | activation tile widened to half once at staging (with `R3_MOE_FP16`); bit-identical, a further 7-8% off the grouped GEMM (1,905 -> 1,771 us on the largest real capture), +4 KB LDS (RAD06 2026-10-18). On by default in the helpers; 0 turns it off. |
| `R3_MOE_NATIVE` | 1 | experimental | native bf16 WMMA inner loop; on by default in the helpers (`r3_moe_worker` main sets it unless set, 2bf231e3): decode 91.4/95.6 to 108.2/108.8. The engine process reads an unset variable as off; component.toml declares the helper default. |
| `R3_MOE_WORKER_SKIP_OWNED` | (rad0002 patch variable, not a component setting) | | helper-owned experts are not placed on the R9700; set in the shared environment in `helper-ab.json` |

Flags and profile: the `flashnext-2xtx` profile in `recipes.toml` (`--tp 2 --tp-wire exact`, `expert_tiered`,
`--ngram-placement ram`, `--host-pool-mib 14336`, 6 seqs, 4096 chunks, fp8 KV, MTP 3, production context 32768) describes
the older two-XTX configuration; there is no runtime profile yet for the R9700 helper layout (the harness config is the
reference).

Why the supporting flags are set (lab log, "Flash-Next: radiance's own prefill stager" and neighbours):

- `--max-num-seqs 6` arms radiance's prefill stager (1.8x prefill at 2.6K, 2.2x at 7.2K, runs `fnsu1`-`fnsu4`).
- 4096-token chunks: 1,250-1,320 tok/s at both 2.6K and 7.2K (runs `fny1`-`fny4`); 8K chunks only suit long prompts.
- `--ngram-placement ram`: as fast as the R9700 split, and the R9700 is for compute (runs `fnz1`-`fnz5`; owner,
  2026-10-13).
- Exact wire, not `wht6`, on the two-XTX engine: wht6 gave KLD 0.0746 / top-1 87.18% on this model (run `fnk1`) for
  about 5% prefill (lab log). Not relevant to the R9700 engine layout, which has no cross-card sum.

## Tried and shelved

| Item | Result | Evidence |
|---|---|---|
| Few-rows-a-expert prefill kernel, `R3_MOE_WORKER_ROWS=1` (RAD06; commits aceef746, 8cae20dc) | correct (`--compare` rel_l2 0.0011/0.0012) but slower: prefill 1719.8/1682.1/1726.8 vs 2757.9/2764.0/2761.3 (-38%), decode 96.1-96.3 vs 108.1-108.8 (-11%, MTP verify batches use the large lane). Cause: scalar FMA against bf16 WMMA, and every N tile of 32 re-dequantises the same activation rows. Default off, evaluated/shelved. | RAD06 notes (lab); harness repeat `men11-fire-10171108`: pp4096 472-485 vs 1989-1991, tg512 76.3-77.2 vs 128.3-128.6 (MEN11). Activation marker added in #250. |
| Two-stream helper halves | hung the lane: the shared libr3 MoE GEMM is not safe to run concurrently on one card; reverted (563455f8) | RAD02 03:50 |
| R9700 as n-gram storage (`R3_NGRAM_DEV`, helper `libr3_ngram_helper.so`) | exact and faster than disk (+11% decode, 1.6-2.1x prefill vs the disk table) but a poor topology; the owner's rule is that the R9700 is compute, lookup work goes to RAM/CPU. Settings kept as `evaluated`. | lab log, run `fnng7`; owner decision 2026-10-13 |
| CPU-mediated R9700 helper (slice 2b) | superseded by GPU lanes (RAD03); kept as the evaluated baseline | RAD03 description |
| Higher-precision draft head (`RAD_MTP_FULL_HEAD`, rad0003) | acceptance 0.63-0.65 vs 0.62-0.64; not the cause of low acceptance | lab log runs `fnmh3-1`, `fnmh3-0`, `fnr97` |
| `R3_HC_NATIVE=1` (native E4M3 hc_read) | 58.2 to 50.7 us a call, decode 29.2 to 29.7 tok/s (inside spread); declared experimental, off | lab log, `fnhc1`/`fn7`/`mps2` section |
| Stopping the heat engine (`--deterministic`) on the XTX pair | worse: decode 19.9 vs 29.2 | lab log, run `fn7` |
| Small lane 16 rows (`R3_MOE_WORKER_SMALL_ROWS=16`) | decode 50-81 vs 131 tok/s: the pre-queued small lane pads every step | RAD06 2026-10-18 |
| Large-lane GEMV up to 32 rows (`R3_MOE_WORKER_GEMV_LARGE_ROWS=32`) | bit-exact on replay; prefill flat (pp4096 2145-2148 vs 2148-2149) but decode 80.4-81.1 vs 130.5-142.4 (-39%); default off, evaluated | run `rad06-gemvl-10180418` (2 pairs) |
| Long-run tile (`R3_MOE_WORKER_MOE_LONG_ROWS=65`): 64-row slices and an MT=4 tile for runs of 65+ rows | bit-identical (rel_l2 0) but slower on replayed 2048-row requests: layer 29 3.57 ms baseline vs 3.72 ms (8 waves, 240 VGPR, 140 B scratch) and 3.86 ms (16 waves, 176 VGPR, no scratch); layer 11 2.65 vs 3.15 ms (16 waves). Spill-free MT=4 still loses, so re-staging the weight tile across 32-row blocks is not the bottleneck. Default off, evaluated. | RAD06 2026-10-18, `tools/lab/radiance/helper-long-rows.sh` |
| Per-layer cost-balanced helper ownership (`R3_MOE_WORKER_BALANCE`, `_SPEED`; RAD06/RHP) | the same jointly owned experts split per layer by predicted cost: offline prediction 3.5-5.6% off the helpers' per-layer worst case; measured the slower helper's large request 1,719 -> 1,535 us (level with the other), but prefill unchanged (pp4096 3,314 / 3,314 vs 3,316 / 3,312 tok/s: the helpers no longer limit prefill). Quality unchanged: KLD 0.0644 / p99 0.705 / top-1 88.46% (helper-off 0.0646 / 0.735 / 88.39%). The decode tok/s the A/B showed (88 against 132-144) is a benchmark-content effect, not a slowdown: the new owner split changes the order in which the two helpers' partial sums are added, the greedy continuation diverged and ended after 282 tokens (tg512 and tg2048 both) instead of 512 / 2,048, with fewer speculative tokens accepted a step; per-call cost of every engine op and the time of a step (~31 ms) are the same in both arms (profiled), the run just did half the steps. Any ownership change can do this, so compare decode speed between ownership layouts only on a fixed-text benchmark. Default off, evaluated. | rad06 balance A/B and profile runs, KL run rad06-fp16kld-10181903, `tools/lab/radiance/heat-placement/balance_score.py` |
| Few-rows kernel on the decode lane (`R3_MOE_WORKER_ROWS_SMALL`; RAD06) | an 8-token decode request sends ~2.4 routes to each distinct expert (145 routes over 60 experts in four real small-lane requests), and the per-route GEMV reads an expert's weights once a route, so reading them once for all the expert's rows looked worth up to 2.4x of the weight traffic. Measured on the small-lane replay it is slower: lane kernel time 830-1,392 us against 302-654 us a run (1.7-2.2x), and not exact (1-3 of 20,480 outputs differ). The per-route GEMV evidently gets its repeats from the Infinity Cache, and the kernel's one-block-an-expert grid is mostly empty at 8 rows. Default off, evaluated. | `helper-sweep.sh` with `small_capture.py` captures, 4 captures x 2 repeats |
| Helper at 20 GiB with the final kernels | decode 89.2 (vs ~108.5 at 22 GiB), no prefill gain | RAD06 08:10 |

## Where the critical path is (RAD06 unprofiled trace, 2026-10-18)

From the helper TRACE lines of a full benchmark cell (`R3_MOE_WORKER_TRACE=1`, 36,643 of 36,729 layer-steps paired, no throughput cost). Exposed wait is the engine's wait for the slower helper, on the R9700 clock.

| lane | publish -> wait (mean) | exposed wait (mean / p95) | slower helper GPU time (mean) | total exposed wait in the cell |
|---|---|---|---|---|
| decode (small lane, 36,336 paired steps) | 75 us | 121 / 185 us | 166 us | 4.4 s, about 23% of decode time |
| prefill (large lane, 307 steps) | 596 us | 1,170 / 3,516 us | 1,676 us | 0.36 s, about 18% of prefill time |

Helper request time is local work plus exposed wait, so the helpers are the critical path and only partly hidden. Each kernel gain converts at about 0.7-1% of prefill per 0.1 ms off the helper request. In decode the request is a few small kernels and gaps; fusing the routing prologue (R3_MOE_WORKER_FUSEROUTE) took about 25 us off it for +5-8% decode.

Caveats: a layer's exposed wait is the larger of the two helpers' waits (the wait kernel polls them in order); the publish -> wait window is not isolated local-expert time. Comparing decode speed between ownership layouts needs a fixed-text benchmark: a different split changes the summation order and the greedy text (see the balanced-ownership row under Tried and shelved).

## Multi-stream baseline (radiance 1.8.0, RAD10-01, 2026-10-19)

First concurrent-stream measurement: `bigcherry stream-bench` inside the ab-benchmark lifecycle
(`tools/lab/radiance/helper-streams-ab.json`; evidence `docs/evidence/RAD10/streams-180-baseline/`). R9700 engine plus two XTX
helpers, the validated helper settings, `--max-num-seqs 6`, 512-token random-word prompts (a different one per stream), 128
generated tokens, temperature 0, two repetitions a round, two identical arms (A/A) in two balanced pairs, so the spread between
the four cells is the noise floor.

| Streams | Aggregate tok/s (four cells) | Per-stream decode tok/s | TTFT p95 (the slowest stream) | Event-gap p95 |
|---|---|---|---|---|
| 1 | 85.9 - 95.4 | about 93 | 318 - 415 ms | 26.6 - 32.3 ms |
| 2 | 131.7 - 132.4 | about 66 | 542 ms | 32.1 - 32.4 ms |
| 4 | 165.2 - 168.4 | about 64 - 70 | 1,052 - 1,087 ms | 47.9 - 48.8 ms |
| 6 | 193.1 - 199.3 | about 33 | 1,581 - 1,592 ms | 55.2 - 56.3 ms |

- Aggregate throughput scales 2.1x from one stream to six; each stream slows to about a third. Above one stream the A/A spread
  is under 1.5%; the single-stream figure is noisy (+-7%).
- The single-stream decode here (about 90 tok/s) is below the 145 tok/s of the tg512 acceptance cell: these prompts are random
  words, so speculative-decode acceptance is low. Compare within this table, not with the acceptance numbers.
- These figures came from the first version of `stream-bench`: the gap column is the time between streamed text events (one
  event can carry several tokens with MTP, so it is not a token latency), the aggregate is finite-batch makespan throughput
  (all tokens over first request start to last response end), and the per-stream rate is a client-observed estimate. The
  harness now separates server token counts from text events, fails without a usage block, and reports the maximum TTFT.
- Identity: at one stream the text is identical across arms. At four streams the first repetition matches across arms but a
  second repetition of the same prompts and seed produced different text, so concurrent decode is not bitwise reproducible
  from run to run; per-stream correctness gates for RAD10 must use quality (KL, top-1), not exact text.
- Not measured yet: mixed long-prefill plus decodes, open-loop arrivals, queueing beyond `--max-num-seqs`, engine-side helper
  counters in the same cell. For scale, Paiton reports 565 -> 685 tok/s at eight streams on two P2P R9700 (a different setup).

### Small-lane row limit 8 against 24, with the engine's own counters (RAD10-02, 2026-10-19)

`tools/lab/radiance/helper-smallrows-ab.json`; evidence `docs/evidence/RAD10/smallrows-180/`. Two balanced pairs, 2, 4 and 6 streams,
`R3_MOE_WORKER_SMALL_ROWS` 8 (production) against 24. Each round also reads the server's `/metrics` before and after (decode steps and
step time, committed tokens a step, draft acceptance), so the table below is the decode phase alone.

| Streams | Arm | Decode step | Committed tokens a step | Decode throughput (engine) | Client aggregate | Draft acceptance |
|---|---|---|---|---|---|---|
| 2 | 8 | 31.7 ms | 5.8 | about 183 tok/s | 129.5 | 0.64 |
| 2 | 24 | 33.0 ms | 5.8 | about 176 | 123.9 | 0.64 |
| 4 | 8 | 44.1 ms | 11.3 | about 256 | 167.6 | 0.67 |
| 4 | 24 | 40.9 ms | 10.8 | about 265 | 173.2 | 0.63 |
| 6 | 8 | 50.3 ms | 16.3 | about 324 | 194.9 | 0.66 |
| 6 | 24 | 49.2 ms | 16.6 | about 337 | 200.0 | 0.65 |

- **The decode phase scales much better than the client aggregate showed.** From 2 to 6 streams the step time goes 32 -> 50 ms while
  the tokens a step triple, so decode throughput rises about 1.8x (183 -> 324 tok/s). The client-side aggregate (129 -> 195) is lower
  because a short 128-token round also contains every stream's 512-token prefill and the ramp; at 6 streams the first token already
  takes 1.6 s. The earlier "2.1x" was largely prefill and round length, not decode.
- **Each stream commits about 2.8 tokens a step** (1 plus 3 drafts at 0.64 acceptance), steady across stream counts, so MTP is not what
  degrades with concurrency here.
- **The lane limit is a small lever.** 24 against 8: about -4% at 2 streams (the small lane is padded to 24 rows), about +3.5% at 4 and
  +4% at 6 streams in decode throughput. Not a free win; the threshold would have to follow the number of active streams. Two pairs
  only, so treat the percentages as direction.
- **What was not measured:** how much of the 32 -> 50 ms step growth is the helper path. That needs the helpers' own per-request counters
  in the same cell (rows, lane, wait), the next RAD10-02 step.

### Where the decode step goes: the helpers' own view (RAD10-02, 2026-10-19)

`tools/lab/radiance/helper-smallrows-ab.json` with `R3_MOE_WORKER_TRACE=1` in both arms; evidence `docs/evidence/RAD10/traced-180/`. Streams
1, 2, 3, 4 and 6 (each stream sends about 4 verify rows a step with MTP depth 3, so rows a helper request are 4, 8, 12, 16 and 24), two
pairs. `trace_report.py --windows` buckets the helpers' TRACE lines by the stream-bench round that produced them. Clean cells only: one
cell (`pair-002-rows8`) ran 40-60% slower than its twin for 1 to 4 streams, the intermittent host-side slow mode seen before (and flagged
there); the 24-row arm's two pairs agree within 2%.

| Streams | Rows a request | Arm | Small-lane share | Exposed helper wait a layer-step (p50) | Decode step | Wait as a share of the step* | Engine decode tok/s |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 8 | 96% | 118 us | 25.8 ms | 22% | 123 |
| 1 | 4 | 24 | 96% | 152 us | 28.6 ms | 26% | 111 |
| 2 | 8 | 8 | 97% | 189 us | 32.4 ms | 29% | 177 |
| 2 | 8 | 24 | 97% | 218 us | 33.0 ms | 32% | 174 |
| 3 | 12 | 8 | **25%** | 370 us | 39.6 ms | 46% | 194 |
| 3 | 12 | 24 | 95% | 258 us | 36.4 ms | 35% | **235** |
| 4 | 16 | 8 | **7%** | 446 us | 49.8 ms | 44% | 236 |
| 4 | 16 | 24 | 95% | 345 us | 40.8 ms | 41% | **265** |
| 6 | 24 | 8 | 13% | 543 us | 53.1 ms | 50% | 315 |
| 6 | 24 | 24 | 94% | 481 us | 49.2 ms | 48% | 336 |

*wait p50 x 49 MoE layers / decode step; an estimate, since the median ignores the tail.

- **The helpers are what limits concurrency.** From one stream to six the decode step grows 26 -> 53 ms, and the exposed helper wait (the
  engine waiting for the helpers after its own work) grows from 5.8 ms to about 26 ms of it, roughly 75% of the growth. As a share of the
  step it rises from about a quarter to about half. This matches the 23% exposed wait measured at one stream.
- **The 8-row small lane is the wrong size for concurrency.** At three and four streams the requests (12 and 16 rows) fall onto the
  large lane (small-lane share 25% and 7%). Raising the limit to 24 keeps them on the small lane and gives +21% (3 streams) and +12%
  (4 streams) decode throughput, +6% at 6, at the cost of -10% at one stream, because every pre-queued small step is padded to the
  limit. So the lane should follow the rows: several pre-queued small-lane sizes (4, 8, 16, 24), the engine choosing the smallest that fits.
- **The two helpers are not balanced:** the median difference between them is 10-27% of the slower one, largest at one stream.
- **Not tested:** whether faster helper kernels for 12 to 24 rows, or fewer dispatches (aggregation), would cut the wait more than the
  lane sizing; the wait per layer-step still grows 3 to 4 times with rows even on the small lane.

## Open levers

- **RAD06, prefill (the ranked remaining levers):** (1) multi-buffered chunk pipeline of the hand-off; analysis
  (RAD06 08:20) says a within-layer pipeline cannot go below the helpers' ~4.5 ms a layer, so it may be of limited
  value; (2) WMMA-based few-row tiles (16-row WMMA, activations shared across N in LDS); (3) prefill-load ownership
  rebalance; (4) cross-layer row pipelining (RAD04-style engine change; upstream's simpler overlap lost 6%).
  RAD06 08:20 gave ~3,500-4,000 tok/s as the ceiling for the few-rows kernel, which then lost (see Tried and shelved); two-R9700 ~6,500 is "unlikely on this hardware".
  Protocol hardening (v11, e26c1b94) is done.
- **RAD06 note 2026-10-17, hipBLASLt/Tensile ideas** (idea source, not measured): per-shape tile-variant tables selected
  at runtime (like the llama.cpp hip-autotune record/tune/replay), split-K / stream-K for skinny GEMMs (the
  few-rows case the FMA kernel lost on), grouped GEMM with per-group tile choice (cold vs hot experts),
  workgroup-to-tile mapping for Infinity Cache reuse, double-buffered / direct-to-LDS loads; mine the gfx1100 Tensile
  logic YAMLs under /opt/rocm/lib/hipblaslt/library for tile sizes near our shapes before writing a tuner.
- **RAD05 remaining scope:** use the decode GEMV for the XTX engine's own decode MoE and the R9700 engine's small
  steps; RDNA2 (gfx1030) build for a 6900 XT helper.
- **RAD02:** KL re-check for the full-width helper before making it default (full-width rounding differs from the
  TP-split reference; KLD 0.0672 at 04:45).
- **RAD08:** report the full KL metric set (and the text+assistant split) for the helper layout.
- **RHP01-RHP06 (patching-radiance-heat-placement):** replace the static heat file with shared, cost-aware, restart-safe
  ownership from Radiance's own router histograms (shadow mode first).
- **RWQ01-RWQ06 (patching-radiance-wire-quality):** wire quality between cards; relevant to two-XTX layouts, not the
  helper layout.
- **MEN11:** KLD runs through the harness (`helper-kld.json`, `quality` block) and component settings have a promotion
  path (see the radiance README); the lab scripts `fn-r9700-engine.sh`, `fn-bench-once.py`, `fn-helper-gate.sh`,
  `fn-helper-kld.sh` and `fn-r9700-kld.sh` are retired in favour of `helper-ab.json` and `helper-kld.json`.
- **Promotion:** none of rad0001-rad0003 is validated; owner policy is lightweight promotion (mechanics tests,
  activation evidence, balanced A/B with identity). The universal-default-on question for helper settings is open.
- **Decode headroom:** the helper's per-layer round trip bounded decode before RAD05; no current measurement of what
  bounds it now (gap).

## Quality measurements

Corpus: 4 documents, 64-token steps; reference: the radiance XTX-pair exact run of the same container (reference PPL
9.1034). Noise floor: re-chunking the stock XTX path alone gives KLD 0.0624 / 0.0582 at 128 / 256 tokens (top-1
88.4%); helper-only-bit-identical rows give KLD 0.000000, so the divergence is the model's numerical sensitivity, not a
helper defect (RAD03 2026-10-14). The bar for helper layouts is therefore "inside 0.058-0.067", not bit-exact.

| Configuration | KLD | Top-1 | PPL | Source |
|---|---|---|---|---|
| R9700 helper on the XTX pair (GPU lanes) | 0.0616 | 88.8% | | RAD03 2026-10-14 |
| R9700 helper on the XTX pair, protocol v8 | 0.0665 | 88.46% | 8.987 | RAD02 2026-10-14 late |
| two helpers, small lane 64 | 0.0592 | 88.48% | 8.987 | RAD02 03:50 |
| full-width helper | 0.0672 | 88.53% | 9.0885 | RAD02 04:45 |
| + heat ownership | 0.0665 | 88.54% | 9.118 | RAD02 05:10 |
| + decode GEMV | 0.0605 | 88.62% | 9.003 | RAD05 07:20 |
| + native MoE in helpers | 0.0642 | 88.73% | | RAD06 07:45 |
| ROWS=1 prefill kernel (shelved) | not measured; compare tolerance only | | | RAD06 |

The final configuration after the 08:10 budget re-tune (22 GiB, GEMV, native) has no KL record of its own; the
nearest are the two rows above (Gaps). Earlier two-XTX engine measurements, 42,761 positions against the same engine
with every native form off (lab log, run `fnk1`): n-gram helper, exact wire 0.000000 / 100%; `wht6` 0.0746 / 87.18% /
PPL 10.436 against 10.106; wht6 plus native forms 0.0751 / 87.52%.

## Gaps (could not be sourced)

- Run IDs for the RAD02/RAD05/RAD06 lab A/Bs (the plan items quote build hashes and completion hashes, not run names).
- KL for the final 22 GiB + GEMV + native configuration, and the full metric set for any helper layout.
- MTP acceptance for the helper layout.
- Harness (MEN11) numbers for decode at 7.3K prompt length; harness prompts are pp4096/pp1024/tg512/tg2048.
