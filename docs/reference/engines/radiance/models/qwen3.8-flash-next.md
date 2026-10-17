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
libr3moe, process `r3_moe_helper`):

- Engine on the R9700, `--tp 1 --kernels libr4d,libr3moe,libref`, radiance's own kernels.
- One `r3_moe_helper` per XTX (`R3_MOE_HELPER_COUNT=2`, devices "0 1"), `R3_MOE_HELPER_ENGINE_RANKS=1` (full-width
  experts), 22 GiB of experts each (`R3_MOE_HELPER_MIB=22528`), `R3_MOE_HELPER_SKIP_OWNED=1` (helper-owned experts are
  not placed on the R9700), ownership by route heat (`R3_MOE_HELPER_HEAT=<histogram>`), hottest experts to the
  helpers (`perf51/heat.rank0.hot-first`, RAD06 2026-10-17: +4% prefill against coldest-first, decode not worse), `R3_MOE_HELPER_MAX_ROWS=4096`, `R3_MOE_HELPER_SMALL_ROWS=8`.
- `--ngram-placement ram`, `--host-pool-mib 30720` (RAD02 used 30 GiB; `helper-ab.json` 30720), `--max-num-seqs 6`,
  `--max-num-batched-tokens 4096 --checkpoint-interval 4096`, `--kv-cache-dtype fp8`, `--num-speculative-tokens 3`.
- Harness form of exactly this: `tools/lab/radiance/helper-ab.json` (MEN11).

Radiance 1.7.0, harness (`helper-ab.json`, 2 balanced rounds, natives on; baselines in
`docs/reference/engines/radiance/baselines/`):

| Ownership | pp4096 tok/s | pp1024 tok/s | tg512 tok/s | tg2048 tok/s | KLD mean / P99 / top-1 | Runs |
|---|---|---|---|---|---|---|
| hottest experts to helpers (current) | 2,134 / 2,140 | 2,144 / 2,143 | 136 / 130 | 143 / 140 | 0.0633 / 0.673 / 88.3% | rad06-heat-10172150, rad06-kld-hot-10172208 |
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
| + decode GEMV (rad05, `R3_MOE_HELPER_GEMV`) | 2197 (warm-up) / 2524 | 95.7 / 95.5 | RAD05 07:20 |
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
| rad0002 MoE remote hand-off | untested | the seam through which helper-owned routes are masked locally and published to the helper; the whole helper layout depends on it. Enabled by `R3_MOE_HELPER`. Reconciled with the libr3moe hook on main (RAD09, #236). |
| rad0003 MTP full-head switch | untested | not used: `RAD_MTP_FULL_HEAD=1` did not change acceptance (0.63-0.65 vs 0.62-0.64). |

Component settings (`engines/radiance/components/libr3/component.toml`; default and status as declared there):

| Setting | Default | Status | Role here |
|---|---|---|---|
| `R3_MOE_HELPER` | unset | experimental | turns on the rad0002 GPU mailbox ops |
| `R3_MOE_HELPER_DEVICES` | unset | experimental | harness-only: one helper process per listed HIP index ("0 1") |
| `R3_MOE_HELPER_COUNT` | 1 | experimental | 2 = two XTX helpers behind a one-rank engine |
| `R3_MOE_HELPER_ENGINE_RANKS` | 2 | experimental | 1 = one engine (the R9700) asking for full-width experts |
| `R3_MOE_HELPER_MIB` | 28672 | experimental | helper expert budget; 22528 (22 GiB) in the best layout (20 GiB gives decode ~89 and no prefill gain, RAD06 08:10) |
| `R3_MOE_HELPER_MAX_ROWS` | 64 | experimental | 4096 so no step falls back to the engine path |
| `R3_MOE_HELPER_SMALL_ROWS` | 64 | experimental | 8: the pre-queued small lane sized for decode and MTP verify steps |
| `R3_MOE_HELPER_MOE_DOWN_TILE` | unset (production configs: 1) | experimental | narrow down tile on gfx1100: +0.6% prefill, bit-identical (RAD06 2026-10-18) |
| `R3_MOE_HELPER_HEAT` | unset | experimental | route-heat file; helpers own the experts it ranks first - production uses `heat.rank0.hot-first` (hottest first, +4% prefill over coldest-first, RAD06 2026-10-17; coldest-first was +36% over unordered, RAD02) |
| `R3_MOE_HELPER_GEMV` | 1 | experimental | decode MoE GEMV for the small lane; decode 69.6 to 95.7 tok/s (RAD05) |
| `R3_MOE_HELPER_HISTOGRAM` | unset | diagnostic | produces the heat file (profile run) |
| `R3_MOE_NATIVE` | 1 | experimental | native bf16 WMMA inner loop; on by default in the helpers (`r3_moe_helper` main sets it unless set, 2bf231e3): decode 91.4/95.6 to 108.2/108.8. The engine process reads an unset variable as off; component.toml declares the helper default. |
| `R3_MOE_HELPER_SKIP_OWNED` | (rad0002 patch variable, not a component setting) | | helper-owned experts are not placed on the R9700; set in the shared environment in `helper-ab.json` |

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
| Few-rows-a-expert prefill kernel, `R3_MOE_HELPER_ROWS=1` (RAD06; commits aceef746, 8cae20dc) | correct (`--compare` rel_l2 0.0011/0.0012) but slower: prefill 1719.8/1682.1/1726.8 vs 2757.9/2764.0/2761.3 (-38%), decode 96.1-96.3 vs 108.1-108.8 (-11%, MTP verify batches use the large lane). Cause: scalar FMA against bf16 WMMA, and every N tile of 32 re-dequantises the same activation rows. Default off, evaluated/shelved. | RAD06 notes (lab); harness repeat `men11-fire-10171108`: pp4096 472-485 vs 1989-1991, tg512 76.3-77.2 vs 128.3-128.6 (MEN11). Activation marker added in #250. |
| Two-stream helper halves | hung the lane: the shared libr3 MoE GEMM is not safe to run concurrently on one card; reverted (563455f8) | RAD02 03:50 |
| R9700 as n-gram storage (`R3_NGRAM_DEV`, helper `libr3_ngram_helper.so`) | exact and faster than disk (+11% decode, 1.6-2.1x prefill vs the disk table) but a poor topology; the owner's rule is that the R9700 is compute, lookup work goes to RAM/CPU. Settings kept as `evaluated`. | lab log, run `fnng7`; owner decision 2026-10-13 |
| CPU-mediated R9700 helper (slice 2b) | superseded by GPU lanes (RAD03); kept as the evaluated baseline | RAD03 description |
| Higher-precision draft head (`RAD_MTP_FULL_HEAD`, rad0003) | acceptance 0.63-0.65 vs 0.62-0.64; not the cause of low acceptance | lab log runs `fnmh3-1`, `fnmh3-0`, `fnr97` |
| `R3_HC_NATIVE=1` (native E4M3 hc_read) | 58.2 to 50.7 us a call, decode 29.2 to 29.7 tok/s (inside spread); declared experimental, off | lab log, `fnhc1`/`fn7`/`mps2` section |
| Stopping the heat engine (`--deterministic`) on the XTX pair | worse: decode 19.9 vs 29.2 | lab log, run `fn7` |
| Small lane 16 rows (`R3_MOE_HELPER_SMALL_ROWS=16`) | decode 50-81 vs 131 tok/s: the pre-queued small lane pads every step | RAD06 2026-10-18 |
| Large-lane GEMV up to 32 rows (`R3_MOE_HELPER_GEMV_LARGE_ROWS=32`) | bit-exact on replay; prefill flat (pp4096 2145-2148 vs 2148-2149) but decode 80.4-81.1 vs 130.5-142.4 (-39%); default off, evaluated | run `rad06-gemvl-10180418` (2 pairs) |
| Long-run tile (`R3_MOE_HELPER_MOE_LONG_ROWS=65`): 64-row slices and an MT=4 tile for runs of 65+ rows | bit-identical (rel_l2 0) but slower on replayed 2048-row requests: layer 29 3.57 ms baseline vs 3.72 ms (8 waves, 240 VGPR, 140 B scratch) and 3.86 ms (16 waves, 176 VGPR, no scratch); layer 11 2.65 vs 3.15 ms (16 waves). Spill-free MT=4 still loses, so re-staging the weight tile across 32-row blocks is not the bottleneck. Default off, evaluated. | RAD06 2026-10-18, `tools/lab/radiance/helper-long-rows.sh` |
| Helper at 20 GiB with the final kernels | decode 89.2 (vs ~108.5 at 22 GiB), no prefill gain | RAD06 08:10 |

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
