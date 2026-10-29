# Qwen3.8-27B (dense) on radiance (dev)

Current-state page. Rule: every measurement or decision updates this page in the same change. Dated history and raw
tables: [engines/radiance/README.md](../../../../../engines/radiance/README.md) (the "lab log"; section names are
quoted below). Overview: [../README.md](../README.md). Gap analysis against RDNA4: [../PARITY.md](../PARITY.md).

## Container and quantisation

Source: lab log, "Models and quantisation behind the measurements" and "An 8-bit Qwen3.8-27B".

| Container | Size | Weights |
|---|---|---|
| `qwen3.8-27b-mxfp4.rad` (the served one) | 19 GB | linear weights MXFP4 (fp4 e2m1, group 32, e8m0 scale); block fp8 128x128 (1.2 GiB); bf16 (3.6 GiB); source quantisation fp8 E4M3. 14.15 GiB of weights on one card, 7.51 GiB a rank on two. |
| `qwen3.8-27b-fp8-df2.rad` (8-bit, DFlash2 drafter merged) | 29 GB | block fp8 (E4M3, bf16 scale per 128x128 block); does not fit one 24 GB XTX |

Activations are E4M3 with an f32 scale per 128 of K. KV cache `fp8` where stated; the first serve runs (2026-10-10)
used bf16 KV. Q8_0 GGUF and bf16 GGUF are refused by radiance as sources; int8 codes are refused by `qwen35_fp8`.
Everything below is the MXFP4 container unless it says 8-bit.

## Reference quality against the full model

Not recorded. No bf16 container of the 27B fits the cards, so the references are quantised models on the R9700 with
libr4d (lab log, "KL divergence: 4-bit against 8-bit"). Those relative numbers are in "Quality measurements" below.

## Current best layout and numbers

Two RX 7900 XTX, tensor parallel, libr3 (gfx1100), `--tp 2 --tp-wire wht6` (profile `qwen38-27b-2xtx`, production
context 8192), native prefill attention on, native delta-net scan off. The 27B is dense, so RAD02 states to run it on
the R9700 alone rather than in the helper layout (RAD02 notes, 2026-10-14).

| Measure | Result | Source (run, date) |
|---|---|---|
| Decode, no drafter | 46.7-47.2 tok/s | `r3-tp2-auto`/`r3-tp2-off` 47.0/47.2 (2026-10-10); `r3-wht6` 46.7 (2026-10-10); wht6 vs exact 46.7 vs 46.8 |
| Decode, with radiance's drafter | 131.1 tok/s (wht6); 141.7 (exact wire, mixed prompts `p4-*`); one run each, difference not established | `r3-wht6spec`, `p4-*` (2026-10-10) |
| Prefill, 7.2K prompt, wht6 + native attention | 2,251-2,256 tok/s (alternating starts) | runs `kqw4` series, lab log "Native prefill attention and delta-net scan" (2026-10-12); PARITY.md quotes ~2,250 |
| Prefill, 7.2K, compatibility attention | 2,135-2,188 | `rr10d`-`rr10f` (2026-10-12) |
| Prefill, 7.2K, wht6 + 4-bit wire (`R3_AR_WIRE4=1`, opt-in) | 2,376-2,387 (+5.6%, complete separation, three starts an arm) | run `kqw4`, KLD cost below |
| Prefill, exact wire | 1,546-1,561 (2.6K/7.2K) | `knob1-*` (2026-10-12) |

Reference points: R9700 with libr4d: prefill 3,104-3,214, decode 38.3 (114.8 with drafter) (lab log, mixed-prompt
bench, 2026-10-10); one XTX: prefill 1,472-1,497, decode 27.9-28.4. Gap to the R9700: the pair reaches 72-76% of its
prefill (PARITY.md). 8-bit container on two XTX (`fp8e-*`): decode 48.0, with drafter 124.6, prefill 1,185 (lab log);
R9700 on the 8-bit container: 22.8 / 79.7 / 2,805-2,711.

Where prefill time goes (two XTX, 7.2K, per rank, `pp3`, 2026-10-12, profiled; shares are what to read): MXFP4 GEMMs
(`gemm_nt_q`) 70.1%, fused all-reduce+norm (`ar_rmsnorm_quant_fp8`) 20.0%, attention 3.3%, delta-net scan 1.6%. RAD04
measured the all-reduce at 27.8% of prefill on 2026-10-14. The tiled GEMM runs about 100 TFLOP/s per card against a
measured 142 TFLOP/s roof (run `rr09c`). The all-reduce is link time for its bytes: 7.17 MB per large call, p50
2.0-2.15 ms on wht6, about 3.4-3.6 MB/ms (run `art3`).

### radiance 1.13.0 (RAD32, PR #451, 2026-10-29)

Same harness as the v1.9.0 baseline (`tools/lab/radiance/rad32-perf.sh 27b 3`: engine-bench, 7.2K prompt, 128 decode
tokens, no drafter, `--tp 2 --tp-wire wht6`, fp8 KV) unless stated.

| Measure | v1.13.0 | v1.9.0, same harness | Source |
|---|---|---|---|
| Prefill, 7.2K | 2,580 / 2,598 / 2,792 tok/s | 2,155-2,287 | `rad32-perf-10291310` (14616743f); baseline build 67a435be.../1cb6feb8... |
| Decode, no drafter | 46.2 / 46.2 / 46.1 tok/s | 42.6-42.7 | `rad32-perf-10291310` |
| Decode, dflash2 drafter (depth 7), 2.3K and 6.5K context, 256 tokens, greedy | 58-91 tok/s, typically ~80; 30-36% of drafts accepted | not recorded in this harness | `rad32-drafter-10290405` (038fdb2af: 66-87), GEMV runs under `/work/development/tmp/bigcherry/rad32-gemv/` (f6ce657d4) |

The drafter is short of radiance 1.3.0's 102.8 / 109.1 tok/s after a 2.3K / 6.5K prefill (`p4-tp2spec`, exact wire,
same dflash2 depth 7). The wire is not the cause (exact and wht6 measure the same, 2026-10-29); acceptance is not either
(~3.4 tokens a step). The verify step is: about 48 ms against 22 ms for a plain step (profiles
`rad32-drafter-prof-10290408` / `rad32-nodraft-prof-10290409`), almost all in the target's MXFP4 GEMMs at eight rows.
#451 stages the GEMV activation in K slabs (1-row K=25600 207 -> 417-553 GB/s), pads staged rows and takes two column
tiles a block at 2-16 rows (8 rows: plain 188-204 -> 198-242 GB/s, gated 185-220 -> 242-288), bit-identical; an
8-row GEMV is still about half the 1-row bandwidth. Follow-up: RAD36.

## What is on and why

No engine patch is part of the 27B configuration in any record read for this page (rad0001 concerns the Flash-Next prefill
stager; `tools/lab/radiance/ab-27b-engine-fix-0001.json` exists but its result is not recorded in the sources read).

| Setting | Default | Status | Why |
|---|---|---|---|
| `--tp-wire wht6` (engine flag) | | | +38% prefill (1,535 to 2,124 tok/s), decode unchanged; costs +0.0008 KLD, 0.4 points top-1 (`kld4-*`) |
| `R3_ATTN_NATIVE` | 1 | production | native gfx11 prefill attention; KLD 0.0035 / 96.8% vs the compatibility form's 0.0037, +5% prefill (`kqa2`; `rr10d`-`rr10f`) |
| native prefill GEMM `gemm_mxfp4a8_tiled` | built in | | 47x prefill over the emulated form (30 to ~1,500 tok/s, commit f68edac1) |
| 1,024-wave split target in libr3 decode GEMMs | built in | | decode GEMM bandwidth to ~650 GB/s a card (`bw-*` runs) |

Settings declared in `component.toml` that touch the 27B and are off by default: `R3_AR_WIRE4` (experimental, opt-in),
`R3_GDN_NATIVE` (evaluated), `R3_AR_PIPE` (evaluated), `R3_BT`/`R3_BT_*` (evaluated or tuning), `R3_DEC11` (experimental),
`R3_AR_TIMING` (diagnostic).

## Tried and shelved

| Item | Result | Evidence |
|---|---|---|
| `R3_GDN_NATIVE=1` native delta-net chunk scan | output wrong in serving after three fixes (KLD 5.52, top-1 11.5%, PPL 1,862); selftests passed but do not cover served use; scan is ~4% of prefill so a correct form is worth 2-3%. Default off, evaluated. | runs `kq0`, `kq1`, `kqg2`, `kqg4`, `kqg5` (2026-10-12) |
| `R3_AR_PIPE=1` sliced all-reduce in one kernel | byte-identical but slower: large wht6 reduce 2.0-2.15 to 2.62-2.74 ms, prefill 2,323-2,358 to 2,170-2,224 (-6%), complete separation. Left off. | runs `arp1`, `art4` |
| RR08 GEMM forms `R3_BT=1`, `R3_BT=2` (lane-owned bf16-tile, eight-wave LDS tile) | pass the selftest, slower than the tiled form at M=512; second form 1,553 vs 854-895 us at 16384x5120; one XTX 984/965 vs 1,399/1,475 tok/s. Not adopted. | runs `rr08a-*`, `rr08b-*` (2026-10-12) |
| `R3_AR_WIRE4=1` 4-bit cross-card wire | +5.6% prefill (2,376-2,387 vs 2,251-2,256) but KLD 0.0126 / 94.5% top-1 vs 0.0044 / 96.6% for wht6 alone; left opt-in, the owner's call. (An earlier build did not reach the 4-bit path at all: identical to wht6.) | runs `kqw2`, `kqw4` (2026-10-12) |
| Chunk size, `--tp-wire-min-kb 16`, fp8 KV as prefill levers | nothing beyond single-run spread (~2%) | runs `knob1-*` (2026-10-12) |
| dec11 decode GEMV form (`R3_DEC11`, eight-tiles-a-block variant) | decode fell to 15.0 (one card) / 24.0 (two) tok/s; not the default | run `p3` |
| Int8 trunk for the 27B | refused by `qwen35_fp8`; not available | lab log, "An 8-bit Qwen3.8-27B" |
| Native fp8 GEMM with codes widened to bf16 | decode 18.5 vs 44.0 with f16 widening | runs `fp8b-*`, `fp8c-*` |

## Open levers

- **RAD04** (pending): overlap the cross-card sum with the next GEMM via two half-steps on the engine's second queue
  (runtime has `Ctx::set_lane`/`lane_join`; no arch uses it yet). Target: hide most of the ~28% all-reduce. Branch
  `feat/rad04-two-half` holds only a seed commit; an earlier GPT request was abandoned with no output.
- **GEMM:** 100 to ~125 TFLOP/s per card (PARITY.md ranked item 3: with the sum halved this reaches ~3,000 tok/s).
  RAD06's hipBLASLt/Tensile ideas (per-shape tile tables, split-K/stream-K, mining the gfx1100 Tensile logic YAMLs) are
  stated for the MoE helper GEMMs; the same shape-table idea applies to the dense GEMMs but has not been scoped for the
  27B.
- **Cross-card bytes:** 4-bit wire is measured; whether to adopt it is the owner's decision (quality cost above).
  RWQ01-RWQ06 (`patching-radiance-wire-quality`) cover wire precision by operation, phase and layer.
- **Arithmetic from the lab log (2026-10-12, `pp1`):** all-reduce to zero ~2,660; attention and scan at the R9700's
  per-call cost +6%; GEMM 100 to 115 TFLOP/s +10%; together ~3,000-3,300 (an estimate, not a measurement).
- **MEN11:** the 27B has no maintained ab-benchmark config yet; its numbers above are lab runs.
- **Layout alternative:** run the 27B on the R9700 alone (3,104-3,214 prefill, decode 38.3) when prefill dominates.

## Quality measurements

radiance KL mode, 42,761 scored positions over 24 pieces, 512-token steps (lab log, `rad-kld.sh`).

| Reference | Subject | KLD mean | p99 | Top-1 | PPL vs ref | Run |
|---|---|---|---|---|---|---|
| fp8, R9700 | MXFP4, R9700 | 0.0509 | 0.354 | 88.7% | x1.0246 | `kld2-mxfp4-r4d` |
| fp8, R9700 | MXFP4, two XTX | 0.0508 | 0.361 | 88.7% | x1.0238 | `kld2-mxfp4-xtx` |
| fp8, R9700 | fp8, two XTX | 0.0044 | 0.029 | 96.4% | x1.0009 | `kld2-fp8-xtx` |
| MXFP4, R9700 | MXFP4, one XTX | 0.0037 | 0.027 | 96.8% | x1.0006 | `kld3-xtx1` |
| MXFP4, R9700 | MXFP4, two XTX | 0.0037 | 0.028 | 96.8% | x0.9993 | `kld3-xtx2` |
| MXFP4, R9700 | two XTX, exact wire | 0.003665 | median 0.001643, p99 0.0275 | 96.79% | x0.9993 | `kld4-xtx2-exact` |
| MXFP4, R9700 | two XTX, wht6 | 0.004422 | median 0.001980, p99 0.0314 | 96.38% | x1.0003 | `kld4-xtx2-wht6` |
| MXFP4, R9700 | two XTX, wht6 + native attention | 0.0044 | | 96.6% | | `kqa2` |
| MXFP4, R9700 | two XTX, wht6 + `R3_AR_WIRE4` | 0.0126 | | 94.5% | PPL 8.605 vs 8.553 | `kqw4` |

The 4-bit container is 0.051 from the 8-bit one (a different top token at 11% of positions, 2.4% more perplexity); the
libr3 vs libr4d difference (0.0037) is rounding carried through 64 layers of 8-bit activation quantisation, not a
less accurate kernel. The Flash-Next helper noise floor (0.058-0.067) was measured on that model and corpus only; no
equivalent re-chunking floor is recorded for the 27B.

## Gaps

- Quality against the original (full-precision) 27B: not recorded.
- Run IDs for the wht6 7.2K figures: quoted as the series name; the series (`kqw*`) is from the lab log.
- A current end-to-end 27B number through the maintained harness: none.
