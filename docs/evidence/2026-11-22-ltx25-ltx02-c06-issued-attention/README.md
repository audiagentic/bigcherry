# LTX02 C06 — issued native checkpoint-weight self-attention

**2026-11-22. Execution PASS on native Radiance 2.2 host and mixed AQL (three GPUs). TP1 parity NOT YET QUALIFIED.**

## Provenance and scope

- Frozen upstream Radiance 2.2 / ABI44 LTX source: `arch/ltx2/ltx2.cpp`, SHA-256 `543490eb70829693f945233d1b66e2d6d87b5e6a2817a72233fa0649218e6b53`.
- Existing source-managed LTX TP3 patch stage; no runtime/arch patch, kernel, transport, scheduler or production config changed by this work.
- Model: read-only real LTX-2.5 IQ8R RAD2 checkpoint; **block 0**, self-attention video 4,096 channels and audio 2,048 channels, 32 heads. One input row per modality, with upstream LTX `positions()` coordinates. The BF16 input is a deterministic seeded activation, **not** a captured TP1 activation.
- Exact same patched `ltx2::Attn::declare`, `Act::step`, `Lin::step`, `attn_core`, native `Builder`, `rad::Ctx`, Registry and collective rows as the prior C05/C06 tests. No alternative math or all-reduce path.
- Existing `rad::weight_share` selects each per-rank model plane. The **kernel-selected native `relayout` callback** produces its actual stored layout; a test-only bounded gather stages those source bytes. This is not the production engine's asynchronous streaming weight loader and should not be promoted as a replacement.

## Execution matrix

The new optional `--c06-issued-host` mode is an always-on native CTest. `--c06-issued-aql` is **opt-in and GPU-lease-only** on the already existing C06 AQL executable; no second executable, linker recipe, or launcher is introduced.

| Modality | Legal TP3 head split | Host `RadCtx` | AQL gfx1100/gfx1100/gfx1201 | AQL gfx1201/gfx1100/gfx1100 | All three rank outputs |
| --- | --- | --- | --- | --- | --- |
| Video | 8/8/16 | PASS | PASS | PASS | exactly equal, 4,096 BF16 |
| Video | 12/12/8 | PASS | PASS | PASS | exactly equal, 4,096 BF16 |
| Audio | 8/8/16 | PASS | PASS | PASS | exactly equal, 2,048 BF16 |
| Audio | 12/12/8 | PASS | PASS | PASS | exactly equal, 2,048 BF16 |

Every rank actually issues input quantization, fused IQ8R QKV and gate projections, **two** existing full-head exact Q/K reductions with RMSNorm, native RoPE, dense attention and head gate, output quantization, IQ8R output projection, **one** existing exact output reduction and one post-reduce BF16 bias. `Ctx::ar_calls()==3` is required, as are finite/nonzero output and exact BF16 equality of the three replicas.

- Native host: four three-rank graphs, **36 initialized and retired instances**, all four issued and verified.
- Two leased AQL device orders: four three-rank graphs each, **108 initialized and retired native instances per run**, both process exits 0. Process wall time **0.706 s** for 0/1/2 and **0.621 s** for 2/0/1; these include setup and staging and **are not operator latency measurements**.
- The AQL kernel set is selected per physical architecture (libr3 on gfx1100 and libr4d on gfx1201), and GPU model weight planes use kernel-selected IQ8R device layouts: four QKV/output code/scale non-identity relayouts per rank, not the host reference `rad.planes` layout. GPU weights and input/coordinates live in native device allocations. Context streams drain before instance/weight teardown.
- Final checked AQL executable SHA-256: `426d9107741c4c380c19ccee64ba8245c3de0d149aef2e7f38be3cd0f8057979`.

## Output numerical comparisons (not acceptance tolerances)

Opt-in `BC_LTX_C06_CAPTURE_DIR` and `BC_LTX_C06_CAPTURE_TAG` emit the validated rank-0 BF16 output in raw little-endian bytes. Captures were compared offline against the same host-reference issued graph (same real checkpoint, input, rotary coordinates and head partition). Neither capture path is used by the production execution path. Raw binary outputs and the comparison JSON remain in the work root, not in Git.

| Modality / split | AQL vs host relative L2 | RMS absolute error | Max absolute error | Cosine similarity |
| --- | ---: | ---: | ---: | ---: |
| Video 8/8/16 | 0.421% | 0.02581 | 0.125 | 0.999991 |
| Video 12/12/8 | 0.422% | 0.02587 | 0.125 | 0.999991 |
| Audio 8/8/16 | 0.330% | 0.00889 | 0.0625 | 0.999995 |
| Audio 12/12/8 | 0.389% | 0.01048 | 0.125 | 0.999992 |

**AQL 0/1/2 and 2/0/1 are bit-for-bit identical over all 12,288 BF16 output positions** (two splits times video 4,096 plus audio 2,048). This is a device ordinal/rank-placement invariant, not TP1 parity.

The host BF16 result **differs between legal head partitions**: video 1,605/4,096 elements changed (max absolute 0.125, RMS 0.017307); audio 736/2,048 changed (max absolute 0.125, RMS 0.0071615). Different BF16 partial accumulations can round differently, but these measurements **do not establish whether all differences are acceptable**. There is deliberately no guessed numerical threshold or claim of TP1 quality.

## Verification

- Frozen model-backed native host CTests, including new issued attention and fail-closed AQL-on-host gate: **10/10 passed**.
- Existing focused Python source/model oracles (numeric, plane extents, shard cuts, patch and reference): **12/12 passed**, no skips.
- AQL C06 binary compiled/linked against the same pinned HIP-enabled Radiance 2.2/ABI44 source/core and existing gfx1100/gfx1201 plugin libraries.
- Exclusive GPU lease acquired and released per bounded AQL replay; two physical device orders pass and contexts retire cleanly.
- No change to the LTX production `world_size > 1` refusal or any non-LTX RUM owner.

## Still required before LTX02 acceptance

1. Capture **real TP1 block-0 input activations and intermediate tensors** (rather than seeded activations), issue the corresponding native TP1 path with matching full checkpoint, and measure TP1/TP3 BF16 output and intermediate parity. Review any tolerance from observed reference and precision budget; do not choose arbitrary acceptance numbers.
2. Extend the one-row proof to 16/64 rows and document the actual model-geometry constraints and memory admission.
3. Validate missing-rank, kernel-refusal, bounded cancel/timeout and recovery against the **RUM06-owned** group lifecycle; no LTX-local cancellation framework.
4. Only after correctness, run leased phase timelines and benchmark LTX02 as required by the plan. LTX03 (full DiT) and LTX04 (serving) remain separate.

The current output is **one block's actual attention**, not a complete DiT or a generated video.
