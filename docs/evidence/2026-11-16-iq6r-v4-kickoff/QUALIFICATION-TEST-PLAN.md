# Qwen3.8-27B IQ6R-V4 — artifact and hardware qualification matrix

Date: 2026-11-16. Owner: XDP11. **Planned tests; no claimed passes until evidence exists.**
This plan qualifies a **new experimental** BF16-derived calibrated RAD2. It does not change the managed runtime, W8A8 production default, model-root installation, or any active TP3 service.

## Immutable inputs and comparison controls

- BF16 source: the separately fingerprinted 18-shard original 27B checkpoint, 64 transformer layers.
- Engine: CI-managed `bc-radiance-1170.26.0`, build ID and runtime SHA checked against the installed release manifest; both serving-plugin sets `libr3,libref` (gfx1100) and `libr4d,libref` (gfx1201) must pass layout planning.
- Quantizer: the managed `rad-convert`, executed on **physical GPU 2 gfx1201**, with `HIP_VISIBLE_DEVICES=2` and the process-local `--gpus 0` ordinal. `libquant.so` must expose a gfx1201 offload bundle. The two XTX gfx1100 devices must not be selected for this particular compiled quantizer.
- Recipe: pinned upstream `q38-27b-iq6r-v4.recipe`, 293 quantized weights in total: 292 calibrated trellis/GPTQ weights (u6:145, u7:145, u4:1, u5:1) plus one RTN u6 token-embedding weight; any unapproved substitution fails.
- Calibration: 257 full 64-layer upstream Hessian sites (10,259,331 rows each) at pinned dataset revision `2dcf0061cb683152742b904e2965ce7cd92b2da6`, expanded into 289 RADGRAM2 converter stems, LFS SHA checks and independent 12,593-cell sampled matrix comparison. Missing calibration and silent rounding are hard failures.
- Baselines: **same-source** BF16, managed W8A8+DFlash2 and MXFP4 RAD with their full identities; DFlash2 and other draft mechanisms off during first quality/performance comparison. Do not treat GGUF Q6 as Radiance IQR.
- Hold-out: public corpus from the same pinned dataset, 596 documents / 1,694,621 tokens, disjoint from calibration by exact prompt; plus fixed short/long instructions in reasoning, code, summarization and multilingual prompts. The upstream published golden data may be used only after tokenizer/hash and score-position alignment are independently verified. If it cannot be aligned, capture BF16 reference logits with a separately qualified harness; do not compare incompatible token positions.

## Gated test sequence

| Gate | Test / required evidence | Pass criterion / blocking condition |
| --- | --- | --- |
| Q0 — provenance | SHA-256 original BF16 shards and staged vhead, recipe SHA, 257 LFS Hessian hashes, RADGRAM2 metadata and sample audit, installed managed release/runtime + converter SHA, observed gfx1201 quantizer offload code, exact command/env and GPU lease | All match; no missing source site, image or Hessian; no unapproved model rewrite |
| Q1 — conversion | End-to-end `rad-convert` exits zero; preserve stderr including every GPTQ tensor; no `no calibration for`; SHA-256 entire candidate; atomic rename only on success; inspect with **same-release** `rad-info` | Valid RAD2, 64 layers, correct architecture/tokenizer/config, no missing/duplicate tensor, all 292 intended GPTQ trellis tensors with expected IQR code/scale planes, plus the separate RTN token-embedding quantization; expected BF16/F32 exceptions preserved; no truncated output, NaN errors or unexpected fallback |
| Q2 — numerical kernel correctness | `rad-kbench` / maintained IQR kernel tests for gfx1100/libr3 **and** gfx1201/libr4d, exact model-representative K/N and 1, 4, 16, 64, 128, 256, 512, 1024, 2048-row bands; compare relevant reference outputs and inspect decoded planes and shape rejection | All supported rows pass numeric tolerances defined by pinned kernel test suite, no silent `libref` fallback, no invalid hardware path. Unsupported shapes must fail closed. Any real failure blocks serving |
| Q3 — isolated bring-up | Load IQ6R under managed Radiance with explicit per-card GFX-compatible plugin and selected device; first single R9700, then each XTX with `libr3`; request single stream, 128-token output, 512/2048/8192 prompt depths; record bring-up log and selected kernel names | Model loads, all required IQR kernels resolve, finite/consistent logits, no segfault, no OOM, no hidden CPU/portable fallback or incompatible activation ABI; never touch running W8A8 service |
| Q4 — held-out fidelity | Same tokenized held-out prompts/score positions: IQ6R vs BF16, repeat-BF16 floor, W8A8 and MXFP4 on matched attention/KV/wire/tuning, draft off; report mean/median/p99/p999 KL, top-1 accuracy/agreement, perplexity delta, category-stratified and first-token/depth probes; include 512, 2048, 8192, 24576 context if supported | Initial **provisional** IQ6R targets: mean KL <= 0.001, top-1 >=98.8%, plus non-catastrophic p99/PPL and category results. These are proposal thresholds, not existing measured baselines. Establish repeat-BF16 floor and freeze final thresholds *before* interpreting the candidate; zero skipped/unaligned tokens. Any significant category regression blocks admission |
| Q5 — TP3 contract | Cold-start exact 3-card rank placement XTX+XTX+R9700, mixed gfx1100/gfx1201 plugin loading, exact host collective, same model encoding and activation semantics on every shard; no-P2P path, GDN/attention/cache, restart, short/long context | Every rank agrees on manifest/tensor encoding; successful admission and teardown with zero unclassified cross-rank data mismatch or silent fallback. TP3 is a new gate, not implied by TP1 |
| Q6 — memory/capacity | Track measured per-card VRAM (weights, KV, GDN, activations, workspaces), host pinned memory and disk-backed load; 1/4/6/8 simultaneous streams, context 512/2048/8192/24576 and longer only if validated; repeated cold and warm tests | No overcommit, paging thrash, hidden service eviction, unexpected KV budget reduction, or reproducible instability. Claim a KV-capacity gain only from actual measured free space/capacity, not file-size arithmetic |
| Q7 — performance | Paired W8A8 vs IQ6R (then MXFP4) same managed release, pool/tier, exact wire, draft off, scheduler and temperature; ABBA repetitions (at least 3 complete blocks) at 1,4,6,8 streams; prefill TTFT at 512/2048/8192/24576, prefill tok/s, decode tok/s and p50/p95/p99 latency; report GPU/host timings separately | No more than **10%** performance regression vs matched W8A8 on main workloads (provisional XDP11 threshold), stable errors, no different settings or mixed baseline configurations; gate should be tightened against run-to-run variation |
| Q8 — optional DFlash2 | Only after Q0-Q7 pass: separately test draft-on compatibility and quality with same drafter and speculation settings. DFlash2-off remains primary control | Draft acceptance/quality do not regress unexpectedly, all startup/TP3 contracts still satisfied; do not blur results with Q7 |
| Q9 — decision | Evidence manifest: hash + commands + calibration/source provenance + all results + failure/waiver record; update XDP03/XDP10/XDP11 | Owner review before any promotion. Production W8A8 stays unchanged; no automatic deployment or default-model replacement |

## Execution and attribution

1. **Artifact gate is first:** `iq6r_v4_convert_experimental.py` checks hashes, offload ISA, both architecture layout plans and the managed CI release. Only a successful converter exit, complete output validation and SHA-256 produce the experimental RAD artifact. Its explicit scratch receipt and `managed-*-stderr.log` are required.
2. **Kernel testing is independent of quantizer device:** R9700 quantizes because `libquant` is gfx1201-only; XTX serving still requires gfx1100-compatible IQR kernels in `libr3`. Do not assume that successful quantization implies the presence or correctness of either serving kernel.
3. **Do not run new GPU tests concurrently with full conversion.** All device-aware steps require a separate exclusive lease and no pre-existing GPU PIDs. The holder's lease must be released before moving from Q1 to Q2/Q3.
4. **Data selection:** use the published public corpus only as held-out text/positions; do not pass its golden logits off as BF16 without verifying exact source checkpoint identity, vocabulary split, BOS/chat formatting, tokenizer hashing and positions. Record a frozen scoring manifest with sorted document IDs, token hashes, prompt depths, seeds, EOS handling and per-arm result checksums.
5. **Fail-closed:** invalid RAD, missing format planes, unsupported IQR, hidden fallback, logit mismatch, OOM or shared-rank divergence = stop. Leave a failed candidate in scratch diagnostic form only, do not load or promote it.
6. **Evidence root:** keep large binaries, scoring arrays and logs under `${BC_WORK_ROOT}/iq6r-v4-qualification/`; commit concise summaries, hashes, command line, score tables, release identity and counterexamples into this evidence directory. Do not commit copyrighted corpus contents or model weights.

## Planned operational records

- `conversion-status-managed.json`: converter status, source/release/model identity, artifact SHA and bytes (Q0/Q1).
- `managed-plan-gfx1100-stderr.log` and `managed-plan-gfx1201-stderr.log`: layout preflight results (Q1).
- `managed-conversion-stderr.log`: full converter error and per-projection relative norm reports (Q1).
- `radinfo-iq6r.txt` and `tensor-encoding-audit.json`: generated **only after** full artifact completion (Q1).
- `kernel-{gfx1100,gfx1201}-*.json`: numerical tests and dispatch trace (Q2).
- `quality-reference-manifest.json`, `quality-iq6r-vs-bf16.json`, `quality-w8a8-vs-bf16.json`: aligned hold-out scores and repeatability (Q4).
- `tp3-admission.json`, `tp3-vram.json`, `tp3-abba.json`: runtime, memory and matched performance (Q5-Q7).

No unexecuted step in this plan is evidence of a passing result.

## 2026-11-16 bounded staged-conversion contingency

The valid R9700 full-recipe run completed 32 of 292 GPTQ tensors over 2,747.87 seconds before a controlled stop, without calibration fallback or HIP errors. Its projected duration exceeded the hard 14,400-second wrapper timeout. The interrupted partial RAD was discarded; logs and its conversion-status-managed-timeout-preempted.json receipt are retained. This was an operator stop, NOT SIGSEGV or a quality failure.

The replacement workflow writes complete RAD2 checkpoints using 15 cumulative recipe prefixes of 20 GPTQ rules. It validates each stage with managed rad-info, exact encoding counts, SHA-256 and a sealed receipt. The next stage uses rad-convert --reuse of the prior checkpoint, requiring exact quantizer, options, shape and encoding equality. The final recipe is SHA-identical to pinned IQ6R-V4. Old checkpoints may be retired after a newer sealed stage to bound scratch space. Intermediate RADs remain UNQUALIFIED and must never be served. Stage 1 and subsequent reuse must be tested before the approach is considered proven.

The held-out quality preflight verified the public corpus SHA-256, 596 documents, 1,694,621 tokens, 15 categories, 1,694,025 scored rows and all golden binary file sizes. The result is quality-public-layout-audit.json, explicitly marked PUBLIC_GOLDEN_LAYOUT_PASS_NOT_BF16_QUALIFIED. It does not prove the reference matches the original BF16 checkpoint or tokenizer, and is not model-quality evidence.

## Corrected executable stage contract

The initial staged headless preflight failed because the qwen35 vocabulary adapter requires all four output.k4..k7 trellis planes. Adding those four rules exposed a second prerequisite: qwen35_w8a8 also requires its trunk inputs to have rotated INT8 or trellis encodings, not BF16. Both invalid partial recipes failed during plan-only, before allocating a model output. Their receipts and stderr are preserved under separate missing-head and missing-int8-trunk names.

The corrected 16-stage design always includes the four GPTQ vocabulary-head rules and the pinned W8A8 128-wide Hadamard rotated-INT8 wildcard rules for trunk weights not yet IQ6R. Exact explicit IQ6R rules precede the wildcards (first-match precedence). Stage 1 establishes the head plus INT8-compatible trunk; stages 2-16 add at most 20 trunk GPTQ rules each, with a fully pinned final recipe identical to the original. Both recipe SHA-256 hashes are enforced. Stage 1 passed its managed gfx1201 plan-only check (25.57 GiB estimated) and began the first actual checkpoint conversion under a new exclusive lease. Success and reuse remain unverified until completed RAD2 checkpoints and stage-2 evidence exist.

## Detached stage controller, 2026-11-16

The long-running controller is tools/lab/radiance/iq6r_v4_controller.py. Launch it ONCE without taking a separate GPU lease while the existing stage-1 worker is running. It uses an exclusive filesystem lock and a JSON status receipt in the isolated experiment work root. It waits for the externally owned stage-1 job to release its lease, verifies its sealed RAD2 receipt and checksum, and then requests a NEW exclusive GPU lease for each stage 2-16 using agent-lease acquire --wait. The stage subprocess uses a bounded timeout; the controller never silently retries a failed or incomplete stage and never starts quality/TP3 GPU jobs. It verifies contiguous recipe-hash receipts, latest model SHA-256 and stage-2-and-later --reuse evidence. Duplicate controllers refuse to start.

After all 16 stages seal, the controller validates the exact final recipe, RAD2 trellis encoding counts and final checksum, then atomically names an experimental IQ6R RAD. The controller explicitly marks quality, TP3 and production promotion unapproved. No kernel, quality or TP3 GPU testing is launched automatically.

Monitor conversion-controller.json and the stage-* logs in the experiment work root. A blocked_no_retry state requires log review; restarts validate the latest sealed checkpoint and refuse orphaned partial files. Do not remove a failed checkpoint before preserving its logs and status.
