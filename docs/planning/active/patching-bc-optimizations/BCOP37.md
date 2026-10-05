---
id: BCOP37
order: 37
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify Strata Flash-Next on local AMD host

## Description

Qualify the Strata execution architecture for Qwen Flash-Next on the local AMD machine (96 GiB aggregate VRAM, 96 GiB RAM), compare it with the current BigCherry/llama.cpp deployment, and preserve enough mechanism-level evidence to decide what BigCherry should adopt.

This item is deliberately prescriptive. Agents MUST NOT spend time trying to make an unsuitable Strata source tree support HIP/Qwen/GGUF. First identify a runnable AMD implementation, prove its capabilities from source, then benchmark it. A failed prerequisite is a routing problem, not an invitation to create a new backend.

BCOP37 is an evidence/disposition item. It does not own a new scheduler, expert router/cache, loader, HIP backend, MTP implementation, or benchmark framework.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

BCOP37 completes when:

- the AMD source lineage is identified and pinned without attempting an unauthorized port;
- a real HIP GPU smoke test passes, or a precise current-source blocker is captured;
- matched Strata/BigCherry evidence is produced where technically possible;
- expert-cache, cold-expert, KV, speculation and topology mechanisms are characterized to the extent exposed by the runtime;
- raw evidence is preserved;
- each useful mechanism is assigned to an existing BigCherry owner;
- no duplicate runtime subsystem was created.

## Notes

2026-10-06: `ro99/strata@015b075079c51a7aec670ee24924f920f5e7bb2b` was tested as if it were the AMD target. Source inspection showed it was unsuitable for this qualification: CUDA-oriented backend/stub behavior locally, no required Qwen Flash-Next/qwen4exp path and no matching GGUF path. No GPU time was spent. This revision is now explicitly architecture-reference-only for BCOP37; agents must not repeat this dead end. The next executable path is current `Maxritz/Strata-rocm` or a demonstrably newer/superseding AMD lineage, pinned after capability verification.

2026-10-06 STEP 0 PASSED FROM SOURCE (no local code changes, no build, no GPU time). Chosen source: https://github.com/Maxritz/Strata-rocm, branch main, SHA 2ed00de617659e3e238bdc0ae7523e1e99ef4e38 (2026-10-01), clone at Brutus /mnt/data/bigcherry-work/external/Strata-rocm. Bundle: /mnt/data/bigcherry-work/runs/bcop37-20261006-0908/source-capability.txt. Prerequisites: (1) real HIP backend - CMake option STRATA_ENABLE_HIP, .cu sources compiled with -x hip --offload-arch=<CMAKE_HIP_ARCHITECTURES>, hipSetDevice/hip allocation in src/core/device.cu, pinned.cu; (2) qwen4exp registered in src, docs/MODEL_SUPPORT.md lists Qwen3.8-Flash-Next 10x512 GGUFs as running; (3) GGUF reader (src/artifact/gguf_reader.cpp), IQ4_XS present in dequant and in the MoE MMQ type switch; packs are built by tools/iq_pack.py, which states it keeps every quantized tensor in its GGUF form (to be proved on our file before a lane is labelled MATCHED); (4) arch list is a CMake variable and the tree says it rejects no arch; externally verified only on gfx1201 (RX 9070 XT) and gfx1031 - gfx1100 is a local unknown; (5) expert cache / hybrid code present (expert_cache.cpp, expert_source.cpp, pinned.cu, --expert-ram-gb, --gpu-miss, --kv-resident, mtp.cpp). Limits seen in source, to confirm at run time: one GPU per process (no multi-GPU split flag found), so BCOP37's multi-GPU step can only be 'not supported by the runtime'; compiled shape 2560 embd / n_ff 640 / top-10, so strata --model-info on the BigCherry UD-IQ4_XS file is the model-compatibility gate; engine takes token IDs (tools/strata_tokenizer.py) rather than text. Next: Steps 1-3 (system.txt, model-compatibility.txt, clean gfx1201 build) once the prefill measurements already running on Brutus finish - no build while a measurement runs.

2026-10-06 STEPS 1-3 (bundle /mnt/data/bigcherry-work/runs/bcop37-20261006-0908 on Brutus; scripts tools/lab/strata/). Step 1 system.txt saved: i7-12700KF, 96 GB RAM (2.7 GB swap), 2x RX 7900 XTX gfx1100 24 GB, R9700 gfx1201 32 GB, RX 6900 XT gfx1030 16 GB, ROCm 7.2. Step 3 build: Maxritz/Strata-rocm@2ed00de builds for gfx1201 on Linux with NO source change after two build-environment corrections - (a) class B: the documented cmake line must be given ROCm clang as the C++ compiler (the tree adds -x hip --offload-arch itself; system g++ rejects it); (b) the tree links ROCm by Windows import-library path (${ROCM_INSTALL_DIR}/lib/amdhip64.lib, hipblas.lib, ~50 sites) - handled outside the tree with a shim ROCm root of symlinks (bcop37-rocm-shim.sh). Binary links libamdhip64/hipblas and carries gfx1201 code objects. Default build prints 'MMQ prefill path disabled (llama.cpp .cu files need hipification)'; the documented option STRATA_HIP_HIPPY_MM=ON was not tried yet. Step 2 model gate: strata --model-info on the BigCherry UD-IQ4_XS file says SUPPORTED (qwen4exp, 2560 embd, 10x512). BLOCKER (class C, stopped for review as the item requires): tools/iq_pack.py refuses the file - 'force-bf16 tensor output_hc_down.weight is Q8_0; no dequant'. The tool rewrites the hyper-connection tensors (hc_attn_up/down, hc_ffn_up/down x 48 layers, output_hc_*) as BF16 and has dequant for Q4_K/Q5_0/Q6_K only; the unsloth UD file stores them as Q8_0. A fix is a ~8-line Q8_0 dequant entry in that Python tool (Strata source change, not applied). Unknown beyond it: whether the engine runs the UD expert mix (experts are IQ3_S x94, IQ4_NL x43, Q8_0 x5, IQ4_XS x2 tensors). Details in model-compatibility.txt. No GPU time spent on Strata yet. Options: (1) approve the Q8_0 dequant in the pack tool and continue on our exact file (hc tensors then served as BF16 dequantized from Q8_0, stated beside every number); (2) download one of the three files the fork lists as running (reap-288 Q4_K_M, ngram-embeddings Q4_0, Swift IQ2_XS) and run UNMATCHED mechanism lanes only; (3) both.

2026-10-06 STEP 4 RESULT - GPU EXECUTION PROVEN, CORRECTNESS FAILED ON THE MATCHED FILE (bundle bcop37-20261006-0908: gpu-smoke.txt, gpu-smoke.raw.log, pack.log, iq_pack-ud.diff, model-compatibility.txt, commands.txt). Path taken, all owner-approved or outside the Strata tree: (a) pack tool conversion (tools/lab/strata/iq_pack-ud.diff, 2 small additions to tools/iq_pack.py): Q8_0 dequant for the FORCE_BF16 hyper-connection tensors; F32 hc_*_inject tensors written as BF16 (tool reports 0 values not exactly BF16 - lossless). (b) missing Python dependency 'regex' installed in a private venv. (c) the 3-shard GGUF splits layer 14's expert tensors across shards, which the native-expert format refuses; merged the shards with llama-gguf-split --merge into /mnt/data/bigcherry-work/external/Qwen3.8-Flash-Next-UD-IQ4_XS-merged.gguf (93,682,584,000 bytes, same tensors and quants). (d) a native (IQ) pack requires --native, --spec T>=2, --prefill, and --spec requires --expert-cache plus --expert-profile; used --spec 2 --prefill auto --expert-cache auto --expert-profile data/expert-profile.bin (the fork's shipped ranking). GPU proof on the R9700 (HIP index 2, gfx1201): VRAM 57 MiB -> 32,354 MiB peak; 'expert cache 11454 slots, 25.84 GiB of VRAM', 'R4 hit path ON - resident experts are computed on the GPU', 55.43 GiB of experts loaded into the pinned host arena, verify windows captured, 301 native projection matrices; engine-reported decode 64 tokens at 23.84 tok/s, prefill 21 tokens at 25.2 tok/s, expert-cache hits 86.9%; no HIP errors, rc 0. CORRECTNESS: FAIL. Two independent defects: (1) Strata's Python tokenizer encodes our prompt as single-byte tokens (93 ids instead of llama.cpp's 22: 248045,846,198,7734,...), so its tokenizer export is wrong for this GGUF vocabulary; worked around by passing llama-tokenize ids (IDS override in bcop37-smoke.sh). (2) With the correct 22 prompt ids the engine still generates garbage (decoded with the GGUF vocabulary: 'edafalaftelicalsslaslysvaslysingassat...'), i.e. the forward pass is numerically wrong for this file. Not diagnosed further (anti-diversion rule). Likely area, unverified: the unsloth UD quant mix differs from every file the fork lists as running - non-expert projections (attn_qkv, attn_gate, attn_q/k/v/output, ssm_out, shared experts, ple_key/value) are all Q8_0 here, output.weight is Q6_K (the log says 'experimental native Q5_K head'), token_embd Q8_0, experts IQ3_S / IQ4_NL / Q8_0 / IQ4_XS. Per the correctness gate the throughput lines above are NOT valid performance results and must not be compared with BigCherry. BigCherry single-R9700 reference taken the same day on the same file (1336 run, --n-cpu-moe 41, no MTP, correct output): ~18 t/s decode, 90.4 t/s prefill at 4.7K tokens. Disposition options: (A) run the fork on one of the files it lists as running (reap-288 Q4_K_M, ngram-embeddings Q4_0, Swift IQ2_XS; none is on Brutus, 40-60 GB download) as UNMATCHED mechanism lanes, with BigCherry on that same file as the control so the pair is matched to each other; (B) stop the Strata arm here and keep only the mechanism observations (expert cache sizing 11.4K slots / 25.8 GiB on 32 GB, 87% hit rate, CPU pool 6-7 ms per round, adaptive tier swaps) as hypotheses for MET01.

2026-10-06 OWNER DECISION: download a model the fork supports and bench Strata on it. Chosen: ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ2_XS (Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf 39.79 GB + 00002-of-00002 28.36 GB) - the one file the fork documents as verified end-to-end on gfx1201 (RX 9070 XT), 10 x 512 experts, same qwen4exp geometry. Destination /mnt/data/llm-models/strata-swift-iq2xs on Brutus, with the repository's SHA256SUMS checked after download (log runs/bcop37-download.launch.log); queued behind the MET01 cache lanes so the 68 GB write does not evict a model under measurement. Lane class: UNMATCHED to the production UD-IQ4_XS quant; BigCherry will be run on the SAME Swift IQ2_XS file on the same single R9700 as the control, so the Strata-vs-BigCherry pair is matched to each other (same weights, same card) and no speedup is computed against the production quant. Order once it is on disk: pack (unmodified tool path first), GPU smoke with the correctness gate (llama-tokenize ids, output compared with BigCherry greedy on the same prompt), then the short-decode / short-prefill / agent-mixed lanes, speculation off first.

## Authoritative source selection

### Architecture reference only

`ro99/strata@015b075079c51a7aec670ee24924f920f5e7bb2b`

Use this exact revision only to understand/reference Strata architecture where useful. DO NOT attempt to port, repair or benchmark it on AMD as part of BCOP37.

Repository evidence already established on 2026-10-06:

- its functional GPU backend is CUDA-oriented;
- the local non-CUDA build reaches a CUDA stub rather than a HIP execution backend;
- this revision does not provide the required Qwen Flash-Next/qwen4exp + GGUF execution path for our matched test.

These facts are a CLOSED finding for this item. Do not rediscover them, retry CUDA/HIP CMake permutations, install CUDA, implement HIP shims, add Qwen registration, or add a GGUF loader to this tree.

### Runnable AMD candidate

Primary candidate: `Maxritz/Strata-rocm` (or its current successor if the repository itself explicitly redirects/supersedes it).

The agent MUST pin the exact tested commit SHA and prove, from the checked-out source before building, that it contains ALL of:

1. a real HIP/ROCm GPU backend (not a stub);
2. Qwen Flash-Next/qwen4exp model support;
3. the model/checkpoint format needed for the selected test artifact, preferably the same GGUF/quant used by BigCherry;
4. build targets for the local AMD architecture being tested;
5. actual expert-cache/hybrid execution code required for the Strata mechanism test.

If any prerequisite is absent, STOP. Search current Strata forks/branches/releases for the correct implementation. Do not implement the missing prerequisite under BCOP37.

Before build, save source evidence for these prerequisites to `source-capability.txt`: repository URL, SHA and exact files/symbols/CMake targets proving each capability.

## Anti-diversion rules

Agents MUST NOT, unless a separate approved plan explicitly requests it:

- port CUDA Strata to HIP;
- rewrite CMake to pretend a CUDA backend is HIP;
- implement Qwen model registration in an incompatible tree;
- implement GGUF loading in an incompatible tree;
- convert the production model to a materially different quant and call the result a matched benchmark;
- spend GPU time on a CPU/stub build;
- benchmark a build before proving that GPU kernels are actually executing;
- create a BigCherry Strata clone before measuring the reference mechanism;
- abandon the test merely because the first repository/revision is unsuitable when a known ROCm fork exists.

Maximum source-selection/build triage budget before escalation: 60 minutes of agent work. If no qualifying AMD tree is found, record exact blockers and stop the Strata arm rather than starting a port.

## Step 0 — refresh external source facts

Before cloning/building:

1. Inspect current `ro99/strata`, `Maxritz/Strata-rocm`, relevant forks and releases.
2. Determine whether `Maxritz/Strata-rocm` remains the current runnable AMD lineage or has been superseded.
3. Read recent README/ROCm porting/build notes and relevant source, not only repository descriptions.
4. Record repository, branch/tag and SHA chosen.
5. Record why this source satisfies the five capability prerequisites above.
6. Record any known gfx1100/gfx1201 qualification claims as external evidence only; do not treat them as local measured results.

No local code changes are permitted during Step 0.

## Step 1 — establish actual local hardware/runtime

Save to `system.txt`:

- exact GPU models and VRAM/device;
- HIP device order and gfx target for each GPU;
- negotiated PCIe width/speed;
- ROCm/HIP and driver/kernel versions;
- CPU/NUMA topology;
- physical RAM, free RAM and swap state;
- existing P2P/RCCL capability evidence;
- free VRAM/device before load.

The expected aggregate is 96 GiB VRAM and 96 GiB RAM, but runtime discovery is authoritative.

Do not silently substitute historical topology.

## Step 2 — model compatibility gate

Choose the exact Qwen Flash-Next artifact BEFORE compiling/tuning.

Preferred primary lane: same GGUF file and quant currently used by BigCherry, with SHA256 recorded.

Create `model-compatibility.txt` containing:

- model repository/file;
- architecture identifier;
- quantization;
- byte size;
- SHA256;
- proof that the selected Strata source accepts that architecture and format;
- proof that BigCherry accepts the same artifact.

If exact artifact parity is impossible but Strata is otherwise runnable, create two classes:

- `MATCHED`: identical weights/quant/file semantics suitable for engine comparison;
- `UNMATCHED`: different representation/quant, useful only for architecture/mechanism exploration.

Never calculate an engine speedup from UNMATCHED lanes.

## Step 3 — clean ROCm build

Build only after Steps 0-2 pass.

1. Clone the qualifying AMD source to a fresh directory.
2. Checkout the pinned SHA.
3. Follow its documented ROCm build path before inventing flags.
4. Build for the first target GPU only; R9700/gfx1201 is preferred if supported.
5. Save configure/build commands, compiler versions and complete logs.
6. Verify produced binaries link/load HIP/ROCm libraries as expected.
7. Verify source/build output contains kernels for the intended gfx target.

Do not add unrelated architecture targets to the first build.

### Build failure policy

For a build failure classify it before changing anything:

A. missing documented dependency -> install/fix dependency and retry;
B. stale/wrong build command -> correct to documented command and retry;
C. local ROCm compatibility issue with a small mechanical fix -> document proposed diff and stop for review if it changes source;
D. missing backend/model/format/architecture implementation -> wrong source for BCOP37; stop and re-run Step 0;
E. genuine upstream defect -> capture minimal reproducer and issue/diff; do not turn BCOP37 into a porting project.

No source patch larger than a trivial build compatibility adjustment is authorized by this item.

## Step 4 — prove GPU execution before benchmarking

A successful compile is insufficient.

Run a 32-64 token smoke generation and prove:

- intended HIP device is selected;
- VRAM allocation materially increases;
- GPU utilization/kernel activity occurs;
- output is non-empty and sane;
- no fallback/stub/CPU-only backend is active;
- model/expert runtime reports expected Qwen/Strata path;
- second request in the same process succeeds.

Save `gpu-smoke.txt` and raw log.

If GPU execution cannot be proven, STOP. Do not benchmark.

## Step 5 — memory safety gate

96 GiB RAM is constrained relative to a large host-resident expert pool.

- Avoid swap-backed benchmark results.
- Reserve >=12 GiB RAM for OS/runtime unless measured requirements demand more.
- Abort if available RAM falls below 8 GiB, sustained swap begins, OOM activity occurs, or pinned allocation silently fails.
- Record peak RSS, locked/pinned host memory where observable and VRAM/device.

A disk-paged expert run is invalid for CPU-vs-PCIe conclusions.

## Step 6 — reproduce one known-good Strata ROCm lane

Before tuning our machine, reproduce the closest configuration documented by the selected ROCm source for a supported AMD GPU/model.

The purpose is not to match somebody else's t/s exactly. It is to establish that:

- expert caching is active;
- Qwen Flash-Next executes through the intended kernels;
- reported counters behave plausibly;
- PP/TG are stable across repetitions;
- speculation can be independently disabled/enabled if supported.

Record every deviation forced by our hardware/model.

If this cannot be reproduced, diagnose before proceeding to broad sweeps.

## Step 7 — single-GPU ceilings first

Do not begin with 96 GiB aggregate VRAM.

Test independently where supported:

1. R9700/gfx1201;
2. one 7900 XTX/gfx1100;
3. another materially different installed GPU only if useful.

For each, use one warm-up + >=5 measured repetitions of the short decode lane with speculation OFF first.

Capture:

- PP/TG;
- wall time;
- expert-cache bytes/slots;
- hit/miss rate;
- CPU expert count/fraction/time;
- GPU expert time;
- expert H2D/D2H bytes/time where available;
- host RSS;
- VRAM;
- GPU utilization.

This determines whether Strata's architecture is already competitive without multi-GPU complexity.

## Step 8 — fixed benchmark lanes

Use cache-busted but semantically equivalent fixtures:

| lane | context/input | output | purpose |
| --- | ---: | ---: | --- |
| short-decode | ~4K | 512 | decode/expert ceiling |
| short-prefill | fresh ~4K | 128 | PP control |
| agent-mixed | ~32K | 512 | representative agent workload |
| long | ~128K | 512 | KV/expert tradeoff |
| extreme | ~256K | 256 | optional safety/architecture lane |

Use >=5 measured repetitions after warm-up. Alternate engine order ABBA where practical. Report median and dispersion, not best run.

## Step 9 — isolate expert-cache mechanism

On the fastest stable single GPU, speculation OFF:

1. record Strata auto/calibrated expert cache `E_auto`;
2. test ~50%, 75%, 100%, and if safe 110-125% of `E_auto`;
3. hold KV/prefill policy constant;
4. run short-decode + agent-mixed;
5. plot/report TG against cache GiB and hit rate.

Find the cache knee: additional GiB gives <2% TG improvement.

This is the primary evidence for MET01/RPL01.

## Step 10 — measure cold-expert economics

Where runtime controls/counters permit, determine for cache misses:

`CPU_cost = CPU expert execution`

versus

`GPU_fetch_cost = H2D expert transfer + GPU expert execution + uncovered synchronization`

Do not assume one wins globally. Record expert size/routing density/device/context.

If the implementation already makes this decision dynamically, capture its calibration inputs and decision path from source plus runtime telemetry.

Do not patch in a new policy merely to run this experiment.

## Step 11 — KV versus expert residency

Test supported resident-KV targets around 16K, 32K, 64K and larger/full where safe.

For each setting record:

- KV VRAM;
- remaining expert cache bytes/slots;
- expert hit rate;
- KV streaming/transfer evidence;
- PP/TG;
- peak VRAM/RSS.

Run 32K and 128K lanes, 256K only if safe.

Select by end-to-end throughput, not maximum KV residency or maximum expert cache independently.

## Step 12 — speculation/MTP sweep

Only after obtaining the best non-speculative configuration, test supported depths around 0/2/3/4/5/6.

Capture proposed/accepted tokens, acceptance by depth if available, draft/verify time, rollback/recompute, VRAM and effective target TG.

A deeper setting with higher acceptance but lower end-to-end TG loses.

Findings map to existing FMTP owners.

## Step 13 — prefill tuning

Starting from Strata auto/default:

- record chosen chunk/batch/cache borrowing;
- test nearest smaller/larger supported values;
- include an ~8192 chunk only if supported/relevant;
- run fresh 4K/32K and optional 128K PP;
- record PP, peak memory, expert-cache borrowing, PCIe traffic and post-prefill TG.

Reject PP-only wins that reduce representative end-to-end throughput.

## Step 14 — multi-GPU only after single-GPU explanation

Construct combinations from runtime-discovered GPUs. Likely candidates include best single GPU, 2x XTX, R9700+XTX, R9700+2xXTX, then all suitable GPUs.

For each capture per-device allocation/utilization plus host-mediated/inter-device traffic and synchronization.

Require >=5% end-to-end improvement over the best simpler configuration. More resident weights alone is not a win.

Negative scaling is important evidence for RPL01/PHA03.

## Step 15 — matched BigCherry controls

Run current promoted BigCherry/llama.cpp against the SAME artifact for MATCHED lanes.

Capture exact:

- commit/patch stack;
- tensor/layer split/device order;
- FA state;
- KV types/context;
- batch/ubatch;
- MTP/spec settings;
- PP/TG;
- host RSS and VRAM/device;
- existing copy/transfer/graph telemetry.

Run both best-known BigCherry deployment and, where practical, single-R9700 BigCherry. This separates engine architecture from topology effects.

## Correctness gate

No performance claim survives without correctness:

- greedy/deterministic fixed prompt >=128 generated tokens;
- repeated requests in one process;
- multi-ubatch prompt where applicable;
- 32K and 128K integrity;
- no NaN/Inf/runtime errors;
- logits/top-token/KLD comparison where both engines expose compatible logits;
- speculation results must prove verification/accepted work;
- reject truncated context, accidental prompt-cache hits, missing expert work or different-quant comparisons.

## Reproducibility bundle

Create a timestamped immutable result directory containing:

- `source-capability.txt`
- `system.txt`
- `strata-version.txt`
- `bigcherry-version.txt`
- `model-compatibility.txt`
- `commands.txt`
- `env.txt`
- `gpu-smoke.txt`
- `runs.csv`
- `summary.md`
- raw build/runtime/profiling logs

`runs.csv` minimum fields:

`timestamp,comparison_class,engine,commit,model_sha,quant,gpus,context,input_tokens,output_tokens,repeat,prefill_tps,decode_tps,wall_s,spec_depth,accepted_tokens,proposed_tokens,expert_cache_bytes,expert_hit_pct,cpu_expert_pct,h2d_bytes,d2h_bytes,host_peak_gib,vram0_peak_gib,vram1_peak_gib,vram2_peak_gib,vram3_peak_gib,kv_resident,correctness_status,notes`

Unavailable counters are `NA`, never zero.

## Required local-agent report

Return:

1. exact runnable Strata ROCm repository + SHA and capability proof;
2. build/smoke status;
3. fastest valid Strata configuration per lane;
4. fastest matched BigCherry configuration per lane;
5. MATCHED PP/TG speed ratios only;
6. RAM/VRAM footprint;
7. expert-cache knee/hit curve;
8. CPU-execute vs H2D+GPU miss evidence;
9. KV/expert crossover;
10. speculation curve;
11. single vs multi-GPU scaling;
12. correctness failures/unsupported counters;
13. raw result path/commit;
14. mechanisms worth transferring to BigCherry, each with evidence.

Do not hide failed or negative lanes.

## Ownership after evidence

Disposition findings to existing owners:

- MET01: expert residency/cache/hot-cold policy;
- MET02: sparse/indexed expert execution;
- MET05: host/aux transport and overlap;
- FMTP02-07: MTP/speculation;
- RPL01: device/VRAM/cross-capability placement;
- PHA03: topology/P2P/RCCL qualification;
- BRVP: external backend integration boundary;
- existing telemetry owner: missing counters.

Do not create a second technical owner when one exists.

## Promotion gates

A mechanism merits BigCherry implementation work when matched evidence shows at least one of:

- >=10% end-to-end PP or TG improvement at equal quant/correctness;
- >=20% lower required VRAM at <=5% throughput loss;
- a measured mechanism-level saving large enough to justify a bounded implementation experiment.

Multi-GPU/adaptive complexity requires >=5% over the best simpler qualified configuration.

Reject/defer gains that disappear after matching quant, context, prompt caching, speculation, paging and correctness.

## Change Log

- 2026-10-05T22:08:13.471487+00:00 (updated-by): Updated: section:notes
- 2026-10-05T22:40:11.385605+00:00 (updated-by): Updated: section:notes
- 2026-10-05T23:23:55.742027+00:00 (updated-by): Updated: section:notes
- 2026-10-05T23:52:18.406728+00:00 (updated-by): Updated: section:notes
