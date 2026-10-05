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