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

# Qualify Strata Flash-Next execution on the local 96 GB VRAM / 96 GB RAM host

## Purpose

Run a controlled local Strata qualification against the current BigCherry/llama.cpp Flash-Next baseline and return enough evidence to determine which Strata mechanisms should be adopted, reproduced, rejected, or assigned to existing technical owners.

This is an evidence/action item. It does not authorize a second BigCherry scheduler, expert router, cache policy, model loader, or benchmark framework. Technical follow-up must be assigned to the existing owner (MET01/MET02/MET05/FMTP/RPL01/PHA03/BRVP or another existing plan) wherever possible.

## Machine contract

Record the actual host at run time rather than assuming historical topology.

Required facts before testing:

- 96 GB total physical GPU VRAM across all installed GPUs.
- 96 GB system RAM.
- exact GPU models, gfx targets, VRAM/device, PCIe negotiated width/speed and device ordering;
- ROCm/HIP version, kernel/driver version and Strata commit;
- BigCherry/llama.cpp commit/build used for the control;
- CPU model, NUMA topology, RAM speed if available, swap configuration;
- P2P/RCCL capability from existing PHA evidence where available;
- free host RAM and free VRAM before model load.

Do not infer aggregate memory from old documentation. Save the command outputs used to establish the run-time topology.

## Safety/resource gate

96 GB RAM is not an unlimited expert backing store. Before full model load:

1. Disable or avoid swap-backed benchmarking. Record swap state.
2. Reserve at least 12 GiB host RAM for OS/runtime unless measurement shows a larger requirement.
3. Estimate model/expert pinned-RAM requirement before enabling all-expert pinning.
4. Abort a lane if host available RAM falls below 8 GiB, sustained swapping occurs, OOM killer activity occurs, or the runtime silently drops pinned/locked allocations.
5. Record peak RSS, pinned/locked memory where observable, and peak VRAM per GPU.

A result obtained while paging to disk is not a valid CPU-expert/PCIe comparison.

## Reproducibility bundle

Create one timestamped result directory containing:

- `system.txt` — CPU/RAM/kernel/ROCm/GPU/PCIe/topology;
- `strata-version.txt` — repository URL, commit SHA, dirty status, build flags;
- `bigcherry-version.txt` — repository/llama.cpp SHA, patch set/build flags;
- `model.txt` — exact model repository/file, quant, file size and checksum;
- `commands.txt` — exact command for every lane;
- `env.txt` — relevant environment variables;
- `runs.csv` — one row per repetition;
- `summary.md` — median, dispersion, failures and observations;
- raw Strata/BigCherry logs.

Do not overwrite previous results.

## Model/control rule

The primary comparison MUST use the same Qwen Flash-Next model weights/quant wherever both engines can consume them. Record exact file checksum.

If Strata requires a different representation, report that as a separate non-equivalent lane. Do not present different quants as an engine speedup.

Use identical prompt text, context construction, output length, sampling/greedy settings and chat template where technically possible.

## Phase 0 — build and smoke test

1. Build/install Strata at a pinned commit using its supported AMD/ROCm path.
2. Save complete build configuration and compiler output.
3. Run the project's tests/smoke tests applicable to AMD.
4. Load the target model and generate 32-64 deterministic/greedy tokens.
5. Confirm output is non-empty/coherent and no HIP/runtime errors occur.
6. Record model load time, RSS and VRAM/device.
7. Run the same basic prompt twice in one process to catch state/cache corruption.

Stop here if the build is unstable or deterministic output is corrupted.

## Phase 1 — establish single-GPU reference ceilings

Do not begin with all GPUs. Test each primary architecture independently so multi-GPU gains/losses can be explained.

Run, where supported:

- R9700/gfx1201 alone;
- one 7900 XTX/gfx1100 alone;
- other materially different GPU alone if useful.

For each device:

1. Start with Strata's automatic/calibrated expert-cache and prefill settings.
2. Keep host expert policy constant.
3. Run the short decode lane below five measured repetitions after one warm-up.
4. Record TG, PP if emitted, expert-cache slots/bytes, hit/miss rate, CPU-expert fraction, H2D/D2H bytes, host RSS, VRAM, draft acceptance if enabled, and wall time.
5. Repeat with speculation disabled to separate Strata execution architecture from speculative gain.

Select the fastest stable single GPU as the tuning reference, but retain all single-GPU results.

## Phase 2 — baseline benchmark lanes

Use fixed prompt fixtures with deterministic cache-busting suffixes so repeated runs do not accidentally become prompt-cache measurements.

Minimum lanes:

| Lane | Effective input/context | Output | Purpose |
| --- | ---: | ---: | --- |
| short-decode | ~4K | 512 | decode ceiling/expert behavior |
| short-prefill | fresh ~4K | 128 | prefill control |
| agent-mixed | ~32K | 512 | normal mixed workload |
| long | ~128K | 512 | KV/expert residency crossover |
| extreme | ~256K | 256 | optional; only if both engines support it safely |

For every promoted comparison use >=5 measured repetitions after warm-up and report median plus min/max or p25/p75. Alternate engine order where practical (ABBA) to reduce thermal/background bias.

## Phase 3 — expert-cache/residency sweep

On the best single GPU, determine whether Strata's advantage is expert residency/execution rather than speculation.

1. Disable speculation for the first sweep.
2. Record the calibrated/auto expert cache as `E_auto`.
3. Test approximately 50%, 75%, 100% and, if memory permits, 110-125% of `E_auto` using supported cache controls.
4. Keep KV-resident setting and prefill policy fixed.
5. Run short-decode and agent-mixed lanes.
6. Record:
   - expert cache capacity in bytes/slots;
   - hit/miss rate;
   - CPU-executed expert count/time;
   - GPU expert time;
   - expert H2D bytes/time;
   - total TG;
   - host RSS and VRAM;
   - synchronization/idle evidence if available.

Decision: identify the knee where another GiB of expert residency produces <2% TG improvement. That is the initial VRAM opportunity cost for RPL01/MET01 comparison.

## Phase 4 — cold-expert policy experiment

If Strata exposes both CPU execution and GPU-fetch behavior for cache misses, compare them directly. If not exposed, use available telemetry to infer the active path and document the limitation rather than patching Strata during this first qualification.

For representative expert misses estimate/measure:

`CPU_miss_cost = CPU expert execution time`

`GPU_miss_cost = H2D expert transfer time + GPU expert execution time + uncovered synchronization`

Run at least short-decode and agent-mixed. Determine whether the preferred path changes with expert size/routing density/device.

This is the critical evidence for MET01/MET05. Do not implement a BigCherry hybrid policy until this comparison exists.

## Phase 5 — KV versus expert-residency sweep

On the best single GPU test supported KV-resident targets near:

- 16K;
- 32K;
- 64K;
- larger/full where safe.

For each setting:

1. Recalibrate or explicitly record expert-cache capacity left after KV allocation.
2. Run 32K and 128K lanes; include 256K if stable.
3. Record KV VRAM, expert-cache VRAM/slots, KV transfer/streaming evidence, cache hit rate, PP, TG and peak memory.

Decision: select settings by end-to-end throughput, not maximum resident KV or maximum expert slots independently.

## Phase 6 — speculation/MTP sweep

Using the best non-speculative residency/KV configuration, sweep supported speculation depths around:

`0, 2, 3, 4, 5, 6`

Do not force unsupported values.

Record:

- proposed tokens;
- accepted tokens by draft position if available;
- acceptance ratio;
- draft time;
- verification time;
- rollback/recompute evidence;
- TG wall time;
- effective accepted target tokens/s;
- VRAM/RSS change.

Run short-decode, agent-mixed and long lanes.

Select by effective end-to-end target throughput. A deeper draft that has higher acceptance but lower TG is a rejection.

Technical findings belong with FMTP owners; BCOP37 only records disposition.

## Phase 7 — prefill tuning

Starting from auto/calibrated prefill:

1. Record Strata's chosen chunk/batch/borrowing values.
2. Test the nearest smaller and larger supported settings, including an 8192-class chunk if applicable.
3. Run fresh 4K and 32K prefills; optionally 128K if runtime is reasonable.
4. Record PP tok/s, temporary expert-cache borrowing/reduction, peak VRAM, host RSS, PCIe traffic and subsequent decode TG.

Reject a prefill setting that wins PP but materially damages steady-state TG without an overall workload win.

## Phase 8 — multi-GPU scaling

Only after the single-GPU optimum is understood, test available combinations in increasing complexity. Construct the actual matrix from installed devices, normally including:

- best single GPU;
- 2x7900 XTX if present;
- R9700 + one XTX;
- R9700 + 2xXTX;
- all suitable GPUs;
- auxiliary/6900-class GPU only as an explicit experiment, not assumed beneficial.

For each combination record:

- per-GPU VRAM allocation;
- per-GPU utilization/kernel time where available;
- H2D/D2H and inter-device/host-mediated traffic;
- synchronization/idle gaps;
- PP/TG;
- expert hit/miss and placement;
- total power if readily available.

Calculate scaling efficiency relative to the best constituent single GPU:

`efficiency = multi_gpu_TG / best_single_TG`

Do not call extra GPUs beneficial solely because more model data fits. Require >=5% end-to-end gain to promote a combination. Explicitly retain negative scaling evidence for RPL01/PHA03.

## Phase 9 — BigCherry/llama.cpp matched control

Run the current promoted BigCherry/llama.cpp configuration on the same model/quant and prompt fixtures.

At minimum capture:

- exact tensor/layer split and device order;
- flash-attention state;
- KV types and context;
- batch/ubatch;
- MTP/spec settings;
- PP/TG;
- peak VRAM/device;
- host RSS;
- H2D/D2H/graph-copy telemetry already available;
- correctness/output checksum or logits comparison where available.

Run the best known BigCherry deployment AND, where practical, a single-R9700 BigCherry control. This distinguishes Strata architecture from multi-GPU topology effects.

## Correctness gate

Performance is invalid unless correctness survives.

Required:

1. deterministic/greedy fixed-prompt comparison for at least 128 generated tokens;
2. repeated requests in one process;
3. multi-ubatch prompt where applicable;
4. 32K and 128K context integrity;
5. no NaN/Inf/runtime errors;
6. if engines permit logits capture, compare top-token agreement and divergence/KLD using the same quant; otherwise record token/output divergence without claiming bit identity;
7. speculation-on results must show accepted/verified work rather than skipped verification.

A faster run caused by missing expert work, truncated context, prompt-cache reuse, failed KV residency, or different quant is rejected.

## Required `runs.csv` fields

At minimum:

`timestamp,engine,commit,model_sha,quant,gpus,context,input_tokens,output_tokens,repeat,prefill_tps,decode_tps,wall_s,spec_depth,accepted_tokens,proposed_tokens,expert_cache_bytes,expert_hit_pct,cpu_expert_pct,h2d_bytes,d2h_bytes,host_peak_gib,vram0_peak_gib,vram1_peak_gib,vram2_peak_gib,vram3_peak_gib,kv_resident,correctness_status,notes`

Use blank/NA for unavailable counters; never encode unknown as zero.

## Analysis to return for review

The local agent must commit or attach the result bundle and provide a concise report containing:

1. fastest valid Strata configuration for each benchmark lane;
2. fastest valid BigCherry configuration for the same lane;
3. same-quant speedup ratio for PP and TG;
4. RAM and VRAM footprint difference;
5. single-GPU versus multi-GPU scaling;
6. expert-cache knee and hit-rate curve;
7. measured CPU-execute versus H2D+GPU miss economics where observable;
8. KV-residency/expert-cache crossover;
9. speculation depth/acceptance curve;
10. correctness failures or unsupported counters;
11. exact raw-result path/commit.

Do not summarize away negative results.

## Ownership/disposition after results

After review, classify each demonstrated mechanism:

- **MET01** — expert residency/cache economics and hot/cold expert policy;
- **MET02** — sparse/indexed expert execution primitives;
- **MET05** — host/auxiliary-device expert transport and overlap;
- **FMTP02-07** — MTP/speculation implementation and acceptance economics;
- **RPL01** — cross-capability/device/VRAM placement cost comparison;
- **PHA03** — topology/P2P/RCCL admission evidence;
- **BRVP** — external/native backend integration boundaries;
- existing telemetry owner — missing measurement primitives.

Prefer updating those owners over creating another implementation plan.

## Promotion/rejection gates

A Strata mechanism is worth BigCherry implementation work only when at least one matched lane demonstrates either:

- >=10% end-to-end PP or TG improvement at equal quant/correctness; or
- >=20% lower required VRAM at <=5% throughput loss; or
- a clearly measured mechanism-level saving large enough to justify an implementation experiment.

For multi-GPU or adaptive mechanisms, require >=5% end-to-end gain over the best simpler qualified configuration.

Reject/defer mechanisms whose gain disappears after matching quant, context, prompt caching, speculation, RAM paging and correctness.

## Completion criteria

BCOP37 is complete when:

- Strata builds and the supported AMD path is qualified or its failure is captured reproducibly;
- matched Strata versus BigCherry results exist;
- expert/KV/speculation/multi-GPU sweeps are completed to the extent supported by the runtime;
- raw evidence is preserved;
- every material mechanism is assigned to an existing technical owner or explicitly rejected/deferred;
- no duplicate BigCherry runtime subsystem has been created as part of the test.
