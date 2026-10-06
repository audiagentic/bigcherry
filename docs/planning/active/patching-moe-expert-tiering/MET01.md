---
id: MET01
order: 1
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:44:44.458623+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Per-layer routed-expert profile + canonical residency policy

## Description

MET01 is the single policy/accounting owner for routed-expert residency. It profiles routing, derives measured per-device expert budgets, and chooses among whole-layer residency, static-hot experts, upstream demand cache, ROCm3 auxiliary residency, and host fallback. It must not own a second scheduler, cache transport, auxiliary transport, or loader.

The immediate implementation gate is upstream llama.cpp #29943 + #29887. #29943 moves selective host-expert copying out of generic `ggml_backend_sched_compute_splits()` into a public scheduler copy callback. #29887 is explicitly intended to become user-code-only after that refactor. BigCherry should therefore qualify the callback boundary before carrying any private scheduler cache fork.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- One canonical placement JSON/accounting table covers static, LRU, aux and host tiers.
- #29943 callback seam is either present upstream or qualified as a minimal temporary backport; MET logic does not live in generic scheduler code.
- Observation-only callback proves semantic transparency before selective-copy/cache testing.
- #29887-style cache proves nonzero activity on gfx1100/gfx1201 and passes correctness/integrity gates.
- The Stew/Shali R9700 4 GiB expert-cache observation is preserved with source/configuration traceability and either reproduced directionally on local gfx1201 or dispositioned with measured reasons.
- Multi-request correctness covers the reported DEV_GATHER corruption class; no first-request-only result can promote.
- Static/LRU/hybrid/aux are compared at equal expert-VRAM budget; decode and large-batch prompt effects are both reported.
- MET05 remains sole auxiliary transport owner; MET06 remains blocked until its explicit placement-gap gate passes.
- RPL01 may compare MET01's selected candidate against other system placements but cannot introduce another expert residency policy or mutate MET01 placement state.
- Unsupported or inactive paths fail closed rather than silently becoming baseline measurements.

## Notes

2026-10-05 audit: #29943 materially improves the integration boundary for MET. Its diff removes expert-ID parsing/copy grouping from generic scheduler compute and exposes a host-weight copy callback, while `llama_context` becomes the user-code owner. #29887 states that after #29943 it should be entirely user-code. This reduces BigCherry's reason to carry scheduler-core MoE cache patches and makes the first local mock cheap: an observation-only callback can validate ordering/lifetime on HIP before cache code is introduced.

2026-10-06 cross-capability audit: RPL01 was added as a read-only whole-system cost/recommendation owner. MET01 remains authoritative for expert residency; it exports measured candidate cost components rather than surrendering placement ownership.

2026-10-06 provenance audit: the existing expert-cache plan was traced back to `stew675/llama-cpp-rdna-boosts` and the R9700 Flash-Next reproduction/report in `Shali12/r9700-flash-next-notes`. The reported `-ncmoe 41` + 4096 MiB cache result, lazy/no-load host mode, MTP interaction and second-request DEV_GATHER corruption are now explicit experimental/correctness gates. Upstream #29943/#29887 remain the preferred implementation seam; source lineage is retained even when implementation lineage converges upstream.

External references:
- llama.cpp #29943 `ggml: refactor selective expert copying to user code`
- llama.cpp #29887 `add a GPU cache for MoE experts kept in host memory`
- https://github.com/stew675/llama-cpp-rdna-boosts
- https://github.com/Shali12/r9700-flash-next-notes

2026-10-06 PREREQUISITE PATCHES - STATUS. Upstream state at b11402+30 (origin/master 7049ff0cb): neither #29943 nor #29887 is merged; both fetched as PR heads (pr-29943 528e0a3fc based on b11379; pr-29887 6b7b03aab single commit on bed0a8566), so step 2 is a backport. (1) AUTHORED: patches/1336_sched_copy_callback (kind upstream-backport, origin upstream-pr, state untested, experiment moe-copy-callback). Carries #29943's public API (ggml_backend_sched_copy_callback, ggml_backend_sched_set_copy_callback), host weights copied last, and llama_context::sched_copy_experts verbatim, with the embedded MUL_MAT_ID expert selection removed from ggml_backend_sched_compute_splits. Deviation from upstream form: the scheduler loop is edited in place (two-pass loop + callback call) instead of extracted into ggml_backend_sched_copy_input, because validated patch 1326 edits the same loop; offline tests prove 1336 applies alone and with 1326 in either order to the same result, is idempotent, and fails closed if the expert block changes. BigCherry additions inside the callback only (no MET policy in ggml-backend.cpp): BIGCHERRY_MOE_COPY=0 observation-only control (the step-3 probe: counts, returns false); BIGCHERRY_MOE_COPY_DENSE_PCT (default 90) - when >= that share of a layer's experts is used, one whole copy replaces per-range copies (existing evidence: prefill touches ~496/512 experts per layer); exit counters under BIGCHERRY_PATCH_TRACE (calls, host_weight_bytes, selective_calls, dense_calls, experts_used/total, copied_bytes). Not yet migrated: 1328 (MET05, untested) and 1293 (evaluated) anchor in the same loop and will need re-anchoring when combined with 1336. (2) RUNNING: queue-moe-copy.sh b11402g - build b-moecopy-b11402g, then moe-copy-ab.sh on the R9700 alone with --n-cpu-moe 41, ctx 16384, f16 KV: arms O (observation-only), S (selective + dense shortcut), R (selective, no shortcut), S2 (repeat); each arm serves short, short again, a ~4K-token prompt, short a third time in one process; reports md5 identity across arms and requests, prefill/decode t/s and the counters. This covers steps 2-3 (callback transparency, non-zero selected-copy counters, second-request and workload-shift integrity). (3) FINDING on #29887 as published: it is NOT expressed on the #29943 callback. It adds its own scheduler hooks (ggml_backend_sched_set_moe_cache with resolve/begin/prepare callbacks, a moe_cache_entry table, changes in ggml_backend_sched_backend_id_from_cur and ggml_backend_sched_split_graph that re-home MUL_MAT_ID onto the cache backend and substitute a remapped ids tensor) - about 190 lines in ggml-backend.cpp plus src/llama-moe-cache.cpp (459 lines), cparams/common plumbing and a moe_cache_size context parameter; it refuses pipeline parallelism and more than one device. The copy callback alone cannot implement it: the cache needs the graph rewritten at split time (cached weight tensor + remapped ids), which the callback does not see. So the cache patch is a second, larger backport (overlay file for llama-moe-cache.* + ~20 anchored edits), to be authored after 1336's lanes pass; the 'entirely user code after #29943' statement is the PR author's intent, not the current diff.

2026-10-06 1336 FIRST HARDWARE RESULT (run moecopy-b11402g-r9700-2, build b-moecopy-b11402g = production + 1336, R9700 alone, --n-cpu-moe 41, ctx 16384, f16 KV, 15,978 MiB VRAM in use; the first attempt, moecopy-b11402g-r9700, is WITHDRAWN - the script set both HIP_ and ROCR_VISIBLE_DEVICES, no device was left and the server ran on the CPU). Arms: O = BIGCHERRY_MOE_COPY=0 (observation-only, whole copies), S = default, R = BIGCHERRY_MOE_COPY_DENSE_PCT=0, S2 = S again; each serves short / short / 4,702-token prompt / short in one process. (1) Correctness: greedy text identical across all four arms for every request (short x12 one md5, long x4 one md5), including second-request and after-workload-shift - the seam is semantically transparent and the multi-request integrity gate passes. (2) Activation: callback calls=1230 per arm, host_weight_bytes=503.3 GB eligible; selective arms copied 298.9 GB (experts_used 372,888 of 629,760 = 59%); observation arm selective_calls=0. (3) Speed: 4,702-token prefill 59.7 t/s with whole copies vs 90.4 / 90.4 / 90.4 t/s selective (+51%); short-request decode 17.2 - 18.5 t/s in every arm (one S2 sample 13.46, a stray). The callback fires only for prompt batches: with --n-cpu-moe the scheduler offloads host-expert MUL_MAT_ID to the GPU only for large batches, single-token decode runs those experts on the CPU, so decode is not on this path at all - which is exactly the regime #29887's cache targets. (4) The dense shortcut never triggered (dense_calls=0, S == R): at ub 512 on this prompt 59% of experts are used per layer, below the 90% threshold; it is inert here and stays as a guarded default. Not yet done: comparison against a build without 1336 (expected equal to arm S, since S is upstream's own selective copy moved behind the callback), gfx1100 lane. NEXT: 1337 cache lanes (experiment moe-expert-cache, ARMS=cache: no cache vs --moe-cache-mib 2048 / 4096 / 8192 at the same --n-cpu-moe 41), queued behind the Strata bench (BCOP37).

2026-10-06 1337 (#29887 EXPERT CACHE) FIRST HARDWARE RESULT - SLOWER THAN NO CACHE AT 2-8 GiB (run moecopy-b11402h-r9700-cache, build b-moe-expert-cache-b11402h = production + 1336 + 1337, patch 1337_moe_expert_caching, R9700 alone, --n-cpu-moe 41, ctx 16384, f16 KV, no MTP; the R9700 sits on a PCIe x4 link at 16 GT/s). Patch builds and runs on HIP; cache active ('ROCm0 MoE cache size = N MiB for 48000 MiB of host experts'). Decode, 128-token greedy requests (three per arm, in one process): no cache 18.3 t/s (VRAM 15,978 MiB); 2048 MiB 9.0 t/s (hit rate 45.4% for ubatch <= 8, 217 GB uploaded over the run, VRAM 18,019); 4096 MiB 11.4 t/s (61.0%, 152 GB uploaded, VRAM 20,065); 8192 MiB 16.7 t/s (79.0%, 82 GB uploaded, VRAM 24,155). Prefill of 4,702 tokens unchanged at 90.2 - 90.4 t/s in every arm (the cache is gated to small batches; prompt batches use the 1336 selective copy: counters identical to the no-cache arm). So at equal --n-cpu-moe the cache costs -51% / -38% / -9% decode: on this host the CPU expert path (i7-12700KF) is faster than uploading misses over PCIe x4 and computing them on the GPU, and the upstream NVIDIA gains (25.0 -> 39.4 t/s on a 4090) and the reported R9700 4 GiB result (17-20 -> 36-39 t/s) are NOT reproduced by this mechanism here. The trend is monotonic in cache size, so larger caches are being measured (12288 / 14336 MiB) together with the equal-VRAM alternative MET01 requires - the same VRAM spent on whole resident layers (--n-cpu-moe 34 and 30, no cache). Correctness observations: within every arm the three repeats of the short request are identical (no second-request corruption); the 4.7K-token request matches the no-cache text in 3 of 4 cache arms; the short-request text differs between no-cache and each cache size (three different texts for 2048 / 4096 / 8192). GPU-vs-CPU expert rounding explains a difference from no-cache, but different text per cache size means the result depends on slot layout (accumulation order over remapped ids) - a fidelity gate (flash-fidelity style, vs CPU f32) is required before any cache configuration could be promoted.

2026-10-06 1337 LARGER CACHES - CROSSOVER FOUND (run moecopy-b11402h-r9700-cache-big, same build and config: R9700 alone, --n-cpu-moe 41, no MTP). Decode of the 128-token greedy request: no cache 18.4 / 18.3 t/s (VRAM 15,977 MiB, measured before and after); 12288 MiB cache 22.15 t/s (+20%; hit rate 87.9% for ubatch <= 8, 47.5 GB uploaded, VRAM 28,264 MiB); 14336 MiB cache 24.53 t/s (+33%; hit rate 90.2%, 38.2 GB uploaded, VRAM 30,308 MiB). Prefill of 4,702 tokens 90.2 - 90.4 t/s in every arm. Full size curve on this host: 2048 -> 9.0, 4096 -> 11.4, 8192 -> 16.7, 12288 -> 22.2, 14336 -> 24.5 t/s against 18.4 without a cache; the break-even is between 8 and 12 GiB (roughly an 85% hit rate) - below it miss uploads over the PCIe x4 link cost more than running the experts on the CPU. Equal-VRAM alternative (whole resident layers instead of a cache): --n-cpu-moe 34, no cache, VRAM 23,953 MiB: 19.1 t/s (+4%) - so at ~24 GB of VRAM whole layers (19.1) beat the 8 GiB cache (16.7), while at 28-30 GB the cache gives 22-24.5 t/s; the --n-cpu-moe 30 lane (the ~28 GB equal-VRAM point) is still running. Text: the 12288 and 14336 arms produce the same short-request text as the 8192 arm (md5 4e2b11d2e58e) and the same 4.7K-request text as no cache; the 2048 and 4096 arms each produced a different text. Agreement among the three large caches, with divergence only where the cache is thrashing (45% / 61% hit rate), points at eviction under pressure rather than plain GPU-vs-CPU rounding - to be checked (possible slot reuse while still referenced in the same compute) before any small-cache configuration is trusted. Interim reading for MET01's objective: an LRU cache only pays on this topology when sized to hold most of the routed working set (>= ~12 GiB on the 32 GB card); static-hot placement and hybrid static + LRU are the next candidates because they avoid paying uploads for the hot set. MTP interaction and the gfx1100 lane are not measured yet.

2026-10-06 EQUAL-VRAM COMPARISON COMPLETE (runs moecopy-b11402h-r9700-ncmoe34 / -ncmoe30, same build, R9700 alone, no MTP, no cache). Whole resident layers: --n-cpu-moe 34 -> VRAM 23,953 MiB, decode 19.1 / 20.1 t/s, 4.7K prefill 105.9 / 106.6 t/s; --n-cpu-moe 30 -> VRAM 28,912 MiB, decode 21.0 / 21.0 / 20.9 t/s, prefill 118.1 / 118.6 / 118.5 t/s. Against the cache at the same VRAM: ~24 GB: 8 GiB cache 16.7 t/s decode / 90 t/s prefill vs whole layers 19.1-20.1 / 106 -> whole layers win both; ~28-29 GB: 12 GiB cache 22.2 t/s decode / 90.3 t/s prefill vs whole layers 21.0 / 118.5 -> cache +5% decode, -24% prefill; 14 GiB cache (30.3 GB) 24.5 t/s decode / 90.2 prefill (no whole-layer point at that VRAM). VERDICT against MET01's promotion rule (>= 5% at equal expert-VRAM budget with <= 2% regression in the unaffected regime): the pure LRU cache of #29887 does NOT qualify as a general setting on this host - its decode gain at equal VRAM is marginal (+5%) and it gives up the prefill gain that the same VRAM buys as resident layers. It remains a candidate only as a decode-only policy, or combined with resident layers (hybrid static + LRU: resident layers for prefill, cache over the remaining host experts), which is the next configuration to test. Text is identical (md5 8bab42c40cc2 / 899360b579ec) across every no-cache arm at --n-cpu-moe 41 / 34 / 30. Queued now (run moehop-b11402h): the MET04 probe requested by the owner - every expert in VRAM, production tensor split against a layer split over the two XTX with the routed experts of the last 28 / 14 layers on the R9700 (activations hop XTX -> R9700 -> XTX through host memory), without and with the MTP sidecar on the 6900 XT; then the 12 GiB cache and the --n-cpu-moe 30 baseline with MTP on the 6900 XT (the draft is a separate model on its own device, so it does not trip the cache's single-device guard).

2026-10-06 MTP LANES + PREFILL DIAGNOSIS (run moehop-b11402h, build b-moe-expert-cache-b11402h). (1) Cache with MTP on the 6900 XT works (the draft is a separate model on its own device; MTP verify batches fall in the cache's ubatch <= 8 regime). R9700 + 6900 XT draft, --n-cpu-moe 41: no cache 23.0 / 22.9 t/s decode (R9700 VRAM 16,293 MiB); 12 GiB cache 26.3 / 26.6 t/s (+15%, hit rate 82.8%, VRAM 28,580 MiB); equal-VRAM whole layers, --n-cpu-moe 30, no cache (VRAM 29,230 MiB): 26.8 - 29.1 t/s decode (six samples) and 115 t/s prefill against the cache arm's 89 t/s. So with MTP the cache no longer beats resident layers even on decode at equal VRAM; the earlier verdict stands and is strengthened - pure LRU cache does not qualify here. (2) Production reference from the same run and requests (2x XTX + R9700 tensor split, all experts resident, 4.7K-token prompt): prefill 1246 / 1254 t/s without MTP, 961 / 999 t/s with MTP; decode 40.8 t/s without MTP, 85 - 86 t/s with MTP on the 6900 XT. (3) OWNER CONCERN - host-expert prefill of 90 - 118 t/s on the single R9700 is far below what Strata publishes for single cards (README: ~1,100 - 1,300 t/s on an RTX 5070, 380 - 580 on an RX 9070 XT; not reproduced locally). Diagnosis from the 1336 counters: the 4,702-token prompt is 10 ubatches of 512; the callback copied 298.9 GB of experts for them, i.e. ~30 GB per ubatch; the R9700's PCIe x4 link measures ~6.7 GB/s (Strata's own probe on this card), so ~4.5 s of every ~5.7 s ubatch is expert upload (~80%). Prefill here is bounded by bytes uploaded per token, not by kernels. Levers in order of cost: larger micro-batch (the same upload serves more tokens - Strata uses chunks up to 8,192), more resident layers (already measured: 90 -> 106 -> 118 t/s at --n-cpu-moe 41 / 34 / 30), a wider PCIe slot for the card (x4 today), pinned staging of the expert arena, and not re-uploading an expert that is already on the device from the previous ubatch (a prefill-side cache: the upstream cache's large-batch hit rate was only 7 - 49%). RUNNING (moeub-b11402h): ARMS=ub sweep of -ub 512 / 1024 / 2048 / 4096 on a 16K-token prompt at --n-cpu-moe 41 and 30. (4) The MET04 hop probe failed to start in the first attempt (layer split -ts 1,1,0 put all the expert-bearing layers on the first XTX: 25.8 GiB allocation, out of memory); fixed by giving each XTX half of the heavy layers, rerunning in the same queue without and with MTP.

2026-10-06 HOST-EXPERT PREFILL vs MICRO-BATCH (run moecopy-b11402h-r9700-ub41, 11:16-11:28, clean; R9700 alone, --n-cpu-moe 41, ctx 32768, f16 KV, no MTP, 23,798-token prompt, -b = -ub): -ub 512 -> 93.9 t/s (VRAM 16,476 MiB); 1024 -> 159.0 t/s (16,584); 2048 -> 265.8 t/s (16,798); 4096 -> 420.6 t/s (17,228 MiB). Decode of the 128-token request unchanged, 17.2 - 18.0 t/s in every arm, same text (md5 8bab42c40cc2). Prefill scales 4.5x from ub 512 to 4096 for +0.75 GB of VRAM, confirming the diagnosis that this lane is bounded by expert bytes uploaded per token (one selective upload per ubatch): the fix for the 'disgusting' host-expert prefill is first a deployment flag (-ub / -b), then a wider PCIe link (estimated 2-3x more at ub 512, less at large ub), then pinned staging. Estimate recorded for the owner's question (R9700 alone on a PCIe 5.0 x16 slot, 20-40 GB/s effective instead of 6.7): host-expert prefill at ub 512 ~190-260 t/s (ceiling ~430), 12 GiB cache decode 22 -> ~26-27 t/s, 4 GiB cache 11 -> ~20-24 t/s (cache break-even moves down to ~4 GiB); production tensor-split decode no change expected; on this board the x16 slot currently feeds both XTX at x8, so the move trades their link width. WITHDRAWN, to be rerun: moecopy-b11402h-r9700-ub30 (125 / 342 / 523 t/s at ub 512 / 2048 / 4096 with --n-cpu-moe 30), moecopy-b11402h-r9700-hop2 and -hop2-mtp. A second queue (the PRBE52 adaptive-MTP build and its first lane) started while they ran - its wait condition matched the ALL_JOBS_DONE line that every sub-run prints - so those measurements shared the CPU and then the GPUs with another server. All affected processes were stopped and everything was requeued as one strictly sequential chain (chain-b11402h.launch.log): hop probe with the experts of the last 24 / 14 layers on the R9700 (28 layers do not fit: 33 GiB of experts on the 32 GB card), the same with MTP, the --n-cpu-moe 30 ubatch sweep, -ub 8192 at --n-cpu-moe 41, then the adaptive-MTP lanes.

2026-10-06 MICRO-BATCH SWEEP COMPLETE, CLEAN RERUNS (chain-b11402h, strictly sequential; R9700 alone, ctx 32768, f16 KV, no MTP, 23,798-token prompt, -b = -ub). --n-cpu-moe 41: 512 -> 93.9, 1024 -> 159.0, 2048 -> 265.8, 4096 -> 420.6, 8192 -> 609.5 t/s (VRAM 16.5 -> 18.9 GB). --n-cpu-moe 30 (run -ub30b, replaces the withdrawn -ub30): 512 -> 124.8, 2048 -> 341.9, 4096 -> 522.2 t/s (VRAM 29.4 -> 30.2 GB). Decode of the 128-token request is unchanged by the micro-batch (18.3 t/s at 41 host layers, 21.1 - 21.2 t/s at 30) and its text is identical in every arm. So host-expert prefill on this card goes from 94 to 610 t/s (6.5x) by micro-batch alone; the remaining gap to the production three-card split (1,330 t/s on the same prompt) is the upload itself. Recommended deployment setting for any host-expert configuration on this host: -ub / -b 4096 or 8192, sized to the VRAM left after the resident layers. Estimates for a PCIe 5.0 x16 slot stay estimates (not measured).

2026-10-06 1336 VALIDATED AS A NEUTRAL ENABLER (chain9 lanes moecopy-b11402g-r9700-prodref / -xtx / -xtx-prodref; details in patches/1336_sched_copy_callback/README.md). The two missing pieces of evidence are in: (1) comparison against the production build WITHOUT the patch on the same lanes - R9700, --n-cpu-moe 41: production prefill 90.1 / 90.5 / 90.3 / 90.5 t/s and decode 18.3-18.4 t/s against the patched selective arm's 90.4 t/s and 17.2-18.5 t/s, identical greedy text for every request; (2) a gfx1100 lane - 7900 XTX alone, --n-cpu-moe 45 (11.5 GB VRAM): production 144.7 / 146.8 / 146.6 / 146.9 t/s prefill, 15.4-16.1 t/s decode; patched selective 146.9 / 147.0 / 146.9 t/s, 15.7-15.9 t/s; identical text; observation arm 103.7 t/s (whole copies, -29%); counters calls=1350, selective copies 324.9 GB of 551.0 GB. So the relocated selective copy equals upstream's embedded one in output and speed on both architectures, the callback is demonstrably the path in use, and repeated requests in one process are identical. Patch state -> validated and added to validated-enhancements (it never fires in the production tensor split, where every expert is resident); experiment moe-copy-callback removed, moe-expert-cache now adds only 1337. STATUS OF MET01's STEPS: step 2 (callback seam) and step 3 (transparency before cache policy) DONE; step 4 (#29887 at equal VRAM, R9700 0 vs 4096 MiB lane) DONE on gfx1201 with the verdict that a pure LRU cache does not qualify as a general setting (see earlier notes), not run on gfx1100; steps 5-8 (canonical objective, aux tier, large-batch separation, MTP interaction) partly informed by today's data (micro-batch sweep, MTP lanes) and by GPT's marginal-value rule (req_3b5bb3c3e4834bc8: replay a routing trace, take the candidate with the highest time saved per GiB among resident layer / pinned expert / LRU slot, re-simulate; from today's numbers cache capacity is worth ~2-4 ms per token per GiB for decode between 8 and 14 GiB against ~0.1-0.2 for resident layers, and nothing for prefill). Open: hybrid static + LRU (pinned hot experts - small change to llama_moe_cache: a pinned flag per slot that eviction skips), the small-cache text difference, 1337 on gfx1100.

## Repository evidence

- Existing routing evidence on Flash-Next UD-IQ4_XS, ub512: mean 496/512 experts touched per layer in mixed prose+code prefill; hottest 10/25/50% cover 41.5/68.8/91.3% of picks; mean experts needed for 50/80/90/95% picks are 73/174/235/287. Static expert-count heuristics are therefore insufficient.
- MET05/patch 1328 already owns auxiliary 6900 execution and host staging. MET01 may select that tier but must not reproduce its transport.
- MET06 owns standard-GGUF materialization only if normal host-weight execution/cache mechanisms leave a measured placement gap.
- RPL01 owns cross-capability topology/compute/transfer/VRAM cost comparison. MET01 remains the authoritative expert-residency solver and exports candidate costs/evidence to RPL01; RPL01 must not choose experts independently or replace MET01 placement JSON.

## External provenance and traceability

The expert-cache/static-hot/hybrid direction predates the current upstream seam in this plan and must retain its source lineage rather than appearing as an internally invented BigCherry mechanism.

1. **Stew AMD implementation lineage:** `https://github.com/stew675/llama-cpp-rdna-boosts`. This repository is an RDNA/HIP-focused llama.cpp patch set and is the practical AMD lineage for the expert-cache idea being evaluated here. Stew-specific environment variables, scheduler behavior and patch details are evidence/mechanism references, not BigCherry API contracts.
2. **Single-R9700 evidence lineage:** `https://github.com/Shali12/r9700-flash-next-notes`. The report covers Qwen3.8 Flash-Next on one Radeon AI PRO R9700/gfx1201, 32 GB VRAM, 64 GB RAM and ROCm 10. The reported configuration included `--lazy-mode on --load-mode none`, `-ncmoe 41` and `MOE_EXPERT_CACHE_MIB=4096`. Reported decode increased from roughly 17-20 t/s without expert cache to roughly 36-39 t/s with the 4 GiB cache. With an MTP draft head, reported throughput reached roughly 52-55 t/s at an additional ~3-8 GB VRAM. These numbers are external evidence and must not be represented as BigCherry measurements until reproduced.
3. **Correctness lineage:** the same R9700 notes report an r30 MoE failure where `GGML_SCHED_DEVGATHER` could allow a correct-looking first request but corrupt the second request into `////////`; setting `GGML_SCHED_DEVGATHER=0` was the reported workaround, and r31 changed the default. This directly motivates the existing multi-request integrity gate and makes first-request-only cache qualification invalid.
4. **Upstream convergence lineage:** llama.cpp #29887 supplies the upstream GPU cache for host-resident MoE experts, while #29943 moves selective expert copying behind a public scheduler copy callback owned from user code. BigCherry's preferred implementation path is therefore `Stew/R9700 evidence -> reproduce/measure -> upstream #29943/#29887 seam where sufficient`, rather than importing a second private scheduler cache.

For every externally derived candidate, preserve at least: source URL, source commit/revision when available, observed hardware/software configuration, exact relevant flags/environment, claimed result, local reproduction configuration, and deviations. If a source idea is later implemented through a different upstream mechanism, preserve both `idea/evidence provenance` and `implementation provenance`.

## Upstream mechanism: #29943 + #29887

#29943 adds `ggml_backend_sched_copy_callback` in `ggml/include/ggml-backend.h`, stores it on `ggml_backend_sched`, and funnels split inputs through `ggml_backend_sched_copy_input()` in `ggml/src/ggml-backend.cpp`. Non-weight inputs are copied first; host-backed weight inputs are copied last so user code can inspect already-copied routing inputs. If the callback returns false, the scheduler performs the normal whole-input copy.

It also removes the scheduler's embedded `MUL_MAT_ID` expert-selection implementation and installs `llama_context::sched_copy_experts` from `src/llama-context.cpp`. This is the important consolidation boundary: selective expert-copy/cache policy belongs above generic scheduler machinery.

#29887 then supplies the policy candidate: per-layout GPU cache banks, selected-expert miss upload, remapped expert IDs and a <=32-token cache gate. Published NVIDIA Flash-Next Q4_0 results are strong for decode (4090 25.0 -> 39.4/40.7 t/s; 5090 30.8 -> 54.5/67.8 t/s) but prompt processing regresses as cache VRAM displaces whole resident layers. Treat those numbers as mechanism evidence only; AMD promotion requires gfx1100/gfx1201 measurements.

## Implementation plan

1. **Profile and budget.** Fix/retain `tools/lab/flash-next/routing-profile.sh` so `ffn_moe_topk` collection is safe under the meta callback. Capture decode and prefill separately across code, chat/prose, retrieval/long-context and MTP. Emit `hits[l][e]`, `weight_sum[l][e]`, `ubatch_present[l][e]`, assigned tokens and route mass.
2. **Adopt the callback seam, not a private cache.** On the current pin, first determine whether #29943 is present. If absent, qualify a minimal backport containing only the public copy-callback seam and moved selective-copy logic. Do not add MET-specific logic to `ggml/src/ggml-backend.cpp`.
3. **Mock callback correctness before cache policy.** Install a diagnostic callback that returns false for every host weight while counting callback invocations, backend, tensor bytes and graph first-op. Require byte/token/logit identity with callback disabled. Then enable the upstream selective-copy implementation and require identical greedy output plus nonzero selected-copy counters.
4. **Qualify #29887 at equal VRAM and reproduce the AMD provenance lane.** Compare whole-layer baseline, pure LRU, static-hot, and hybrid static+LRU. Preserve <=32-token gating initially. Prove the callback/cache is active under production scheduler copies; zero cache activity is a failed experiment. On gfx1201 include an explicit 0 versus 4096 MiB cache lane with `-ncmoe 41` and the closest compatible lazy/no-load controls to the Shali12/Stew report, MTP disabled for primary attribution. Treat failure to reproduce the claimed direction as useful evidence requiring transfer/cache-activity explanation, not as permission to tune until it appears.
5. **Canonical objective.** Rank each `(layer,expert,tier)` by avoided critical-path milliseconds per resident byte, using measured H2D, compute, sync and auxiliary staging costs. Do not optimize hit rate in isolation. Export this decomposition to RPL01 for whole-system counterfactual scoring; RPL01 consumes the MET01 result rather than re-solving expert placement.
6. **Aux tier.** Feed MET05/1328 candidates from the same placement JSON. Keep 1328 whole-layer until hardware evidence proves expert-granular auxiliary placement worthwhile.
7. **Large-batch separation.** Keep pp512/2048/8192 separate from <=32-token decode/MTP. A cache configuration that improves decode by consuming VRAM but materially hurts prefill is not globally promoted; allow workload-specific feature-set policy only with explicit budgets.
8. **MTP interaction after cache attribution.** Once the non-speculative cache lane is stable, repeat with the production MTP sidecar/depth. Record draft VRAM, acceptance, effective target TG and cache working-set/hit-rate change. The reported ~52-55 t/s R9700 result is a hypothesis to reproduce, not a promotion target.
9. **Only then consider MET06 slicing.** Runtime standard-GGUF slicing is blocked unless the callback/cache/static/aux matrix leaves >=5% TG/PP opportunity attributable to coarse placement or >=1 GiB avoidable resident expert memory at equal throughput.

## Code-level mock

Use the upstream seam as the test double; no cache implementation is required for the first discriminator:

```cpp
struct met_copy_probe {
    uint64_t calls = 0;
    uint64_t host_weight_bytes = 0;
};

static bool met_probe_copy(
        ggml_backend_t backend,
        const ggml_tensor * src,
        ggml_tensor * dst,
        ggml_cgraph * graph,
        void * opaque) {
    auto & p = *static_cast<met_copy_probe *>(opaque);
    ++p.calls;
    p.host_weight_bytes += ggml_nbytes(src);
    // Observation-only control: normal scheduler copy must still execute.
    return false;
}
```

Build/test this control before any cache port. Required assertions: callback fires only for host-backed weight split inputs; ordinary input tensors are already copied before callback; `return false` preserves baseline output; no callback-owned pointer survives beyond the compute call; multiple scheduler copies do not share mutable probe/cache slot state without generation/copy indexing.

For the selective path, preserve upstream MMQ safety: grouped expert copies must include required guard/padding bytes. Never infer a performance win from missing/corrupt expert work; use the BCOP/QFP28 multi-request integrity gate when qualifying Flash-Next. Explicitly include a two-or-more-request same-process lane capable of detecting the Stew/Shali `GGML_SCHED_DEVGATHER` corruption class; first-request correctness is insufficient.

## Ownership and files

- `tools/lab/flash-next/routing-profile.sh`, `tools/lab/flash-next/expert-placement/`: profiling/solver tooling.
- `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp`: upstream #29943 callback seam only; no MET policy.
- `src/llama-context.cpp` / context-owned helper: selective-copy/cache user-code owner from #29943/#29887.
- MET05 / patch 1328: auxiliary ROCm3 execution and staging.
- MET06: standard-GGUF materialization fallback only after the measured gate.
- RPL01: whole-system advisory cost model; consumes MET01 candidate/decomposition and must not duplicate expert selection/residency state.

No second dispatch table, residency map, cache allocator, prefetch scheduler or ROCm3 transport is permitted.

## Validation matrix

Hardware: gfx1100 XTX, gfx1201 R9700, then production 2xXTX+R9700; 6900 only through MET05 controls. No-P2P means every cache bank/upload is target-device-local.

Correctness before performance:
- callback-disabled vs observation-only callback: greedy identity and logits/KLD contract;
- selective copy/cache: multi-request same-process, cold->warm->workload-shift, MTP on/off;
- explicit second-request corruption check covering the reported r30 `GGML_SCHED_DEVGATHER` failure class;
- <=32-token decode plus pp512/2048/8192 bypass lanes;
- long context and scheduler-copy reuse;
- counters prove selected experts and uploaded bytes are nonzero and physically plausible.

Performance: ABBA >=5 repetitions for TG128/512 and PP512/2048/8192; report median/dispersion, static/LRU/aux GiB, route mass by tier, cache hit, H2D bytes/token, miss-upload ms/token, auxiliary service/staging, CPU fallback, peak VRAM and locked host memory. Include an explicit gfx1201 0/4096 MiB provenance lane before broader cache-budget optimization.

Promotion: >=5% end-to-end TG/effective-TG or PP improvement at equal expert-VRAM budget with <=2% regression in the unaffected regime. A workload-specific decode policy may be retained if prompt regression is explicitly isolated and feature-gated. Reject any result whose implied transfer/work exceeds measured physical limits or whose correctness/integrity gate fails.

## Change Log

- 2026-10-02T04:44:44.458623+00:00: created.
- 2026-10-04: consolidated DwarfStar persistence and #29887 into MET01.
- 2026-10-05: consolidated MET05 auxiliary residency with canonical solver.
- 2026-10-05: made #29943 user-code copy callback the required integration seam; added observation-only mock, HIP qualification matrix and explicit no-duplicate-scheduler boundary.
- 2026-10-06: added explicit RPL01 boundary: export expert candidate cost evidence to the whole-system scorer without duplicating MET01 residency policy.
- 2026-10-06: added explicit Stew/Shali provenance, 4 GiB R9700 reproduction lane, MTP interaction and multi-request DEV_GATHER correctness gate.
- 2026-10-05T22:40:38.378182+00:00 (updated-by): Updated: section:notes
- 2026-10-05T23:19:09.902376+00:00 (updated-by): Updated: section:notes
- 2026-10-05T23:44:31.435693+00:00 (updated-by): Updated: section:notes
- 2026-10-05T23:52:12.689345+00:00 (updated-by): Updated: section:notes
- 2026-10-06T00:00:39.940430+00:00 (updated-by): Updated: section:notes
- 2026-10-06T00:15:41.239330+00:00 (updated-by): Updated: section:notes
- 2026-10-06T00:53:47.475427+00:00 (updated-by): Updated: section:notes

## 2026-10-06 PCIe-x4 cache-policy audit

Local 1337 evidence makes the topology gate decisive. The R9700 PCIe 4.0 x4 path measures about 6.7 GB/s effective H2D. No-cache decode is 18.4 t/s; 2/4/8 GiB LRU gives 9.0/11.4/16.7 t/s at 45.4/61.0/79.0% hit rate; 12/14 GiB gives 22.2/24.5 t/s at 87.9/90.2%. At equal VRAM, resident layers beat 8 GiB LRU in decode and prefill. Near 28-29 GB VRAM, 12 GiB LRU is only about 5% faster in decode but about 24% slower in prefill than resident layers. With the 6900 XT MTP sidecar, 12 GiB LRU is 26.3-26.6 t/s versus 26.8-29.1 t/s for equal-VRAM resident layers. Pure LRU therefore fails the general promotion gate on this topology; keep 1337 as an experimental/reference implementation, not a default policy.

Host-expert prefill is separately transfer-amortization bound: ubatch 512/1024/2048/4096 measured 93.9/159.0/265.8/420.6 t/s with decode unchanged. Larger ubatch is the first prefill lever before new cache code.

### Next bounded candidate

Test static-hot residency before adding cache complexity. At equal total expert-VRAM budgets of 8/12/14 GiB compare whole-layer residency, static-hot experts from existing route profiles, pure 1337 LRU, and static-hot plus residual LRU only if static-hot leaves a useful miss residual. Rank static candidates by avoided H2D bytes times route frequency per resident byte. Report bytes/token, hit rate, promotions/evictions, measured link GB/s, decode t/s and prefill t/s. Do not repeat the already-negative 2/4 GiB pure-LRU sweep.

The first gate is hardware-free: use existing route traces to predict H2D bytes avoided by static-hot placement. Require at least 20% fewer miss bytes/token than pure 1337 LRU at the same budget on held-out traces, without starving later layers/tensors. On fail, retain whole-layer residency and close static-hot for this topology. On pass, reuse MET01 placement state and the existing 1337 storage/cache seam; do not create another cache or scheduler owner.

### External mechanism assessment

ap03906101/moe-hotcache provides relevant HIP mechanism evidence: persistent per-expert VRAM slots, graph-visible route counters, per-tensor budgets, bounded background promotion, and static seed plus dynamic refinement. Its published compute GPU measured about 14 GB/s H2D and the project explicitly found a much slower chipset/x4 GPU unsuitable as a cache tier. That supports a topology-specific gate rather than assuming its gains transfer to BigCherry's about 6.7 GB/s R9700 link.

Do not port its direct host-memory miss path before measured miss bytes/token show it can beat CPU expert execution here. Preserve its strongest lesson: decode and prefill require separate paths and every batch-size regime needs output verification. Correctness must cover batch sizes 1, 2-8, 9-31 and large prefill; at least three requests in one process with a workload shift; greedy fidelity; cache-size invariance; graph replay after refresh; and transfer-byte plausibility against measured PCIe bandwidth.

External references:
- https://github.com/ap03906101/moe-hotcache
- https://github.com/ggml-org/llama.cpp/pull/29887
- https://github.com/ggml-org/llama.cpp/pull/29943
- https://github.com/ggml-org/llama.cpp/pull/29963
- 2026-10-06T01:15:14.315144+00:00 (updated-by): Updated: section:notes

## Ledger-events


- chg_20261006_025038_groundwork-for-running-moe-mod_5459
- 2026-10-06T02:50:41.953811+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-06T04:52:30.132399+00:00 (updated-by): Updated: section:notes
- chg_20261006_061539_scheduler-copy-callback-for-ho_3573
- 2026-10-06T06:15:42.820366+00:00 (updated-by): Updated: section:ledger-events
