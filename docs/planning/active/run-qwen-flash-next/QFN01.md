---
id: QFN01
order: 0
plan: run-qwen-flash-next
state: pending
created-at: '2026-10-01T06:20:09.091717+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# Qwen3.8-Flash-Next: memory layout across 2x XTX + R9700 + 6900 XT + RAM, MTP and ngram speculation

## Description

Model: /mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1..3}-of-00003.gguf, 87.2 GiB, arch qwen4exp (present in the b11233 pin). 48 layers, hidden 2560, 512 experts with 10 active plus a shared expert, expert FFN 640, hybrid SSM/GDN with full attention every 4th layer, sparse-attention indexer, per-layer input embeddings (160 per layer), MTP head.

Size by role: routed experts 55.4 GiB (IQ3_S 31.6, IQ4_NL 18.9, Q8_0 4.2, IQ4_XS 0.8); per_layer_token_embd 26.8 GiB (one lookup table, [160 x 320M rows]); attention/SSM 3.0 GiB; embed/output 1.1 GiB; shared experts 0.2 GiB; router/other ~0.6 GiB.

Hardware: GPU0/1 RX 7900 XTX 24 GiB each (gfx1100, ~960 GB/s); GPU2 R9700 32 GiB (gfx1201, ~640 GB/s; currently held by the radiance-vllm 27B container, ~30 GiB used); GPU3 RX 6900 XT 16 GiB (gfx1030, ~512 GB/s); host RAM 91 GiB; 20 CPU threads.

Key observation: the 26.8 GiB per-layer embedding table is a per-token lookup (each token reads 48 x 160 values), so it belongs in host RAM at near-zero cost. That leaves ~60 GiB of compute weights (experts + dense), which fits entirely in VRAM: 2x XTX + 6900 XT = 64 GiB without the R9700 (tight once KV/compute buffers are counted), or with the R9700 = 96 GiB (comfortable). So all experts can stay resident; no expert swapping should be needed.

## Steps

1. Load test: confirm qwen4exp + its MTP head load and run on the pin; check where llama.cpp places per_layer_token_embd (force to CPU with --override-tensor if not); record per-device VRAM.
2. Layouts to compare (single-request decode, prefill, MTP acceptance): (A) 2x XTX + 6900 XT, -sm layer, experts spread by -ts; (B) add the R9700 (needs vLLM stopped or moved); (C) dense layers on the XTX pair, experts on the slower cards via --override-tensor (ffn_*_exps to 6900 XT/R9700); (D) -sm tensor on the XTX pair if the 6900 XT is used only for experts.
3. Speculation: MTP draft depth sweep (as for 27B), then ngram speculation (draft-free, built from the prompt/context; low memory, CPU) alone and combined with MTP if llama.cpp allows; measure on code/repetitive and free-text prompts separately (ngram helps mostly on repetitive text).
4. AllReduce/split: for any tensor-split layout, reuse the adaptive AllReduce findings; 3+ GPU mixed-arch rules from PGC10.
5. Pick a preferred layout and runtime settings; record as the Flash-Next production profile.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Every layout: A/B with balanced rounds; VRAM headroom recorded; MTP acceptance per arm; prompt-type split for ngram.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner request 2026-10-01: main model in VRAM, as many experts as possible resident or cheaply swappable, consider the 6900 XT, ngram in RAM. vLLM on the R9700 must keep booting on startup; any layout using the R9700 needs a decision on where the 27B vLLM service goes.

Owner 2026-10-01: vLLM is not run while Flash-Next runs, so all four GPUs (96 GiB VRAM total: 2x XTX 24 + R9700 32 + 6900 XT 16) are available to this model. Test jobs still stop/restart radiance-vllm around runs because it must keep booting on startup.

Owner 2026-10-01: PCIe topology: 2x XTX and the R9700 hang off CPU PCIe lanes; the RX 6900 XT is on chipset PCIe (shares the chipset uplink, higher latency, lower bandwidth). Consequence: keep the 6900 XT out of any per-token AllReduce / tensor-split group; give it only whole layers or whole expert blocks so cross-device traffic is one activation hand-off per layer boundary, or use it last if VRAM is short. Prefer layouts where the three CPU-attached cards carry all hot traffic.

Owner 2026-10-01: link widths: 2x 7900 XTX on PCIe 4.0 x8 each (~13 GB/s effective), R9700 on PCIe 4.0 x4 (~6.5 GB/s), 6900 XT on chipset PCIe. No P2P, so host-staged collectives cross each link twice. 27B dual-XTX prefill AllReduce measures 5-10 GB/s effective, i.e. near the x8 host-staged ceiling: RCCL tuning cannot fix prefill; overlap or fewer/smaller ARs can. Any tensor-split group including the R9700 is AR-bound at about half the XTX link rate.

2026-10-01 probe flashnext-probe-1 (b11233 + validated set, gfx1100/gfx1201/gfx1030 build, PLE -ot to CPU, -c 16384): (1) every -sm tensor layout fails at load: 'LLAMA_SPLIT_MODE_TENSOR not implemented for architecture qwen4exp' -- tensor split needs a patch adding qwen4exp to the tensor-split architecture support (per-tensor split rules for the hybrid SSM/GDN, sparse-attention indexer, PLE and MoE tensors, as for qwen35). (2) -sm layer on XTX,XTX,R9700 -ts 3,3,2: out of memory on device 0 allocating a 231 MiB compute buffer (pp graph), i.e. GPU0 got more than its share of resident weights. (3) -sm layer on all four: model loaded, then no health within 10 min (hang or very slow warmup; log stops at threadpool init). Next: verbose (-lv 4) load with per-device buffer sizes, lower -ts weight on device 0 and -c 8192, check where per_layer_token_embd lands; design the qwen4exp tensor-split patch (owner: tensor split is much better than layer split).

2026-10-02 MTP under -sm tensor resolved. The unsloth self-contained MTP sidecar is legacy-format: qwen4exp.attention.compress_ratios[48]=0 (dense MTP) although blk.48 carries indexer tensors; c061df198 (#29761) expects the MTP layer QSA (ratio 4). Load asserts (dead k-pool inputs) and the first-draft get_rows OOB both came from that. Corrected copy mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf (ratio[48]=4, edited in place on a copy) on pristine b-flash-c061, -sm tensor -ts 3,3,2: baseline 421 pp / 28.3 tg; MTP depth 2: 384.5 pp / 41.8 tg, acceptance 68.9%; depth 3: 385.6 pp / 42.9 tg, 63.8%; greedy identical to baseline for both. Patch 1280 (forward-expand k-pool inputs) rejected as a mask. Plain -sm layer segfaults in warmup on both pristine and patched builds (separate upstream issue). Next: 192K + MTP layout (v9 OOM'd by 1.19 GiB on device 1), MTP depth/ngram sweep, consider reporting the converter mismatch to unsloth.

2026-10-02 sweeps 1-3 (pristine c061, qsa4 sidecar, MTP --no-spec-draft-backend-sampling), greedy identical to baseline in every run. 8K: baseline -ts 3,3,2 425 pp / 28.3 tg; even -ts 1,1,1 431 / 28.0. MTP depth 3/4/5 at 3,3,2: 43.8/43.6/43.8 tg (acc 63.8/60.4/56.7%) -> depth 3. MTP3+ngram-mod 38.1 (worse). ABAB MTP3: 3,3,2 43.3/43.8 (acc 63.8%), even 45.6/45.8 (acc 70.5%) - acceptance differs between splits (deterministic numerics), so not a pure split-speed attribution, but end-to-end better. Draft on R9700 (-devd ROCm2) 41.7. BEST: even split on ROCm0-2 with MTP draft on the 6900 (-dev ROCm0,ROCm1,ROCm2 -devd ROCm3, HIP_VISIBLE 0-3): 401.9 pp / 48.5 tg, acc 70.9%; 6900 uses 3.7 GiB. 192K (-c 196608, q8_0 KV + q8_0 draft KV, -ts 2,2,3, draft on 6900): 391.9 pp / 45.1 tg, acc 74.7%, VRAM 20.2/20.5/31.6/4.9 GiB - 192K + MTP target met. 192K without 6900 draft OOMs at every split tried (2,2,3 / 3,3,4 / 5,5,6 / 5,5,7). -ts 5,5,6 with 6900 draft also OOMs.

2026-10-02 fit sweeps (MTP depth 3, qsa4 sidecar, greedy identical everywhere). Draft on the 6900 (-dev ROCm0,ROCm1,ROCm2 -devd ROCm3), 8K: -ts 1,1,1 48.6 tg (acc 70.9%), 3,3,2 44.4 (60.4%), 5,5,4 50.1 (73.1%), 4,4,3 50.3 (73.1%); 1,1,1 depth 2 47.8 (79.8%), depth 4 44.9 (61.2%); q8_0 draft KV 48.0 (no speed change). At 8K VRAM is ~identical across -ts; at 192K -ts clearly moves memory (KV/compute buffers follow -ts). 192K with 6900 draft: only 2,2,3 fits (47.0 tg, acc 74.7%, VRAM 20.2/20.5/31.6/4.9 GiB); 3,3,4 OOM at first request, 4,4,5 no result, 1,1,1 OOM at load. Draft R9700 vs 6900 at 4,4,3 8K, ABAB: R9700 48.0/48.3 (acc 72.5%) vs 6900 50.4/50.2 (73.1%) -> 6900 +4.5%. Max context with draft on R9700 (4,4,3): 32K/64K/96K/128K/160K all run at 46.0/45.9/45.7/45.8/45.6 tg (XTX 23.9 GiB at 160K); 176K at 4,4,3 and 192K at 4,4,3/3,3,4/2,2,3/4,4,5 all OOM. Preferred configs: 8K-160K: draft on 6900, -ts 4,4,3 (or draft on R9700 if the 6900 is needed elsewhere, max 160K); 192K: draft on 6900, -ts 2,2,3.

2026-10-02 stock kernel 7.0.13-070013 (P2P-hack kernel halved host<->GPU DMA; thermald had throttled XTX0/R9700 links to Gen1 - all earlier prefill numbers in this item are low). flashnext-stock-1/-3, production build b-flash-c061, greedy identical in every MTP run. No MTP -ts 4,4,3: ub512 1079 pp / 36.2 tg; ub2048 1478 pp / 36.5 tg. MTP3 draft on 6900, -ts 4,4,3: ub512 933 pp / 66.8 tg (acc 73.1%); ub1024 1139 / 68.4 (73.9%); ub2048 1328 / 66.2 (69.5%). Depth 2: 65.2 (81.0%); depth 4: 66.2 (68.5%). -ts 5,5,4 MTP3: 951 / 68.0. 192K -ts 2,2,3 MTP3 (q8_0 KV): ub512 917.5 / 62.5 (74.0%); ub1024 OOM. 224K ub256: 704 / 61.2. Preferred: 8K-160K -ts 4,4,3 (or 5,5,4), MTP depth 3, draft on 6900, ub1024 (best decode, +22% prefill vs ub512) or ub2048 for prefill-heavy use; 192K -ts 2,2,3 ub512. Single-session numbers; differences under ~2% not conclusive.

2026-10-02 native (llama-native:stock:linux-multi, pristine c061, same RCCL HIP build) vs production, same window, 3 requests: Flash-Next 3-card no MTP ub512 969/35.1 vs 1079/36.2 (+11% pp); ub2048 1403/35.6 vs 1478/36.5; best 8K MTP3 ub1024 draft-6900 1090/67.8 vs 1139/68.4; 192K 865/63.3 vs 918/62.5. 27B: no-draft dual-XTX tensor 1430/34.2 vs 1479/38.5 (+13% tg); MTP5 ~1314/77.6 vs ~1316/78.7; 3-card MTP5 1101/78.7 vs 1101/77.3; single R9700 HIP 1223/19.7 vs 1340/19.7 (Vulkan RADV 1120/19.2). BigCherry's gains are mostly plain decode/prefill; with MTP decode is within ~1-2% of native -> the MTP verify path (AllReduce on verify batches) is the remaining lever.

2026-10-02 flashnext-ar-1 (production build, -ts 4,4,3 ub1024 MTP3 draft on 6900, 3 requests): auto 1132/67.9 and repeat 1136/68.5; ccl 1137/67.8; adaptive switch 32K/96K/256K/1M 1135-1138 / 67.0-69.1 (acceptance 73-75%, spread within noise); host-only 608/47.0; butterfly 608/47.5; 4-card incl. 6900 (-ts 4,4,3,2, host provider) no MTP 270/8.1; 3-card ref no MTP ub1024 1266/36.6. On 3 GPUs auto==adaptive==RCCL; the BigCherry host path does not scale to 3 devices (x4 R9700); the 6900 must stay out of the tensor split. Gains must come from structural changes (fewer/fused/overlapped reductions), not provider choice.

2026-10-03 AllReduce census + microbenchmarks (goal: Flash-Next deployment/AllReduce). Census (1277 trace, adaptive dispatcher, -ts 4,4,3 ub1024): decode = 96 AllReduces/token (2 per layer x 48), f32 [2560 x ne1] = 10 KB (ne1=1) or 20-40 KB under MTP3 verify; all RCCL on 3 GPUs; prefill ~96 x 10.5 MB (+6.4 MB tail) per 1024-token ubatch. RCCL microbench (tools/lab/rccl/ar-latency.hip), stream-ordered chain: 2 ranks 10KB 22.4 us, 40KB 25.3, 10MB 1420 us; 3 ranks 10KB 33.1 us, 40KB 41.1, 640KB 270, 10MB 3604 us (~2.9 GB/s), 40MB 14.6 ms. Host-synchronised-per-call numbers (66.7 / 107.7 us) are dominated by launch+sync (empty-kernel floor 40 / 78 us). CPU-root one-shot (tools/lab/rccl/cpu-root-ar.hip: mapped pinned slots, per-rank epochs, AVX2 exact-f32 CPU sum, GPU spin on result epoch), stream-ordered: 3 ranks 10KB 13.8 us, 20KB 21.0, 40KB 35.5; 2 ranks 11.4 / 16.9 / 26.4. Estimates: decode no-MTP 96 x ~19 us = ~1.9 ms of 27.8 ms (~+7%); prefill AR ~350 ms of ~0.8 s per ubatch (~40%) -> a pipelined host/CPU-root path for 10 MB messages is the larger lever (+20-30% prefill if ~1-1.5 ms/10 MB).

## Change Log

- 2026-10-01T06:20:09.091717+00:00 (created-by): Created by agent
- 2026-10-01T06:23:08.097063+00:00 (updated-by): Updated: section:notes
- 2026-10-01T06:24:16.813548+00:00 (updated-by): Updated: section:notes
- 2026-10-01T07:45:50.591371+00:00 (updated-by): Updated: section:notes
- 2026-10-01T08:31:56.666485+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261002_011157_llamacpp-updated-to-include-q_5054
- 2026-10-02T01:12:00.116378+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-02T03:48:27.574308+00:00 (updated-by): Updated: section:notes
- chg_20261002_034831_qwen38-flash-next-mtp-specula_9007
- 2026-10-02T03:48:35.036254+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-02T05:23:42.872770+00:00 (updated-by): Updated: section:notes
- 2026-10-02T07:20:01.905871+00:00 (updated-by): Updated: section:notes
- 2026-10-02T11:50:00.976519+00:00 (updated-by): Updated: section:notes
- 2026-10-02T13:16:39.643772+00:00 (updated-by): Updated: section:notes
- 2026-10-02T13:27:09.420054+00:00 (updated-by): Updated: section:notes
- 2026-10-02T14:35:13.352947+00:00 (updated-by): Updated: section:notes
