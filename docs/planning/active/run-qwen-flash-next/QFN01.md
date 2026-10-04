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

2026-10-03 large-message host AllReduce prototype (tools/lab/rccl/large-host-ar.hip, chunked copy-engine D2H, AVX2 CPU sum exact f32, per-chunk H2D): 10 MB 3 ranks best 2.63 ms (1 MB or 512 KB chunks, 2-6 threads; persistent pool no better) vs RCCL 3.60 ms (-27%); 2 ranks 1.45 ms vs RCCL 1.42 (tie). Link check (bidir-check.hip): XTX H2D 13.5 / D2H 14.2 / simultaneous 10.8 GB/s each way; R9700 6.7 / 7.0 / 6.2 each way (bidirectional overlap works); 6900 3.5 / 3.5 / 2.7. 10 MB floor on 3 ranks ~1.7 ms (R9700 x4). GPT (req_3f7154f755904a49) agrees: CPU-root copy-engine pipeline, 512 KiB-1 MiB chunks; RS/AG through host loses on this topology; bf16 prefill wire is not acceptance-neutral. Small-message path (cpu-root-ar.hip) 10 KB 13.8 us vs RCCL 33.1 us. Integrated provider: patch 1291_ar_cpu_root requested from GPT (req_dab269203605443f).

2026-10-03 patch 1291_ar_cpu_root (--allreduce cpu-root, HIP): CPU-root one-shot AllReduce for f32 messages <= 64 KiB on the backend streams (pinned mapped slots, per-rank device-advanced epochs, persistent exact-f32 CPU worker), RCCL above. flashnext-cpuroot-4 ABBA vs auto, -ts 4,4,3 ub1024: no MTP decode 38.7/39.1 vs 36.5/36.6 (+6.4%); MTP3 draft-6900 70.5/70.4 vs 67.0/67.1 (+5.1%), acceptance 74.6 vs 73.0%; prefill unchanged; greedy identical in all arms. Root cause of earlier garbage: ranks whose node the meta backend left uncomputed must contribute zeros. New best Flash-Next 8K config: -ts 4,4,3 ub1024 MTP3 draft-6900 --allreduce cpu-root = ~1130 pp / 70.5 tg. Next: KLD + contract, large-message host path for prefill (prototype 2.63 ms vs RCCL 3.6 ms per 10 MB), HC-combine fusion.



2026-10-03 (later) patch and tuning status - every patch below is a package in patches/, committed on patch-refactor:
- 1291_ar_cpu_root (evaluated): small path keeps decode +5-7%. Large-message host path (prefill) added then made opt-in default off: in-model it lost to RCCL at every chunk size (1/2/4/8 MiB: 1057/1057/1027/937 vs ~1450 t/s prefill, -ts 4,4,3 ub1024); 4 SIMD sum workers changed nothing. Not graph-capture safe (host generation) - RNX11 RV4210.
- 1292_kpool_tail_truncate (evaluated): Qwen4Exp kpool layout rebuilt O(n_ctx) every MTP step (seq_rm of rejected drafts); truncate instead. -14% ms/MTP step at ~80K (77 -> 65.5-66.4), -2..3% at 10K, two ABBAs; greedy byte-identical at 10K and 80K without MTP on the 1294 base.
- 1293_sched_single_input_sync (untested, neutral): scheduler syncs an event-less split backend once per split; hipStreamSynchronize -18% but ms/step unchanged; not deployed.
- 1294_topk_deterministic_ties (evaluated): HIP radix TOP_K picked tied QSA cells in atomic order (ReLU-sum scores, many exact zeros) -> long-context greedy differed between server starts; lowest-index tie-break makes 32K/80K output identical across starts; speed neutral.
- 1295_qsa_gather_decode (untested): QSA attention on HIP reads the whole cache (upstream sparse FA is NVIDIA-only; attention 0.44 -> 3.36 ms/token target, 0.18 -> 1.41 draft, 10K -> 80K). Gathers each token's selected cells for n_tokens <= 8. A/B run 1 invalid (parallel queues OOMed); rerun flashnext-gather-ab-2 serial; GPT review req_bb197fcc0e734bf3.
- 1296 (not written): HIP sparse flash attention. Prefill uses the WMMA kernel (flash_attn_ext_f16), verify the tile kernel, draft the vec kernel; a tile port only duplicates 1295, so 1296 = sparse WMMA for long-prompt prefill, scheduled after 1295 results. RNX02 RV4213.
- 1297_draft_vocab_trim (untested, opt-in BIGCHERRY_DRAFT_VOCAB_N): MTP draft output.weight (Q8_0, 675 MB) read per draft token = ~31% of 6900 draft time at 80K; trimmed head + -inf scatter. A/B rerun flashnext-trim-ab-2 (serial).
- RNX10 shared-expert gate fusion: not written (upstream already fuses sigmoid*mul; ~96 launches/step, ~1%); GPT requests died in gateway restarts.

Config findings (all MTP3, draft on 6900, cpu-root, ~10K cached prompt unless noted):
- f16 draft KV (-ctkd f16 -ctvd f16): -8.8% ms/step at 80K, -2.5% at 10K (ABBA) -> adopt everywhere.
- RCCL NCCL_ALGO=Tree NCCL_PROTO=Simple: +2.0% prefill at -ts 4,4,3 ub1024 (ABBA, complete separation); neutral at -ts 2,2,3 ub512.
- MTP depth 2/3/4 at 192K: within noise; keep 3.
- Rank balance at 32K ctx: -ts 3,3,2 / 4,4,3 47.5-47.8 ms/step vs 2,2,3 50.1-50.4 (~6%); prefill ~1105-1110 vs ~1068.
- Context tiers (ub512; ub1024 OOMs at every long tier; ub256 costs ~28% prefill, not a RAM spill - prefill flat across context): up to 96K f16/f16 -ts 4,4,3 ~71.7 tg; up to 144K f16-K/q8_0-V -ts 0.31,0.27,0.42 ~1051 pp; up to 160K q8_0 -ts 4,4,3 or 3,3,2 ~68-69 tg ~1100 pp; up to 192K q8_0 -ts 2,2,3 ~64 tg ~1070 pp. f16-K/q8_0-V at ub384 reaches 160K (~901 pp). XTX1 carries ~4 GB more than XTX0 at equal share (unsplit data), so XTX1-light splits fit more. Owner: KV never q4.
- Best prefill measured: ~1477 t/s (-ts 4,4,3 ub1024, no MTP, Tree+Simple, 1.7K prompt); MTP costs ~15%.
- Decode at depth is GPU/launch bound: ~4176 kernels per MTP step per tensor-split GPU (1283 elementwise, 576 quantize_q8_1), cpu_root consume ~121 us x 96 (rank imbalance); HIP graphs active (graphs off +6% step time). VRAM overhead plan: QFN03.
- Queue lesson: parallel queue scripts overlapped (one's servers/vLLM took the R9700) -> use tools/lab/flash-next/chain-serial.sh; queues no longer restart vLLM (owner).



2026-10-03 END-TO-END (flashnext-combined-ab-1, ABBA, 192K deployment flags, q8_0 target KV): this morning's deployment (1291, Q8_0 MTP draft, q8_0 draft KV) vs candidate (1291+1292+1294+1297, Q5_K_M draft, f16 draft KV, BIGCHERRY_DRAFT_VOCAB_N=65536). 10K: 66.8/60.4 t/s (52.1/52.3 ms/step) -> 70.3/69.8 t/s (44.1/44.2) = +10% t/s, -15% ms/step. 80K: 39.6/39.4 (77.0/77.3) -> 49.9/50.0 (57.0/56.3) = +26.5% t/s, -26% ms/step. Prefill unchanged (~1050-1070 at 10K, ~850 at 80K). Complete separation at both depths; acceptance slightly lower (~331 vs 344 accepted per 512 at 80K) but more than offset.

PRODUCTION PROFILE (Flash-Next, 192K): b-flash-deploy-1 = experiment ar-cpu-root-kpool-topk-trim (1291_ar_cpu_root, 1292_kpool_tail_truncate, 1294_topk_deterministic_ties, 1297_draft_vocab_trim). llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS -ngl 99 --fit off -c 196608 -ub 512 -b 2048 --flash-attn on -ot 'per_layer_token_embd\.weight=CPU' -dev ROCm0,ROCm1,ROCm2 -devd ROCm3 -sm tensor -ts 2,2,3 -md mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf --spec-type draft-mtp --spec-draft-n-max 3 --no-spec-draft-backend-sampling -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 --allreduce cpu-root; env BIGCHERRY_DRAFT_VOCAB_N=65536. Shorter-context tiers: up to 96K f16/f16 -ts 4,4,3; up to 144K f16-K/q8_0-V -ts 0.31,0.27,0.42; up to 160K q8_0 -ts 4,4,3. Q5_K_M sidecar = llama-quantize --allow-requantize of mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf.

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
- 2026-10-02T14:49:31.198618+00:00 (updated-by): Updated: section:notes
- 2026-10-02T16:07:10.226148+00:00 (updated-by): Updated: section:notes
- chg_20261002_160710_new---allreduce-cpu-root-optio_9249
- 2026-10-02T16:07:13.385564+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_004922_cpu-root-allreduce-keeps-its-d_3316
- 2026-10-03T00:49:25.615827+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_004925_faster-long-context-decode-wit_4394
- 2026-10-03T00:49:28.688991+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_004928_profiling-and-sweep-tooling-fo_2428
- 2026-10-03T00:49:31.768304+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_040726_long-context-qwen4exp-decode-w_8449
- 2026-10-03T04:07:32.984347+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-03T06:40:44.684261+00:00 (updated-by): Updated: section:notes
- chg_20261003_084245_faster-mtp-speculative-decodin_6706
- 2026-10-03T08:42:48.723602+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-03T09:09:44.321472+00:00 (updated-by): Updated: section:notes
- chg_20261003_104706_faster-long-context-decoding-f_1057
- 2026-10-03T10:47:23.068845+00:00 (updated-by): Updated: section:ledger-events

- chg_20261003_151601_flash-next-now-runs-240k-conte_5065
- chg_20261004_011520_three-more-flash-next-decode-k_5440
## Reviews

- RV4214

2026-10-03 split skew + PRBE115 quick screens (ABA, 24K depth, deployment candidate). Tensor split at 128K ctx vs 2,2,3 (~48.3 ms/step): 2,2,3.4 49.3; 2,2,3.8 49.1; 1.9,2.1,3 48.1; 2.1,2.1,2.8 48.0; 2.2,2.2,2.6 47.4; 2.3,2.3,2.4 46.8; 2.4,2.4,2.2 46.8 -> R9700 is the slow rank, gain saturates at ~-3% from 2.3,2.3,2.4. That split fits only 160K q8_0/q8_0 (176K/192K fail; f16-K/q8_0-V fails at 160K) -> middle tier, neither max context (2,2,3 @192K) nor max speed (4,4,3 tiers); owner: not useful, dropped. 1301 (PRBE115, Q8_0 F32-act at MTP widths 2-4, activation proven): neutral (48.8 vs 48.3/49.3; +RDNA4 48.2 vs 49.3/48.3), acceptance slightly lower -> parked. 1205 / 1206 quick screens: neutral. 6900 draft at 24K: ~7.4 ms per 3-token draft (~15% of step).

2026-10-03 f16-K/q8_0-V max context with all GPUs filled (maxctx-search, token_embd on CPU, -b 512, ub512): max 144K (147456) at -ts 0.31,0.27,0.42 -> 1109 pp / 72.4 tg at 8K depth, VRAM 21.6/22.0/33.9/4.4 GiB. 152K and 168K fail at every split tried (0.31-0.44 XTX0): weights scale with share (~0.35 cap on XTX0), and the 2 KV heads sit on XTX1 + R9700 only (XTX0 holds none), so XTX0's ~4 GB headroom cannot absorb KV. token_embd-on-CPU and -b 512 did not move the limit. Further f16 context needs structural changes (QFN03 compute-buffer reductions, KV placement), or ub384 (160K, -14% prefill). Fixed -ts 2.3,2.3,2.4 cannot run f16/q8_0 at 144K+.
- RV4215

2026-10-04 NEW LONG-CONTEXT PROFILE CANDIDATE (1303 + 1302): f16/f16 KV at -c 245760, BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 (both KV heads on the two XTX), expert -ts 0.31,0.27,0.42, -b 512 -ub 512, token_embd on CPU, deployment draft settings, build b-deploy-1303 (deploy-plus-1302-1303). Max-context searches (8K-deep requests): f16/q8_0 and f16/f16 both load to 248K but prefill collapses there (~1100 -> 85-88 t/s, decode unchanged; 256K OOM at every split) -> usable ceiling 240K. At 8K: f16/f16 ~1100-1116 pp / 74.6-75.4 tg (vs production q8_0/q8_0 192K ~1070/~70). KV pinned to XTX0+R9700 (1,0,1) also reaches 208K (~70.5 tg); XTX1+R9700 (0,1,1) fails below 144K. Deep fills (flashnext-1303-deepfill): 164,470 tokens -> prefill 690 t/s, decode 40.8 t/s, 162/279 accepted (~66.8 ms/step; production at the same depth 636 pp / 46.4 tg / 178/230 = ~70.7 ms/step), 1302 evicted 193 graphs on the R9700; 228,030 tokens -> prefill 593 t/s, decode 44.6 t/s, 181/221 accepted (~76.5 ms/step), no OOM. Single runs; ABBA vs production pending. Before 1303 the f16/f16 ceiling was 96K and f16/q8_0 144K.

2026-10-04 PROFILE ABBA (flashnext-profile-ab-1, 512 decode tokens): production (b-flash-deploy-1, q8_0/q8_0, -c 196608, -ts 2,2,3) vs 1303 profile (b-deploy-1303 = deploy + 1302 + 1303: f16/f16, -c 245760, BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0, -ts 0.31,0.27,0.42, -b 512, token_embd on CPU). ~10K: pp 1032/1038 -> 1065/1058 (+2.5%), tg 68.5/69.6 -> 73.3/74.0 (+6.5%), ms/step 44.8/44.3 -> 43.4/43.5 (-2.5%; rest is acceptance 345-346 -> 351-353 of ~490). ~80K: pp 835/847 -> 903/902 (+7.4%), tg 50.0/49.7 -> 52.9/53.7 (+7%), ms/step 56.9/56.9 -> 54.1/53.6 (-5.4%), acceptance equal (331-334). Complete separation on every metric at both depths. Draft (6900) ~6.5 ms per 3-token draft at 10K, ~9.2 ms at 80K (15-17% of step), unchanged between profiles.

PRODUCTION PROFILE v2 (Flash-Next, 240K, f16 KV): build b-deploy-1303 (experiment deploy-plus-1302-1303: 1291, 1292, 1294, 1297, 1302, 1303). Env: BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_DRAFT_VOCAB_N=65536. llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS -ngl 99 --fit off -c 245760 -ub 512 -b 512 --flash-attn on -ot '^per_layer_token_embd\.weight$=CPU' -ot '^token_embd\.weight$=CPU' -dev ROCm0,ROCm1,ROCm2 -devd ROCm3 -sm tensor -ts 0.31,0.27,0.42 -md mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf --spec-type draft-mtp --spec-draft-n-max 3 --no-spec-draft-backend-sampling -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 --allreduce cpu-root. Deep fill to 228K tokens verified (593 pp / 44.6 tg). Do not exceed 240K: 248K loads but prefill collapses to ~85 t/s.
- 2026-10-03T15:16:07.855086+00:00 (updated-by): Updated: section:ledger-events

2026-10-04 ADOPTION ABBA (flashnext-v2-fusion-ab-3, 512 decode tokens): profile v2 (b-deploy-1303) vs v2 + 1307-1310 (b-v2-q81c, launch reduction: Q8_1 activation reuse + RMSNorm/activation producers + rollback copies without CONT; greedy output identical). ~10K decode 74.3/74.3 -> 76.6/76.3 t/s (+3%); ~80K 53.2/53.4 -> 55.2/55.2 t/s (+3.6%); complete separation both depths; prefill unchanged. Kernels/token 1307 -> 1138 per GPU.

PRODUCTION PROFILE v3 (Flash-Next, 240K, f16 KV): profile v2 command line plus patches 1235, 1307, 1308, 1309, 1310, 1311 (experiment deploy-v2-plus-1311; producers publish only in decode-shaped graphs, the row caps are gone) and env GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1 (in addition to BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_DRAFT_VOCAB_N=65536). 1311 adopted 2026-10-04: speed-neutral in its screen, greedy-identical, removes the hc_pre quantize launches; kept because it cuts launches with no regression (owner policy: small/neutral non-regressing wins are taken). Candidate next: 1312 (deploy-v3-plus-1312, screening as flashnext-v3-1312-*).

2026-10-04 Qwen3.8-27B drafter placement (owner question; tools/lab/flash-next/draft27b-ab.sh, runs/draft27b-ab-1): 27B Q8_0 on the two XTX (-sm tensor, cpu-root), built-in MTP head (drafts on the TP XTX) vs the Q4_0 MTP sidecar on the 6900 (-devd, f16 draft KV), spec-draft-n-max 4, 256 greedy tokens. ~10K: built-in 70.2/72.8 t/s (46.2/44.5 ms/step) vs sidecar 68.0 (49.5); ~32K: built-in 71.9/71.7 (47.5/47.6) vs sidecar 61.2 (53.0). Acceptance similar. Verdict: keep the 27B's built-in MTP on the XTX - its single dense MTP layer is cheap there; on the slower 6900 its compute + attention (grows with depth) + hidden-state handoff exceed the AllReduces saved. Opposite of Flash-Next, whose large MTP model did not fit/benefit on the TP cards.
- 2026-10-04T01:15:27.393851+00:00 (updated-by): Updated: section:ledger-events
