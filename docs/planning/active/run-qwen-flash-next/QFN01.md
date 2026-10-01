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

## Change Log

- 2026-10-01T06:20:09.091717+00:00 (created-by): Created by agent
- 2026-10-01T06:23:08.097063+00:00 (updated-by): Updated: section:notes
- 2026-10-01T06:24:16.813548+00:00 (updated-by): Updated: section:notes
- 2026-10-01T07:45:50.591371+00:00 (updated-by): Updated: section:notes
- 2026-10-01T08:31:56.666485+00:00 (updated-by): Updated: section:notes
