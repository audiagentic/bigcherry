---
id: QFP45
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-08T11:01:17.709872+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Reference lane and portable ideas from the vLLM + radiance + libr4d stack (Qwen3.8-27B on one R9700)

## Description

A public report (reddit, 2026-10-08, repo mike2153/mbea-qwen38-dflash) claims Qwen3.8-27B at 125-134 t/s greedy decode and 2,750-2,970 t/s prefill on ONE Radeon AI PRO R9700, 216-260K context, with vLLM + radiance (codeberg ggz14/radiance-vllm-mxfp4 @ 7d9a15a) + libr4d (StillDeadcode, b9e42ab-rx9), AMD's Quark AWQ MXFP4 checkpoint and tcclaviger's DFlash2-FP8 drafter (7 speculative tokens, ~60% acceptance; by position 91/80/70/61/53/46/39%; 5.1-5.4 tokens per verify). Long context: 32K 2,977 t/s prefill / 165 t/s decode; 98K 2,378 / 136; 164K 1,970 / 145; 258K 1,577 / 115. Their own llama.cpp baseline on the card: 52-68 t/s (different prompts). Our best llama.cpp on the same model: 79-80 t/s decode, ~1,300 t/s prefill at 10K on TWO 7900 XTX with Q8_0 and MTP. Brutus already carries this stack: container `radiance-vllm` (image stilldeadcode/vllm-radiance:0.9.3, r4d build b9e42ab-rx9, model Qwen3.8-27B-MXFP4-mtpfp8, drafter Qwen3.8-27B-DFlash2-FP8, R9700, host port 8081, same tuned settings: rerank 80, verify head, dynamic width, W4A8). Two jobs: (1) measure it our way as a reference lane; (2) decide which of its mechanisms are worth porting to llama.cpp patches.

## Steps

1. Reference lane (queued 2026-10-08): `tools/lab/reference-vllm/run-radiance.sh` starts the existing container, measures prefill / decode with the lab corpus at 2K / 8K / 24K / 98K (uncached, streamed, greedy), records the server's speculative-decode metrics, stops the container.
2. Quality: the checkpoint is 4-bit (MXFP4 weights, 8-bit activations, FP8 KV). Before any conclusion about which stack to prefer, run the same probes / task set on both (their report: 8/8 facts at 258K; 46-test Rust task 4-6 of 6 runs perfect).
3. Map their mechanisms to ours and rank what is missing (see Detailed Solution).
4. For each candidate worth porting: design note against the composed b11474 source, patch slice, A/B with queue-env-ab.sh, second model check.

## Detailed Solution & Technical Design

Mechanisms named in the report's config (config/dflash.env) and entry script, against BigCherry:

| Their mechanism | What it is | BigCherry today |
|---|---|---|
| `--max-num-batched-tokens 2048` (owner's container: 4096) | prefill batch 4-8x ours | ubatch 512; ubatch 1024 only fits with 1330 (QSA mask in place) or 1328 (aux expert backend); +13% prefill measured with 1330 at ub1024 |
| MXFP4 W4A8 kernels (hand-written, weights 4-bit, activations 8-bit), HOIST_QUANT / TRACED_QUANT / WPERM | quantised-activation GEMM with activation quantisation hoisted and reused | Q8_1 activation cache chain 1235 / 1307 / 1309-1312 (validated). No 4-bit-weight x 8-bit-activation kernel family of this kind; IQ4_XS / Q4_K MMVQ paths exist (PRBE111, PRBE74) |
| `RADIANCE_RMS_QUANT_FUSION` | RMSNorm fused with activation quantisation | 1309_rms_norm_mul_q81 (validated) |
| `RADIANCE_SKINNY_GEMM` | thin GEMM path | 1347 thin-F32 transposed MMVF (validated); QFP36 router GEMM open |
| `R4D_ATTN_FP8=3`, `--kv-cache-dtype fp8`, FP8 stream | 8-bit attention and KV | owner rule: KV q8_0 minimum, f16 preferred; production runs f16. An 8-bit KV lane (q8_0) is allowed and would roughly halve KV memory: not measured at this pin |
| DFlash2 drafter, 7 tokens, `DRAFT_RERANK=80`, `VERIFY_HEAD=1` (max M 32), `DRAFT_TAU` | draft re-ranking and a small verify head: +3.6% to +9% decode in their table | DFlash/DSpark in llama.cpp broken by 1340 until the fix (PR #63); no rerank or verify-head equivalent. 1297 trims the MTP draft head vocabulary, which is the nearest idea |
| `DYNAMIC_WIDTH` (alpha 0.35, margin 2, min 2) | confidence-driven draft length | 1255 / 1268 adaptive MTP depth (under test, PR #51); QFP43 item 3 proposes one controller for all drafters |
| GDN (gated delta net) fused update, merged in-proj, norm quant, 'both' paths | fused linear-attention recurrence for the 27B model | 1221 chunked GDN (untested), 1253 / 1254 (NRO04 / NRO05); PRBE18 / 21 / 41 SSM fusions parked as experiments |
| Top-k via Triton above a row threshold | fast top-k | 1294 deterministic ordered top-k, 1256 / 1257 |
| Pinned host memory for async H2D (`VLLM_WSL2_ENABLE_PIN_MEMORY`, ~17 ms per copy without) | async host-to-device input copies | 1326_sched_async_host_inputs (validated). QFP42 found ~25 ms per chunk of host-side cost with deferred catch-up, the same class of problem |
| AllReduce with quantised wire (`AR_QBITS=6`), overlap slices | multi-GPU collective compression | 1250 q8 AllReduce wire (evaluated, undecided), cpu-root provider in production |
| Dynamic KV sizing from available memory (`available_memory.py`, reserve 1.0-1.5 GiB) | fit the KV pool to free VRAM at start | llama.cpp sizes KV from -c; our fit / balance sweeps are manual |

First candidates by expected value on our hardware: (a) larger prefill batch (1330 + ub1024 confirmation is queued; then ub2048 needs more memory work); (b) draft rerank + verify head for whichever drafter is in use; (c) an 8-bit KV lane as a memory lever to reach larger ubatch at long context, subject to the owner's KV rule (q8_0 allowed, never q4); (d) GDN fused update for the 27B models.

## Code Samples & Guidance



## Files



## Validation

Reference lane numbers recorded with the server's own acceptance metric. Any ported mechanism follows the normal patch route: mechanics test on the composed production set, compile check, activation marker, ABBA with complete separation or a stated rationale, second-model run.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

The report's llama.cpp comparison is not like-for-like (their words). Weights differ in precision from our Q8_0 / IQ4_XS, so speed alone does not decide anything. Their run-to-run noise is quoted as about +-4% on decode.

2026-10-09 like-for-like side of the reference lane (tools/lab/reference-vllm/run-llamacpp-r9700.sh, PR #75): BigCherry llama-server (production build, 1356 present but off) on the R9700 alone, Qwen3.8-27B UD-Q4_K_M, f16 KV, 180,000 context, same client, corpus and depths as the container run, two repeats. Prefill t/s at ~9K / ~35K / ~111K prompt tokens: built-in MTP (4) 1,035-1,089 / 1,003-1,013 / 764-773; DFlash2 Q8 (7) 1,023-1,046 / 954-970 / 737-747; no drafter 1,259-1,291 / 1,141-1,164 / 844-857. Decode t/s: MTP 50.8-68.9 / 49.0-51.8 / 32.4-43.9 (acceptance 58.7%); DFlash 42.6-44.0 / 35.1-41.2 / 23.7-26.6 (acceptance 24.8%); none 30.5 / 28.1 / 23.0. f16 KV fitted in all arms (28.0-31.8 GB used). Against the container (MXFP4, fp8 KV, DFlash 7: ~3,000 prefill to 35K, ~2,470 at 111K; decode 57-79 to 35K, ~50 at 111K; acceptance 40.7%, earlier session): prefill is 2.3-2.9x ours even with no drafter, so the gap is in the single-device kernels, not the tensor split (the dual-XTX Q8_0 split does ~1,290) and not the drafter. Our drafter costs 15-20% of prefill on one card. Our DFlash path accepts 24.8% against the container's 40.7% with the same drafter family and is slower than MTP at every depth. Owner decision 2026-10-09: the second engine for the multi-engine work is standalone radiance (group run-multi-engine, MEN01-MEN08); libr4d kernels are gfx1200/gfx1201 only and expect MXFP4 / pre-permuted 4-bit layouts, so they are a source of mechanisms to port, not a library to link into llama.cpp.

## Change Log

- 2026-10-08T11:01:17.709872+00:00 (created-by): Created by agent
- 2026-10-08T21:29:03.426287+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261008_212949_added-a-like-for-like-single-c_9270
- 2026-10-08T21:29:55.909605+00:00 (updated-by): Updated: section:ledger-events
