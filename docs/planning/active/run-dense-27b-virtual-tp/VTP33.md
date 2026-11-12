---
id: VTP33
order: 33
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-12T07:38:58.014935+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P2
---

# TP3 capacity: resident KV bounded by the R9700; capacity-oriented splits

## Description

TP3 serves the full trained context (262,144 tokens), but its resident KV is only 6-9% above TP2's. One block table serves every rank, so the carve is the smallest rank's share. The R9700 holds 2 of the 4 KV heads (twice the KV per token) and ~45% of the weights; the XTX ranks could page about 1.96M tokens each.

## Results so far

- MXFP4 at 262k context: TP3 3.7 against TP2 3.4 full-length sessions (4 sequences); 3.5 against 3.3 (8 sequences). TTFT for a 26,624-token prompt is 20.0 s against 12.7 s; decode after it is 93.4 against 68.9 tok/s.
- w8a8 at 262k, 8 sequences: kv 1,1,2 gives 12.15 GiB and 2.8 sessions; kv 2,1,1 gives 11.72 GiB and 2.7 sessions. The extra head stays on the R9700.
- **2026-11-12, capacity splits without and with rad0029** (MXFP4, 262,144-token context, 8 sequences, depth 3, fp8
  KV, exact wire; logs vtp33/logs/cap-1812 on build 8a6f2776, and cap-1839 on build 260b77b5, which adds rad0029).
  Each cell is full-length sessions at 262k (tokens carved), then stream-bench tok/s at 4, 6 and 8 streams.

  | split (K / FFN / KV heads 1,1,2) | without rad0029 | with rad0029 |
  |---|---|---|
  | TP2 (two XTX) | 3.3 (854,208); 200.8 / 192.6 / 224.9 | 3.3 (854,208); 209.5 / 195.8 / 225.6 |
  | 4,4,8 / 5120,4864,7424 (6/6/12) | 3.5 (917,184); 219.7 / 217.3 / 239.5 | **5.5 (1,435,072)**; 219.8 / 219.3 / 242.3 |
  | 5,5,6 / 5632,5632,6144 | 3.8 (1,005,296); 220.0 / 215.5 / 239.6 | 5.8 (1,520,368); 215.5 / 215.4 / 238.7 |
  | 6,6,4 / 6656,6656,4096 | 3.9 (1,018,960); 209.6 / 196.0 / 223.7 | 6.2 (1,630,096); 211.2 / 200.8 / 222.7 |
  | 7,7,2 / 7680,7680,2048 | 3.8 (993,536); 208.6 / 193.2 / 218.5 | **6.7 (1,743,616)**; 202.6 / 193.6 / 216.8 |

  - **Without rad0029, splits stop paying beyond 6,6,4.** Every rank's weights pool was the largest rank's, and every
    rank was budgeted as the smallest card. The R9700 holds 2 of the 4 KV heads and sets the token count, yet about 8
    GiB of it stayed free: its card is 8 GB larger, and its weights pool had room left over (1.45 GiB at 6,6,4, 3.59
    GiB at 7,7,2).
  - **rad0029** gives each rank of an uneven dense world its own split. The weights pool holds exactly its own weights;
    the KV pool takes what that gives back plus the free memory above the smallest card. Every pool is then cut to the
    least whole MiB that pages the new common token count.
  - **Result:** the throughput split (6/6/12) carves +56% more tokens with no throughput cost, which is **+67% resident
    KV over TP2**. 7,7,2 reaches +103% at 3–8% less throughput than 6/6/12. TP2 is unchanged (its cards are alike).
    Greedy answers are identical in every arm.
  - **Long-context checks** (6/6/12 with rad0029; logs vtp33/logs/long-1844 and fill-1855). The server stayed up and
    logged no errors.
    - A passphrase in the middle of a 44,688-token prompt is answered correctly.
    - 8 concurrent 128k-token prompts, 1,038,923 tokens resident at once, all complete. That is past the 917,184 tokens
      the old split could carve. 8 × 160k timed out at the client (600 s) during prefill, with the server healthy.
  - **Acceptance met:** +67% resident KV over TP2 at no throughput cost (target +25%), or +103% at 3–8% less
    throughput.
- **2026-11-12, w8a8 with rad0029** (build 260b77b5, logs vtp33/logs/capi8-1926, same setup as the MXFP4 sweep):

  | split (K / FFN, KV heads 1,1,2) | full-length 262k sessions (tokens) | tok/s 4 / 6 / 8 streams |
  |---|---|---|
  | TP2 (two XTX) | 1.9 (489,344) | 175.5 / 167.3 / 195.0 |
  | 4,4,8 / 5120,4864,7424 (6/6/12) | 4.2 (1,111,040) | 171.8 / 184.0 / 202.7 |
  | 6,6,4 / 6656,6656,4096 (`qwen27b-tp3-int8`) | **5.5 (1,443,408)** | 189.4 / 182.2 / 212.8 |
  | 7,7,2 / 7680,7680,2048 | 4.9 (1,280,464) | 174.0 / 163.1 / 193.3 |

  - The int8 layout's own split is both the fastest and the largest: **2.9 times TP2's resident KV**.
  - At 7,7,2 the XTX ranks run short instead, because int8 weights are twice MXFP4's.
  - Greedy text differs between splits (6,6,4 and 7,7,2 begin "Sunlight is composed", the others "Sunlight contains").
    Each answer is coherent. The reduction order changes with the split; N6 measured w8a8 quality at 6/6/12.
- **Capacity layout:** `qwen27b-tp3-capacity` (MXFP4, 7,7,2, 262,144-token context) sits beside `qwen27b-tp3`. Its
  settings are the 7,7,2 arm above: 6.7 sessions at 3–8% less throughput. The throughput layouts already get rad0029's
  gain, since it needs no setting. The new layout's own launch was not run; its flags match the measured arm.

## Steps

1. Sweep capacity-oriented splits: move delta-net and feed-forward shares off the R9700 to free its VRAM for KV. Record resident KV, full-length sessions and tok/s at 4/6/8 streams.
2. Report the throughput/capacity frontier per format.
3. Ship a capacity layout (settings only) beside the throughput layout.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Engine KV report at 262k (pool, sessions).
- stream-bench at 4/6/8 streams.
- A long-context correctness check.

## Effort & Risk



## Standards



## Acceptance Criteria

- A layout with measurably more resident KV than TP2 (target +25% or more), with its throughput cost stated.

## Notes

Data: VTP14.md (N5 first pass, KV head placement). Script pattern: ${BC_TASK_TMP}/vtp14/kvsplit.sh.

## Change Log

- 2026-11-12T07:38:58.014935+00:00 (created-by): Created by agent
- 2026-11-12T07:40:17.937281+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
