# Flash-Next KLD of the packed KV caches (VTP44)

## Summary
Flash-Next's KLD against a bf16 reference cannot measure KV-cache error: re-chunking the plain bf16 cache alone lands
where the packed caches do. The packed caches therefore neither pass nor fail the 27B bar on this model. A noise-immune
metric is the follow-up (below). Serving and the kernels are proven by the selftest (DEVICE-AB.md), not by this KLD.

## Outcome
| Arm (short corpus, 24 docs, 42,761 positions) | mean KLD | p99.9 | top-1 |
|---|---|---|---|
| bf16 (same build, repeat) | 0 (bit-identical) | 0 | 100% |
| bf16_i8 | 0.0446 | 1.754 | 89.3% |
| bf16_fp8 | 0.0495 | 1.756 | 88.8% |
| fp8 | 0.0539 | 2.044 | 88.2% |

Long corpus (3 x ~25K and 2 x ~100K tokens, 276,900 positions): bf16 repeat 0.00032 (p99.9 0.024); bf16_i8 0.0293
(p99.9 3.258, top-1 94.7%). The long bf16_fp8 and fp8 arms did not finish (a session restart killed the run at 52,548
of 276,900 positions); they were not rerun, as the calibration below shows they would not discriminate.

Calibration, same two documents, same bf16 reference recorded at 64-token chunks: bf16 cache at 128-token chunks
(no KV change) 0.0372 / p99.9 1.36 / top-1 90.8%; bf16_i8 at 128 chunks 0.0390 / 1.43 / 90.9%; bf16_i8 at 64 chunks
0.0382 / 1.58 / 90.8%. This matches the band RAD39 recorded for Flash-Next: any arithmetic change, even re-chunking the
stock path, moves mean KLD to 0.055 - 0.065 and p99.9 to 1.6 - 2.35 (MoE routing flips amplify tiny differences).

## Detail
Setup: two 7900 XTX, TP2 exact wire, experts tiered through the host pool, no helper, one sequence, no speculation,
64-token chunks (the model's minimum: its linear-state boundary needs 64), build `kv-bf16-packed` (rad0037, 38, 40,
41, 42). The only change per arm is `--kv-cache-dtype`. Each arm is scored against a bf16 reference recorded on the same
build.

Checks that ruled out other causes:
- Expert placement. The arms differ in free expert slots (16799 to 16840) because smaller caches leave more VRAM. A
  rerun with `--deterministic` (placement pinned to the loader's plan, no moves) gave bf16_i8 0.04458, identical to the
  unpinned run, and bf16 exactly 0 against its pinned reference.
- Recurrent state. The linear-attention state is f32 whatever the cache dtype.
- Position profile. bf16_i8 mean KLD by position in a document: 0.032 at position 0, 0.12 at 1 - 4, 0.11 at 16 - 64,
  0.074 at 64 - 256, 0.042 at 256 - 1024, 0.034 at 1024 - 4096. On Qwen3.8-27B the same arm is 0.0004 early, rising to
  0.0019 at 1024 - 4096, which is quantisation noise accumulating. Flash-Next's profile is the opposite, the signature
  of routing divergence, not of rounding. int8 V has about ten times less rounding than fp8 yet scores within 20% of it.
- A decode-only comparison is not possible: `--max-num-batched-tokens` below 64 is refused for this model.

## Follow-up
A metric the routing noise does not swamp: a long-context retrieval or task probe (around 100K tokens) with bf16
against bf16_i8 and bf16_fp8, the capacity gain being the case for the packed caches here; or the KLD of attention
outputs on a single layer. The speed A/B on the full helper config and the default-cache identity check remain open.
