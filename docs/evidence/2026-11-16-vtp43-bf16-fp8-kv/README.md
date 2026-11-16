# VTP43 evidence: bf16_fp8 and bf16_i8 KV caches (2026-11-16)

Argv: qwen27b-tp3-int8-556 (rank 0, 1 gfx1100; rank 2 gfx1201), one start each. KLD harness: `--kld-record` / `--kld-ref`
with a bf16 reference; short corpus 24 documents (42,761 positions, 8K context), long corpus 3 x ~25K and 2 x ~100K
tokens scored past 4096 (276,900 positions, one sequence).

## Layouts (head_dim 256, bytes a token and head)

| cache | K | V | bytes | cache tokens (qwen27b -long, 8 x 262,144 argv) |
|---|---|---|---|---|
| bf16 | bf16 | bf16 | 1024 | 624,048 |
| bf16_i8 (rad0038) | bf16 | int8, f16 scale per 32 dims | 784 | 818,896 (1.31x) |
| bf16_fp8 (rad0037) | bf16 | E4M3 | 768 | 835,952 (1.34x) |
| fp8 | E4M3 | E4M3 | 512 | 1,253,936 |

## Quality, long corpus, against a bf16 reference recorded by the same build

bf16 against that reference: exactly 0 (the build is deterministic).

| cache | mean KLD | p99.9 | top-1 | PPL ratio | doc 00 / 02 p99.9 |
|---|---|---|---|---|---|
| fp8 | 0.00495 | 0.591 | 98.23% | 1.00052 | 3.92 / 6.13 |
| bf16_fp8 | 0.00323 | 0.318 | 98.62% | 1.00051 | 2.46 / 4.71 |
| bf16_i8 | 0.00269 | 0.186 | 98.67% | 0.99959 | 1.87 / 3.33 |

Short corpus (against the earlier bf16 reference, whose floor is ~0 there: bf16 again 5e-11): fp8 0.00221, bf16_fp8
0.00157, bf16_i8 0.00111 mean KLD.

### How the investigation went

- bf16_fp8's first long run (0.00336 mean, docs 00/02 p99.9 3.3/4.0) missed the plan's bar (mean <= 0.0027, no document
  p99.9 above 1). 79% of the mass sat in 0.1% of positions (277 tokens), KLD 14-21 nats, in long identifiers and
  numbers repeated from far earlier; the same tokens flipped under fp8 (log-KLD correlation 0.94).
- That suggested E4M3 V precision, so bf16_i8 (int8 V, 8x finer) was built. Against the same old reference it scored
  0.00309, barely better.
- The missing control: plain bf16 against the old reference scored 0.00315 with the same tail (docs 00/02 p99.9
  1.9/5.2). The reference had been recorded by an older build; across builds the bf16 cache itself moves by that much on
  this corpus. A fresh reference from the current build gives bf16 vs bf16 = 0 and the table above.
- Conclusion: the per-document p99.9 < 1 bar cannot be met by any change on this corpus (docs 00 and 02 carry exact-copy
  positions that flip under any numeric perturbation). Relative to that floor bf16_i8 is the closest of the three and
  the only one whose mean is inside the original 0.0027.

## Speed (long argv, 2 reps, 128 tokens, 512-token prompts; TTFT is a 2048-token prompt)

| cache | 1 stream tok/s | 4 | 8 | TTFT |
|---|---|---|---|---|
| fp8 | 92.1 | 225.6 | 279.2 | 776 ms |
| bf16_fp8 | 94.7 | 227.7 | 288.4 | 778 ms |
| bf16_i8 | 93.3 | 227.0 | 285.3 | 778 ms |
| bf16 | 93.3 | 225.9 | 281.1 | 753 ms |

Within the run-to-run spread of one another (all within 3% of fp8).

## Kernel checks

gfx1201 (r4d_selftest) and gfx1100 (r3_selftest through the libr3 rewrite): numeric prefill/decode cases against an
independent oracle with negative controls, split-KV and mixed steps, byte-exact stores, fused qk_norm_rope_gate store
byte-identical to the unfused pair. Spills (gfx1201): bf16_i8 prefill 0 (bf16_fp8 5), decode 22/0/0/0 (60/16/3/0).

## Not done

No VTP42 mixed-workload run on these caches; streaming (host-tier) caches warn, as for the other packed layouts;
libhalo has no bf16_i8/bf16_fp8 form.
