# Candidate B — vocabulary head weight-prefetch lifetime

## Decision

Retain shorter-lookahead code only for a *fully populated long-token activation staging ring* on gfx1100. The compile-time condition compares staged activation columns with available weight fetch units: `kBP >= kU && FN == kBP * kRPP`. With the existing tile geometry, this selects NT4 but excludes NT1–NT3. Other GPU architectures retain the original lookahead.

The revised path fetches the next K128 pair after the K256 fold/barrier, eliminating an overlapping pair of live decoded weight records. The integer multiply-accumulate order and the F32 fold are not changed. The unresolved cost is a potentially serialized next-iteration global fetch.

## Resource gate

The two experimental libraries and the final NT4-guarded library compiled and linked with the complete head unit. All six head symbols match. The final guarded resource table is `candidate-b-resources.csv`.

Only NT4 changed in the guarded library. It holds 256 VGPR per lane, 51,200 bytes LDS and the same six-wave register floor as baseline; SGPR increases 84 to 86, without residency loss. Compiler VGPR spill count decreases **797 to 663**, private-segment size stays **204 bytes**. The remaining five symbols (including NT3, which otherwise dropped from 35 to 9 spills) preserve their original resource records exactly. No new spill or wave loss.

## GPU equality

The experimental baseline and shorter-lookahead versions each ran the `logits_gemm_vocab_iqr` selftest without a failure. The test-only harness captured 12 complete F32 head tensors across M=1, 8, 16, 33, 37, 48, 56, 64, 100, plus two offset and one single-class head cases. All 12 tensor pairs matched byte-for-byte, with zero differences. Testing used a separate one-unit plugin, so the harness skipped the unrelated diagnostic symbols that an unrestricted plugin exports. No test-hook modification is part of the candidate.

The final guarded library passed the full GPU byte gate separately: 12/12 tensor files byte-identical to the baseline, zero selftest failures (exit 0).

## Cold and warm GPU replay

`candidate-b-replay.csv` contains two separate orders of baseline/candidate experiments for K=5120 and vocabulary classes 4:1024,5:1024,6:4096,7:4096 at M=33,40,48,49,56,64. Both cold (256 MiB cache flushing, 12 event samples) and warm (32 launches) were measured by GPU events under queued leases. Geometric means of opposite-order timing ratios:

| M | NT | Cold candidate/base | Warm candidate/base |
|---:|---:|---:|---:|
| 33 | 3 | 0.6543 | 0.9989 |
| 40 | 3 | **1.0127** | **1.0013** |
| 48 | 3 | **1.0129** | 0.9919 |
| 49 | 4 | 0.8845 | 0.8328 |
| 56 | 4 | 0.9290 | 0.8579 |
| 64 | 4 | 0.9021 | 0.9374 |

**NT3 was rejected** as a measured cold-cache regression despite lower spills. **NT4 improved at all three tested row counts in both cold and warm paired-order aggregates.** Its final guarded code is identical to the corresponding unguarded experimental instantiation. The final guarded library compiled, passed byte identity and was separately timed at M=49/56/64 in cold and warm runs. All six cold and warm medians were no slower: cold final/base ratios 0.9394/0.9446/0.8906; warm 0.9984/0.9982/0.8510. The per-run data are in `candidate-b-final-replay.csv`. NT1–NT3 remain unchanged. These are kernel timings, not serving-level time measurements.

No serving run, model A/B or KLD was performed. Other IQR kernels were not modified by this candidate.
