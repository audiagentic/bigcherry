# b10901 -> b11126 throughput (post-bump regression check)

**Question.** The single post-bump smoke at b11126 (Brutus RX 7900 XTX) showed
pp512 3428.6 / tg128 85.8 t/s vs 4174.5 / 100.2 at b10901. Is that real, and
is it upstream or BigCherry?

**Arms** (same GPU 0 gfx1100, same model tierA Qwen3.5-4B Q6_K, `-p 512 -n 128
-r 3 -ngl 99`, llama-bench exits normally):

| Arm | Source | Build |
| --- | --- | --- |
| U10901 | llama.cpp 28ff09582 (b10901), no patches | stock, CMake options identical to U11126 |
| U11126 | llama.cpp b1ff4ca23 (b11126), no patches | `llama-native:stock:linux-multi` |
| B10901 | bigcherry release composition at b10901 | `bigcherry:native:linux-multi` (build plan 79170e04) |
| B11126 | bigcherry release composition at b11126 | `bigcherry:native:linux-multi` (build plan 130f3908) |

Composition read from the builds: stock arms' CMake options identical; the two
BigCherry arms differ only in their build-local generated-inputs path;
`build_commit` in every row confirms the source revision.

**Design.** 8 rounds, arm order rotated each round (every arm in every position
twice). Harness: `tools/lab/bump-b11126-regression/run_arms.py`. Raw rows:
`results.jsonl` (sha256 ca9a37fa7594a6bec4a0822a97be13f6ca81e8c9527cb0d86029e30cc142cbb1).
Position drift: pp512 p0 0.994 .. p3 1.012 of arm mean (balanced out by
rotation); tg128 within +/-0.1%.

**Result** (means, n=8/arm, exact two-sided Mann-Whitney):

| Test | U10901 | U11126 | B10901 | B11126 |
| --- | --- | --- | --- | --- |
| pp512 t/s | 4550.0 | 4656.3 | 4455.2 | 4631.8 |
| tg128 t/s | 100.07 | 100.11 | 99.95 | 99.96 |

- Upstream b11126 vs b10901: pp512 **+2.34%** (p=0.003), tg128 +0.04% (p=0.88).
- BigCherry b11126 vs b10901: pp512 **+3.96%** (p=0.007), tg128 +0.02% (p=0.88).
- BigCherry vs upstream at the same pin: pp512 -2.08% (p=0.80) / -0.53%
  (p=0.72), tg128 -0.13% / -0.15% -- no measurable difference.

**Conclusion.** No regression: b11126 is faster for prompt processing (upstream
gain, carried through by BigCherry) and unchanged for generation. The single
b11126 smoke result was not reproduced in 32 controlled runs and is withdrawn
as a throughput signal; its cause was not determined (single uncontrolled
run; cold caches right after the build are the likely explanation).

**Not attributed.** MTP/server speculative-decode workloads, other GPUs
(gfx1201, gfx1030), other models, and tuned replay-cache performance were not
measured.
