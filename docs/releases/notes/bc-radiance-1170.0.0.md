# bc-radiance-1170.0.0

BigCherry on Radiance `v1.17.0` (upstream `19d4e8fe0e18`).

## Radiance v1.17.0

BigCherry's pinned Radiance moves from v1.16.0 to v1.17.0. All managed patches apply without change, and the libr3 self-test rewrite blocks follow upstream's new performance counters.

**Measured on the Flash-Next four-card layout (two alternating passes, temperature 0, fixed seeds):**
- Cold prefill of a ~48K-token prompt: 3,626-3,733 tokens/s, the same as v1.16.0 (3,720-3,739). At 1M context with YaRN, a ~396K prompt prefills at 3,212 tokens/s and ~783K at 2,458.
- Decode step time is level at four and six streams. At one and two streams it is about 1.7% longer (24.9 to 25.4 ms); per-kernel times are identical, so the gap is host-side and not yet explained.
- Draft acceptance is identical on RAD23's five prompts (list 96%/97%, code 87%/89%).

**Checks:** compile, plugin ABI, component tests, lane build and CPU smoke pass.

**Known limitations:**
- The two validated patches (remote-MoE hand-off, hyper-connection up-mix tile) are accepted by carry-forward from earlier pins (v1.14.0, v1.9.0), not re-measured at v1.17.0.
- The 6900 XT tail still shows one persistent outlier position in the KLD check, so its p99.9 limit is not met on this pin (mean, p99, top-1 and perplexity are within limits).
