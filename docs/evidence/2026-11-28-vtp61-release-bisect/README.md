# VTP61: TP3 short-prompt throughput, 2020.8.5 to 2020.24.1

Question: the stored 2020.8.5 record for qwen27b TP3 at eight streams, 512/192, is 360.8 aggregate t/s, and installed
2020.24.1 measured 351.3. Which release introduced the loss?

Answer: none. A fresh 2020.8.5 build is level with 2020.11.4 and 2020.24.1 on decode step time, TTFT, draft acceptance
and running requests. The stored 360.8 does not reproduce on the 2020.8.5 code. Verdict: accept, no fix item.

## Cohort and settings

- Layout qwen27b-tp3 (XTX, XTX, R9700), Qwen3.8-27B IQ8R with the DFlash drafter.
- Every arm runs the 2020.8.5 shipped native config:
  - exact wire, speculation depth 3, max-num-seqs 8, max-num-batched-tokens 2048;
  - pipelined prefill on with min rows 256, the all-reduce tune pin, the 6/6/4 ownership split;
  - fp8 KV, prefix cache as shipped.
- Exceptions to that config:
  - 2020.24.1 lists libbciqr ahead of libr3. Since 2020.17 the IQR rows live only there.
  - 2020.24.1 ships power profile `efficient`. It is pinned back to `stock` here, as in every other arm.
- Workload: stream-bench at 8 streams, 512-token prompts, 192 new tokens, temperature 0, 3 repetitions per seed, seeds 1 to 3.
- The seed-1 prompt SHA-256 matches the historical cohort. `summary.json` lists the hashes.
- Order: ABAB, one managed start per arm, one short gpu lease per arm. Runs on 2026-11-28, 15:45 to 16:17 AEDT.

## Build identities

| arm | release | how it was obtained | archive or build | engine sha256 (first 16) |
|---|---|---|---|---|
| b085 | bc-radiance-2020.8.5 | local release-lane build (radiance:stock:linux-radiance) from a clone at the tag; no archive survives | build plan 339bada30607693032f63857191c77b4, radiance pin b6b8751 | c27b3d20e0255697 |
| r114 | bc-radiance-2020.11.4 | cached tag download with its receipt, `bc-release install` into an isolated root | bigcherryrad-6fe8e142f41d935cc88fa95dcdaf11de | 3f62658361090642 |
| r241 | bc-radiance-2020.24.1 | release assets, `bc-release install` into an isolated root | bigcherryrad-2512de74e548b195140341c6ab17aa5b | 6f7d7dd8a5151057 |

Full engine and bc-serve hashes are in `summary.json`.

## Results

Medians over 3 seeds × 2 launches (n = 6 per arm). Aggregate t/s is context only. Across versions it follows TTFT
and draft acceptance.

| probe | arm | decode ms/step | TTFT max ms | draft acceptance | running requests (mean) | event gap p95 ms | aggregate t/s |
|---|---|---|---|---|---|---|---|
| 1 | 2020.11.4 | 31.41 | 2227 | 0.708 | 5.00 | 39.3 | 350.0 |
| 1 | 2020.24.1 | 31.72 | 2253 | 0.711 | 4.97 | 39.9 | 347.2 |
| 2 | 2020.8.5 (fresh build) | 31.86 | 2261 | 0.714 | 4.97 | 39.8 | 346.7 |
| 2 | 2020.11.4 | 31.36 | 2260 | 0.708 | 4.98 | 39.4 | 348.0 |
| power pair | 2020.24.1 stock | 31.73 | 2260 | 0.704 | 5.00 | 39.9 | 346.1 |
| stored record | 2020.8.5 (T1/T2 seed 1) | 31.9-32.1 | about 2050 | 0.70-0.72 | 5.1-5.6 | 38.5 | 360.8 |

- Decode ms/step spread within an arm is 1.5 to 2.0 ms. One 2020.24.1 run (seed 2) read 37.2 ms with a TTFT of 1632 ms.
- Seed 1 alone, aggregate t/s:
  - fresh 2020.8.5: 353.6 and 351.5;
  - 2020.11.4: 348.6 to 352.7;
  - 2020.24.1: 348.4 and 353.3.

## Conclusion and caveats

- The versions do not differ in decode step time or TTFT. The stored record's extra throughput comes mostly from a
  lower TTFT max, about 2050 ms against about 2250 ms today, which gives more running requests per step. That
  difference appears on the 2020.8.5 code too, so no release in 2020.8.5..2020.24.1 introduced it.
- The 2020.8.5 arm is a local lane build. The CI archive the record ran on is gone: GitHub retention removed the
  bundles of most releases in this range. Only 2020.11.1 and 2020.20.0..2020.24.1 keep one, and 2020.11.1 has no
  receipt, so `bc-release install` refuses it.
- Power was not measured. 2020.24.1's bc-serve refuses to apply the `efficient` profile without a qualified apply,
  readback and restore receipt for these boards (BC_CFG_CALIBRATION_INVALID). The qualification was deliberately not
  run, because the power-cap profile is an open owner decision. Production serves `stock` in observe mode, so power is
  not part of this gap.
