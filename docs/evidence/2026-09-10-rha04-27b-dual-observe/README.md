# RHA04 dual-XTX 27B exploratory capture

Date: 2026-09-10  
Model: Qwen3.8-27B Q8_0  
Topology: XTX GPU0+GPU1, gfx1100, PCI-BDFs `0000:03:00.0` and
`0000:06:00.0`  
Runner: maintained `bench/run_bench.py --bench-type server-bench`  
Schedule: six balanced permutation rounds, stock/native/replay, 18 cells  
Status: **exploratory only; physical execution evidence missing**

The required capture was attempted first and failed closed before timing:
with `-sm tensor --fit off`, the server emitted only `Meta()`/layer-assignment
diagnostics and no ordered physical PCI locators for the two-device process.
That required failure is retained as `required-capture-failure.json`.

The exact same production-shaped workload was then run explicitly in
`execution_evidence=observe` mode to obtain engineering direction. All 18
cells completed with clean SIGINT teardown, but all 18 have
`execution_evidence_status=missing`; none is decision-grade.

Median metrics across the six samples per arm:

| arm | pp512 | pp2048 | tg128 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| stock | 929.250 | 1287.315 | 34.000 | 34.190 |
| native | 929.815 | 1286.975 | 33.870 | 34.100 |
| replay | 929.065 | 1286.185 | 34.175 | 34.265 |

Direct paired replay-versus-native effects (10,000 bootstrap resamples):

| metric | effect | 95% interval |
| --- | ---: | ---: |
| pp512 | -0.056% | -0.377% to +0.298% |
| pp2048 | -0.024% | -0.185% to +0.131% |
| tg512 | +0.657% | +0.411% to +0.990% |

These measurements indicate near-native prompt parity and a small replay tg512
increase, but they cannot establish a production result until the dual-device
physical attestation gap is solved. Do not promote replay from this capture.

Raw machine-readable artifacts are `run.json`, `config.json`, and
`required-capture-failure.json`. Per-cell logs remain on Brutus under
`/tmp/rha04-27b-observe/` and are not committed as release artifacts.
