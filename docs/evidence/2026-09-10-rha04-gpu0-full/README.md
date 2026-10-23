# RHA04 GPU0 full 9B three-arm capture

Date: 2026-09-10  
Model: Qwen3.5-9B Q6_K  
Topology: GPU0, gfx1100, PCI-BDF `0000:03:00.0`  
Runner: maintained `bench/run_bench.py --bench-type server-bench`  
Schedule: six balanced permutation rounds, stock/native/replay, 18 cells  
Status: **complete exploratory capture; performance admission remains false**

All 18 cells passed the identity-bound attestation preflight and completed
with clean SIGINT teardown. The preflight used the same binary/model/common
arguments and visibility as each timed arm, adding only `--verbosity 5` in the
untimed process. No cell failed and no forced termination occurred.

Median metrics across the six samples per arm:

| arm | pp512 | pp2048 | tg128 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| stock | 1290.355 | 2512.050 | 84.300 | 84.985 |
| native | 1279.175 | 2520.325 | 84.265 | 85.030 |
| replay | 1268.820 | 2523.355 | 84.260 | 84.725 |

Direct paired replay-versus-native effects (10,000 bootstrap resamples):

| metric | effect | 95% interval |
| --- | ---: | ---: |
| pp512 | -2.420% | -5.316% to +0.201% |
| pp2048 | +0.026% | -0.249% to +0.214% |
| tg512 | -0.360% | -0.935% to +0.385% |

This GPU0 result does not establish a global conclusion: replay is clearly
not a pp512 win here, while pp2048 and tg512 are statistically near parity.
The capture remains unadmitted because the existing admission contract still
requires complete work-equivalence/replay-activation proof. It is suitable as
current-source engineering evidence and must be followed by the other GPUs
and the matched dual-XTX 27B run.

Raw machine-readable artifacts are `run.json` and `config.json`. Per-cell
logs remain on Brutus under `/tmp/rha04-gpu0-full/` and are not committed as
release artifacts.
