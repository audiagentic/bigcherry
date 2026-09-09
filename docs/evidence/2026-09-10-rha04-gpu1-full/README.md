# RHA04 GPU1 full 9B three-arm capture

Date: 2026-09-10  
Model: Qwen3.5-9B Q6_K  
Topology: GPU1, gfx1100, PCI-BDF `0000:06:00.0`  
Runner: maintained `bench/run_bench.py --bench-type server-bench`  
Schedule: six balanced permutation rounds, stock/native/replay, 18 cells  
Status: **complete exploratory capture; performance admission remains false**

All 18 cells passed the identity-bound attestation preflight and completed
with clean SIGINT teardown. The timed process received the same production
arguments as its untimed preflight; only the preflight added `--verbosity 5`.

Median metrics across the six samples per arm:

| arm | pp512 | pp2048 | tg128 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| stock | 1303.420 | 2519.355 | 84.385 | 85.080 |
| native | 1296.765 | 2526.685 | 84.490 | 85.185 |
| replay | 1299.100 | 2529.420 | 83.835 | 84.755 |

Direct paired replay-versus-native effects (10,000 bootstrap resamples):

| metric | effect | 95% interval |
| --- | ---: | ---: |
| pp512 | +0.205% | -0.841% to +1.195% |
| pp2048 | +0.063% | -0.165% to +0.242% |
| tg512 | -0.464% | -0.587% to -0.303% |

GPU1 does not reproduce GPU0's pp512 regression: replay is near native for
prompt processing, while generation is consistently about 0.46% slower. This
is still not a universal replay win and remains unadmitted pending the
work-equivalence/replay-activation gate and the remaining GPUs/topology.

Raw machine-readable artifacts are `run.json` and `config.json`. Per-cell
logs remain on Brutus under `/tmp/rha04-gpu1-full/` and are not committed as
release artifacts.
