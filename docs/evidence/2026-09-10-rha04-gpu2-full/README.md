# RHA04 GPU2 full 9B three-arm capture

Date: 2026-09-10  
Model: Qwen3.5-9B Q6_K  
Topology: GPU2, gfx1201, PCI-BDF `0000:09:00.0`  
Runner: maintained `bench/run_bench.py --bench-type server-bench`  
Schedule: six balanced permutation rounds, stock/native/replay, 18 cells  
Status: **complete exploratory capture; performance admission remains false**

All 18 cells passed physical attestation and completed with clean SIGINT
teardown. The timed process retained production arguments; only the untimed
preflight used the diagnostic verbosity delta.

Median metrics across the six samples per arm:

| arm | pp512 | pp2048 | tg128 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| stock | 1240.345 | 2246.085 | 67.745 | 68.255 |
| native | 1388.285 | 2793.680 | 67.815 | 68.095 |
| replay | 1388.955 | 2791.335 | 67.825 | 68.310 |

Direct paired replay-versus-native effects (10,000 bootstrap resamples):

| metric | effect | 95% interval |
| --- | ---: | ---: |
| pp512 | +0.451% | -0.438% to +1.496% |
| pp2048 | -0.717% | -2.500% to +0.322% |
| tg512 | +0.409% | +0.169% to +0.684% |

The stock prompt-processing baseline is substantially below both BC arms
(about 11% lower), while native and replay are close. This is a stock
execution/build anomaly that must not be interpreted as a BC tuning gain.
Replay is modestly faster in tg512 here, but this remains topology-specific
and unadmitted pending work-equivalence/replay-activation evidence.

Raw machine-readable artifacts are `run.json` and `config.json`. Per-cell
logs remain on Brutus under `/tmp/rha04-gpu2-full/` and are not committed as
release artifacts.
