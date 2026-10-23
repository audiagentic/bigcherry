# RHA04 GPU3 full 9B three-arm capture

Date: 2026-09-10  
Model: Qwen3.5-9B Q6_K  
Topology: GPU3, gfx1030, PCI-BDF `0000:17:00.0`  
Runner: maintained `bench/run_bench.py --bench-type server-bench`  
Schedule: six balanced permutation rounds, stock/native/replay, 18 cells  
Status: **complete exploratory capture; performance admission remains false**

All 18 cells passed physical attestation and completed with clean SIGINT
teardown. Timed processes retained production arguments; only untimed
preflights used the diagnostic verbosity delta.

Median metrics across the six samples per arm:

| arm | pp512 | pp2048 | tg128 | tg512 |
| --- | ---: | ---: | ---: | ---: |
| stock | 905.130 | 1281.160 | 55.260 | 55.815 |
| native | 892.860 | 1279.880 | 55.290 | 55.700 |
| replay | 888.725 | 1281.450 | 55.250 | 55.785 |

Direct paired replay-versus-native effects (10,000 bootstrap resamples):

| metric | effect | 95% interval |
| --- | ---: | ---: |
| pp512 | -0.800% | -2.299% to +0.482% |
| pp2048 | +0.018% | -0.210% to +0.177% |
| tg512 | +0.033% | -0.251% to +0.318% |

Replay is slightly lower in pp512 but statistically overlaps native; the
other measured replay/native effects are effectively flat. This is not a
universal replay promotion and remains unadmitted pending work-equivalence and
replay-activation evidence.

Raw machine-readable artifacts are `run.json` and `config.json`. Per-cell
logs remain on Brutus under `/tmp/rha04-gpu3-full/` and are not committed as
release artifacts.
