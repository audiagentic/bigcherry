# RHA04 GPU0 native timed-cell smoke

Date: 2026-09-10  
Role: Brutus build server, GPU0 (`gfx1100`, `0000:03:00.0`)  
Runner: maintained `bench/run_bench.py --bench-type server-bench`  
Status: **cell completed; exploratory single sample**

This cell used the pushed identity-bound preflight and then timed the same
binary/model/common arguments with diagnostics unchanged. It used
`HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, `-sm none`, `--fit off`,
`-ngl 99`, `-c 4096`, and the valid harness selectors
`pp512,pp2048,tg128,tg512`. Attestation was verified and SIGINT teardown was
clean (`requested=true`, `forced=false`, return code 0).

Observed metrics (one repetition, not a parity estimate):

| metric | tokens/s |
| --- | ---: |
| pp512 | 1198.19 |
| pp2048 | 2460.25 |
| tg128 | 84.83 |
| tg512 | 85.36 |

The cell remains `performance_admitted=false`; it is a smoke/contract result,
not evidence for native-vs-BC parity. Full balanced native/replay/stock matrix
and uncertainty analysis remain required by RHA04.

Raw machine-readable files are retained in this directory. The server log is
retained on Brutus under `/tmp/rha04-timed-gpu0-native3/pair-001-native/` and
is not committed as a release artifact.
