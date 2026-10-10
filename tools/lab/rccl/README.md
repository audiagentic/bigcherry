# RCCL / inter-GPU transport lab

Plan item: PGC14 (RCCL transport), PGC11 (AllReduce wire)
Status: active
Owner: bigcherry
Question state: open

## Question

How fast can the dual-XTX (and 3/4-GPU) AllReduce transport go on Brutus, and which RCCL settings are best?
Established 2026-10-02: on the `7.0.13-dirty` P2P-hack kernel, XTX<->XTX direct P2P ran at 0.9 GB/s and
host<->GPU at 7.5/8.5 GB/s; on stock `7.0.13-070013-generic` P2P is off (RCCL uses SHM) and host<->GPU is
13.3/14.3 GB/s, giving 27B pp4096 1511 t/s vs 918.

## Inputs

- `p2p-check.hip`, `h2d-check.hip`: HIP peer-access/peer-copy and pinned host<->device bandwidth per GPU pair.
- `rccl-env-sweep.sh` (+ `queue-rccl-env-sweep.sh`): llama-bench pp512/pp4096/tg128 on Qwen3.8-27B Q8_0
  across RCCL env settings (SHM copy engine, protocol, channels, buffer size), two reversed-order passes.
- `build-rccl-nohostcall.sh` (+ queue): hostcall-free RCCL build for the 6900 XT (no PCIe atomics).

## Outputs

Run outputs under `/mnt/data/bigcherry-work/runs/<run-name>/` on Brutus; nothing in the repo.

## Runtime

GPU required: yes
Real compilation required: yes (hipcc for the checks; campaign builds for the sweep)
Mutates canonical BigCherry state: no

## Safety

- Canonical-state mutation: none.
- Queue jobs only (host + GPU locks; PCIe link preflight); R9700/vLLM untouched by the dual-XTX jobs.
- Do not import this experiment from `bigcherry` production, tests, or maintained analysis.

## CE failure gate (2026-10-10)

Independent commit `9ded4c6e84b1` reported `NCCL_SHM_USE_CUDA_MEMCPY=1` hanging `llama-bench` on 2026-10-02. It is historical failure evidence, not a current-pin throughput measurement. RCCL `NCCL_PARAM` requires the `NCCL_` prefix; the existing sweep already uses it and starts a separate process per arm.

`rccl-env-sweep.sh` now writes `status.tsv` (`pass, arm, bench_exit, csv_exit`) and fails the overall run if any arm times out, crashes or emits invalid CSV. `SWEEP_DONE` means all arms passed; otherwise `SWEEP_INCOMPLETE`. The previous script could report success after a failed arm. No hardware rerun occurred in this audit.

Before any new CE experiment, check Brutus host/GPU queue, force RCCL rather than adaptive host AllReduce, verify actual SHM and Simple protocol, and test direct, send-only, receive-only and both in bounded separate processes. Do not implement the prior unsafe PGC14 FIFO coalescing sketch. The gfx1030 hostcall lane is separate.

## Disposition

Open. Winning RCCL settings graduate into the production launch profile / adaptive AllReduce defaults via
a separate validated change.
