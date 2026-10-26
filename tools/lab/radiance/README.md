# Radiance: second-engine lab (MEN01, MEN02)

**Question.** Can standalone radiance (codeberg.org/StillDeadcode/radiance, Apache-2.0, C++/HIP) be built and run from
source on the lab host, on which of our cards, and how fast is it measured our way? Plan group `run-multi-engine`.

**Why.** On one R9700 with a 4-bit Qwen3.8-27B the vLLM + radiance container prefills about 2.3-2.9x faster than
BigCherry llama.cpp (2026-10-09, `tools/lab/reference-vllm/`). Radiance is the engine behind that result and now a
standalone server with a llama-server-compatible API.

## Files

- `build.sh`: configures and builds radiance at the checked-out (or given) commit for `gfx1201`, runs its card-free
  tests, and records commit, toolchain, binaries and plugins. Run as a queue `SCRIPT` job.

- `fetch-gcc14.sh`: unpacks Ubuntu's g++-14 packages into a private directory for hosts that only have g++ 13
  (radiance does not build with g++ 13 or with ROCm's clang as host compiler). Nothing is installed system-wide.

- `run-radiance.sh`: serves a model from that build on the R9700 alone (`--tp 1`, fp8 KV) and measures it with
  `bigcherry engine-bench --engine radiance` at the reference-lane depths, with the drafter on and off. Each arm
  writes `<arm>.engine-bench.json`, the same record `run-llamacpp-r9700.sh` writes for llama.cpp.

- `helper-ab.json`, `helper-kld.json`: `ab-benchmark --server-config` examples for the R9700 engine with two XTX helper
  sidecars: speed (R3_MOE_WORKER_ROWS 0 vs 1) and KL quality (MEN11). They replaced the `fn-r9700-engine.sh`,
  `fn-helper-gate.sh`, `fn-helper-kld.sh` and `fn-r9700-kld.sh` lab scripts.

- `fn-helper.sh`, `fn-profile.sh`, `fn-profile-compare.py`: rocprofv3 kernel/copy traces of one prefill and their
  comparison; the harness does not trace, so these stay.

- `rad20-barrier-wake.cpp`, `rad20-barrier-wake.sh`: RAD20 option B measurement. How long after the R9700 (or a host
  thread) releases a wait on an XTX queue does the next XTX kernel start? Arm A waits with an AQL `BARRIER_AND` on an
  HSA signal (raw HSA queue), arm B with `hipStreamWaitValue32` on host-registered shared memory; single process and
  the production two-process split (helper creates an IPC signal, engine attaches it). Needs `R9700_BDF` and
  `XTX_BDF`; hold the `gpu` lease. In two processes `hsa_amd_ipc_signal_attach` maps the signal for the host only,
  so the engine attaches the signal's page again with `hsa_amd_ipc_memory_attach` for its R9700 kernel's store.
  `--dump FILE` writes one CSV row per iteration. In two processes the first arm released by an R9700 kernel,
  whichever wait it uses, pays a one-time warm-up: about 350-530 us on its first iteration and 11-39 us for the next
  ~15 ms; a single process does not show it. Compare arms after that warm-up, or put a throwaway arm first.
  Run it with `bash` (lab scripts are not executable in the repo):
  `R9700_BDF=0000:09:00.0 XTX_BDF=0000:03:00.0 agent-lease acquire gpu --ttl 30m --purpose "RAD20 barrier wake" -- bash tools/lab/radiance/rad20-barrier-wake.sh`.
  Result (2026-10-26): the barrier wakes about 0.8 us slower than `hipStreamWaitValue32` at p50 (two processes, R9700
  release, n=2000: 7.55 vs 6.77 us), and a pending wait of either kind holds the XTX at about 96-98 W against 12 W
  idle, so option B saves no power while a wait is armed.

## Build result on dev (2026-10-09)

Builds in 104 s for gfx1201 with ROCm 7.2.4 and g++ 14.2 (unpacked by `fetch-gcc14.sh`). 38 of 39 card-free tests
pass; `plugin_test` fails because it probes device 0, an RX 7900 XTX, and the log states the reason: libr4d has code
objects for gfx1201 only, so its 201 device kernels are left out of selection on a gfx1100 card.

## Facts from the source (radiance 1.3.0, commit 89cee7ce)

- Kernel library `libr4d` covers gfx1200 / gfx1201 only. `libref` is the reference implementation and `libavx` the
  CPU host backend. A 7900 XTX (gfx1100) or 6900 XT (gfx1030) needs a kernel library plugin (`docs/PLUGIN.md`).
- Tensor parallelism (`--tp N`) needs every rank on a covered card, so this host runs `--tp 1` on the R9700.
- Flash-Next ships as `qwen3.8-next-flash-fp8-iq4r-moe.rad` (114 GiB) with serve flags written for two R9700s.
- Its `ggml` quantiser reuses llama.cpp's quantize functions, so GGUF quantisation choices carry over.

## Disposition

Experiment-only until MEN01 and MEN02 conclude. If radiance stays, these scripts move behind the engine adapter
(MEN03); if not, the topic is removed.
