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

## Build result on Brutus (2026-10-09)

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
