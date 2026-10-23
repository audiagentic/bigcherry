# RCCL / inter-GPU transport lab

Plan item: PGC14 (RCCL transport), PGC11 (AllReduce wire)
Status: active
Owner: bigcherry
Question state: open

## Question

How fast can the dual-XTX (and 3/4-GPU) AllReduce transport go on dev, and which RCCL settings are best?
Established 2026-10-02: on the `7.0.13-dirty` P2P-hack kernel, XTX<->XTX direct P2P ran at 0.9 GB/s and
host<->GPU at 7.5/8.5 GB/s; on stock `7.0.13-070013-generic` P2P is off (RCCL uses SHM) and host<->GPU is
13.3/14.3 GB/s, giving 27B pp4096 1511 t/s vs 918.

## 2026-10-15 PGC14 coalescing disposition (BCOP219)

The 2026-10-06 in-model protocol/algorithm sweep already rejected Simple/LL/LL128/Tree/Ring as a performance lever (LL was ~3.6x slower). Do **not** requeue it. The untested question is whether the actual SHM CE proxy path is selected and exposes small-copy/proxy overhead; pinned RCCL already prints `via SHM/CE/direct` per channel. Record the selected channel, actual device/link topology and a matching run ID before requesting additional instrumentation. Direct vs CE mode results must not be inferred from standalone H2D/D2H throughput.

The former multi-slot coalescing sketch in PGC14 is **not safe to implement** without byte-contiguity, `sliceSteps==1`, destination FIFO-size publication, event/tail and slot-generation proofs. The host-only 9-case discriminator demonstrates two unsafe geometry cases and missing metadata; no RCCL build or GPU test was run. Do not queue coalescing unless the existing CE path is active, many adjacent full slots are ready, and a ≥3% optimistic E2E prefill opportunity is supported by measured traces. PGC15 token-tiled overlap remains separate. External gfx1030 AllGather issue ROCm/rocm-systems#12145 is a correctness-negative test to consider, not an observed BigCherry regression.

## Inputs

- `p2p-check.hip`, `h2d-check.hip`: HIP peer-access/peer-copy and pinned host<->device bandwidth per GPU pair.
- `rccl-env-sweep.sh` (+ `queue-rccl-env-sweep.sh`): llama-bench pp512/pp4096/tg128 on Qwen3.8-27B Q8_0
  across RCCL env settings (SHM copy engine, protocol, channels, buffer size), two reversed-order passes.
- `build-rccl-nohostcall.sh` (+ queue): hostcall-free RCCL build for the 6900 XT (no PCIe atomics).

## Outputs

Run outputs under `${BIGCHERRY_ROOT}/runs/<run-name>/` on dev; nothing in the repo.

## Runtime

GPU required: yes
Real compilation required: yes (hipcc for the checks; campaign builds for the sweep)
Mutates canonical BigCherry state: no

## Safety

- Canonical-state mutation: none.
- Queue jobs only (host + GPU locks; PCIe link preflight); R9700/vLLM untouched by the dual-XTX jobs.
- Do not import this experiment from `bigcherry` production, tests, or maintained analysis.

## Disposition

Open. Winning RCCL settings graduate into the production launch profile / adaptive AllReduce defaults via
a separate validated change.
