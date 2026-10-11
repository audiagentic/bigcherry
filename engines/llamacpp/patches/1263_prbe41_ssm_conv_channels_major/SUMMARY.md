# 1263_prbe41_ssm_conv_channels_major

**Status:** untested (historical validation records exist; no b11474 promotion)
**Plan item:** PRBE41; tensor-split split-state dependency PRBE114

## Actual implementation

Adds CPU and CUDA/HIP channels-major SSM_CONV, replacing the single-GPU Qwen3.5/Qwen3Next conv-state assembly with `[channels,time,seq]` concat on time axis. Under `LLAMA_SPLIT_MODE_TENSOR`, patch 1263 **declines** channels-major and uses the pristine time-major graph; its `ssm_conv_channels_major_declined_split` marker is a negative control, not a positive kernel activation. Saved conv-state layouts cannot be interchanged across subject/control.

The patch guards CANN, Hexagon, OpenCL, SYCL, Vulkan and WebGPU; the AMD fork's Metal rejection was not ported. Do not qualify unsupported Metal execution.

## Source and performance evidence

Port of [AMD-Ecosystem llama.cpp #52](https://github.com/AMD-Ecosystem/llama.cpp/pull/52), merged 2026-07-16. That PR's gfx1151 pp4096 roofline attributed its largest change to **CONCAT 1166 -> 34 ms**; CONT was 10 -> 10 ms; total prefill 3825 -> 2717 ms. The old summary's “CONT transpose ~221 ms” is not supported by that table.

BigCherry `evidence/validation.json`: 24 b11126/b11233 historical records, with withdrawals separately tracked. Four validated gfx1100 b11233 pp512 sessions measured +1.13%..+1.43% geometric throughput, CI95-low +0.90%..+1.20%; the maximum historical gfx1201 CI95-low is +2.59%. **None meets the >=3% CI95-low promotion gate.** No new b11474 performance result exists.

## Disposition

Do not reimplement the op, alter Meta split ownership, or launch a duplicate QFP49 campaign. First measure exposed copy/concat cost on current-pin single GPU; reject if below the 2.9126% theoretical E2E floor. If it survives, run the existing producer and state/graph/MTP correctness matrix before paired performance qualification. See PRBE41 and TESTING.md.
