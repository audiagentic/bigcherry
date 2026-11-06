# RAD23 lane gated quantise (branch perf/rad23-lane-quant, PR #482, 2026-11-01)

**What:** the gated activation and E4M3 requantise between the helper's gate/up and down run in a lane kernel that stops at the request's route count (R3_MOE_WORKER_LANE_QUANT), plus the few-rows lane kernel without the accumulator spill and a guard so the rotated form is taken only at a 128-multiple width.

**Outcome:** all of it is on main: the bounded gated quantise (1d74b1206), the spill-free few-rows kernel (25a0cd9e3), and the rotated-width guard (engines/radiance/components/moe-remote/moe_remote_gpu.h, `gated_quant_ != gated_had_ || mid_w() % kGroup == 0`). Measured on the RX 6900 XT 14 GiB tail: 616 to 28 us a request, replay hashes identical on 14 GiB, 4 GiB and XTX captures (PR #482 body).

**Why the PR was closed:** "Replaced by a single clean commit (a commit message named a machine)": the same change landed again without the machine name. Nothing remains to archive; no patch.
