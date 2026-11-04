# Radiance lab operations

Start with the [Radiance runbook](../../../docs/reference/engines/radiance/RUNBOOK.md) for production/lab source composition, launcher arguments, plan-check cache, layouts, and exit codes. The lab source is **production plus evaluated fixes and explicit experiments**; the validated remote-MoE hand-off is already in production. Build without a hand-off experiment:

```sh
PYTHONPATH=tools python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance
```

`lab-build.sh` provides `rad_build LOG` to sourced lab scripts. It reuses builds only for matching recorded engine/config trees and ROCm SDK, requiring both the engine and helper binaries. For an explicit build, give `--build DIR` to `launch-flashnext-best.sh`.

## Launch and quality

Hold the exclusive **gpu lease** for launches and measurements, not just the benchmark request: the first launch of a new configuration measures the engine's memory plan with helpers running (**about 1–2 minutes**). Every start arms `--plan-check` even when a cached plan exists, waits for `/health`, then runs the mandatory start canary. The launcher returns **96** for a failed canary, **97** for plan measurement/storage/setup failure, or **98** for a held launch lock/already-running server. Stop before clearing `$BC_WORK_ROOT/flashnext/plan-cache` to force remeasurement.

Use `--layout NAME` (`bigcherry layout list|show`) for device composition.

The helper-free dense Qwen3.8-27B MXFP4 layout is `qwen27b-2xtx`: it selects both
`gfx1100` cards from the host inventory and launches one engine with two tensor
ranks, `libr3`, `--tp-wire wht6` and `--p2p off`. It needs no route-heat file or
remote-MoE helper. From the repository root, after sourcing
`tools/env/bigcherry-env.sh` and acquiring the GPU lease, start it with:

```sh
bash tools/lab/radiance/launch-flashnext-best.sh start --layout qwen27b-2xtx
```

The dense path requires a successful fresh engine `--plan-memory` preflight and
the normal response canary. The interim `bc-serve` calibrated plan-check only
understands Flash-Next helper layouts, so it is **not** yet enforced for TP dense.
Do not treat this as a hardware-validated launch or a calibrated safety result.

The RDNA2 RX 6900 XT tail can improve placement capacity but has a known **worst-case quality penalty (RAD46)**; compare outputs against the two-XTX layout rather than inferring quality from throughput.

For KL quality, use `helper-kld.json` with `PYTHONPATH=tools python3 -m bigcherry.campaign.benchmark --server-config tools/lab/radiance/helper-kld.json`. Record the reference **once** with `quality.record` and `quality.corpus` present, then remove those keys and reuse the same reference directory for subsequent deterministic captures. See [patch validation](../../../docs/reference/testing/PATCH_VALIDATION.md). `helper-ab.json` is the speed harness; `fn-profile.sh` and `fn-profile-compare.py` support GPU kernel/copy traces.

## Comparative benchmark harness

Prefer the maintained `bigcherry ab-benchmark` for paired experiments. For a specialised engine/helper hardware bisect, follow the **structure**, not the host-specific paths, of the RHP11 C2 `run.sh` and `analyse.py` reference:

1. Pin **each arm** to a fixed commit in its own throwaway **detached worktree**. Never build from or reset the live slice worktree. Record the commit, build directory, source/composition and settings; clean up only the throwaway worktrees.
2. Build **all arms before acquiring the gpu lease**. Coordinate with concurrent build/measurement work using a **hold file**: the runner signals `builds-ready`, then waits while `hold` exists; the other owner clears it only when no build can overlap the measurement. Acquire the lease after the hold is cleared.
3. Rotate arm order (for example ABC/CBA/ABC); use equal workloads and temperatures, check each start/stop, and collect per-rep output, helper logs and activation counts. When isolating engine versus helper costs and the shared-memory ABI/layout is unchanged, **swap only the helper binary** between builds.
4. In the analyser, assert that **every arm has exactly one nonempty completion text SHA-256 hash and that it equals main's single hash**. Fail otherwise; an aggregate `all ok` line is not a correctness assertion.
5. Report **decode** (engine step ms and tokens/s) and **prefill** (TTFT or prefill tokens/s) **separately**, by stream count. Give medians, ranges, sample counts, per-position comparisons, and identified outliers. Occasional whole-rep stalls of about **+60% occur on main too**; explicitly identify/exclude stalled reps under a stated rule rather than treating them as a patch effect.
6. Bisect helper features **cumulatively** (add one component at a time), and use engine `--profile-ops` plus host `perf` to distinguish host scheduling from device time. Retain raw per-rep evidence and a summary of the caveats.

Do not copy the reference harness's absolute paths, device identifiers or mutable head selections into new runs. The benchmark skill (`.claude/skills/bigcherry-benchmark/SKILL.md`) describes the general attribution rules.
