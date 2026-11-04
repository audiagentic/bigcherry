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

## Kernel A/B before end-to-end A/B (KAB01)

Test proposed kernel changes first with tools/lab/radiance/kbench_ab.py on a
single recorded fixture, then run an end-to-end benchmark only for plausible
kernel improvements. Build both libraries before taking the GPU lease. The
caller holds the lease for the entire timing pass: this driver never acquires
or releases it and does not launch a server.

Provide sides.json with exactly two entries, a and b, each containing a unique
name, kernels (list or colon-delimited string), radiance_home or build_dir, and
optional env overrides. For a build directory, bin/rad-kbench and
radiance_home/ must exist. Example:

    {
      "a": {"name": "baseline", "radiance_home": "build-a/radiance_home",
            "kernels": ["libr3", "libref"]},
      "b": {"name": "candidate", "build_dir": "build-b",
            "kernels": ["libr3d", "libref"]}
    }

The caller must choose a finite, positive timeout in seconds for **each**
rad-kbench invocation. There is no default timeout. Use a one-case probe
under the caller-held GPU lease to measure one operation and size the full
run; the probe runs only side a, only one --op, and forces --max-cases 1:

    PYTHONPATH=tools python3 tools/lab/radiance/kbench_ab.py probe \
      --sides sides.json --fixture fixture.rkb --out probe-results \
      --device 0 --side a --op gemm_nt --timeout-s "$PROBE_TIMEOUT_S" \
      --seed 23 --bench-flush 256

The probe prints its wall seconds and Speed row count. It writes a per-op
report, logs, run.json and progress.log. It uses --timing-only and does not
perform a correctness check. After acquiring or continuing to hold the lease,
run the full A/B experiment from the repository root:

    PYTHONPATH=tools python3 tools/lab/radiance/kbench_ab.py run \
      --sides sides.json --fixture fixture.rkb --out results \
      --device 0 --op gemm_nt --op gemm_nt_q --seed 23 \
      --max-cases 4 --bench-flush 256 \
      --timeout-s "$PER_OP_TIMEOUT_S" --repeat-a

    PYTHONPATH=tools python3 tools/lab/radiance/kbench_ab.py run \
      --sides sides.json --fixture fixture.rkb --out results \
      --device 0 --op gemm_nt --op gemm_nt_q --seed 23 \
      --max-cases 4 --bench-flush 256 \
      --timeout-s "$PER_OP_TIMEOUT_S" --repeat-a --resume

    PYTHONPATH=tools python3 tools/lab/radiance/kbench_ab.py compare \
      results/baseline results/candidate \
      --again results/baseline.again \
      --graph graph.txt --rows 4 --out results/ab

One invocation is executed for each requested op on each side, in sequence,
followed by an optional repeat of the first side. Reports are written to
results/<side>/<op>.report.md, and repeat reports to
results/<side>.again/<op>.report.md. Each invocation has its own timeout;
failures and timeouts are recorded in run.json and do not stop later ops.
progress.log has one immediately flushed tab-separated line for each
finished invocation: side, op, wall seconds and exit code (or timeout).
Each operation also has its own stdout/stderr logs in its side directory.
The runner returns a nonzero status if any invocation in that pass fails.

Use --resume with the *same* fixture, binaries, side definitions, ops and
benchmark settings to skip existing **nonempty** reports; empty or missing
reports are retried. The timeout may be resized between attempts. A nonempty
report is skipped even if its earlier process failed: inspect run.json before
accepting it. Resume refuses conflicting run settings to avoid mixed evidence.
Comparison accepts either a single .report.md on each side (existing
behaviour) or a directory of per-op reports, including mixed forms, and
checks fixture provenance on **every** report. It explicitly lists operations
missing from either side, including ops absent on **both** sides when a
matching run.json provides the expected op list. Otherwise, supply repeatable
--expected-op flags to compare for an explicit expected set. Incomplete or
malformed reports fail closed.

The run uses --timing-only, which DOES NOT check correctness or produce
margins. Obtain checked rad-kbench --bench --report results separately under
the lease before interpreting a speed gain as a correctness-validated win.

The comparison reuses kbench_step.py speed parsing and graph matching.
Without --graph, the summary uses equal calls per matched case and says so.
With --graph, it uses actual graph instance counts at the largest timed
M/q_len step size not above --rows, excluding vision as in kbench_step.py.
It lists unmatched graph instances and all A-only or B-only rows. Different
kernel names, shapes and operand footprints are NOT joined speculatively.
The time-weighted B/A is sum(calls * B_us) / sum(calls * A_us).
Failed or vacuous cases cannot be wins. A/B/A noise is measured from the
per-case A vs A-again spread, never assumed from a fixed percentage.

Fixture recording is separate (rad-kbench --skeleton/--record). Keep raw
fixtures, graphs, run.json, logs, and reports in the measurement work root,
not in git. run.json stores exact child argv, explicit child environment,
wall time, exit code, and report path; its runtime paths are local by design.
Do not place credentials in side environment overrides. Use equal fixture,
seed, warm-up, flush footprint, filters and runtime policy on both sides.
An omitted flag uses rad-kbench's own default, so specify it to make a
measurement fully pinned.
