# Radiance lab operations

Start with the [Radiance runbook](../../../docs/reference/engines/radiance/RUNBOOK.md) for production/lab source composition, launcher arguments, plan-check cache, layouts, and exit codes. The lab source is **production plus evaluated fixes and explicit experiments**; the validated remote-MoE hand-off is already in production. Build without a hand-off experiment:

```sh
PYTHONPATH=tools python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance
```

`lab-build.sh` provides `rad_build LOG` to sourced lab scripts. It reuses builds only for matching recorded engine/config trees and ROCm SDK, requiring both the engine and helper binaries. For an explicit build, give `--build DIR` to `launch-flashnext-best.sh`.

## CI-seeded ccache (BU01)

A successful **main** Radiance CI build publishes a seven-day `radiance-ccache-main` artifact. It includes a compiler fingerprint, compressed ccache entries and `SHA256SUMS`. To use it before a branch kernel edit/build:

```sh
# Requires GitHub CLI access to this repository, ccache, GCC/G++ 14 and ROCm.
bash tools/lab/radiance/seed-ci-ccache.sh
# Or choose a particular successful main build:
bash tools/lab/radiance/seed-ci-ccache.sh --run-id RUN_ID
```

The importer refuses a non-main run, a bad archive SHA-256, an unsafe archive member or any mismatch in the exact host/HIP/ccache executable hashes, ROCm version, or recipe files. **A refusal is not a cache hit**; install the same toolchain as the CI image (or build in that image) before retrying. No branch PR compiler objects are published as main cache artifacts. Imported files live only in `BC_WORK_ROOT/.ccache`. The lab `rad_build` function sets `CCACHE_BASEDIR` to the checkout, `CCACHE_NOHASHDIR=true`, and `CCACHE_COMPILERCHECK=content` to match CI. It resets ccache statistics before a real compile and appends `ccache -s` to the build log after it, so hits, misses and elapsed build times can be measured on the same source edit.

For offline verification, `python3 tools/lab/radiance/ci_ccache.py import --source ARTIFACT_DIR --work-root WORK_DIR` accepts a previously downloaded artifact with the same validation. It never downloads an engine runtime or substitutes a binary. The `rad_build` source-tree cache check remains independent; an edit to engine/config always requires a build. Measured CI-to-lab hit rate and under-10-minute target remain to be established on a matching toolchain.

## Verified release bundles (BU01)

The release record phase resolves and builds the **current** `radiance:stock:linux-radiance` lane before packaging; it will not reuse the historical pin-bump receipt after a patch promotion. `patch-promote --release` defers tagging until the exact release asset has been staged through `pin-release`. The next release is the first opportunity to validate real GitHub asset upload and stock-host reuse.

After a new Radiance `pin-release`, the release record's `runtime_assets[bc-radiance-<version>]` identifies the exact pin-bump lane build plan, source-tree IDs, ROCm SDK version, binary SHA-256, manifest SHA-256 and tarball SHA-256. The release phase attaches a deterministic `radiance-<plan>.tar.gz` and matching `.sha256` file to the GitHub release; retries verify an existing attachment rather than overwriting it.

```sh
# Stock pinned Radiance checkout, using GitHub CLI and the configured ROCm SDK:
bash tools/lab/radiance/fetch-release-bundle.sh
# A specific recorded release:
bash tools/lab/radiance/fetch-release-bundle.sh --tag bc-radiance-1160.0.0
```

The command first attempts to download and verify both release assets, the entire runtime file manifest, source/config tree identity, compiler platform version and the recorded engine executable. It extracts into a staging directory and atomically moves the validated files under `BC_WORK_ROOT/builds/<source-slice>/<plan>`; `rad_build` then reuses it without compiling. A missing asset, digest mismatch, SDK drift or branch changes means **no prebuilt binary is trusted**: the wrapper falls back to `rad_build` and logs the compile to `artifacts/logs/manual/bu01-fallback-build.log`. Unlike `rad_build` itself, the fetch command requires a network connection and GitHub CLI credentials. It never replaces an existing build-store directory. This prebuilt path remains unvalidated against a published GitHub release until the next release run.

## Launch and quality

Hold the exclusive **gpu lease** for launches and measurements, not just the benchmark request: the first launch of a new configuration measures the engine's memory plan with helpers running (**about 1–2 minutes**). Every start arms `--plan-check` even when a cached plan exists, waits for `/health`, then runs the mandatory start canary. The launcher returns **96** for a failed canary, **97** for plan measurement/storage/setup failure, or **98** for a held launch lock/already-running server. Stop before clearing `$BC_WORK_ROOT/flashnext/plan-cache` to force remeasurement.

Use `--layout NAME` (`bigcherry layout list|show`) for device composition. The RDNA2 RX 6900 XT tail can improve placement capacity but has a known **worst-case quality penalty (RAD46)**; compare outputs against the two-XTX layout rather than inferring quality from throughput.

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
