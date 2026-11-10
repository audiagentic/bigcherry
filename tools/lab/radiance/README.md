# Radiance lab operations

Start with the [Radiance runbook](../../../docs/reference/engines/radiance/RUNBOOK.md) for production/lab source composition, launcher arguments, plan-check cache, layouts, and exit codes. The lab source is **production plus evaluated fixes and explicit experiments**; the validated remote-MoE hand-off is already in production. Build without a hand-off experiment:

```sh
PYTHONPATH=tools python3 -m bigcherry build --lane radiance:stock:linux-radiance
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

Hold the exclusive **gpu lease** for launches and measurements, not just the benchmark request: the first launch of a new configuration measures the engine's memory plan with helpers running (**about 1–2 minutes**). Every start of a helper layout starts the engine and the helpers from the measured serving manifest and the engine checks the plan it carries, even when a cached plan exists, waits for `/health`, then runs the mandatory start canary. The launcher returns **96** for a failed canary, **97** for plan measurement/storage/setup failure, or **98** for a held launch lock/already-running server. Stop before clearing `$BC_WORK_ROOT/flashnext/plan-cache` to force remeasurement.

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
the normal response canary. The manifest-based plan check only
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


## VTP06 — original-BF16 27B IQ6R calibration and holdout qualification

Owner: VTP06 (experiment only; no production default or model changes).
Inputs: the original 64-layer Qwen3.8-27B BF16 checkpoint, existing
full 18-shard SHA-256 audit, tested isolated VTP11 host/device guard
engine, and a manually source/licence-reviewed JSONL dataset.
Outputs: an immutable corpus bundle, bounded layer-range receipts,
packed F32 Hessians and a separate provisional KLD evaluation receipt
under the configured BC_WORK_ROOT.

Lab commands:
  - xtx27_calib_corpus.py audit --input SOURCE.jsonl --mode candidate
  - xtx27_calib_corpus.py prepare --input SOURCE.jsonl --checkpoint CHECKPOINT --out BUNDLE --mode candidate
  - xtx27_calib_corpus.py verify --bundle BUNDLE --checkpoint CHECKPOINT
  - xtx27_calib_ranges.py plan --bundle BUNDLE --checkpoint CHECKPOINT --audit FULLSHA_RECEIPT --build EXACT_ISOLATED_BUILD --reference MEASURED_LAYER0_HESSIANS --out CAMPAIGN --work-root WORK_ROOT --chunk-size 2 --tp 2 --devices 0,1 --host-pool-mib 16384
  - xtx27_calib_ranges.py status --campaign CAMPAIGN --full-verify

On a genuinely idle GPU, first acquire the exclusive gpu lease with a
unique purpose beginning VTP06/, and configure only one planned range:
  - agent-lease acquire gpu --ttl 2h --purpose VTP06/27b-candidate-00-02
  - xtx27_calib_ranges.py run-range --campaign CAMPAIGN --lo 0 --hi 2 --attempt 1 --lease-purpose VTP06/27b-candidate-00-02
  - xtx27_calib_ranges.py verify-range --campaign CAMPAIGN --lo 0 --hi 2 --attempt 1
  - agent-lease release gpu

Release the GPU lease even if validation fails (shell trap/finally).
Never run a hidden multi-layer campaign loop or override another
agent's lease. Retry with a new attempt number; existing outputs are
immutable. The range runner checks free RAM, disk and exact model,
tokenizer, corpus and binary identities; kills only the subprocess
it owns on monitored failure. Full verification rehashes every
Hessian, including lm_head.input on the 62:64 run.

The original 2-document pilot lives at
${BC_WORK_ROOT}/xtx27-prefill/VTP06-pilot-two-doc-20261111/
and is NOT representative; no quality inference may use it.
The existing measured layer-0 output is 0.75 GiB; 64 layers project
to roughly 46.88 GiB before final head, safety margin or
checkpoint-wide variability. Plan-time approval never replaces a
fresh run-time RAM/GPU admission.

Once a native IQ6R model actually exists and is separately qualified
for kernel/ABI/dispatch, use Radiance's BF16 --kld-record --kld-top 128
on the independent BUNDLE/holdout.jsonl and the candidate's --kld-ref
--kld-out on the same holdout. Then call
xtx27_iq6r_quality.py --bundle BUNDLE --checkpoint CHECKPOINT
--reference-dir KLD_BF16_REFERENCE --report KLD_IQ6R_REPORT
--candidate IQ6R_RAD_FILE --policy PROPOSED_POLICY_JSON --out REPORT.
The reference is sparse top-128+tail, NOT full-vocabulary KL.
The policy is explicitly not owner-approved and no emitted status
alone can approve a production checkpoint. See VTP06 for acceptance.

## VTP12 prior-run FP8 versus INT8 comparison (card-free)

`xtx27_vtp12_compare.py` reads **existing** VTP05 `--perfw8`
receipts and logs. It checks the paired rows, geometry, per-GPU
plugin provenance, selected INT8 receipt and complete three-device
case groups. Inputs: `BC_WORK_ROOT/xtx27-prefill/quant-shape/
placement-w8a8-20261110/runs/`; output: optional separate JSON
report via `--output` under BC_WORK_ROOT. No GPUs, conversion,
serving or mutation of receipts. It estimates only ideal FFN stage
barriers with *synthetic warm-weight* timings: FP8 row selection,
activation production, collective, attention/GDN and language-model
quality remain untested. Owner VTP12; see VTP05 for source results.
Card-free test: `tools/tests/lab/test_xtx27_vtp12_compare.py`.

## VTP12 leased per-GPU FFN ratio ABBA and corrected replay

`xtx27_vtp12_pilot.py` is a bounded **single-physical-GPU-at-a-time**
Radiance `--perfw8` microbenchmark, with 24 ordered A/B/B/A cases
per campaign (3 physical GPUs, gate/up + down). It verifies exact
shared build/plugin hashes and ROCm inventory, rejects resident KFD
processes, acquires and releases the global gpu lease through
`agent-lease`, and writes immutable case receipts/stdout/stderr
under the configured `BC_WORK_ROOT`. Never run on a busy GPU.

`python3 tools/lab/radiance/xtx27_vtp12_pilot.py --phase plan --challenger r9700-45 --m 4`

For a hardware run, add `--phase run --run --build-dir EXACT_BUILD
--output-dir BC_WORK_ROOT/xtx27-prefill/UNIQUE_RUN_DIR`; explicit
`--m 4` or `--m 64`, fresh output directory only. The tool does
not run serving, conversion, three-way collective or numeric quality.

`xtx27_vtp12_replay.py --campaign-dir EXISTING_RUN --output
EXISTING_RUN/summary-replay-v1.json` is **card-free** and verifies
all raw case/log hashes. Required for the M64 campaign: its original
summary used narrow INT8 across all cards, but VTP05 declares tiled
INT8 above M16. Corrected replay reports both VTP05-declared and
best-observed INT8 dispatch; original raw evidence remains intact.

The three 2026-11-11 hardware campaigns and exact results are in
VTP12. Tests: `test_xtx27_vtp12_pilot.py` and
`test_xtx27_vtp12_replay.py`, plus earlier VTP12 comparison tests.

### VTP12 finer 43% R9700 candidate

The VTP12 pilot now accepts `--challenger r9700-43`: 4992/4992/7424
physical FFN widths, using 128-column legal K tiles. The older 256-tile
ABBA candidate geometries and source receipts remain unchanged. Run the
new M4 ABBA only when `agent-lease status` is free and no KFD process
is resident; the runner acquires/releases the lease itself. See VTP12
for exact commands, M64 corrected-replay requirement and promotion gates.

### VTP12 unequal XTX 42.65%-R9700 evidence

The pilot also accepts `--challenger r9700-43-skew` with **5120 /
4864 / 7424** physical FFN widths (256-aligned down-K on each GPU),
versus symmetric `r9700-43` (4992 / 4992 / 7424). The eight
2026-11-11 exclusive-lease runs at M=1/4/8/64 contain **192
validated selftest cases**, with immutable `summary.json` and
separate corrected `summary-replay-v1.json` in
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43[-skew]-m{M}-20261111/`
(the bracketed expression is a *folder-family illustration*, not
a literal pathname). VTP12 has the exact per-run folders and
results. Single-card warm synthetic GEMM timing is not concurrent
TP or quality; the next gate is matched model-shaped numerical
operator and no-P2P three-GPU collective measurement.

## VTP04 physical TP3 F32 host collective and VTP13 ownership checker

`xtx27_vtp_micro.py --physical-tp3` adds an opt-in 3-contribution
mode (one partial for XTX0, XTX1, R9700) beside legacy virtual TP4
local-combined/two-packet modes. `--phase plan` never touches GPUs;
`--phase compile --run` builds in BC_WORK_ROOT scratch without a GPU;
`--phase gpu --run --physical-tp3 --local-mib 4 --rows 1,4,64
--iterations 9` acquires and releases the exclusive lease. It refuses
pre-existing KFD processes or a different lease owner; never stop
other agents' processes. One output oracle is checked on every GPU.
Three leased starts passed; see VTP04 for exact receipts/timings.
This is F32 host transport only, NOT actual mixed INT8/FP8 compute.

`xtx27_vtp13_ownership.py --checkpoint ORIGINAL_BF16 --output
${BC_WORK_ROOT}/xtx27-prefill/NEW_OWNERSHIP_RECEIPT.json` reads only
original safetensors headers and audits 1/1/2 KV heads, 6/6/12
query+gate, 4/4/8 GDN key, 12/12/24 GDN value and 5120/4864/7424
FFN offsets. It proves structural coverage of 640 text tensor sources,
including 96 noncontiguous GDN QKV/conv source repacks. No GPU,
quantisation or full weight hashes. See VTP13 and
`test_xtx27_vtp13_ownership.py` for adversarial tests and limits.
