# Radiance on BigCherry: overview

Radiance is the second serving engine in this project (beside llama.cpp): a standalone inference server,
[radiance 1.3.0](https://codeberg.org/StillDeadcode/radiance), pinned at `89cee7ce414486c717aa2085c99e1f522e4c0ea6`
(`engines/radiance/recipes.toml`). Upstream ships its device kernels (libr4d) for gfx12 only. BigCherry's work here is
to serve the Qwen3.8 containers on the RDNA3 cards (gfx1100) and to use all of Brutus's GPUs together, through the same
build, patch, benchmark and evidence platform the llama.cpp engine uses.

**Rule: every measurement or decision updates the model page in the same change.** The per-model pages below are the
current statement of state; `engines/radiance/README.md` stays the dated lab log (history, never rewritten).

## Model pages

| Model | Page | Current best (see the page for the run, date and source) |
|---|---|---|
| Qwen3.8-Flash-Next (512-expert MoE) | [models/qwen3.8-flash-next.md](models/qwen3.8-flash-next.md) | R9700 engine + two XTX expert helpers: prefill ~2,755-2,764 tok/s, decode ~108.5 tok/s (7,306-token prompt) |
| Qwen3.8-27B (dense) | [models/qwen3.8-27b.md](models/qwen3.8-27b.md) | two XTX, MXFP4: prefill ~2,250 tok/s (7.2K prompt), decode 46.7-47 tok/s without a drafter |
| Qwen3.6-35B-A3B FP8 (MoE) | [models/qwen3.6-35b-a3b.md](models/qwen3.6-35b-a3b.md) | R9700 engine alone: MTP acceptance collapses at 4+ streams with the prefix cache on; `--no-prefix-cache` is the workaround |

## Reference documents

- [PARITY.md](PARITY.md): where the two-XTX path stands against RDNA4 and what closes the gap (written 2026-10-12/13,
  before the helper layout; the Flash-Next numbers in it are superseded by the model page).
- [RUNBOOK.md](RUNBOOK.md): how the engine is built, configured and measured.
- [PLATFORM_DESIGN.md](PLATFORM_DESIGN.md): the design decisions behind the engine, patch, component and profile layout.
- [../RADIANCE_WIRE_QUALITY_DESIGN.md](../RADIANCE_WIRE_QUALITY_DESIGN.md): cross-card wire quality plan (RWQ01-RWQ06).
- Lab log: [engines/radiance/README.md](../../../../engines/radiance/README.md) (dated entries, run IDs, raw tables).

## Hardware (Brutus)

| Card | Arch | Role in radiance work |
|---|---|---|
| 2x RX 7900 XTX (24 GB) | gfx1100 | the two-XTX tensor-parallel engine (libr3 kernels), and the expert helpers in the R9700-engine layout |
| Radeon AI PRO R9700 | gfx1201 | runs radiance's own libr4d kernels; the engine in the Flash-Next helper layout |
| RX 6900 XT | gfx1030 | unused (an RDNA2 helper build is listed as remaining scope in RAD05) |

Host: about 86 GiB RAM (RAD02 notes record "86 GiB RAM"). The two XTX have no peer-to-peer access and sit on PCIe 4.0
x8 (PARITY.md), so every cross-card exchange goes through pinned host memory. Owner rule (2026-10-13): the R9700 is
compute, not storage; lookup-type side work goes to host RAM/CPU.

## Layout in the repo

| Piece | Location | Notes |
|---|---|---|
| Engine declaration | `engines/radiance/engine.toml` | patch id prefix `rad`, build and serve contract |
| Recipes | `engines/radiance/recipes.toml` | pin, patch sets, sources, experiments, runtime profiles `flashnext-2xtx` and `qwen38-27b-2xtx`, platform `linux-radiance` |
| Engine patches | `engines/radiance/patches/rad000N_*` | packages against the pinned radiance (see below) |
| Components | `engines/radiance/components/libr3/` (gfx1100 kernels), `engines/radiance/components/moe-remote/` (remote-MoE helper and mailbox plugin) | BigCherry source built against the installed engine; components, not patches |
| Lab scripts, A/B configs | `tools/lab/radiance/` | `helper-ab.json` (speed A/B, helper sidecars), `helper-kld.json` (KL quality), `fn-profile.sh` (rocprofv3 traces) |

Engine patches (all package state below is from each `SUMMARY.md`):

| Patch | State | What it does |
|---|---|---|
| `rad0001_drain_mover_before_arena_free` | evaluated | drains each rank's mover stream before the activation arena is freed; fixes a shutdown hang |
| `rad0002_moe_remote_handoff` | untested | optional whole-expert MoE hand-off as thin hooks into one BigCherry extension point (14 call sites in 4 upstream files): helper-owned routes are masked from local GEMMs and published to the helper |
| `rad0004_moe_remote_diagnostics` | untested | shadow compare and mask-only for the hand-off; diagnostic build only (`BC_MOE_REMOTE_DIAG`, source `radiance-diag` with build `moe-diag`); never in a production source |
| `rad0003_mtp_full_head_switch` | untested | `RAD_MTP_FULL_HEAD=1` drafts through the trunk head instead of the container's 2-bit head; off by default |

Components and plugins produced by `libr3` and `moe-remote` (each `component.toml` `outputs`; `libr3moe.so` and `r3_moe_worker` belong to `moe-remote`, the rest to `libr3`):

| Output | What it is |
|---|---|
| `kernels/libr3.so` (libr3) | gfx1100 kernel plugin: native WMMA forms for the GEMMs, attention, MoE, all-reduce; loaded beside or instead of libr4d (`--kernels`) |
| `kernels/libr4d.so` (upstream) | radiance's own kernels, gfx12 (R9700); not built by BigCherry's component |
| `kernels/libr3moe.so` (libr3moe) | small plugin, built for gfx1100 and gfx1201: the mailbox rows and the registration (`bc_moe_remote_register`) with the extension point rad0002 adds; used with the R9700 engine as `--kernels libr4d,libr3moe,libref` |
| `helpers/r3_moe_worker` (helper) | separate process per XTX that owns whole routed experts and computes them for the engine through a pinned-host mailbox with GPU-side flags |
| `kernels/libr3ngram.so`, `helpers/libr3_ngram_helper.so` (ngram-helper) | n-gram gather (list libr3ngram in `--kernels` to use it) on a spare card (evaluated, shelved: the R9700 is compute) |

Runtime settings (`R3_*`) are declared in `component.toml` with a default, a status (`production`, `experimental`,
`evaluated`, `diagnostic`, `tuning`) and, where it exists, an activation log marker. Profiles in `recipes.toml` list
only what they switch.

## Workflow

Everything goes through the engine-aware commands (details in [RUNBOOK.md](RUNBOOK.md)):

    python -m bigcherry --engine radiance ...                          # engine selector for the maintained commands
    python -m bigcherry build --lane radiance-lab:stock:linux-radiance # patched engine + libr3 into radiance_home
    python -m bigcherry ab-benchmark --server-config tools/lab/radiance/helper-ab.json --pairs N

Build lanes name a source (`radiance-native` = the pin, `radiance` = validated patch sets, `radiance-lab` = evaluated
engine fixes), a build (`stock`) and a platform (`linux-radiance`). `helper-ab.json` is the maintained example of a
multi-process A/B: it drives the R9700 engine with two XTX helpers, started and stopped by the harness
(`R3_MOE_WORKER_DEVICES`), arms differing in one component setting, with order-balanced rounds and an activation marker
for the setting under test (MEN11). Set `RAD_BIN`, `RAD_HOME` and `R9700_BDF` before use.

`helper-kld.json` is the quality counterpart: a `quality` block (`"quality_only": true` skips the serving cells) runs
radiance's offline KL mode for each arm with that arm's helpers up, gates every arm against the control arm, and records
the result in `run.json` under `quality`. With `quality.record` it first records the reference
(`--kld-record <dir> --kld-corpus <jsonl>`), otherwise it scores against an existing one
(`--kld-ref <dir> --kld-out <json>`).

### Acceptance after a bump (MEN17)

    python -m bigcherry --engine radiance accept --output <new dir> [--only helper-ab] [--only helper-kld]

`accept` runs each `[[acceptance]]` entry of `engines/radiance/engine.toml` through the same server-comparison path as
`ab-benchmark --server-config` (speed cells and the quality block), then compares `run.json` with the stored baselines in
`baselines/`: a table of metric, baseline, measured, delta and pass/fail, an `acceptance.json` evidence file in
the output directory, and a non-zero exit on any regression (1: a regression, a failed run or a metric the run did not
measure; 2: a bad declaration or baseline, before anything runs). It is an ordinary bigcherry command, so it runs on the
lab host with `bigcherry remote run --host brutus --ref <ref> -- --engine radiance accept --output <dir>`. Set `RAD_BIN`,
`RAD_HOME` and `R9700_BDF` first; the KLD entry records its reference into its own output directory (`KLD_REF_DIR` is set by
the entry unless you set it). `pin-bump --accept` runs it after the CPU smoke.

| Baseline file (pin 1.7.0, 2026-10-17) | Arm | Metric | Baseline | Tolerance (one-sided) |
|---|---|---|---|---|
| `helper-ab.json` | `rows0` | `pp4096_tps` | 2049.5 | 3% below |
| | | `pp1024_tps` | 2099 | 3% below |
| | | `tg512_tps` | 124 | 5% below (runs 122-126) |
| | | `tg2048_tps` | 140.5 | 2% below |
| `helper-kld.json` | `helper-on` | `kld_mean` | 0.0642 | 0.01 above |
| | | `kld_p99` | 0.684 | 0.1 above |
| | | `top1_percent` | 88.7 | 1.0 below |

Baselines are updated deliberately, by a new measured run with its pin and run id, never by the run being judged. The 27B
attention-native KLD (0.0040, top-1 96.6%, `27b-native-kld.json`) is not an entry yet: its control reproduces its own
reference exactly, so it needs a stored reference. Record one once (the config's `record` block), write
`<reference dir>.meta.json` with its `pin`, `layout` and `recorded_on`, then point the config at it with
`"reference_metadata"` and `"gate": "baselines"` and add the baseline file.

A `quality` block can name a stored reference (`reference_dir` plus `reference_metadata`, a JSON file giving the pin and
layout it was recorded on) instead of recording in the same run, and with `"gate": "baselines"` the arms are not gated
against a control at all (no `thresholds`); the numbers are judged by `accept`. See `tools/bigcherry/campaign/quality_capture.py`.

### Promoting a component setting

A setting in `component.toml` has the patch lifecycle in its lightest form: `experimental` -> `validated` (or
`evaluated` when it did not pan out). A `validated` setting, and a `production` one whose default is not off/0, names
`evidence = ["<repo-relative run.json>", ...]`: ab-benchmark captures in which the setting is the only varied variable
between two arms, the arm at the setting's default shows an `executed` activation record (the setting's `activation`
marker), and no cell failed or breached the quality gate. The balanced separation and greedy identity or KL quality
inside noise are read from that same capture; no per-setting campaign is required. `bigcherry patch-lint` enforces it
(a `validated` setting without valid evidence fails; a `production` setting without evidence, today only
`R3_ATTN_NATIVE`, is reported as a warning). Changing a default or which profile switches it does not invalidate the
evidence; a change to the component code behind the setting is re-confirmed with a fresh capture.

Radiance has its own release line (`bc-radiance`, package `engines/radiance`); the release stages are common to every
engine and described in [../../build/RELEASES.md](../../build/RELEASES.md).

## Quality bars

- Layout and kernel changes are judged by KL divergence against a named reference (radiance `--kld-ref`;
  `tools/bigcherry/experiment/engine_quality.py`), not by bit-exactness. Flash-Next helper layouts are compared with
  the two-XTX exact run of the same container: re-chunking the stock XTX path alone gives 0.058-0.062 mean KLD on the
  4-document corpus, so 0.058-0.067 is the noise floor (RAD03, RAD02 notes).
- Container against the full model: see the Flash-Next page.
- KV cache: `--kv-cache-dtype fp8` is what every radiance number here used; f16 has not been measured on radiance
  (the owner's floor is q8/8-bit, f16 preferred).

## Open plan items

Patching plan `patching-radiance-rdna3`: RAD02 (inverted expert-parallel layout: working end to end, item still open), RAD03 (GPU lanes),
RAD04 (27B overlap the cross-card sum), RAD05 (decode MoE GEMV; done as helper small lane), RAD06 (prefill hand-off
pipeline and gfx11 grouped GEMM), RAD08 (full KL metric set). Related: `patching-radiance-heat-placement` (RHP01-RHP06),
`patching-radiance-wire-quality` (RWQ01-RWQ06), `run-multi-engine` MEN11 (radiance on the evidence tooling).

## CPU smoke (MEN15)

`python -m bigcherry --engine radiance smoke-cpu --build <build dir> [--model-root <dir>] [--out <dir>]` serves a tiny
`.rad` container with `--kernels libavx,libref --tp 1` (the CPU host backend and the reference kernels, one rank) and the
HIP/ROCR/CUDA device variables empty, so it runs while the GPUs are busy. The checks are the same as for every engine:
sha256 of the model, health, three 16-token greedy `/v1/completions` requests run twice (model id taken from
`/v1/models`), non-empty and identical text, SIGINT shutdown with a clean exit, and no device-initialisation line in the
log; the result is `cpu-smoke.json` beside `server.log` under `--out`. The container is declared in `[smoke.cpu]` of
`engines/radiance/engine.toml` (path relative to the model root; `args` are its serve flags), with a PLACEHOLDER sha256
until a tiny container is converted and pinned. Not yet confirmed on hardware: that `libavx` serves a full model, and the
exact device-init log lines the assertion looks for (`engines/radiance/engine.toml` and
`engine_adapters/radiance.py` are the places to adjust).

## Concurrent-stream benchmark (RAD10-01)

`bigcherry stream-bench --url http://HOST:PORT --streams 1,2,4,8 --prompt-tokens 512 --max-tokens 256 --reps 3 --output out.json`
drives N simultaneous streaming chat completions at an already-running OpenAI-compatible server (radiance, llama-server) and
writes one `bigcherry.stream-benchmark.v1` record: per stream time to first token, decode tokens/s and a hash of the text; per
round aggregate tokens/s, per-stream mean, TTFT and inter-token percentiles; and a summary per stream count. Prompts differ per
stream unless `--shared-prefix`. At temperature 0 the text hashes compare helpers on against off. A failed stream fails the run.
It starts no server: the lifecycle belongs to the build and campaign code.
