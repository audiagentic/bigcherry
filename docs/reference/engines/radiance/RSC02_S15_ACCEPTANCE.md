# RSC02 S15 — same-build Bash-to-native acceptance

> **Archived 2026-11-20.** The S15 tools moved unchanged to `tools/lab/radiance/archive/rsc02_s15/` (see `tools/lab/radiance/archive/ARCHIVE.md`): they compared the native owner with the retired Bash launcher, which no longer exists, so nothing here can be re-run. The text below is the record of how the equivalence was tested.

**State: operator hardware acceptance attempted 2026-11-18; dense passed, helper blocked.** This is the RSC11 S15 gate for retiring
`launch-flashnext-best.sh`, `lib/plan-check.sh` and the Python supervisor at
S16. S13 must have merged, S14 must provide attributable serving/campaign
receipts, the step-5 stored-manifest review fixes must be merged, and the RSP01
Python-free tagged release must pass its own gate before retirement. The
RSC12 named-config loader is the source of effective settings, not a new
launcher-flag mapping. There is **no GPU fallback** in card-free CI.

## 2026-11-18 leased acceptance finding (helper cohort blocked)

The operator's leased S15 short A/B found that the dense native cohort passes, but no
helper parity has been accepted. The native `bc-serve serve flashnext-4gpu` measuring
start reached a measured owner map, then the final manifest's strict S10 live
VRAM admission refused helper0 (23,074 MiB of experts required, 391 MiB
live claimable). Measuring helper processes had exited, but their BDF-scoped
`mem_info_vram_used` counters had not necessarily returned to the pre-start
baseline. The native owner requires a bounded post-reap VRAM release barrier
before the final writer; neither a stale pre-measurement snapshot nor a
larger helper budget may bypass live admission.

For `flashnext-tp2xtx-r9700`, the measuring engine reported **host RAM**
rather than VRAM exhaustion: the RAM-pinned n-gram table requires 47.69 GiB,
another 64.60 GiB was projected for pools, checkpoint slots and other engine
allocation, and 4.59 GiB headroom was required against 82.00 GiB
reported available. The native owner's derived pool was 32,005 MiB.
The Bash launcher's RAM admission also refuses this host; do not shrink the
pool merely to turn the start green, infer nonreclaimable page cache without
evidence, or claim helper equivalence. Both arms need admissible identical
settings and new leased confirmation after the CPU-only fixes pass. No GPU
or lease was used to develop these changes.

## One-command leased short A/B acceptance (operator only)

The lab-only `tools/lab/radiance/archive/rsc02_s15/rsc02_s15_ab.py` driver is **not a
production dependency** and is **not run by CI or by this agent on GPUs**.
Run from the repository root, with the completed build, the archived control and absolute inputs
recorded first. Replace the example paths with the real paths of **one**
immutable build and selected model; all paths must already exist:

```bash
python3 tools/lab/radiance/archive/rsc02_s15/rsc02_s15_ab.py accept \
  --build /absolute/completed/BUILD \
  --legacy-runner /absolute/archived-control.sh \
  --bc-serve /absolute/completed/BUILD/radiance_home/bin/bc-serve \
  --configs /absolute/repo/engines/radiance/configs \
  --site /absolute/SITE.toml \
  --model-root /absolute/model-root \
  --model-file /absolute/model-root/actual-model-file.rad \
  --work-root /absolute/BC_WORK_ROOT \
  --out /absolute/BC_WORK_ROOT/rsc02-s15-short-ab \
  --power-profile stock \
  --confirm I-WILL-RUN-ON-LEASED-GPUS
```

The driver checks `rocm-smi` and `free -g` before **each** arm,
then acquires an individual `agent-lease acquire gpu --ttl 30m --est 15m`
lease and releases it when that arm exits. It alternates A/B, B/A, A/B,
B/A for **each** dense Qwen 27B and helper Flash-Next cohort; A is the
existing Bash launcher and B the foreground native owner. It refuses
unleased startup or any unclean stop, and halts the sequence at the first
failure. To test only the fake-engine/card-free path, run
`python3 tools/lab/radiance/archive/rsc02_s15/rsc02_s15_ab.py dry-run --native-test BUILD-TEST/bc_session_stored_test`
(a lab command; it is no longer a CTest target, the engine component is C++ only).

Every paired short run **passes** only if the same build/engine/helper/model
digests remain unchanged, both start/health/canary succeed, all manifest
bytes are bound to observed `/proc` launch arguments and complete
engine/helper role inventories, the existing strict semantic comparator
reports **zero** mismatches, one fixed temperature-zero greedy completion
has the **same SHA-256** on both arms, power observations/receipts are
retained, and all owned children and power state have a verified clean
stop. Each cell writes its own preflight, snapshot, output hash, power
and stop evidence under `--out`. Failed starts, missing receipts where
required, process races, mismatches and incomplete stops **block** the
next lease; inspect and restore the host before any retry. The short
run's `SHORT_AB_PASS` is **not** the full S15 PASS: S14 attribution and
the RHP12 quality, retained-session, energy, safety and performance
requirements below remain mandatory. Do not merge S16 until the full
leased hardware acceptance and release-start gate pass.

## Owners and build identity

Run from a clean candidate checkout on a leased host, supplying `--legacy-runner` as an absolute archived pre-S16 control script from an immutable, independently verified historical checkout. The candidate checkout itself no longer contains that script. Do not substitute the candidate native owner for control arm A. Use exactly **one immutable completed
build** and record its full build metadata/source digest, `bc-serve`, engine,
helpers, kernels and model SHA-256, ROCm/driver/runtime versions, ordered GPU
PCI BDFs and architectures. Both arms must resolve the **same ordered cards,
model bytes, calibration key, live-policy decision, heat file bytes, power
profile, budgets and active settings**. No build, SDK, model, work root or
device can change between the arms. A debug or exploratory timing arm is not
production evidence.

The two configurations in the S15 acceptance matrix are:

| Cohort | Bash control A | Native candidate B | Compulsory evidence |
| --- | --- | --- | --- |
| Dense TP | `launch-flashnext-best.sh start --layout qwen27b-tp3-q8 --build BUILD` | `bc-serve serve qwen27b-tp3-q8 --configs DIR --site SITE --model-root ROOT --run-dir RUN --engine-binary BUILD/bin/radiance --power-profile PROFILE` | Three ordered engine ranks; disk ngram, KV, MTP replay, returned canary and shutdown |
| Helper | `launch-flashnext-best.sh start --layout flashnext-tp2xtx-r9700 --build BUILD` | `bc-serve serve flashnext-tp2xtx-r9700 --configs DIR --site SITE --model-root ROOT --run-dir RUN --engine-binary BUILD/bin/radiance --power-profile PROFILE` | Real engine/helper ranks, heat producer/consumer format, histogram, owner map, live policy, measured plan and startup admission |

Here `DIR`, `SITE`, `ROOT`, `BUILD`, `RUN`, and `PROFILE` are
**actual recorded operator inputs** selected before the lease. They must
resolve to exactly equal semantic execution projections; do not invent a
profile or substitute a layout just to make a start pass. If production power
intent is not stock/observe, use the same named profile for both and pass
`--allow-power-apply` to the native owner **only after lease validation**.
The archived Bash control's options must match the exact historical code and receipt; legacy options have no compatibility alias in native code. A cell with a missing
native typed mode is **BLOCKED**, not equivalent.

## Automated admission, observations and schedule

1. **Card-free prerequisite:** build `bc_session_stored_test` and run the lab `dry-run` command above.
   It runs the production native stored-manifest owner/fake engine and its
   comparator injects differences in speculation, histogram, helpers and
   role identity.
   Run the two longer real Bash launcher/fake-engine paths separately on a
   suitable card-free host with
   `python3 tools/lab/radiance/archive/rsc02_s15/rsc02_s15_projection.py dry-run --native-test
   BUILD/bc_session_stored_test --legacy-launch-tests`. They hang under
   the CI build-container launcher environment, so no Bash success is
   inferred from the component CTest. The card-free prerequisite
   **cannot** assert hardware, latency, power or equal end-to-end owner
   behaviour. The unit test `tools/lab/radiance/archive/rsc02_s15/test_rsc02_s15_projection.py`
   checks byte-binding and semantic tampering.
2. **Before every GPU cell:** verify identical build SHA, model SHA and
   ordered device facts; record a signed/verified active lease and current
   availability, physical power caps and saved pre-run state. Preflight the
   production owner and its actual strict-v2 manifest reader, SDK, native
   planner, real measured plan and current S8/live-room gates. Persist the
   original manifest bytes, SHA-256 and producer digest. Refuse admission
   on a missing/contradictory field, changed weights/heat/calibration, or
   missing source provenance.
3. **Capture every arm, do not infer from labels.** Record the exact
   `/proc/<pid>/cmdline` and selected environment **for every engine rank
   and helper process**, the opened manifest path and bytes, process/BDF/role
   mapping, actual runtime receipts, power receipts, ordered canary,
   log/output digests, live placement counters and shutdown evidence. The
   process launch capture is a typed JSON with keys
   `schema_version:1`, `manifest_path`, `manifest_file_sha256`,
   `manifest` and `processes.roles[]: {role,argv,env}`. Capture the
   minimal *execution-relevant* environment (GPU selectors, runtime home,
   kernel/transport and any settings reaching children), never the entire
   shell, secrets or sensitive host environment. The observed process set
   must contain **exactly** `engine` plus every `spec.processes.helpers[].role`
   declared in the captured manifest; each actual argv must have exactly one
   `--role NAME` agreeing with its captured label. Two arms missing the same
   helper are both invalid even if their semantic projections otherwise match.
   The observed `--manifest`
   argument must refer to captured bytes; a missing, substituted or mutable
   manifest fails closed. Use the same selection rules on both arms.
4. **Compare the two captures before measurements:**
   `python3 tools/lab/radiance/archive/rsc02_s15/rsc02_s15_projection.py compare --legacy
   A-snapshot.json --native B-snapshot.json --output projection.json`.
   It requires full equality of the execution spec, model/producer identity,
   measured host resources, plan, power, calibration, ordered cards, owner map,
   helper count/budget, handoff heat, telemetry period, MTP/engine options
   and observed process roles/argv/env. It normalises **only** per-run
   manifest path (after exact-byte binding), instance/digest/provenance
   bookkeeping, free-memory fact changes, per-owner SHM namespace,
   histogram output path and HTTP endpoint. The latter have separate
   collision, path and port validation. Verify **both** native strict-v2
   reader acceptance and independent physical BDF and build identity; the
   JSON comparator is not a substitute for digest/signature admission.
   Every changed path is reported; **zero** semantic mismatches is required.
5. **Interleave A/B; never group by launcher.** For each dense and helper
   cohort use at least **four clean paired repetitions** with two-order
   blocks (e.g. `A B / B A / A B / B A`) and a fresh start/shutdown for
   every cell. Use `bigcherry ab-benchmark` and its
   `schedule_named_arms`, `--pairs`, `--schedule-seed` and
   `--settle-seconds` controls for balanced positions, audit per-position
   means and exclude only predeclared, evidenced stalls. In particular,
   `--server-config` currently drives the engine's existing server
   lifecycle directly; it does **not** automatically wrap the Bash
   launcher or native `bc-serve` session owner. **Do not run the hardware
   contrast through that path until S14 wires a serving-owner lifecycle
   adapter that produces these exact capture receipts.** Record any
   missing adapter as a blocking dependency rather than treating direct
   engine starts as S15 evidence.
6. **Run RHP12 workloads per fresh/retained cohort:** eight-session
   prefill, six-stream decode with engine step counters, a fresh ~42k
   token prefill probe, quality KLD/top-1 and semantic canary, then the
   retained-session cohort after the declared ~1M-token eight-session fill,
   capturing host KV/loan bytes, mover traffic, helper expert residency,
   host RAM, card VRAM and histogram/epoch counters. For performance
   evidence cite the exact `workload-id@version` and JSONL `row_id`
   from `docs/reference/benchmarks/WORKLOADS.md`. Use the maintained
   benchmark client and stable corpus digest, not a newly invented
   throughput formula; report prefill and decode separately.
7. **Teardown every cell, including failures:** native `bc-serve stop
   --run-dir RUN` through its owner, Bash `... stop`, verify each
   recorded PID plus process start time is gone, no orphan helper/SHM or
   GPU lease escapes, no un-restored power profile, no stale manifest replay
   and no writes outside the assigned run directory. A failed cleanup
   ends the acceptance. Save the failure logs and do not promote S16.

## Lease window sizing (short, bounded, never guess a duration)

Perform **card-free** enumeration of the two layouts, estimated startup
memory/plan room and build identity first. On the hardware lane, use one
separately authorised, bounded preflight lease to time one startup,
health/canary, probe and teardown for each topology; do not call it an A/B
pass. Let `S_L` be measured p95 startup/health and `D_L` measured p95
verified shutdown for layout `L`; `W_L` is the fixed timed workload
duration from the canonical benchmark manifest and `Q_L` the agreed
inter-cell settle period. For `n_L` predeclared cells, request:

`lease_seconds = T_preflight + Σ_L n_L × (S_L + W_L + D_L + Q_L) + T_restore + T_reserve`.

The reserve must come from the owner/lease policy or observed worst-case
startup/shutdown variability, not a hand-picked universal number.
S15 is one *same-session* comparison if that computed request fits the
platform lease maximum. Otherwise file an explicit **BLOCKED** gate and
rescope the measured cohort with the owner; never silently split hardware
identity across days or overrun/extend a lease. The harness must hard-timeout
start, canary, every workload, stop and restore independently within the
lease; the final deadline terminates only the known test-owned PIDs and
restores saved state before releasing the lease. No simultaneous GPU
build or measuring job may compete with the cells.

## Pass/fail and rollback

Every gate is individually PASS/FAIL/BLOCKED with links to raw receipts.
No launcher-only mode, missing process role, digest, power/room, canary or
quality evidence is waived. Require **four or more clean ABBA pairs per
layout**, all roles bound to the same build and exact semantic projection,
zero guard bypass, crashes, memory/pin/VRAM safety breaches, missing/duplicate
or stale-epoch expert work, and verified clean shutdown/power restore.

The RHP12 proposal bands apply, judged against paired variability: median
engine step no worse than the reviewed 1–2% allowance, p95 no worse than
3%, prefill chunk completion within 2%, unchanged kernels within 2%,
migration-period sustained p95 within 5% (separate transient spikes),
and cached startup below 0.5 s where the plan says that gate applies.
Quality must stay within the KLD/top-1 acceptance band; user-facing
outputs and canary must be semantically correct. Retained-session probes,
actual power/energy samples, CPU and GPU occupancy and full log faults
are mandatory before any decision. See RHP12 for baseline identity and
the requirement that returning to a known layout reproduces its cohort.

On any failure restore the original power and placement baseline, stop
only test-owned processes, retain the complete evidence under the work
root and **do not** merge the S16 deletion PR. The rollback target is
the last proven Bash launcher with the same build and original verified
profile, not a Python-powered engine asset.

## S16 proof before deletion

The separate deletion branch is eligible for review only after this S15
record, S14 serving-owner benchmark adapter, all active lab/CI/test
callers migrated, release installer/workflow handoff to RSP01, and a
`git grep -n` audit of every direct and indirect invocation (including
legacy environment owner names in `config/serving-settings.toml`) finds
**no executable dependency** on the launcher, plan-check library or Python
supervisor. Preserve historical experiment patches and evidence as history.
Retain `bc-serve` alone in the single Python-free engine release.
