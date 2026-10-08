# Patch development pipeline

The repeatable loop for finding, authoring, testing and promoting performance patches for a
target workload (e.g. Qwen3.8-27B Q8_0 on 2x 7900 XTX, `-sm tensor`, MTP). Any agent or tool
can follow it. Mechanics live in the linked docs; this page is the order of operations and
the rules that keep runs trustworthy. Agents follow it through the
`bigcherry-perf-pipeline` skill (`.claude/skills/bigcherry-perf-pipeline/SKILL.md`).

## Stages

| # | Stage | Tool / doc | Exit condition |
|---|---|---|---|
| 1 | **Triage** candidates against the workload (model, quant, archs, split mode, decode vs prefill) | reviewer agent; `patches/*/SUMMARY.md` | Each candidate rated TEST-NOW / TEST-LATER / SKIP with the marker that proves it fires |
| 2 | **Author** or repair the package | [PATCH_AUTHORING.md](PATCH_AUTHORING.md) | `patches/<id>/` with anchored edits, marker, `validation.toml`, focused test |
| 3 | **Mechanics** (no hardware) | `bigcherry patch-lint`; `unittest discover -s tools/tests/patch` | Lint clean, focused tests pass |
| 4 | **Build** | queue `BUILD` rows ([BUILD.md](../build/BUILD.md#queued-qualification-campaigns-and-the-firing-pre-flight)) | `BUILD_EXIT=0`, binary recorded |
| 5 | **Fire first** | queue `PREFLIGHT` row on the target model/config | Marker hit >= 1; otherwise stop: the patch cannot fire here (retarget or drop) |
| 6 | **Measure** | `REQUIRES=<preflight>` `AB` rows (exploratory) or contract campaign rows (evidence) | Balanced pairs; activation and work-equivalence (e.g. MTP acceptance) equal across arms |
| 7 | **Decide** | `bigcherry-patch-lifecycle` skill; reviewer sign-off | Win -> fresh contract sessions at the current pin -> `validated` -> `[patch-set.validated-enhancements]`; neutral -> leave untested; regression -> `rejected` with the evidence |

Record a ledger event after each substantive stage and keep the plan item current.

## Rules

- **Queue everything.** Builds, preflights, A/Bs and campaigns go through
  `tools/lab/plan-qualification/queue.sh`; no ad-hoc runs. The queue dedupes builds, holds
  host and GPU locks, and is restartable.
- **No timed lane without firing proof.** Producers call the activation check before any
  timed lane; a failed check exits 3 with `producer-blocked.json` and no evidence record.
- **One variable per comparison.** Same binary with env arms (e.g. AllReduce provider), or
  same lane with one extra patch. Read composition from the builds, not labels.
- **Promotion needs current evidence.** A pin bump or patch.py change makes prior evidence
  stale; run fresh sessions before adding a patch to the production patch-set.
- **Scripts and configs live in `tools/lab/<topic>/`**, never `/tmp` on the bench host.
  Don't edit or pull scripts on the bench host while one is executing in place.
- **Models** for new comparisons go under `/mnt/data/llm-models/<model>/gguf/<vendor>/`.
  Quant comparisons use one vendor's full set so only the quant differs.
- **Multi-arch**: build with an arch list (`gfx1100,gfx1201`) and declare every device's
  architecture and PCI locator in the A/B config's `expected_execution`.

## Working with a reviewer agent (GPT gateway)

Keep **two reviewer sessions busy at all times**, one per workstream (e.g. transport
patches, kernel/MTP patches). On each check (every ~10-15 min):

1. If a request is still running, leave it.
2. If it finished: apply what it delivered (stage 2-3), commit and push, add its queue rows
   (stage 4-6), then immediately submit that session's next backlog item.
3. If it failed or was interrupted, resubmit it.

Reviewer constraints:

- It only sees **pushed** repo files. Commit and push before every request.
- If a reviewer cannot see `vendor/llama.cpp`, provide the exact pinned revision, path, function/anchor text, and any necessary excerpt in the request. Do not commit copied upstream source under `tools/lab/`; materialize or inspect the configured pinned checkout instead.
- Ask for complete, applyable files (patch.toml, patch.py, tests, validation.toml) with each
  anchor's expected match count, plus the queue rows to run. Verify everything locally
  before trusting it.
- Useful standing work when a session has no patch to author: deep review of recent
  commits, design of the next backlog item, lifecycle sign-off on finished evidence.
