---
id: WSL01
order: 1
plan: run-workspace-lifecycle
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Standard work-root layout and one run-directory template

## Description

Owner direction 2026-10-19: build and test in standardised folder structures with common templates, and clean up automatically once what we need has been extracted. Today runs land in ad-hoc
directories (4,635 entries under runs/, 554 build and 618 source directories, one-off output folders named by hand) so nothing knows what is safe to remove.

Define ONE layout under the work root and ONE way to create a run:

    work/
      builds/<lane>/<plan-id>/            build outputs, keyed by plan id (existing)
      sources/<slice-id>/                 materialised sources (existing)
      runs/<kind>/<run-id>/               every experiment, benchmark, acceptance, sweep, capture
      cache/                              tune caches, KL references, ccache (size-capped, rebuildable)
      evidence-hold/                      harvested bundles waiting for the repo (small)
    run-id = <yyyymmdd-hhmmss>-<topic>-<short-hash>;  kind in {ab, accept, stream, replay, capture, kl, profile, build, bump, release}

Every run directory carries run.manifest.json, written by the command that made it (never by hand): purpose, plan item, engine, command line, builds used (plan ids), inputs (configs with hashes),
host and device summary, start and end, exit state, size, and the harvest and retention fields (WSL02, WSL03). One function (paths.run_dir(kind, topic)) creates it; commands and the lab launchers use
it, and a check rule rejects a command that writes a run output outside it.

## Steps

1. Specify the layout, the run-id rule, the manifest schema (versioned JSON) and the kinds; document in docs/reference/build.
2. Implement paths.run_dir and the manifest writer and reader; migrate ab-benchmark, accept, stream-bench, build, pin-bump, pin-release and the replay sweep to it (every caller, no old path left).
3. Run template: `bigcherry run new <kind> <topic>` creates the directory, manifest and a README stub (question, inputs, outputs, runtime, safety), replacing hand-made folders.
4. Lint: no hand-named run output directories in lab scripts; lab launchers must call the run template.

## Validation

Offline tests for layout, id uniqueness, manifest round trip and schema versioning; one real ab-benchmark and one accept run produce a manifest that names their builds and configs.

## Notes

Foundation for WSL02 to WSL06. Existing llama.cpp tuning outputs (artifacts/, tune journals) keep their own schemas but gain a manifest.

## Design review (GPT, 2026-10-19) - adopted

- Layout additions: `work/trash/<deletion-id>/` (quarantine, same filesystem as runs/) and `work/objects/sha256/<xx>/<digest>/` (valuable immutable blobs: captures, KL references; not cache).
  Builds have one canonical place (`builds/<lane>/<plan-id>/<build-id>/`); a build run carries a manifest but its output is not duplicated under runs/. A plan id alone is not a build identity.
- Run id = UTC stamp + topic slug + 12 hex of a uuid4; the uuid is the identity, ids are never reused, the directory is created with an exclusive mkdir.
- Manifest v1: run lifecycle, harvest lifecycle and retention are separate fields (`completed` is not `harvested`; `harvested` is not `deletable`). Written atomically (tmp + fsync + os.replace + fsync dir) with a
  generation number; a crashed process leaves `running`, which only reconciliation may turn into `interrupted`.
- First slice (PR #311): core/workspace.py, core/run_manifest.py, core/retention.py (read-only), `bigcherry run new|status`, `bigcherry clean` (dry run only), tests on temporary roots.
  Producer migration (ab-benchmark first, then accept) is the next step, one at a time.
