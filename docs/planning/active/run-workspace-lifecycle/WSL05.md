---
id: WSL05
order: 5
plan: run-workspace-lifecycle
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P1
work: S
---

# Experiment templates: config, launcher, analysis and evidence in one pattern

## Description

New experiments start from a template, not from a copy of the last script: a config (ab-benchmark server config or stream or accept entry), a launcher that uses only bigcherry commands and the run template,
an analysis step declared in the harvest recipe, and the evidence README skeleton. This replaces the per-experiment .sh files in tools/lab/radiance (helper-evidence.sh and friends) as they are retired under
MEN20 and MEN21. Configs live beside the setting they test and are named for it; arms are identical except the variable under test.

## Steps

1. Template set under tools/lab/_template (extend the existing lab template) for ab, accept, stream, replay and capture experiments.
2. `bigcherry run new` copies the right template and fills the manifest.
3. Retire the hand launchers as each is covered.
