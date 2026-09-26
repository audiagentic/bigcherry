# PA30: final semantic-equivalence gate receipts (G1/G4/G5)

Plan item: PA30
Status: completed
Owner: PA30 (patching-patch-system)
Question state: answered

## Question

Before deleting the old `framework`/`native` names, do the final
semantic-equivalence gates pass? This topic holds the hardware receipts for
the gate sub-clauses: G1 (representative balanced/rotated A/B performance,
`bigcherry-native` vs `bigcherry-serving-base`), G4 (tuning smoke), and G5
(reduce/telemetry + corpus measurements for `0830`/`1100`).

## Inputs

- `bigcherry ab-benchmark` with `pa30-g1-ab-server-config.json`
  (`--pairs 6 --schedule-seed 0 --settle-seconds 20`).
- Brutus dual `gfx1100` (devices 0+1), `production-dual-xtx`-shaped
  tensor-split config, `qwen3.8-27B Q8_0`.

## Outputs

`pa30-g1-ab-{advisories,run,server-config,summary}.json`,
`pa30-g4-tuning-smoke-receipt.json`,
`pa30-g5-0830-reduce-telemetry-sample.jsonl`,
`pa30-g5-1100-corpus-measurements.jsonl`, `pa30-hardware-receipt.json`.

## Runtime

GPU required: yes (dual gfx1100)
Real compilation required: yes (fresh builds per arm)
Mutates canonical BigCherry state: no (build + receipt artifacts only)

## Safety

- Drives the existing `ab-benchmark`/tuning machinery; never imports into
  production or test.
- Receipts are diagnostic gate evidence for PA30, not a maintained tool.

## Disposition

Retained (TRANSITIONAL) as the PA30 gate receipts; diagnostic gate evidence,
not a maintained tool or the evidence authority itself.
