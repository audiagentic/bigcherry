---
id: XDP12
order: 11
plan: run-xtx27-prefill
state: pending
created-at: '2026-11-27T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# IQ6R-V4 toolchain: current-release compatibility and portability

## Scope and lineage

Follow-up plan from the 2026-11-27 post-merge source review of the #1718 landing (the
`agent/iq6r-converter-fix` content, merged as a preservation exercise). The merge is **not**
reverted: preservation, placement, merge integrity and production isolation were verified sound.
This item owns making the landed toolchain correct for immediate use against the current
Radiance release (2.2.0, pin b6b8751, release `bc-radiance-2020.23.2`) and portable across
hosts, plus the documentation reconciliation the review required. XDP11 remains the
qualification spec; this item is the compatibility and portability work that precedes it.

## Findings this item resolves

| # | Finding | Sites |
|---|---|---|
| 1 | Release identity `bc-radiance-1170.26.0` hardcoded (obsolete; main is on 2.2.0 / 2020.23.2) | `iq6r_v4_controller.py:76,202`, `iq6r_v4_convert_experimental.py:68`, `iq6r_v4_crash_probe.py:39`, `iq6r_v4_micro_convert.py:51` |
| 2 | Hardcoded GPU ordinals (0,1 = gfx1100; 2 = gfx1201), host-specific | `iq6r_v4_convert_experimental.py:186,195,208`, `iq6r_v4_crash_probe.py:85-89`, `iq6r_v4_adaptive.py:186`, `iq6r_v4_micro_convert.py:149` |
| 3 | Original BF16 shard integrity trusts the prior `source-sha256-recheck.json` receipt; RADGRAM2 inputs checked by header and size only | `iq6r_v4_convert_experimental.py:81` (`verified_inputs`), `iq6r_v4_validate_hessians.py` |
| 4 | Host environment still resolved at import time (placeholder-defaulted, but captured at import) | `iq6r_v4_controller.py`, `iq6r_v4_convert_experimental.py` module scope + dependent importers |
| 5 | Lab regression tests are not in normal CI execution; planning and evidence docs carry stale or inconsistent claims | `core-offline-validation.yml` (lab paths trigger but do not run), XDP11 (v1.17.0 / 1170.26.0 narrative), `docs/evidence/2026-11-24-iq6r-v4-*.md` (289 vs 292 tensor counts; 22.53 GiB plan-only vs diagnostic containers) |

## Workstreams

**A. Release identity as data (profile).** Introduce a conversion profile — a tracked data
file (e.g. `tools/lab/radiance/iq6r_v4_profile.toml`) carrying: release tag, source pin,
recipe identities (the existing sha pins), calibration identity, and build-manifest location
(resolved from `BC_RELEASE_MANIFEST`). Keep the historical 1170.26.0 profile as a named
profile for reproducibility of the recorded runs; add a 2020.23.2 profile marked
`unverified` — it must pass the card-free preflights on the current pin before any device
run may use it. Acceptance: no release-tag literal in tool source; profiles parse and are
validated by a card-free test; the current profile's source pin equals
`engines/radiance/recipes.toml`.

**B. Device selection from inventory and lease.** Replace fixed ordinals with resolution
through the existing `tools/bigcherry/hardware/inventory.py` machinery
(`HardwareInventory`, `SeriesGpuBinding`, `InventoryCatalog`): select the gfx1100 pair and
the gfx1201 device by architecture from an accepted inventory, refuse closed on a mismatch,
and record the chosen ordinals in each run receipt. Lease acquisition (already used) governs
exclusivity; the inventory governs *which* devices. Acceptance: no hardcoded ordinal
constants in the toolchain (explicit operator-supplied ordinals as arguments are allowed);
the selection logic is card-free and unit-tested.

**C. Runtime configuration.** Finish the import-safety migration: move the
`WORK`/`ROOT`/`SOURCE`/`ORIGINAL`/`PINNED`/`RELEASE` constants to runtime resolution
(functions) in `iq6r_v4_controller.py` and `iq6r_v4_convert_experimental.py`, migrate the
dependent importers and the test patch-targets together. Acceptance: no `os.environ` read
at module scope anywhere in `tools/lab` (card-free check); behaviour on a configured lab
host is unchanged.

**D. Input integrity without trusted receipts.** `verified_inputs` recomputes the 18
original BF16 shard SHA-256s directly (card-free; the shards are in the work root) instead
of trusting `source-sha256-recheck.json`, which is retained as a historical cross-check
artifact only. RADGRAM2 input validation gains sample-based content verification against
the source Hessians (the discipline `iq6r_v4_verify_grams.py` already applies to outputs)
in addition to header and size checks. Acceptance: a card-free test on a fixture tree
proves independence from any prior receipt; the historical receipt file is untouched.

**E. Targeted offline CI.** Add the card-free lab suite to normal CI execution
(`core-offline-validation.yml` gains a `tools/tests/lab` step for the card-free set);
tests that require host data skip with an explicit reason when the data root is absent, and
run on the lab host. Acceptance: the IQ6R lab suite is green in CI with no host data;
host-data tests skip loudly in CI.

**F. Documentation reconciliation.** XDP11: status section distinguishing the completed
historical work (recorded under its own release identity) from outstanding qualification,
with the current resumption procedure and current release identity; the stale v1.17.0 /
1170.26.0 narrative is preserved as history, not as instruction. Evidence pages: annotate
each tensor count and size projection with its recipe generation and measurement context
(known deltas: 289 final-recipe GPTQ stems vs the projection's 292-tensor two-point fit;
the 22.53 GiB plan-only estimate vs single-tensor diagnostic containers). Acceptance: no
unreconciled numeric claim; XDP11 is the resumption entry.

## Sequencing and constraints

- A, C, E and F are offline-safe and can land as one follow-up PR (or split A+C / B / E);
  B and D are also offline-safe in their logic (device and shard data are read, not
  executed) and join the same or a second PR.
- No GPU work, no full-model conversion, no rad0062 device validation, and no production
  promotion are in scope while the lane is dormant; those remain with XDP11 at resumption.
- Historical receipts, work-root artifacts and the #1718 archive are preserved as-is.
- The 2020.23.2 profile stays `unverified` until the card-free preflights pass on the
  current pin; device qualification follows at resumption.
