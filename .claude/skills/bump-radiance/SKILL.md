---
name: bump-radiance
description: Triage, port, qualify and release a change to BigCherry's pinned Radiance source. Use for upstream tags, Radiance pin moves and cross-pin compatibility investigations.
---

# Bump Radiance without duplicating its contracts

The source pin and version live in `engines/radiance/recipes.toml`. Work in an
isolated bump worktree; never hand-edit vendor sources or modify tracked files
in the primary checkout. See [PIN_BUMP.md](../../../docs/reference/build/PIN_BUMP.md)
for the orchestrator and [BUMP_COUPLING.md](../../../docs/reference/engines/radiance/BUMP_COUPLING.md)
for the executable coupling checklist. Plans under
`docs/planning/active/patching-radiance-upstream/` own unfinished acceptance;
passing patch application never proves runtime compatibility.

## 1. Triage the candidate before moving the pin

1. Identify the **exact commit**, version, `RAD_ABI_VERSION`, layout fingerprint
   and relevant upstream changes. Do not equate the newest tag with the reviewed
   target. Run `python3 -m bigcherry --engine radiance pin-triage <ref> --json`;
   its non-clean verdict is advisory, not an acceptance decision. Run it once
   at the **current pin** first: it must be `clean`. Native rules for sources a
   managed patch creates are listed as checked after rebase, not as drift.
2. Diff upstream against the *registered patch targets* and the libr3
   `rewrites.txt` / `native/*.rw` anchors. Test libr3 rewrites against an
   exact-ref `libr4d` archive with `r3_rewrite.py --check`, without disturbing
   the primary vendor checkout.
3. Check every deployed `engines/radiance/configs/*.yaml` against retired or
   changed **model weight encodings, kernel row constraints, KV formats and
   rank-local shape assumptions**. An upstream format retirement is a
   **model-roster migration**, not a reason to resurrect the old kernel:
   MXFP4/block-FP8 retirement requires an approved replacement container.
   For IQ8R, the container's int8 scale group is 256 despite FWHT128;
   per-rank V and FFN widths and offsets must be whole encoding scale groups.
   Check proposed `tp-ownership` splits *before* GPU starts.
4. Capture the current composed patch footprint before modifying packages:
   `python3 -m bigcherry --engine radiance patch-footprint --check`.
   Review `docs/evidence/radiance-patch-footprint/footprint.md` for the
   upstream files with the most patch attachments; compare a regenerated
   report after the port. Anchor counts are lexical, not semantic coverage.
   Record the before/after delta and the remaining coupling, not just lines moved.

## 2. Port only surviving changes

Use `engines/radiance/recipes.toml` for the authoritative `[source.*]`
selections. On the new exact upstream source, run rebase checks for **each**
deployed source composition, *then* the catalog-wide check:

```sh
python3 -m bigcherry --engine radiance patch-rebase-check --source radiance-native  # unpatched control
python3 -m bigcherry --engine radiance patch-rebase-check --source radiance
python3 -m bigcherry --engine radiance patch-rebase-check --source radiance-diag
python3 -m bigcherry --engine radiance patch-rebase-check --source radiance-routing-evidence
python3 -m bigcherry --engine radiance patch-rebase-check --source radiance-vtp44-evidence
python3 -m bigcherry --engine radiance patch-rebase-check --all
```

Reconcile `FAILED_NEEDS_RECONCILIATION`, `BLOCKED_BY_DEPENDENCY` and
`QUARANTINED` against the actual upstream diff. A clean `--source` does
**not** cover independent diagnostic or unselected registered patches;
`--all` does not substitute for proof of each exact production composition.
When upstream supersedes a patch, move its **whole package unchanged** into
`engines/radiance/patches/archive/` and add an `ARCHIVE.md` row (reason,
successor and evidence); remove it from recipes and re-run the checks.
Never re-port, rewrite or delete a superseded package for convenience.
See `docs/reference/patches/PATCH_SYSTEM.md`.

Keep remaining changes in managed patches or owned components. Prefer a narrow
typed hook over multiple source-body attachments, without duplicating upstream
policy. For KV kernel registrations use the owned `libr4d/bc_kv_rows.h`
(declared through `rad0049`, extended by `rad0051`–`rad0053`): a supported
geometry is described in the shared row inventory, with *actual* query/KV
dtype, head geometry, page size and kernel limits; do not pin per-rank heads
or rotary dimensions that the kernel does not require. Run the KV-row
registration fixture and native rewrite gate on any new row.

Regenerate `patch-footprint` after the final exact recipe composition;
commit its report and verify with `patch-footprint --check`. The archive and
footprint should make the **next** bump smaller to reason about.

## 3. Build and verify one coherent identity

```sh
nice -n 15 python3 -m bigcherry --engine radiance pin-bump <ref> \
    --source radiance --build stock --platform linux-radiance
# After an interrupted phase, use the recorded run:
python3 -m bigcherry --engine radiance pin-bump <ref> --resume
```

The declare phase also handles the release line: when the new pin's release major (2.2.0 is 2020) differs from the
release-please manifest's, it sets the package's `release-as` in the pin commit, and the release-please workflow
removes it after the release (`tools/bigcherry/release/release_line.py`). Nothing to do by hand; a missing or stale
key fails `tools/tests/release/test_release_line.py`.

Follow the reported phase failure, rather than bypassing it. The drift gate
checks component coupling and native rewrites; the compile gate rebuilds
components, checks exported plugin ABI **and layout fingerprint**, and runs
component CTests. Follow with lane build, CPU smoke and the exact patch
application/idempotence tests. All six component `built-against` identities,
the pin, runtime SDK, kernel roster and bundle digest must match.

The serving wire is **edition 3 only**. The single flat registry is
`engines/radiance/components/serving/contract/serving.toml`; generation
produces `generated/v3` schema/codec/fingerprints. Run `bc_contract_check`
and the component tests that verify generated fingerprint equality and v3
reader/writer/stored replay. Do not introduce a second native option mapping,
an old-edition reader or a parallel manifest index.

Every managed serving config is one native `NAME.yaml` with a closed
`x-bigcherry` section. Upstream-owned options stay in the native YAML;
BigCherry intent lives only in `x-bigcherry`. Inspect **all** shipped YAMLs
offline, with the matching candidate engine and `bc-serve explain NAME
--native-config engines/radiance/configs/NAME.yaml --model-root ${BC_MODEL_ROOT}
--engine-binary <candidate-binary> --json`, then validate variant/roster
consistency. No catalogue, layout file, `--layouts` or managed-start
`--set` projection is involved. Fresh resource admission and the live room check run on every start, including
a start that reuses a stored plan.

Pass the platform's absolute work root as `--work-root ${BC_WORK_ROOT}` to
enable the named-config store at
`${BC_WORK_ROOT}/cache/named-configs/`: a named start reuses the last measured
manifest of its config, keyed on the resolved settings, the ordered cards and
VRAM, the power profile, the config inputs and the model's path, size and
mtime. The model is never read, and the engine, helper and bc-serve builds are
provenance, not part of the key. Without `--work-root` every start measures.
A start under a new build or prefix carries the stored measurement and re-seals
it (`plan_carried_from` records the source, `carried-from.json` is written, no
measuring engine runs); only a start with no stored manifest to carry measures.
Never reuse historical free-VRAM estimates as static identity. A reused plan
is protected by admission: it runs on every start, and the engine refuses a plan
that does not fit.

Record ONE fresh measurement per new build, in a separate work root, and report
how its plan differs from the previous build's (it does not gate the A/B).
Parity arms of one comparison share a work root, so both run the same measured
plan.

## 4. Qualify on leased hardware before publishing

Run the **entire deployed YAML roster** from the installed, relocatable
runtime: offline explain, measured start, health, short completion, stop/drain,
and a second named start with `--work-root` (the reuse path). Exercise engine-only, helper-on and
mixed-rank topologies where the roster advertises them. Check ordinal order,
code objects, exact collective rendezvous and host/VRAM admission. Fix a
shared planner defect in the measured owner, not by adding per-config
headroom constants.

For raw model KLD/top-1 against a fixed BF16 reference, make a **temporary
YAML arm** with `num-speculative-tokens: 0` (disable draft) and match
tokenizer, corpus, positions and KV dtype. Do **not** change the production
draft depth merely for this quality test; the performance and replay gates
run at their authored depth.

Use matched old/new builds and identical helper topology/inputs for paired
prefill, cold TTFT and decode comparisons. Run **interleaved pairs, at least
two launches per arm** (the baseline tolerance is derived from the old arms'
spread, so one launch is refused) and a third pair when a result is within
that spread or an arm fails; long-prefill/long-8 can be bimodal, so a single
adverse result is not decisive. On a decode regression compare the engine
log's "the cache can grow to N GiB a card" line between builds before
anything else: a smaller GPU KV pool shows up first in warm (prefix-hit)
rounds. Repeat only to resolve genuine noise or regression. Baselines, KLD, kernel
rows and newly qualified formats must bind exact model, source, build and
runtime identities. Do not claim a clean bump from compile, one canary or
unpaired throughput alone.

Only after runtime, short quality, paired performance and installed-bundle
gates pass: record evidence and use the release workflow in
`docs/reference/build/RELEASES.md`. Do not push or publish as part of
read-only triage.

## 5. After the release: sync and clean up

Once the release PR is merged, the bundle published, and installed bundle
smoke passes (`bc-release install`, `verify`, then a start):

1. In the **primary** checkout run `git fetch` and `git pull --ff-only` only,
   never edit; then `python3 -m bigcherry --engine radiance pin-status` must
   report the vendor clone consistent with the new pin.
2. Close the bump's gateway sessions and cancel orphaned requests, then
   `audiagentic worktree finish` each task worktree (`abandon TIP` only after
   confirming its commits are in the merged branch; archive scratch first).
3. Rename a branch that `audiagentic worktree start` prefixed with the account
   name before its first commit.
4. Record the pin move in the plan and the ledger. Lessons from the 2.2 bump
   are in the PIN_BUMP.md Radiance subsection; add new ones there.
