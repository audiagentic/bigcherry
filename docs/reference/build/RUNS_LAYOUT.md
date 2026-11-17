# Managed and historical run layout

New runs use `<work-root>/runs/<kind>/<run-id>/run.manifest.json`. The
managed nested kinds are `ab`, `accept`, `stream`, `replay`, `capture`,
`kl`, `profile`, `build`, `bump` and `release`. New runs are created
with `bigcherry run new <kind> <topic>`, not by creating directories by hand.
Manifests use schema version **1**, with separate run, harvest and retention
states, typed references, and generation-checked atomic updates.

## Pre-manifest flat runs

Historical `runs/<name>/` directories are *not* nested under a managed kind.
They may be adopted without moving or deleting their data by adding a
`run.manifest.json` with `kind: "legacy"` and `run_id: "<name>"`.
`legacy` is a **reader-only** kind, not an option for creating new runs;
this is the only change to the v1 schema's list of allowed kinds.
The retention inventory accepts a flat manifest only when it is a **regular,
non-symlinked file** with `provenance: "legacy_inferred"`, valid `legacy_scan`
evidence with the migration scanner version, and a matching current tree stamp.
An unadopted, forged/incomplete, corrupt, unreadable or changed flat run is
**unclassified** and blocks **all build GC**; a v1 legacy-shaped manifest by
itself is not proof of adoption. A pre-existing manifest is **never**
overwritten by the migration tool, whether valid or invalid. The evidence and
stamp are metadata checks for a trusted single-user work root, not a
cryptographic attestation against a malicious operator.

Run from the BigCherry repository with `PYTHONPATH=tools`:

```sh
python3 -m bigcherry legacy-runs migrate --work-root /work/bigcherry --json
# Review the per-run JSON and errors before choosing to apply:
python3 -m bigcherry legacy-runs migrate --work-root /work/bigcherry --apply
```

The default is a **read-only dry run**, which reports each run, every
discovered `source-slice-id/build-plan-id` typed build reference, files and
bytes scanned, earliest/latest file mtime, a scanned-tree stamp, and whether it
would write a manifest. The scanner reads **all** regular text and binary files in chunks,
with overlap across chunk boundaries and no file-size ceiling. It matches
both relative `builds/<32-hex>/<32-hex>` and absolute paths containing that
component sequence. Nonexistent builds remain references. Every scan must
complete: a read failure, symlink, special file, device change or concurrent
tree mutation yields **no manifest**, with an error requiring operator review.
Before publication the writer repeats a non-following tree stat walk, comparing
**every** scanned file's device/inode, size, nanosecond mtime/ctime, and all
directory membership and identities with the completed scan. New, removed,
renamed or modified entries fail the adoption. The manifest records
`legacy_scan.tree_stamp` with file count, total regular-file bytes, maximum file
`mtime_ns`, directory count and a SHA-256 digest of file path/identity/size/mtime/ctime and directory path/identity metadata. On every retention/GC inspection, the reader
re-walks the run and checks that stamp; a mismatch, an uninspectable file, or
untrusted provenance makes the run unclassified and blocks all build deletion.
The stamp catches ordinary later changes, but is not a content hash and does
not prevent deliberately restoring metadata values. The migration also fails
closed on Linux mount points at `runs/`, at the candidate run, or anywhere
below it (including same-device bind mounts). An unreadable mount table
prevents adoption and validation.

An apply acquires the usual run manifest lock, writes the manifest at mode
`0644` via an fsynced temporary file and a no-overwrite atomic link, and
never intentionally moves or deletes legacy data. It checks the run's directory
identity under `runs/<name>` again **after publication** and unlinks its
just-published manifest through the original directory descriptor if the run
moved. This detects a rename during publication, but is not a transactional
defence against a malicious concurrent mover. **Do not move or rename runs
while migrating**; only operate on a quiescent single-user work root.

Imported records use `provenance: "legacy_inferred"`, unknown original
execution identity, `harvest.state: "pending"`, and an active run state:
the import **does not assert** a completion or verified harvest. The standard
retention policy therefore protects them from run cleanup; operator evidence
and separate lifecycle work are required before changing that protection.
No default harvest recipe is assigned to the reader-only legacy kind;
reconciliation and explicit evidence review must precede any attempt to
harvest it. Build GC can use the exact typed build edges immediately.

Flat **regular files** directly beneath `runs/` (legacy notes, logs and other
loose evidence) do not get directory manifests and are never eligible for run
cleanup. On each build GC inspection their full contents are scanned
read-only for build edges; a fully scanned file stops being a blanket blocker,
but any detected build remains pinned. An unreadable file, symlink, special
entry or unclassified nested path **continues to block all build GC**.
This rule does not silently discard historical references.

See [Build cache garbage collection](BUILD_CACHE_GC.md) for the additional
independent checks required before any build removal.
