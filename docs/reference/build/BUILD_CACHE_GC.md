# Build cache garbage collection (WSL09)

The work root's build cache consists of
`${BC_WORK_ROOT}/builds/<source-slice-id>/<build-plan-id>/` directories.
`bigcherry clean --builds` inventories candidates and protected entries,
showing path, size in bytes, age in days, metadata status, and reasons.
This is a **dry run** unless `--apply` is explicitly supplied.

A managed build must use lowercase 32-character hexadecimal IDs for
**both** the source-slice recipe directory and its build-plan directory.
Noncanonical recipe/build names and non-directory entries (including logs
and symlinks) at the cache recipe/build listing levels are reported as
protected **unclassified** entries, never as candidates, even with zero
minimum age and newest-K retention. Symlinks or special entries *inside*
a content-addressed build instead protect that entire build.
A partial build lacking a metadata sidecar is eligible only at a valid
content-addressed path. An inventory of unclassified files should not be
interpreted as a count of actual builds.

A build is eligible only if all of the following hold:

- It is a real directory under a real recipe directory, containing no nested
  symlinks, special files or cross-device entries. No Linux mount point
  (including a same-device bind mount) may exist anywhere from
  `${BC_WORK_ROOT}/builds/` down the candidate's path or within the candidate:
  this includes `builds/` itself, its source-slice recipe, the build directory
  and any nested file or directory. Mounts at or above the work root are a
  documented policy choice, not an automatic GC blocker. The cleaner reads
  `/proc/self/mountinfo`; unreadable/invalid mount tables protect every build.
- Its latest `bigcherry-build-metadata-*.json` sidecar and directory
  modification time are at least `--min-age-days` old (default: 7).
  A partial build without sidecars uses directory mtime; malformed
  sidecar JSON is protected. `--min-age-days` must be finite and non-negative,
  and `--keep-newest` must be a non-negative integer.
- It is not among the most recent `--keep-newest` builds of its source
  slice (default: 3, independent of build age).
- No managed run manifest names its build ID or path in a typed build
  reference or input. The `retention.hold` mechanism on a referencing run
  is respected. The work root must contain a genuine, fully readable `runs/`
  directory; missing, symlinked or partially unreadable run trees protect
  **all** builds rather than implying no references. Tracked repository
  release records naming a build-plan ID also pin that build. Installed
  `${BC_WORK_ROOT}/releases/<tag>/install.json` receipts (from CI bundles
  or recorded releases) pin their exact `build_dir`. An installed release
  directory without a valid receipt, or any uninspectable release records,
  prevents GC across all builds.
- It is not held by a build-plan resource lock or an active work-root
  lease. No process holds it as executable, working directory, open
  file descriptor, or memory mapping (scanned via Linux `/proc`).
  An inaccessible surviving process is treated as **unknown**, so
  candidates are protected rather than assuming the process is inactive.

Example:

```sh
bigcherry clean --builds --json
bigcherry clean --builds --min-age-days 14 --keep-newest 5
bigcherry clean --builds --apply
```

`--apply` is available **only** with `--builds`, and cannot be combined
with `--no-size`. Before removing each ordinary candidate, the cleaner claims
the same `resource-locks/build-<build-plan-id>.lock` as campaign builds,
rechecks all protections, and compares the inspected device/inode identities
of the work root, `builds/`, recipe and build under the lock. It opens
these components one at a time with `O_NOFOLLOW|O_DIRECTORY` and verifies
each identity using the previous directory descriptor. It then checks the
mount table and preflights the whole tree **without deleting anything**.

Before any destructive traversal, the verified build is atomically renamed
(using `renameat` with verified parent descriptors, on the same filesystem)
into a fresh random `gc-<device>-<inode>-<uuid>` entry of
`${BC_WORK_ROOT}/builds/.gc-quarantine/`. The quarantine is a private
`0700` directory owned by the cleaner; a symlink, unexpected permissions or
owner, device change, or mount point at or below the quarantine prevents
deletion. A failed rename does not delete the original build. After the
rename, the directory's device/inode identity and mount table are checked
again. The final bottom-up walk uses only verified `dir_fd` operations,
rejects symlinks, special entries, device changes and mounts, and removes
the quarantined directory via its verified parent fd. A later check can
also remove an empty original recipe using the verified `builds/` fd.

The rename removes the tree from its **original publicly reachable path**
before any deletion, so a concurrent rename at that original path cannot
reparent an open nested fd into an external directory during destructive
traversal. Quarantine privacy assumes that untrusted processes cannot
access the cleaner's `0700` directory: another process running as the same
Unix user (or root) may bypass that discretionary permission boundary.
Do not run adversarial same-UID filesystem mutations against the private
quarantine during GC. Mounts at or above the work root remain a policy
choice; mounts anywhere from `builds/` through the live candidate or its
quarantine, and unreadable mountinfo, fail closed.

If deletion fails **after a successful quarantine rename** (a failed
unlink, a newly mounted directory or unreadable mount table), the build is no
longer at its original path. The apply result reports its exact
`quarantine_path` under `skipped`, with a reason, and the audit journal
retains a `pending` event. A hard interruption (a signal that is not
handled, a killed process) returns no result at all; the leftover is then
found by the next inspection. A failure to write the completion journal entry
after the build was deleted is reported as `removed` with a warning. A later `--apply` **never deletes or resumes**
anything found in `builds/.gc-quarantine/`, whether or not there are pins,
locks or apparently valid device/inode fields. This is unconditional:
quarantine entry names and inode hints are informational only, never
authorization for automatic recovery.

Dry-run inventory lists each leftover as a distinct **protected**
`quarantine` entry with its path/name, approximate byte size when safe to
measure, and age from the directory's mtime (not time since quarantine).
Each says `interrupted removal; verify and remove by hand`. Such entries
are excluded from the recipe age/newest-K policy and from all apply
candidacy. Operators must inspect the journal and verify provenance,
active runs, original build-plan locks, references, mounts and filesystem
ownership before manually removing any leftover. Unknown, symlinked,
untrusted or mounted quarantine paths are never followed for deletion.
A post-removal failure when optionally pruning an empty recipe does
**not** reclassify the already removed build as skipped: the removal
and a `removed` journal event are still recorded, with a `warning`.
Empty recipe directories are pruned on a best-effort basis.

The cleaner never removes build-log inventory entries, run bulk or
materialised sources.

A journal of planned and completed removals is written to
`${BC_WORK_ROOT}/cleanup-records/build-gc-*.jsonl`. Each removal records
path, size, age and reason, with any post-removal prune warning on the
completed record. Inspect this journal after any apply,
particularly if deletion is interrupted. The default run cleanup and
legacy inventory remain non-destructive; source cache GC is not enabled.

On multi-user systems `/proc` access may be restricted. When any surviving
process cannot be inspected, the build inventory will show its protection
reason and `--apply` will not remove those builds. Resolve process
visibility through normal administration rather than bypassing the check.
Never use `--apply` on the real work root solely as a validation test.
