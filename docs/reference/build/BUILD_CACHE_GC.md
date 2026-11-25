# Build cache garbage collection

The work root's build cache is `${BC_WORK_ROOT}/builds/<source-slice-id>/<build-plan-id>/`, both names
lowercase 32-character hexadecimal. `bigcherry clean --builds` lists every build as a **candidate** or **protected**,
with path, size in bytes, age in days, metadata status and the reasons. It is a dry run unless `--apply` is given.
Entries that are not content-addressed builds (logs, symlinks, odd names) are listed as protected *unclassified* and
are never removed.

## What keeps a build

A build is kept for exactly these reasons, and for no other:

- **A visible process uses it** as executable, working directory, open file or memory mapping (scanned through Linux
  `/proc`). Processes of other users and of other containers cannot hold a build here and an unreadable foreign process
  does not block anything.
- **An installed release or release bundle points at it**: a receipt `${BC_WORK_ROOT}/releases/<tag>/install.json` names its
  `build_dir`, a native install record names its build identity, or a tracked release record in the repository names its
  build-plan ID.
- **A build-plan resource lock exists** (`resource-locks/build-<build-plan-id>.lock`: a build is in progress or in use).
- **It is younger than `--min-age-days`** (default 1): the newer of the metadata sidecar's time and the directory's
  modification time. A malformed sidecar keeps the build.
- **It is among the newest `--keep-newest` builds of its recipe** (default 0: none is kept for being the newest).
- **Cleanup is blocked**: another cleanup or active work-root activity is running.

Run manifests, run records and stored serving manifests are history, not pins: an old build is reclaimable even when a
run once used it, and an unreadable or unknown manifest never blocks cleanup. What the run measured is kept in the
evidence documents and the run records, not in the build cache.

A build is also refused if it is itself a symlink or a mount point lies on its path or inside it. Symlinks, special files
and files of other filesystems *inside* a build do not matter: removal unlinks them and never follows them.

## Applying

```sh
bigcherry clean --builds --json
bigcherry clean --builds --min-age-days 14 --keep-newest 2
bigcherry clean --builds --apply
```

`--apply` is available only with `--builds`. For each candidate the cleaner takes the same
`resource-locks/build-<build-plan-id>.lock` that campaign builds take, checks it is still a real build directory and
removes the tree. The removals are journalled in `${BC_WORK_ROOT}/cleanup-records/build-gc-*.jsonl`: a `pending` and a
`removed` event per build with its path, size, age and the reason it was eligible. Empty recipe directories are left
alone. The source cache is inventoried read-only and never garbage-collected here, and build logs are never removed.
