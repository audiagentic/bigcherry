# Run layout

Runs live at `<work-root>/runs/<kind>/<run-id>/run.manifest.json`. The kinds are `ab`, `accept`, `stream`, `replay`,
`capture`, `kl`, `profile`, `build`, `bump` and `release`. A run is created with `bigcherry run new <kind> <topic>`, not
by creating directories by hand. Manifests use schema version **1**, with separate run, harvest and retention states,
typed references and generation-checked atomic updates.

## Anything else under `runs/`

A directory directly under `runs/` whose name is not a run kind (a flat run from before manifests), a symlink, and loose
files are listed by `bigcherry clean` as unknown entries. They are never classified, adopted, migrated or removed by the
tools: keep what matters in the evidence documents and delete the rest by hand.

Runs do not pin builds. The build cache is reclaimed by age and use alone (see [Build cache garbage
collection](BUILD_CACHE_GC.md)); a run's results are its manifest, its records and the evidence written from it.

## Run cleanup

`bigcherry clean` without `--builds` reports what run cleanup would reclaim and why every other run is protected: a run is
reclaimable only when its harvest is verified and its retention has expired, and a `retention.hold` keeps it. It never
deletes anything; only `clean --builds --apply` removes (builds).
