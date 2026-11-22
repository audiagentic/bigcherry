# Radiance CI ccache: eviction or change-driven misses

Question: the CI ccache is always full (`5.0 / 5.0 GiB`, 5 to 44 cleanups per run, see
`2026-11-22-radiance-ci-critical-path`). Are the misses of the slow runs caused by evicting entries the next build
needs, so that a bigger cache would help.

## Measurement

One cold lane build (`bigcherry build --lane radiance:stock:linux-radiance`, the lane CI builds) on an empty ccache with
no size limit, the same ccache settings the build sets for CI (stable source and build paths, content compiler check):

| Quantity | Value |
|---|---|
| Cacheable compiles | 846 (CI reports 923 to 933 per job; the rest are the component test targets) |
| Hits / misses | 35 / 811 |
| Cache after the build (the working set of the lane) | 212,205,000 bytes (0.2 GiB) |
| Wall time, cold, 20 cores at the lowest priority | 1169 s |

## Reading

- One build's whole working set is 0.2 GiB, 4 % of the 5.0 GiB limit. ccache evicts the least recently used entries
  first and every run touches its whole working set (direct hits refresh them), so a build's own entries can only be
  evicted if more than 4.8 GiB of other entries are newer than the last run that used them. One run adds about
  0.26 MB per miss on average (0.2 GiB over 811 misses), so the 75 to 98 misses of the slow runs add about 25 MB.
  Eviction of entries a build needs is not a credible cause of those misses; they are changed sources (a changed
  header or pin recompiles the units that include it).
- So no larger size is proposed. A bigger limit would only grow the 5.3 GB the job restores (35 to 72 s on every run)
  and, on main, saves, packages and publishes (about 190 to 260 s).
- The full cache is mostly entries nothing uses any more. Not measured: how much of the 5.0 GiB the last run actually
  touched (`ccache -s` gives the totals only), and the share the component test targets add. A limit derived from
  these (a small multiple of the touched working set) would shrink the restore and the main-only save and packaging;
  that needs a run with the touched size recorded before any value is chosen.
