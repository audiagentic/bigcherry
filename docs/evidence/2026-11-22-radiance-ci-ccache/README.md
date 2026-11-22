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
- The full cache is mostly entries nothing uses any more (measured below).

## Touched size on CI

ccache refreshes the modification time of every entry it hits or writes (checked: a hit makes the entry newer than a
marker taken before it). A measurement step (`ci/rum18-ccache-touched-size`, dispatched on a branch, restoring the
main cache) took a marker before the build and summed the bytes of the cache files newer than it afterwards.

| Quantity | Value |
|---|---|
| Restored cache, total | 5,343,616,614 bytes (5.0 GiB) |
| Touched by the build (hit or written) | 277,762,161 bytes (0.26 GiB), 5.2 % |
| Hits / misses | 923 / 933 hits (98.9 %), 10 misses, 10 cleanups |
| Steps | restore 89 s, build in the image 370 s |

The local cold lane build (0.21 GiB) agrees: the working set of the whole job, lane and component test targets, is
about a quarter of a GiB. 94.8 % of what the job downloads and, on main, saves, packages and publishes (about 190 to
260 s per main run) is entries the build did not use.

## Size rule (change proposed in the PR that follows)

The limit should hold the touched working set of the current build and that of the build before it (a pin or toolchain
bump replaces nearly the whole working set once, and the next run needs the old entries only for the PR runs still
based on the old main): twice the touched bytes of the last successful main build. The factor is the two generations,
not a tuning knob. On main, after a successful build, the job sets the limit to that value and trims the cache before
it is saved, so the cache follows the measured working set instead of a fixed 5 GiB.
