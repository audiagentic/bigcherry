# Radiance CI build: where the time goes

Question: the `radiance / build` job takes 9 to 15 minutes. Is the HIP compile the critical path, and what is worth
changing. Measure only; no workflow or tooling changed. Source: the job logs of three recent main runs and two PR runs
of `radiance-build.yml`, plus a local serial-versus-parallel run of the serving component's tests.

## Phases of one job (seconds, from the log timestamps)

| Run | Lane build | Component tests | of which ctest | ccache hits | Cache size, cleanups |
|---|---|---|---|---|---|
| main, warm (01:44) | 52 | 322 | 271 | 918 / 923 (99.5 %) | 5.0 / 5.0 GiB, 5 |
| main (07:22) | 388 | 479 | 321 | 834 / 932 (89.5 %) | 5.0 / 5.0 GiB, 12 |
| main (07:55) | 192 | 348 | 245 | 857 / 933 (91.9 %) | 5.0 / 5.0 GiB, 44 |
| PR (07:40) | 193 | 376 | 274 | 932 hits listed in the log | 5.0 GiB, full |
| PR (08:47) | 72 | 346 | 291 | 933 calls, all cacheable | 5.0 GiB, full |

(Lane build = `bigcherry build --lane`; component tests = `bigcherry component-tests`: configure and build every
declared test target, then ctest. Job wall also includes about 35 to 50 s pulling the image, 35 to 70 s restoring the
5.3 GB ccache, and on main 24 to 31 s saving it, 109 to 145 s packaging it for the lab and 27 to 56 s publishing it.)

1. **The component tests, not the HIP compile, are the largest phase on a warm cache.** The 245 HIP commands sum to 4.3
   to 15 s of elapsed compile time (the job's own `ci_build_profile` report; commands overlap, so not wall time), and
   the lane build is 52 to 193 s when the cache is warm. The component-tests phase is 322 to 479 s every run.
2. **Inside it, one test is the long pole.** In the serving component the 35 tests ran one after another (`ctest`
   gets no `-j`, `runtime_bundle.py` line 186): `bc_session_test` 146 to 185 s, `bc_release_install_test` 36 to 41 s,
   `bc_serving_test` 27 to 30 s, `bc_session_stored_test` 12 s; everything else is under 7 s. About 245 to 320 s of
   the phase is this serial run.
3. **Running ctest in parallel buys little.** Locally, the same 35 tests: serial 234 s wall, `-j8` 198 s wall, because
   `bc_session_test` alone takes 140 to 146 s. Parallelism would save about 36 s of the roughly 240 s; to go below
   about 145 s the session test itself has to be split or made faster.
4. **The ccache is full and evicting.** Every run shows `Cache size 5.0 / 5.0 GiB` and 5 to 44 cleanups; the main runs
   with the most cleanups (07:22 and 07:55) also had the most misses (98 and 76 of about 930) and the longest lane
   builds (388 and 192 s, against 52 s with 5 misses). The cache is saved and restored whole each run (about 5.3 GB:
   35 to 70 s restore, 24 to 31 s save, plus the lab packaging and publish steps on main).

## Proposals (not done; each needs its own A/B)

- Split `bc_session_test` (146 s) into several CTest tests and run ctest with `-j`: from about 245 s to roughly the
  longest remaining test. Needs the tests to be independent (they use a private work root already; check the fixture
  ports and the shared `bc-serving-fixture` directory they create relative to the working directory).
- Raise the ccache size limit above what one lane build touches, so a warm main run stops evicting; compare the next
  main runs' miss counts and lane-build times. Do not pay for it in cache size alone: the save and restore are already
  about 100 s per main run, so measure whether a larger cache costs more than the misses it removes.
- Do not touch HIP parallelism (the manual `2` and `6` options): the HIP compile is not on the critical path on a warm
  cache.

## Not measured

Ninja's own critical path (`.ninja_log` is only reported as a sum of HIP durations), per-target ccache misses (the log
has the totals only), and cold-cache runs (the force-cold option was not run).
