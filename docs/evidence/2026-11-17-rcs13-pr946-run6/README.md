# RCS13 / PR #946 — Run 6 source-isolated builds and strict-v2 stop

**2026-11-17 | Verdict: NOT CLEARED | Valid A/B/B/A serving samples: 0**

This evidence is a *failed-admission record*, not a decode or prefill benchmark.
The normal source-isolated bundles built successfully. Their baseline helper
could not start under the normal measured serving manifest. No production server
started; neither a speedup nor a slowdown is established.

## Exact source isolation and full builds

| Attribute | A (pre-merge parent) | B (PR #946) |
|---|---|---|
| Source SHA | `5c3c77f31498ab5f01ac132d22af27b32c69869a` | `5cfefdd19525bbbec04831961702ac24b55e91c7` |
| Campaign | `rcs13-run6-a` | `rcs13-run6-b` |
| Full Radiance lane | PASSED | PASSED |
| Build plan ID | `7d6ccd84ac007fddea5235f0d32fbb9f` | `8fb69437193df5a28f3b8ee197dd011a` |
| Installed/exported ABI | 35 / 35 | 35 / 35 |
| Engine SHA256 | `a5704dea94b8b7397ae0360a00bcfc48f311ae5c596a1abf3c62a77c8886a4ec` | same |
| libr4d SHA256 | `e8d357890afbe221537cbc04f713f730d3d970e002cb2a93f2e26edc0114f726` | same |
| Helper SHA256 | `c047bf5fc83e7eea4b0952f11e1dab979aaa4c6e1701dfb416c4579849ee0d51` | `893594f3d0765d0d56277b3579d12f76beadec1a2c6e00d77bf9c8239ef6e95f` |
| gfx12 plugin SHA256 | `c68975a078b076c1f401f3ce6994fb405877292e9a413b6ad1510be1e914ba7e` | `7226dfb3187af6f9ed5fe990e416270505069d1a460bc4319e34aff1de5eda5d` |

Both invoked the official `python3 -m bigcherry build --lane
radiance:stock:linux-radiance --run-id rcs13-run6-{a,b}` separately,
with detached PID/terminal receipts and no cross-arm component mixing.
Both have the same upstream slice `833e240cd588a59e91ba7960fcbbda9f`, Radiance v1.17.0,
30 validated patches, zero pending. The seven-file source difference is
limited to RCS13 GEMV implementation, geometry tests/headers and model guard.
Patch files, recipe bytes, build tool and effective CMake configuration match
except for the absolute location of identical `bc-ccache` binaries.
Other source-identical shared libraries can have different whole-file hashes
because source-path debug sections/build IDs vary; do not interpret such
whole-file differences as independent code modifications. See
`build-attestations.json` for full module hashes and the source-diff list.
Live helper ABI handshake was not completed.

## Normal strict-v2 planner gate

Used normal `RadianceAdapter.write_serving_manifest()` ->
`radiance_plan.write_manifest()` (not a scratch replacement), layout
`flashnext-tp2xtx-r9700`, the model's Flash-Next container and repo heat
`tools/lab/radiance/heat/perf51.rank0.hot-first`. Repository heat SHA256:
`cbf8e17ad4e0e57470da2c000904d863621d5affcf8876c9b87960b9472ba06e`.
The stock manifest requests one session, 8192 context, 64 max handoff rows,
8 small rows, disk ngram, W4H2 enabled and an exclusive bounded GPU lease.

The CPU owner-map budget preflight derived host pool **32005 MiB**, host KV
**0 MiB**, helper expert budget **29656 MiB**. It published **12269** owned
experts with digest `c89150871acd9404cf96255644df017842ddc80143f9e2b0819daf69a1b96c71`.
The measuring launch instead published **12641** owned experts (372 extra)
with digest `744477bc8f39f18bdc68f472b4f83d6cbb2b576c0095439fe463cd84427ca391`
and helper budget **30555 MiB**. These are different placements.
Historic run-3 figures of 32301 MiB/755 owners were **not** reproduced here.
The CPU-only reproducer re-invokes the `_collect` owner-map command and asserts
the resulting digest matches the independently derived budget but not the
measuring draft.

The **first terminal failure** was helper0 preparing lanes, before engine
measurement (helper exit 1). Original log excerpt:

```text
r3_moe_worker: R3_MOE_WORKER_GEMV_W4H2 on - packed f16 W4 decode GEMV
r3_moe_worker: prepared lanes take 16924112 bytes, the shared footprint says 16923992 (worker_footprint.h out of step with the lane buffers)
r3_moe_worker: GPU lane preparation failed
```

The 120-byte drift is independently reconstructible without GPU execution:
`moe_remote_gpu.h:492` allocates large-lane chunk counters using the
**protocol maximum** 4096 rows / 256 rows per chunk = 16 `uint32` counters =
64 bytes *per lane*. `worker_footprint.h:84-113` derives the fixed
budget using the **configured** 64-row large lane, counting 1 `uint32`
counter = 4 bytes per lane. Two engine-rank large-lane slots contribute
`(64 - 4) * 2 = 120` extra runtime bytes. At
`moe_remote_gpu.h:513-521` the runner explicitly refuses the mismatch.
Protocol constants: `moe_remote_shm.h:36-41`. Helper row selection:
`worker_shape.h:11`, `r3_moe_worker.cpp:282-302`.
Owner-map derivation: `tools/bigcherry/engine_adapters/radiance_launch.py:103-166`,
measurement `radiance_plan.py:108-146`; publishing:
`engines/radiance/components/serving/manifest_writer.cpp:482-510`.

## CPU-only reproducer

`cpu_gate_repro.py` is a parameterized, offline variant of the executed
run-6 reproduction. It requires the source-matched build, its source checkout,
the published repo heat, and the original measuring draft and budget.
It never launches the engine, starts GPU helpers, adjusts an expert budget,
or writes a measured manifest.

```sh
python3 cpu_gate_repro.py --source-root "$BASELINE_SOURCE"   --build "$BASELINE_BUILD" --heat "$HEAT"   --site "$PLAN_DIR/site.toml"   --budget "$PLAN_DIR/budget.json"   --measuring "$PLAN_DIR/manifest-measuring.json"   --helper-log "$PLAN_DIR/helper0.log"
```

## Measurements, disposition, retention

| Gate | Result |
|---|---|
| Source/build composition and installed ABI | PASS |
| Owner-map identity | **FAIL** (12269 vs 12641) |
| Helper footprint exact match | **FAIL** (120 bytes) |
| Measured strict-v2 manifest | **NOT CREATED** |
| A/B/B/A server samples across prompts/seeds | **0** |
| Decode tok/s, engine ms/step | N/A |
| Prefill tok/s, TTFT | N/A |
| MTP draft/accept counts | N/A |
| Cross-arm output hashes | N/A |
| Live gfx1201 K2560 W4H2 activation | None |

The W4H2 startup switch is **not** executed-kernel proof. The baseline
failed before engine launch, so the candidate GPU test and all A/B/B/A
cells were deliberately skipped. No guessed host pool, alternate heat, patched
runtime allocation, mixed ABI, or unmeasured fallback was used. The GPU
lease automatically released and no KFD processes remained. Source
PR #946 was not changed, committed, or pushed.

**Builds and failing run files are intentionally retained**, not deleted:
the explicit no-deletion requirement lasts until a valid measurement is
completed and its results are in evidence. No decode/prefill verdict should
be issued until the upstream helper/footprint and owner-map discrepancies
are resolved independently.
