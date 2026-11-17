# RCS13 / PR #946 — run 5 full-build and strict-v2 preflight evidence

Date: 2026-11-17 UTC. **Verdict: NOT CLEARED; zero valid end-to-end samples.**
This is a failed-admission diagnostic, not a performance result. Nothing here
supports a decode/prefill regression or gain claim.

## Source isolation and production builds

| | Arm A, pre-merge main | Arm B, PR #946 |
|---|---|---|
| Git revision | `5c3c77f31498ab5f01ac132d22af27b32c69869a` | `5cfefdd19525bbbec04831961702ac24b55e91c7` |
| Full production build | succeeded | succeeded |
| Build plan | `1729da15df800a9cad2fe9891ccf8fb0` | `d225ef62f99dc617f89c973e235bc9bd` |
| Content-addressed Radiance source slice | `833e240cd588a59e91ba7960fcbbda9f` | `833e240cd588a59e91ba7960fcbbda9f` |
| Upstream pin and patch admission | v1.17.0; 30 validated, zero pending | v1.17.0; 30 validated, zero pending |
| `libr4d` exported ABI / installed header | 35 / 35 | 35 / 35 |
| Engine SHA-256 | `a5704dea94b8b739...` | `a5704dea94b8b739...` |
| `libr4d.so` SHA-256 | `e8d357890afbe221...` | `e8d357890afbe221...` |
| `r3_moe_worker` SHA-256 | `ff51ff0ee7cf9888...` | `893594f3d0765d0d...` |
| `libbcgfx12.so` SHA-256 | `0dd1bc70b92f5a03...` | `7226dfb3187af6f9...` |

Both were built through the normal detached
`python3 -m bigcherry build --lane radiance:stock:linux-radiance --run-id rcs13-run5-{arm}`
campaign from their own clean checkouts. The work-root helpers and plugins were
**not** swapped or hand-assembled. The global `--engine radiance` option is rejected
by this version of the build CLI; Radiance is selected by the lane instead.

`git diff 5c3c77f31498..5cfefdd19525` comprises exactly seven RCS13 GEMV
implementation/geometry/guard files, recorded in `run5-composition.json`.
The recipes, patch source files (excluding ignored generated `__pycache__`),
build tool and ccache wrapper have byte-identical hashes; full patch admissions
and effective CMake settings agree. The only CMake settings difference is the
absolute source-worktree path to the **byte-identical** `bc-ccache` launcher.
The same content-addressed upstream source slice independently confirms this.

The installed engine header and the exported `libr4d.so` function both report
ABI 35 in each independently built runtime. Each `r3_moe_worker` was compiled
from the corresponding arm's checkout against that build's SDK include tree.
A live helper ABI handshake was **not** reached because the manifest gate failed.

Full-file hashes in several source-identical libraries also differ: static ELF
section comparison shows `.text` and `.hip_fatbin` identical, with changes only
in debug sections and some build-ID notes. Those are path-dependent rebuild
artifacts, not proof of changed executable kernels. The intended helper and
gfx12 plugin differ in executable code and HIP fatbins, as expected.
See `run5-section-comparison.json` for per-section digests.

## Strict-v2 gate: deterministic failure BEFORE the measured-manifest phase

Arm A gate began 2026-11-17 08:46:34 UTC under a bounded, automatically
released `agent-lease acquire gpu --ttl 30m --est 20m --wait --`.
GPU status had shown no KFD PIDs and 77 GiB available host RAM.

The launcher stops at `radiance_launch.py:103–120` while calling
`bc-serve --write-manifest --unmeasured` for the owner-map
needed to derive the RSC03 resource budget. It returns exit **2**:

```text
r3_moe_worker: R3_MOE_WORKER_HEAT ${BC_WORK_ROOT}/runs/perf51/heat.rank0.hot-first: want 65536 counts, read 0
bc-serve: the declared placement cannot be planned for this model: helper 0 (gfx1201): owned-expert selection refused
```

The referenced heat file and even its `${BC_WORK_ROOT}/runs/perf51`
parent no longer exist. This is a missing **prerequisite input**, not the
32005/32301 MiB measured-ledger failure or the 12641/755 owner-map discrepancy
observed in run 3. In `engines/radiance/components/serving/experts.cpp:151–162`,
`fopen` failure leaves the number of read uint32 counts at zero, while the
selected model requires 65536 counts; the function returns false.
`experts.cpp:366–380` surfaces helper 0 owner-map selection refusal.
`tools/bigcherry/engine_adapters/radiance_launch.py:108–120` refuses to
continue without this owner map.

### Minimal reproduction

`run5_minimal_repro.py` reproduces *only* the launcher's unmeasured
`bc-serve` owner-map command from the same source, model, binary, layout,
site fields and absent heat path; it does not start the engine or any helpers.
It was executed and asserted exactly exit 2, the 65536/0 error, and the
owner-map refusal. To reproduce against a rebuilt source-matched arm:

```bash
python3 run5_minimal_repro.py --source-root <arm-checkout> --build <arm-build>
```

The production adapter's prior failure is in `run5-measured-plan-a.log`;
the original setup and settings are in `run5_plan.py`. The minimal script
asserts the heat file remains **absent**, so it fails loudly if the
precondition changes rather than silently claim a reproduction.

## STOP and non-results

**No alternative heat, guessed pool, altered helper budget, or unmeasured
substitute was used.** The strict-v2 measured manifest was not generated,
no production server session started, and no arm B GPU gate or A/B/B/A
measurement was attempted after arm A's failed prerequisite.
The GPU lease was free after the failing invocation.

Decode ms/step, prefill TTFT/tok/s, MTP drafts/accepts, cross-arm output hashes,
and executed gfx1201 K=2560 W4H2 dynamic-LDS kernel activations: **N/A;
zero samples**. Historical card-replay identity evidence remains separate.

The run's two builds and diagnostic run files are staged in this Git evidence
slice, not retained in the shared work root after archival. The source
PR #946 branch was not committed to.
