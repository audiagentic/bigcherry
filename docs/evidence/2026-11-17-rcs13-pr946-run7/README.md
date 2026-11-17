# RCS13 / PR #946 — Run 7 strict-v2 admission

2026-11-17. **NOT CLEARED; 0 valid end-to-end A/B/B/A samples.** This is a fresh test after worker-footprint fix #1056, not a replay. Both official source-matched Radiance campaigns completed successfully: A main `8c10d1cea356ea4e1e41e2df4ef630e1bb9fa1d0`, B #946 `aa7fd2ead1fd50ba72908c7ce0fef7eedcaca2a3`. Exactly the seven RCS13 files differ. Both use pin v1.17.0 and 30 validated / 0 pending patches; ABI and all binary hashes are in `build-attestations.json`.

Normal `RadianceAdapter.write_serving_manifest` admission: Flash-Next, layout `flashnext-tp2xtx-r9700` (TP2 gfx1100 engine + gfx1201 helper), stock, one session, context 8192, repo heat `tools/lab/radiance/heat/perf51.rank0.hot-first` SHA-256 `cbf8e17ad4e0e57470da2c000904d863621d5affcf8876c9b87960b9472ba06e`. The sourced preflight owner map differs from the normal measuring-draft owner map:

| | CPU preflight | Measuring draft |
| --- | --- | --- |
| Owned experts | 12,269 | 12,641 |
| Owner-map SHA-256 | `c89150871acd9404cf96255644df017842ddc80143f9e2b0819daf69a1b96c71` | `744477bc8f39f18bdc68f472b4f83d6cbb2b576c0095439fe463cd84427ca391` |
| helper VRAM budget, MiB | 29,656 | 30,555 |
| max_rows | 4096 | 64 |
| host_pool, MiB | 9216 | 32005 |

A no-GPU replay with the source-matched `bc-serve --write-manifest --unmeasured` and the exact preflight flag set reproduces the official budget's owner digest. Its owner map differs from the measuring draft by preflight-only 0 and measuring-only 372 experts.

## First terminal gate: host-pool reconciliation (exit 11)

After the corrected helper started and the engine wrote its normal measured plan,
the normal adapter failed at the final measured-manifest writer: **the plan was
measured with 32005 MiB host pool but the measured inputs need 32301 MiB**.
No strict-v2 measured manifest was produced. The exit is deliberate in the
C++ writer: `engines/radiance/components/serving/main.cpp:684-711` recomputes S8
host-pool demand with `v2/s8_rules.cpp:57-66`, emits
`derived_host_pool_mib=32301`, and returns 11. Its paired C++ owner
`serving/session.cpp:1928-1942` handles exit 11 by remeasuring the plan
with the new host pool (bounded retries); **the Python serving adapter does not**,
and raises `PlanError` on any nonzero result at
`tools/bigcherry/engine_adapters/radiance_plan.py:151-159`. This is a real
planner retry-protocol mismatch, not the fixed GPU lane-footprint defect.

An independent CPU-only replay, using the original measured engine plan and
rank demand, the real cached residence inputs (helpers 509 MiB, engine 183 MiB),
and exactly the same serving overrides in an isolated scratch plan cache,
reproduced **exit 11, 32005 vs 32301 MiB** without launching a helper or engine.
See `cpu_host_pool_repro.py` and `planner-gate.json`.

## Independent owner-map identity divergence

**Cause to fix in a separate lane:** `tools/bigcherry/engine_adapters/radiance_launch.py:103-166` derives the host budget using a preflight owner map while omitting the serving overrides later sent by `tools/bigcherry/engine_adapters/radiance_plan.py:108-146`. `engines/radiance/components/serving/manifest_writer.cpp:490-519` recomputes the owner map from the actual draft/helper budgets. Differing input policies produce differing identities. This divergent owner-map identity is an additional admission defect, independent of the 120-byte GPU lane-allocation mismatch fixed by #1056. The 32301 MiB re-derivation must be handled by remeasuring; the input reconciliation must be fixed without substituting a guessed budget. Do not paper over it by inventing budgets or replacing manifests.

This STOP precedes production serving. Decode ms/step and throughput, prefill/TTFT, MTP draft/accept counts, output-hash parity, and live K=2560 W4H2 kernel activation are all **not measured**. The historical GPU replay does not establish any speedup. PR #946 had no new commits. The GPU lease released after the bounded planner attempt.

Artifacts: `build-attestations.json`, `planner-gate.json`, portable `cpu_gate_repro.py` and `cpu_host_pool_repro.py`. Source/runtime paths and host details are intentionally excluded.
