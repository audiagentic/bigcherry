# S8 measured resource and handoff rules (inactive until S13)

S8 is a **GPU-free resolver** in `engines/radiance/components/serving/v2/s8_rules.{h,cpp}`. It consumes the S9 `v2::MeasuredResourceInputs` type and qualified calibration records. It is linked only by `bc_v2_s8_test` and **does not alter the running v1 resolver, manifest writer, shell launcher, or engine**. A caller may compute candidates offline; the single v2 production activation is S13.

## Measured interface and derivations

S10 supplies engine `--plan-memory` byte measurements, the real owner-map digest, S7 host reserve, checked actual helper owner sets and worker-allocated nonexpert footprints, live MemAvailable/effective cgroup/pinnable bytes, current per-BDF live claimable VRAM and architecture reserve with evidence, a measured transport row limit, and the objective's context/retention target. **Neither a RAD metadata approximation nor the historical 9,216 / 12,288 / 22,528 MiB recipes can fill a missing measurement.** `resolve_s8` checks all supplied quantities and returns these closed v2 types: `MeasuredResourceInputs`, `HostResources`, `HandoffTransport`, `HandoffKernels`, `HandoffTelemetry`, per-BDF helper expert budgets, `CalibrationEntry` records, and `ProvenanceRecord` records by pointer.

- **Host pool:** `ceil_mib(engine.host_embedding_bytes + owner_map.unowned_expert_bytes + staging_bytes + calibrated host_ledger_margin.mib × MiB)`. Pool reservation is driven by the full measured ledger. Do not invent an extra headroom margin. Only `host_ledger_margin.mib` (the existing RSC10 key) expresses host movement between resolution and startup.
- **Host KV:** `ceil_mib(retention_target_tokens × engine.kv_bytes_per_token)`. The context/retention target comes from service intent and the measured engine KV geometry; no 12 GiB tier default. S10 decides which retention target to admit against the objective and free host room. Reserve and other measured resident consumers remain separate.
- **Host admission:** `(pool_mib + kv_mib) × MiB + ngram_bytes + checkpoint_bytes + resident_other_bytes + S7 host_reserve_bytes` must fit the *minimum* of live MemAvailable and cgroup headroom. Pinned pool and KV must fit current pin headroom. All arithmetic is checked for I-JSON overflow. S10 publishes the same plan into the v2 manifest; S11 handles a structured host-room retry.
- **Per-helper expert budgets:** `floor_mib(min(live_BDF_claimable_bytes, physical_BDF_bytes - qualified_arch_reserve_bytes) - sum(worker_footprint_component_bytes))`. Every component (lanes, pointer banks, slot map, incoming slot) and each role/BDF must match exactly; otherwise refuse. The engine claimable VRAM is checked without tolerance. The **11 November R9700** example is a refusal: required 33,242,744,312, claimable 33,241,708,024, shortfall **1,036,288 bytes**. It cannot be fixed by a hard-coded 1 MiB buffer.

## Five RSC09 handoff decisions

| Field | S8 rule | Source and qualification |
|---|---|---|
| `transport.max_rows` | Qualified calibrated row limit, at most the currently measured engine/worker dispatch capacity | Worker/engine row-batch sweeps; compare throughput, tail latency, output identity, queue depth and overflow to select an admissible maximum. The capacity is an independent measured upper bound, not a default. |
| `transport.small_rows` | Qualified calibrated small-lane crossover, positive and below GEMV large crossover | Same-build, same-layout row-shape microbench and end-to-end A/B with throughput, latency and correctness. |
| `transport.gemv_large_rows` | Qualified calibrated GEMV large crossover, above small-lane limit and at most max rows | A/B at the actual helper BDFs across decode row distributions, output correctness and 4/6/8 streams. |
| `transport.skip_owned` | **Derived Boolean:** any helper owns an expert in the actual map; no calibration or default | Actual authoritative helper-owned expert sets, digest-equal to `resource_inputs.owner_map_digest`; no hand-picked `true`. |
| `kernels.moe_down_tile` | Qualified calibrated choice among permitted v2 modes 0, 1 and 2 | Per-arch/kernel-build A/B on measured MoE downprojection shapes, correctness checks and throughput; no historical `1` implied by profile. |

Other handoff values are explicit too. `telemetry.period_s` and live-placement `cadence_ns` are **separately calibrated** if histogram/live placement is configured; they must have independent freshness/coverage, histogram write cost, effect-window and migration-stability evidence. An explicitly disabled histogram produces `histogram_file=null`, `period_s=0` and no live cadence, not a fabricated measurement. With a histogram but live policy off, only the histogram period must be calibrated. Engine-only layouts need only the resource/margin ledger; no handoff fields or handoff calibrations are emitted. S10 must make RHP18's `live_policy_state` reflect configured/off/not-applicable and the measured cadence, and refuse live policy without the required qualified record.

## Calibration record and procedure

The v2 registry contains six S8 calibration keys for the four handoff integer tunables plus histogram period and live cadence. The existing `host_ledger_margin.mib` is reused, never duplicated. Every selected integer is attached to the generated `spec.calibration[]` type: `{key, value, source: "seed"|"cache", seeded: "YYYY-MM-DD", evidence}`. S8 requires a richer source record *before* constructing those fields:

```text
key, value, evidence_id, raw_run, method,
scope: {model_digest, engine_build_digest, helper_build_digest,
        layout_digest, inventory_digest, owner_map_digest, context_target_tokens,
        retention_target_tokens, kv_bytes_per_token},
measured_at_unix_s, valid_until_unix_s, measured_candidates[],
source: seed|cache, recorded_date: YYYY-MM-DD
```

The implementation refuses unknown/duplicate keys, an untested selected value, missing evidence or raw run, stale or future evidence, missing scope digests and mismatched model/build/layout/owner-map/target identity; a calibration is **not** valid merely because a prior manifest had that value. If hardware inventory, device assignments or runtime/kernel selection changes, The required `inventory_digest` changes (and invalidates the record) when the selected BDF/arch/VRAM map changes.

The production calibration procedure is to capture immutable model/container and engine/helper build digests; selected role-to-BDF map, owner-map digest and measured plan; conduct ABBA or stronger matched sweep with exact request/row shapes, output correctness, health/thermal context and RHP12 comparable performance gates; record all tested settings and per-run evidence identifiers; select the winning setting under the stated objective; and attach an explicit, limited validity horizon. **No qualifying RSC09 A/B results are supplied in this PR.** The test fixtures are deliberately synthetic and prove logic only. Old v1 values are not automatically seeded or made qualified v2 defaults.

For `host_ledger_margin.mib`, collect at least a window of starts' *measured* engine host-room figures (required, available, planned reserve, shortfall), preserving the largest actual shortfall in the chosen measurement window, plus the pre-existing qualified margin if a failed start requires refit, as specified in RSC10. Store the observation, matching plan/owner-map digest, UTC time and evidence run. The November R9700 claimable VRAM shortfall is a separate **VRAM** issue and cannot be used as a host-margin sample. A margin without a valid cache/seed evidence record refuses. Hardware qualification of seed floors/refit remains RSC10/S11/S13.

## S10 handoff and S13 activation

S10 consumes the typed `S8Resolution` and writes `spec.plan.resource_inputs`, `spec.resources.host`, each helper budget, the handoff transport/kernel/telemetry fields, the corresponding `spec.calibration[]` entries (deduplicating existing ones), and their closed `provenance` pointers. These values may not be independently recomputed in a second writer. S10 must validate that every source's effective covering provenance obeys v2 allowed origins/confidences, confirm authoritative owner-map and engine-plan parity, and use S5's out-of-band typed refusal for any live pre-weight miss. S13 turns this path on once and removes v1 instead of adding a fallback; v1 remains byte-identical until then.

Local card-free verification: build `bc_v2_s8_test` through the component CMake target **and** declare the target in `component.toml` so CI executes it. `bc_contract_check` checks the regenerated v1/v2 schemas and semantic fingerprints; it must stay green after registry changes. The real BDF/engine/histogram measurements and RHP12 A/Bs are post-S13 hardware gates, not claimed by this slice.
