# 1350_mmq_few_tile_streamk

QFP37 implementation against llama.cpp b11474 (`b9acf138a1e2`).

b11474 already has the Stream-K MMQ algorithm and deterministic fixup. This package only compiles a forced specialization and narrowly selects it for ordinary few-tile Q8_0 prefill on qualified gfx1100/gfx1201 shapes. MUL_MAT_ID/MoE remains on the validated compact-grid path.

## Switch

- Build/runtime default: `BIGCHERRY_MMQ_FEW_TILE_STREAMK=0`.
- Qualified Flash-Next profile: `BIGCHERRY_MMQ_FEW_TILE_STREAMK=1`.
- Marker: `BIGCHERRY_PATCH_HIT patch=1350_mmq_few_tile_streamk`.

The default remains off because qualification covers one model/topology.

## Promotion record

QFP18 lightweight tier; pin b11474; Brutus 2026-10-08. Build `b-metamem-qfp37c` composed 49/49 patches, compiled cleanly, smoke reported 0 error lines, and emitted:

`BIGCHERRY_PATCH_HIT patch=1350_mmq_few_tile_streamk cc=16781568 I=128 J=128 tiles=12 cu=48 k=10240 n=512`

Fully separated ABBA, A = production/off, B = `BIGCHERRY_MMQ_FEW_TILE_STREAMK=1`:

| depth | A/off prefill t/s | B/on prefill t/s | gain |
|---|---|---|---|
| 8192 | 1162.9, 1168.0 | 1185.8, 1186.0 | +1.8% |
| 24576 | 1173.7, 1168.3 | 1184.9, 1185.1 | +1.2% |
| 98304 | 1069.3, 1084.3 | 1095.3, 1097.1 | +1.9% |

Complete separation at all three depths. Decode was not worse.

Stream-K changes F32 reduction grouping, so bit identity is not expected. Against the CPU F32 reference over 24 probes: production/off was 23/24 top-1 with total-variation mean 0.0764 (max 0.1926); subject/on was 24/24 top-1 with total-variation mean 0.0738 (max 0.3001). Repeated production D2 vs D was exact.

## Native llama.cpp baseline

No separate native build was required for this mechanism: the off arm (`BIGCHERRY_MMQ_FEW_TILE_STREAMK=0`) is the upstream/native MMQ dispatch on the same binary. The ABBA control therefore exercises native llama.cpp dispatch behavior for the mechanism being promoted.

## Coverage limit

Verified on Flash-Next IQ4_XS with the production Brutus layout only. A second model was not measured; keep the build/runtime default off outside the Flash-Next profile.
