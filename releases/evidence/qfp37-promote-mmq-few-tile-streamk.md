# QFP37 — promote 1350_mmq_few_tile_streamk

## Promotion

QFP18 lightweight tier; llama.cpp pin b11474; Brutus 2026-10-08.

Build `b-metamem-qfp37c` of the 1350 experiment at d840d8f0 composed 49/49, compiled cleanly, smoke reported 0 error lines, and emitted `BIGCHERRY_PATCH_HIT patch=1350_mmq_few_tile_streamk cc=16781568 I=128 J=128 tiles=12 cu=48 k=10240 n=512`.

Fully separated ABBA, A = production/off, B = `BIGCHERRY_MMQ_FEW_TILE_STREAMK=1`:

| depth | A/off prefill t/s | B/on prefill t/s | gain |
|---|---|---|---|
| 8192 | 1162.9, 1168.0 | 1185.8, 1186.0 | +1.8% |
| 24576 | 1173.7, 1168.3 | 1184.9, 1185.1 | +1.2% |
| 98304 | 1069.3, 1084.3 | 1095.3, 1097.1 | +1.9% |

Complete separation at every depth. Decode was not worse.

Stream-K is not bit-identical because it changes F32 reduction grouping. Against the CPU F32 reference over 24 probes, production/off was 23/24 top-1 with TV mean 0.0764 (max 0.1926); subject/on was 24/24 top-1 with TV mean 0.0738 (max 0.3001). Repeated production D2 vs D was exact.

## Native llama.cpp baseline

The off arm (`BIGCHERRY_MMQ_FEW_TILE_STREAMK=0`) is the upstream/native MMQ dispatch on the same binary, so the control arm is the native llama.cpp dispatch for this mechanism.

## Scope

Evidence covers one model/topology: Flash-Next IQ4_XS on the production Brutus layout. A second model was not measured. Therefore the package default remains off and only `src/profile/flashnext.ini` enables it.
