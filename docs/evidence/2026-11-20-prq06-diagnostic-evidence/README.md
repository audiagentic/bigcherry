# PRQ06: qualifying the VTP44 evidence tools as diagnostic patches

rad0042 (packed-KV selftest cases), rad0043 (complete expert stats), rad0044 (expert route record) and rad0045
(expert route replay) are lab tools for scoring KV caches on Flash-Next, not production features. They are qualified at
the diagnostic tier (mechanics and activation, no production admission) with the evidence of the VTP44 runs
(`docs/evidence/2026-11-16-vtp44-generic-kv-layouts/`); the four text files here are the `--evidence` inputs of
`bigcherry --engine radiance patch-promote --diagnostic`, one per patch. Their SUMMARY "not yet run" lines described the
state before those runs and are corrected.
