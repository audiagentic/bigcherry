# Qwen3.8 Flash-Next lab

Temporary hardware/diagnostic drivers for active Flash-Next plan items under
`docs/planning/active/patching-qwen-flash-next/`.

This directory is not a permanent script archive. A plan-specific driver stays
only while an active plan names or needs it. When the owning plan completes,
retain compact decision-grade evidence under `docs/evidence/` or the owning
patch package and delete the one-off driver.

Shared helpers currently referenced by active plans include:

- `ar-segment.py`, `ar-boundary.py` — AllReduce boundary/segment analysis.
- `long-ctx-profile.sh` — long-context profiling used by active QFP work.
- `q81-trace-run.sh` — Q8_1 trace support.
- `queue-prefill-profile.sh`, `prefill-provider-sweep.sh` — prefill profiling/provider comparison.
- `queue-env-ab.sh` — common active A/B launcher.
- `queue-moe-cache.sh`, `moe-cache/README.md` — MET01 host-expert 0/4096 MiB cache + profile qualification.
- `submit-timing-table.py` — timing-table submission used by QFP31.
- `rank-census.py` — rank census used by QFP09.

Generated run output belongs under ignored `artifacts/lab/` or host-local
storage. Commit only compact evidence selected for `docs/evidence/`. Do not
commit copied upstream source; materialize the pinned `vendor/llama.cpp`
checkout instead.
- `cross-model-rel.sh`: cross-model check of a released production set (Qwen3.8-27B built-in MTP, Gemma without a draft, Flash-Next with a second request text) for the MTP deferred catch-up and look-ahead switches; reports per-model activation.
- `review-with-model.sh`: starts Flash-Next in the production layout and sends each prompt file in a directory as one chat request (code review while no external reviewer is available); saves answers and timings. Also a real-workload measurement: long uncached prompt, long generation.
