# Radiance build and serving runbook

Radiance is the primary serving engine. Its upstream pin, patches, sources, and build platforms are declared in `engines/radiance/recipes.toml`; separate components are installed alongside the engine in `radiance_home`.

## Source composition and build

| Source | Composition |
| --- | --- |
| `radiance-native` | Unmodified pinned upstream source |
| `radiance` | Production: `serving-core`, `host-compat`, and `serving-manifest` |
| `radiance-diag` | Production plus `moe-remote-diag`, built with `moe-diag` |

The validated remote-MoE hand-off (`rad0002_moe_remote_handoff`) belongs to production `serving-core`, not an experiment. There is one build: lab scripts, the Flash-Next launcher and CI all build `radiance:stock:linux-radiance`, and experiments are switched on a run, never a separate source. Diagnostic patches remain isolated from production (`test_radiance_sources_share_production.py`).

From a configured checkout:

```sh
PYTHONPATH=tools python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance
```

For builds with helper offload, confirm both `bin/radiance` and `radiance_home/helpers/r3_moe_worker` exist. See the [enhancement catalogue](ENHANCEMENTS.md) for each patch's validation state; recipe inclusion alone is not evidence.

## Launch Flash-Next

Use `tools/lab/radiance/launch-flashnext-best.sh` with an explicit `--build DIR` when reproducibility matters:

```sh
bash tools/lab/radiance/launch-flashnext-best.sh start --build DIR --layout flashnext-2xtx
bash tools/lab/radiance/launch-flashnext-best.sh stop
```

Obtain the exclusive `gpu` lease before any launch or measurement (including the first launch). A cold configuration first starts helpers and measures the engine memory plan, usually **1–2 minutes**, before starting the serving engine. Every start, including a cached start, starts the engine from the measured serving manifest, and the engine checks the plan that manifest carries (mandatory; no flag skips or replaces it). After `/health`, every start also sends a completion **canary**; failure stops the processes and restores applied power settings.

| Exit | Meaning |
| --- | --- |
| `96` | Start canary failed (wrong, malformed, or failed completion) |
| `97` | Memory plan measurement, cache/store, or plan-check setup failed |
| `98` | Launch lock is held by another start/stop, or a server is already running |

Plans are cached under `$BC_WORK_ROOT/flashnext/plan-cache`. To force a fresh measurement, **stop the server first**, then remove that directory (`rm -rf "$BC_WORK_ROOT/flashnext/plan-cache"`). Different layouts, contexts, memory budgets, and other launch parameters have distinct cache entries. Logs and the launch lock live in `$BC_WORK_ROOT/flashnext`.

## Layouts and measurement

Select a topology using `--layout NAME` or `FLASHNEXT_LAYOUT`; enumerate definitions with `bigcherry layout list` and `bigcherry layout show NAME` (from `engines/radiance/layouts.toml`). The default uses all available cards; `flashnext-2xtx` excludes the RX 6900 XT, while `flashnext-2xtx-rdna2-tail` includes the RDNA2 tail. `--budget ARCH=MIB` overrides a card group's memory budget. Missing cards or plugins fail the layout check.

**RAD46 quality caveat:** putting experts on the RX 6900 XT tail has a known *worst-case quality cost*. Do not treat an improved throughput result as an unconditional quality win; evaluate the workload's output/quality and, when needed, compare with the two-XTX layout.

Use `bigcherry ab-benchmark --server-config` for paired experiments and `bigcherry stream-bench` for stream-count studies. Report **prefill** and **decode** separately, with model, topology, prompts, sample count, build identity, and text/quality checks. See [Flash-Next](models/qwen3.8-flash-next.md), [27B](models/qwen3.8-27b.md), [35B-A3B](models/qwen3.6-35b-a3b.md), and the [lab guide](../../../../tools/lab/radiance/README.md).

The strict [serving manifest v1 contract](../../../design/serving-manifest-v1/spec.md) describes the target configuration interface; current launcher flags remain the operational reference until cutover.
