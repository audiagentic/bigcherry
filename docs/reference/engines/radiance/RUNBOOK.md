# Radiance build and validation

The pinned source is **Radiance v1.14.0**. BigCherry selects upstream
patches through `engines/radiance/recipes.toml` and builds separate
components against the same pin. Those components are installed in
`radiance_home` alongside the engine binary and its kernel libraries.

## Source selection

| Source | Contents |
| --- | --- |
| `radiance-native` | Unmodified pinned upstream engine |
| `radiance` | The declared serving-core, host-compatibility and serving-manifest patch sets |
| `radiance-lab` | Host compatibility and evaluated fixes; the remote-MoE hook is added by the `moe-remote-handoff` experiment |
| `radiance-diag` | Diagnostic patch set, including remote-MoE diagnostics |

A source's inclusion in a recipe does **not** imply every patch has
validated hardware evidence. The [patch catalogue](ENHANCEMENTS.md) and
each package's `patch.toml` give the recorded states.

## Build and measurement

The Flash-Next helper layout currently uses the lab source and
`moe-remote-handoff` experiment. For example:

```sh
PYTHONPATH=tools python -m bigcherry build \
  --lane radiance-lab:stock:linux-radiance \
  --experiment moe-remote-handoff
```

The build includes the pinned engine, `libr3`, `libr2` and remote-MoE
components needed for the selected targets. A/B measurements use
`bigcherry ab-benchmark` with model-specific configurations under
`tools/lab/radiance/`. Quantitative comparisons must identify the
source, model container, GPU layout, prompt/decode workload, cache
policy and quality checks.

The current supported results are on the
[Flash-Next](models/qwen3.8-flash-next.md),
[27B](models/qwen3.8-27b.md) and
[35B-A3B](models/qwen3.6-35b-a3b.md) pages. Historical experiments and
old pinned builds remain in the
[dated lab record](../../../../docs/archive/radiance/LAB_HISTORY.md), not in the
current result tables.

## Serving configuration

`bc-serve` supplies strict configuration loading and manifest inspection;
the engine has a manifest entry point. The [v1 manifest contract](../../../design/serving-manifest-v1/spec.md)
is implemented incrementally. Until process-role loading and supervision
land together, the legacy launcher/configuration paths are not a stable
public interface. The source and build recipes remain authoritative for
what is actually packaged.
