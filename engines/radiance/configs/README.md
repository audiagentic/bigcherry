# Radiance serving configs

A config says what to serve and how: the model, the cards its engine ranks and helpers run on, and every setting that
differs from the resolver's defaults. One file per model; a variant of a config is a named entry that lists only what
it changes. Configs are run-time data: nothing compiles them in. Plan: RSC12
(`docs/planning/active/run-serving-configuration/RSC12.md`).

Configs replace the layout files in `../layouts/` one model at a time. Until the conversion is complete both exist;
a converted config keeps its layout's name (until the v2 manifest switch), so the two are never read together.

## Where configs come from

`bc-serve --configs FILE|DIR` (repeatable), else `BIGCHERRY_CONFIGS` (a `:`-separated list; a directory gives its
`*.toml` files in name order). When configs are given, layout files are read only from an explicit `--layouts`. A config
is selected like a layout, by `layout.name` (`--set layout.name='"qwen27b-tp3-int8-556"'`).

The config's `model` is its lowest `engine.model` setting: a relative path is under `--model-root DIR` (bc-serve reads
no environment variable for it); a site file, profile or `--set` that names a model wins. A config with a relative
model and no root is refused when it is selected.

## Format

```toml
schema = 1

[config.NAME]
description = "what this config is for"          # required
model       = "radiance/.../model.rad"           # relative to the model root
ranks       = [{ arch = "gfx1100", count = 2 }, { arch = "gfx1201", count = 1 }]   # engine ranks in order
tp_wire     = "exact"                            # required when there is more than one rank

[[config.NAME.helpers]]                          # remote-MoE helper card groups, in order (optional)
arch = "gfx1100"
count = "all"                                    # a number, or "all" unused cards of the arch
placement = "head"                               # "head" or "tail" (only the last group, after a head group)
budget_mib = 23500                               # optional explicit override; left out, derived from the card
share = 1                                        # weight in a head group (optional)

[config.NAME.service]                            # settings tables: the resolver's own keys,
max_context = 8192                               # the names --set and bc-serve --explain use
[config.NAME.engine]                             # (service, engine, tuning, memory, helper, handoff)
kv_cache_dtype = "fp8"
[config.NAME.tuning]
tp_ownership = "kv=1,1,2;ffn=..."

[config.VARIANT]
base = "NAME"                                    # inherits every value of NAME
description = "what differs"
[config.VARIANT.tuning]
tp_ownership = "kv=1,1,2;ffn=..."                # only what changes
```

Rules:

- **Strict.** An unknown key, a setting outside a settings table, a missing base, a cycle of bases or a name defined twice
  is refused, naming the file and the key. A setting the resolver does not know is refused by the resolver.
- **Variants.** A variant inherits every value of its base (a chain of bases is allowed). Tables merge key by key; a
  scalar or a list (`ranks`, `helpers`) replaces, so a variant that changes the helper groups restates the list.
- **No flag spellings.** There is no `server-args`: a context length is `service.max_context`, not `--max-model-len`.
- **Not in a config:** what belongs to the machine (ports, model root, run root, power profile definitions; the site
  file) and what is derived (host pool, host KV tier, checkpoint slots: the memory plan).
- **Precedence:** config < site file < profile < `--set`. A `--set` is for a one-off experiment; a setting worth keeping
  becomes a variant.
