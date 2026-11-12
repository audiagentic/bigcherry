# Radiance layouts

A layout names which cards of a host play which part for one model: the engine's tensor-parallel ranks, the remote-MoE
helpers, and the settings that go with them. Layouts are run-time data. No binary compiles them in and no code names one,
so a model or a run is defined, changed or tuned without a rebuild.

## Where layouts come from

`bc-serve` and the Python tools read the same list:

1. `bc-serve --layouts FILE|DIR` (repeatable), else
2. `BIGCHERRY_LAYOUTS`: a `:`-separated list of files or directories. A directory gives its `*.toml` files in name
   order.

When `BIGCHERRY_LAYOUTS` is unset, the `bigcherry` CLI and `tools/lab/radiance/lab-env.sh` set it to this directory,
the repository's example layouts. A site or a run uses its own definitions by pointing `BIGCHERRY_LAYOUTS` at them (for
instance a directory beside the models). A name defined in two files is refused, naming both. `bc-serve` itself has
no default: without either source it refuses to resolve.

The layout to use is `layout.name`, set in the site or profile (`[layout] name = "..."`) or with
`--set layout.name=NAME`. There is no default layout.

## Format

Each file is TOML with `schema = 1` and any number of `[layout.NAME]` tables. A layout picks cards from the host's
inventory by architecture, never by device index (the serving resolver refuses `devices`). The card types themselves are
`engines/radiance/archs.toml`. `bigcherry layout show NAME` resolves a layout against this host.

```
[layout.NAME]                  description, model (relative to BC_MODEL_ROOT), engine
engine = { arch = "gfx1201" }  the engine card(s); count = 2 makes two TP ranks
engine = [{ arch = "gfx1100", count = 2 }, { arch = "gfx1201", count = 1 }]
                               a world of more than one card type: groups in rank order, each arch once; each rank
                               runs its own architecture's plugin (radiance --rank-devices, VTP14 rad0016/rad0017)
tp-wire                        required for TP > 1 (host-memory transport, no P2P)
server-args                    optional engine options, each one a manifest setting (context, sessions, chunk, KV
                               dtype, speculative tokens, checkpoint interval, host pool, host KV tier, n-gram
                               placement); for a helper layout they are the launcher's defaults, which its own
                               command-line options override

[layout.NAME.settings]         the layout's own settings as resolver keys, checked like a profile's and layered under the
                               site and profile (which override them, as does --set KEY=VALUE): tuning.tp_ownership,
                               tuning.pipelined_prefill, helper.combine, service.max_context, ... Dotted keys:
                               `tuning.tp_ownership = "kv=..;ffn=.."`, `helper.combine = true`

[[layout.NAME.helper]]         a group of helper cards, in order (helper 0 first; helper 0 creates the shared segment)
  arch       architecture to take from the inventory
  count      how many of that arch ("all" or a number)
  budget-mib VRAM for experts on each card; omitted: derived by bc-serve from the card's VRAM less its archs.toml
             reserve-mib less the helper's own footprint for this model (moe-remote worker_footprint.h). Prefer that
             to a hand-picked number (a literal is an interim value, RSC03)
  placement  "head": the group takes the heat ranking by `share` (round robin); "tail": takes the coldest experts
             that fit what the heads left. Only the last group may be tail, and only after a head group
  share      weight of each card in a head group (default 1; 0 owns nothing)

```

A tensor-parallel engine (count > 1) may have helpers (RAD49): one request slot a (helper, rank), at most the protocol's
`kMaxSlots`, so two helpers behind two ranks.

## Files here

| File | Layouts |
|---|---|
| `flashnext.toml` | Flash-Next (Qwen3.8-Next-Flash MoE): the R9700 engine with XTX and RX 6900 XT helpers, and TP2 variants |
| `qwen27b.toml` | dense Qwen3.8-27B: two-XTX TP2, and native TP3 over XTX, XTX, R9700 (MXFP4 and w8a8) |
