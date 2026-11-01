# Card types and layouts for the remote-MoE offload

Which cards the offload uses, and for what, is configuration. Nothing in the helper, the launcher or the harness adapter names an
architecture, a device index or a VRAM budget.

| What | Where | Holds |
|---|---|---|
| The host's cards | `config/environment.local.toml` `[[host.*.devices]]` | index, `arch`, `vram-mib`, PCI locator (untracked, per host) |
| Card types | `engines/radiance/archs.toml` | per architecture: kernel `plugin` file, weight `formats` its backend runs, `rows-kernel`, `reserve-mib`, `roles` |
| Layouts | `engines/radiance/layouts.toml` | per name: the engine card and ordered helper groups (`arch` or `devices`, `count`, `budget-mib`, `placement`, `share`), and an optional `[layout.NAME.env]` of `R3_MOE_WORKER_*` settings for the engine and every helper |

`bigcherry layout list | show NAME | env NAME | card-env DEVICE` resolves a layout against this host and prints what each process gets. A layout that
needs a card the host does not have, a role the card type does not allow (`roles`), a budget above `vram-mib - reserve-mib`, more helpers than the
protocol's `kMaxHelpers`, or a tail group that is not a single last card after a head group, is refused with the reason before any GPU is touched.

A layout's `env` table carries settings that belong to the whole arrangement rather than to a card: `flashnext-2xtx-combine` sets
`R3_MOE_WORKER_COMBINE = "1"` (the helper-row combine, which adds the rows of two to four helpers and so is refused for a single helper:
there is nothing to add). Names must be `R3_MOE_WORKER_*` and may not be one the cards decide (`COUNT`, `MIBS`, `SHARES`, `PLACEMENT`, `LAYOUT`, `DEVICES`).

## Using one

- Launcher: `launch-flashnext-best.sh --layout NAME [--budget ARCH=MIB]` (default `flashnext-2xtx`; or `FLASHNEXT_LAYOUT`).
- Benchmark harness: an arm setting `R3_MOE_WORKER_LAYOUT=NAME` (new configs), or `R3_MOE_WORKER_DEVICES="0 1"` (explicit indices; the card type of each comes
  from the inventory). Either way the helper's plugin, formats and few-rows kernel come from `archs.toml`; a setting the arm names itself wins.
- Lab scripts that start one helper by hand add `$(bigcherry layout card-env INDEX)` to the helper's environment.

## Changing the topology

- **A different set of the same cards** (say one XTX): edit the host inventory; a layout that takes `count = "all"` follows it. Pin cards with `devices = [..]`.
- **A different use** (budgets, which card is the tail, shares, a third head group): edit or add a layout. Placement `head` deals the heat ranking to the group by
  `share`; `tail` (the last group, one card) takes the coldest experts the heads left that fit its own budget.
- **A new card type**: add an `[arch.gfxNNNN]` stanza (and a kernel plugin that builds for it, a component like `libr3` / `libr2`; its `outputs` must list the file). The
  helper reads `R3_MOE_WORKER_PLUGIN_FILE` and `R3_MOE_WORKER_FORMATS` from the environment and checks the plugin's build target against the card when it opens it.
  `R3MOE_TARGETS` in the moe-remote build (`CMakeLists.txt`) lists the architectures the helper binary itself is compiled for.

## What is still fixed

The engine is one card (one engine rank) and the combine (`R3_MOE_WORKER_COMBINE`) needs exactly two helpers; both are protocol limits, not layout choices. The
protocol allows up to `kMaxHelpers` (4) helpers. A running system does not re-plan its layout; adaptive placement is the `patching-radiance-heat-placement` plan.
