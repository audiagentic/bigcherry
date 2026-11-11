# rad0016 — VTP14 N1: per-rank kernel code eligibility and `--rank-devices`

Radiance v1.17.0 serves one card architecture per process:

- `usable_devices` drops every card whose architecture differs from the first card's.
- The kernel registry checks each library's device code against that first card only (`set_code_device(devs[0])`).

A tensor-parallel world of two RX 7900 XTX cards (gfx1100, served by libr3) and one R9700 (gfx1201, served by
libr4d) needs each rank to select the rows its own card runs. This managed patch adds that. It does nothing unless
`--rank-devices` is given.

## What it changes

- **Registry** (`core/plugin/registry.h`, `loader.cpp`)
  - `set_code_devices(ordered ordinals)` must be called before any library loads. It refuses an empty list, a list
    longer than 32, a negative or repeated device, and any call made after rows exist.
  - Each device row carries `KernelRow::code_ranks`: bit r is set when rank r's card runs the library's code
    objects. The mask is read from the fat binaries the dlopen registered (`dev_code_runs`), not from
    `build_target`.
  - The mask is fixed at load and the rank threads only read it.
  - A library that no rank's card runs loses its device rows, as upstream does.
- **Selection** (`select.cpp`, `bands.cpp`, `core/build`)
  - `select`, `resolve` and `build_bands` take the rank. The Builder passes `RadBuildCtx.rank` through.
  - A device row the rank's card cannot run is refused with its plugin named, and the next library answers.
  - A question asked for no rank gets only rows that every rank's card runs.
- **Engine** (`config.cpp`, `engine_bringup.cpp`)
  - `--rank-devices BDF,BDF,...` puts rank i on the i-th card, named by PCI address. The list must name exactly
    `--tp` cards, each unique, discrete and visible.
  - This list is the only way to get a world that spans architectures.
  - After declare:
    - The reference-kernel gate runs on every rank; upstream checks rank 0 only.
    - Every rank must issue the same collective kernels in the same order. Otherwise startup stops, naming the
      first difference.
    - A per-rank receipt is logged: card, architecture, op bands per library, and collectives.
- **New files**
  - `core/plugin/bc_rank_devices.{h,cpp}` hold the pure planning and agreement functions.
  - `tests/bc_rank_devices_test.cpp` tests them, plus per-rank selection and the unchanged legacy path. It uses
    radiance's `radtest_*` plugins with a code-runs probe, so it needs no card.

## Known limits (later VTP14 slices)

- **N3 collective rendezvous.** The all-reduce rendezvous in `r4d_ar_entry.hip` is static per library, so libr3
  ranks and libr4d ranks never meet: the exchange deadlocks until the 60 s timeout.
  - The agreement check above passes, because the kernel names are identical.
  - The fix is a process-wide shared slot exported by core, which both libraries look up. It needs its own patch.
- **N2 split.** Qwen3.8-27B cannot be split uniformly over three ranks: there are 4 KV heads. The N2 asymmetric
  ownership is required.
- **KV tiers.** `runtime/tierexec.cpp` resolves gather and scatter without a rank, so in a mixed world it fails
  closed.

## Validation

- `tools/tests/patch/test_vtp14_n1_rank_code_selection.py` checks:
  - the contract;
  - that guards are unique and recognise their own text;
  - idempotence;
  - fail-closed behaviour on drift;
  - that the patch applies to the real pinned v1.17.0 source (19d4e8fe) when a checkout is reachable.
- On a host-only build (RAD_WITH_HIP=OFF, GCC 14), the touched units compile without warnings, and
  `bc_rank_devices_test` and `plugin_test` run.
