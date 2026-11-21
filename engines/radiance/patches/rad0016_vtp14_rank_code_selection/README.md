# rad0016_vtp14_rank_code_selection

## Summary

Radiance v1.17.0 serves one card architecture per process:

- `usable_devices` drops every card whose architecture differs from the first card's.
- The kernel registry checks each library's device code against that first card only (`set_code_device(devs[0])`).

A tensor-parallel world of two RX 7900 XTX cards (gfx1100, served by libr3) and one R9700 (gfx1201, served by
libr4d) needs each rank to select the rows its own card runs. This managed patch adds that. It does nothing unless
`--rank-devices` is given.

## Outcome

On cards (with rad0017), mixed XTX+R9700 TP2 serves Qwen3.8-27B fp8 correctly. The patch also makes the AQL backend register fat binaries that HIP gives no handle (HIP judges by device 0 only). Earlier status: Untested (experimental, VTP14 N1). The ROCm lane build and the HIP unit tests pass. On cards, per-rank selection works (libr3 for the XTX rank, libr4d for the R9700 rank), but a mixed-library world stops at the N3 all-reduce rendezvous. The rendezvous is a prerequisite for serving.

## Detail

### What it changes

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
  - `core/plugin/bc_rank_devices.{h,cpp}` hold the pure planning, admission and agreement functions (no engine types).
  - `core/plugin/bc_rank_engine.{h,cpp}` hold what needs the engine's own types: the `Registry` rank-list members and
    per-library rank mask, the inventory and configuration adapter for the admission, and the per-rank program receipt. The
    upstream files carry only the typed calls into them (RUM02).
  - `tests/bc_rank_devices_test.cpp` tests them, plus per-rank selection and the unchanged legacy path. It uses
    radiance's `radtest_*` plugins with a code-runs probe, so it needs no card.

### Code past device 0 (`core/device/aql.cpp`)

HIP's `__hipRegisterFatBinary` returns no handle for a fat binary without code for device 0, even when a later card runs it. With two gfx1100 cards visible first, that covers every gfx1201-only libr4d binary. The AQL backend loads code objects from the bundle per device, so it issues its own handle for such a binary (in `own_handles`) and never passes that handle to HIP's register-function or unregister calls.

### Known limits (later VTP14 slices)

- **N3 collective rendezvous.** The all-reduce rendezvous in `r4d_ar_entry.hip` is static per library, so libr3
  ranks and libr4d ranks never meet: the exchange deadlocks until the 60 s timeout.
  - The agreement check above passes, because the kernel names are identical.
  - The fix is a process-wide shared slot exported by core, which both libraries look up. It needs its own patch.
- **N2 split.** Qwen3.8-27B cannot be split uniformly over three ranks: there are 4 KV heads. The N2 asymmetric
  ownership is required.
- **KV tiers.** `runtime/tierexec.cpp` resolves gather and scatter without a rank, so in a mixed world it fails
  closed.

### Validation

- `tools/tests/patch/test_vtp14_n1_rank_code_selection.py` checks:
  - the contract;
  - that guards are unique and recognise their own text;
  - idempotence;
  - fail-closed behaviour on drift;
  - that the patch applies to the real pinned v1.17.0 source (19d4e8fe) when a checkout is reachable.
- The ROCm lane build (stock, linux-radiance, ROCm 7.2.4: libr4d for gfx1201, libr3 for gfx1100) succeeds.
- In the HIP test build of the same tree, `bc_rank_devices_test`, `plugin_test` and `config_test` pass. Every other
  failure also occurs on the production source.
- On hardware, `--tp 2 --rank-devices 0000:03:00.0,0000:09:00.0` plans the ranks and masks the libraries per rank
  (libr3 for the XTX, libr4d for the R9700).
- Serving then stops at the N3 all-reduce rendezvous (see VTP14.md).

### Promotion record

Model: Qwen3.8-27B MXFP4 and w8a8 (.rad), tensor parallel over two RX 7900 XTX and an R9700 (TP3)
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the TP3 world over (XTX, XTX, R9700) starts and serves only with this chain (rad0016 per-rank code selection, rad0017 shared rendezvous, rad0018/rad0019 uneven ownership); without rad0017 any libr3+libr4d world hangs in all-reduce init (reproduced on cards, VTP14.md), the engine reports the rendezvous as shared by libr3.so and libr4d.so and the ownership split in its startup report.
Identity: TP3 6/6/12 against an exact TP2 reference over 42,761 positions: KLD mean 0.00307, p99.9 0.084, top-1 96.80%, PPL x0.9993; the production TP2 wht6 wire scores 0.00391 / 0.111 / 96.57% on the same corpus (VTP14.md, N7, build a0979ca2). Launchers qwen27b-tp3, -int8 and -capacity start from their layouts with greedy output identical to hand-flag runs.
A/B: control cannot run (mixed-library world hangs without the chain); TP3 6/6/12 serves 64-65 tok/s at about 0.05 s TTFT, correct (VTP14.md milestone table); w8a8 TP3 KLD against BF16 within TP2's median/p99/p99.9 (VTP14.md N6).
