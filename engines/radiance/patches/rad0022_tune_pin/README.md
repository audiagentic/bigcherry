# rad0022_tune_pin

## Summary

Radiance's selector takes each kernel's tunable axes (tile shapes, block and thread counts, fence choices) from the
per-machine tune cache, else the kernel's own defaults. `RADIANCE_TUNE=op:key=value[,key=value][;op:...]` pins axes
for every kernel serving an op (core/plugin/select.cpp, spec section 15). The pin is environment only, so a serving
manifest or layout could not carry it.

`--tune-pin SPEC` takes the same text. Before the plugins load the engine sets `RADIANCE_TUNE` from it, so selection,
validation and the `--debug-graph` label are radiance's own. An axis or value a kernel does not offer still fails the
selection with the kernel's list. If `RADIANCE_TUNE` is already set to something else, the engine refuses to start
rather than drop one of them.

Nothing is built in: the option carries whatever the run gives. The serving manifest field is `tuning.tune_pin`
(contract amendment 20); a layout sets it in `[layout.NAME.settings]`.

## Outcome

Untested (not promoted). First use, VTP31: the exact all-reduce's default grid scales with the message at 1024
threads a block, tuned on two- and four-rank worlds. On a three-card world (two RX 7900 XTX at Gen4 x8, an R9700 at
x4), MXFP4 Qwen3.8-27B, exact wire, DFlash2 depth 3, a 3x ABBA against the same build:

| streams | TP2 (two XTX) | TP3 default grid | TP3 `all_reduce:nblocks=4,nthreads=256` |
|---|---|---|---|
| 1 | 83.0 | 105.9 | 107.8 |
| 4 | 182.0 | 176.4 | 188.2 |
| 6 | 176.9 | 177.4 | 183.6 |
| 8 | 196.1 | 188.5 | 199.3 |

TTFT at 2048/8192-token prompts -7.6% / -11.4% against the default grid. Greedy output identical on every grid (the
grid never changes a word's fold).

## Detail

- `core/rad_core.h`: `Config::tune_pin`.
- `core/config.cpp`: `--tune-pin SPEC` help, parse (form only: `op:key=value`, lower-case, digits, `_=,;:`) and
  explain.
- `core/engine_bringup.cpp`: `Engine::load_plugins` sets `RADIANCE_TUNE` from it, or refuses a disagreeing one.
- `tests/config_test.cpp`: the option inventory value.
- Source-pinned to radiance v1.17.0 (19d4e8fe); independent of the other patches.
