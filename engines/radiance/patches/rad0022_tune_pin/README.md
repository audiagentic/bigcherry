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

### Promotion record

Model: Qwen3.8-27B MXFP4 and w8a8 (.rad), TP3 over two RX 7900 XTX and an R9700
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: rad0031 (diagnostic, on only with its timeline env) emitted the push, peer-wait, reduce and assemble stamps that VTP36.md used to split the three-rank two-shot, and its stamps are the activation proof for rad0032/rad0033; rad0021 weights 4,4,1 and rad0032 duplex weights are reported in the all-reduce setup; rad0024 hands decode-sized messages to the host-mailbox one-shot above ts_min=32768 elements; rad0033 ts_once writes the reduced shard once (XTX 32-row decode call 81 to 65 us); rad0022 pins all_reduce:nblocks=4,nthreads=256 through the tune pin the engine echoes at start (VTP14.md, VTP31.md experiments 1 and 3, VTP36.md).
Identity: rad0031 is read-only timing (unset, the kernel takes no stamp and does the same work); output byte-identical in every arm: rad0021 byte-identical, rad0024 greedy hashes identical, rad0022 greedy identical, rad0032 bit-identical (VTP14.md, VTP31.md, VTP36.md).
A/B: 2x ABBA on one binary. rad0021 4096-token TTFT -11% at one stream, decode unchanged; rad0024 4/6/8 streams +5.0/+3.5/+6.1% vs ts_min=0, one stream and TTFT unchanged; rad0022 +1.7/+6.7/+3.5/+5.7% at 1/4/6/8 streams; rad0032 with rad0033 2048-token TTFT -24% (w8a8) and -27% (MXFP4), 8 streams +14% and +17%.
