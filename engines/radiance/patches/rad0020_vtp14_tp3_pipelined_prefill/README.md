# rad0020_vtp14_tp3_pipelined_prefill

## Summary

`qwen35_fp8` (and the builds that include it: `qwen35_w8a8`) pipelines a wide prefill step (`step_pipelined`). A step
of one sequence runs as two pieces, and each block's all-reduce for one piece runs on the second lane (`rad_lane`)
beside the other piece's GEMMs. The pipelined step computes the serial step's bytes.

Upstream fixes both choices in the plugin: it pipelines at two ranks only ("the only rank count where the step has
been timed"), from 1024 rows (`kPipeMinRows`). The native TP3 split (rad0018, rad0019) therefore prefilled serially
while the TP2 baseline pipelined, and neither choice could be tuned.

This patch makes both configuration, carried in the build context:

| option | values | default |
|---|---|---|
| `--pipelined-prefill` | `auto`, `on`, `off` | `auto`: two ranks, and three ranks under `--tp-ownership` |
| `--pipelined-prefill-min-rows` | 0, or 128 and up | 0: the plugin's own (1024) |

`on` pipelines at any world of two or more ranks; `off` never. A trunk whose norms write rotated int8 codes
(`a_h.q8_fed`) still prefills serially whatever the mode, as upstream. In the serving manifest these are
`execution.pipelined_prefill` and `execution.pipelined_prefill_min_rows`.

## Outcome

Untested (not promoted). On cards, MXFP4 TP3 with 4096-token prompts: time to first token 1395 → 1327 ms (−5%) with
the pipelined prefill at three ranks; TP2 is 902 ms (VTP14.md, pipelined prefill at TP3).

## Detail

- **ABI.** `RadBuildCtx` gains trailing `pipelined_prefill` (`BC_PIPE_AUTO`/`ON`/`OFF`) and
  `pipelined_prefill_min_rows`, zero-defaulted.
- **Core.** Both options in config, help, explain and the `config_test` inventory; the engine passes them on.
- **`qwen35_fp8`.** `Model::pipe_min_rows` replaces `kPipeMinRows` in every gate (declare, `pipe_ready`, the note);
  `kPipeMinRows` remains the default.
- The all-reduces keep their serial declaration at `overlap` 1; selection picks their kernels by world, size and
  peer access, as for the serial step.
- Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0019.

### Promotion record

Model: Qwen3.8-27B MXFP4 (.rad), TP3 over two RX 7900 XTX and an R9700
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the scheduler's "runs serially" note is gone in every arm that carries rad0025; rad0026 and rad0027 are the mixed-step shapes (decode beside two or more prefills, decode beside one cut prefill), each exercised by the workload that makes that shape (VTP31.md experiments 8, 9, 11).
Identity: pipelined steps compute the serial step's bytes; greedy texts differ only where the base build's own runs differ (VTP31.md: rad0025-0027 diverge only where the base build's runs diverge); rad0020 output unchanged.
A/B: 2x ABBA, one binary per arm, same settings. rad0020 4096-token TTFT -5% (VTP14.md); rad0025 four 4096-token prompts -7.0% slowest TTFT, 4 streams +4.0%, 1/8 streams 0; rad0026 4/6/8 streams +7.5/+10.0/+10.6%, four-prompt TTFT -6.7%; rad0027 8192-token prompt beside 4 streams TTFT -19.2%, standard bench unchanged. Logs e8-1223, e9-1328, e11-1421 (VTP31.md).
