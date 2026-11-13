# rad0030_int8_pipelined_prefill

## Summary

The dense Qwen plugin pipelines a wide prefill step: the step's rows run as two pieces, and each block's all-reduce and
the next block's norm run on a second lane beside the other piece's work (rad0020, rad0025 to rad0027). Upstream keeps
this off on a trunk whose norms write rotated int8 codes, which is the w8a8 plugin wherever the kernel library serves
the `_had_quant_i8g` ops. Such a trunk prefilled serially whatever `--pipelined-prefill` said.

The reason was one call. The boundary norm on the second lane (`NormQuantFP8::step_plain`) issued its op with the E4M3
code pair as outputs. On a rotated-int8 trunk the op is `rmsnorm_had_quant_i8g`, which writes the int8 pair, and every
linear reads that pair. Nothing else in the pipelined step depends on the code type:

- each block's piece path reads the pair the trunk feeds (`ActFP8::cq` / `cs`);
- the other ops on the second lane (the all-reduce, the residual add, the final norm) touch only bf16 planes;
- the gated quantiser, the attention gate and the delta-net output quantiser stay on the first lane.

This patch:

- makes `step_plain` write the pair `step()` writes: the int8 pair on a rotated-int8 trunk, the E4M3 pair otherwise;
- names the pair the norms write to `rad_buf_concurrent`, so the int8 codes the second lane writes leave the buffer
  sharing pool (on such a trunk no op touches the E4M3 pair, which stays in the pool);
- pipelines a rotated-int8 trunk only under `--pipelined-prefill on` (`tuning.pipelined_prefill = "on"` in a layout).

It does not apply when:

- `--pipelined-prefill` is `auto` (the default) or `off`: a rotated-int8 trunk prefills serially as before, and under
  `auto` the startup note names the option;
- the trunk is fed int8 codes that are not rotated: it stays serial under every mode.

An E4M3 trunk is unchanged under every mode.

## Outcome

Untested (not promoted). VTP37, w8a8 Qwen3.8-27B at TP3 (XTX, XTX, R9700; exact wire, three speculative tokens, fp8
KV), 2x ABBA of the build with this patch and `--pipelined-prefill on` against the build without it:

| | without | with, `on` | change |
|---|---|---|---|
| 1 / 4 / 6 / 8 streams, tok/s | 83.1 / 187.4 / 179.6 / 214.3 | 88.1 / 201.2 / 193.7 / 232.3 | +6.1 / +7.4 / +7.8 / +8.4% |
| one prompt, time to first token, 512 / 2048 / 6144 tokens | 342 / 1252 / 3757 ms | 281 / 1040 / 2959 ms | -18.0 / -17.0 / -21.3% |
| four 4096-token prompts, slowest time to first token | 10121 ms | 8492 ms | -16.1% |
| 6144-token prompt beside 4 generating streams | 4049 ms | 3380 ms | -16.5% |

Output. A pipelined step computes what the serial step computes over the same two pieces as separate chunks: where a
prompt's pieces fall on the chunk bounds of a serial run at half the chunk size, the first token's top-20
log-probabilities are bit-identical to that run's. Against the serial step of the full chunk the bytes can differ in
the last places (the top token stayed the same on every prompt tried, its log-probability within 0.001), so a greedy
continuation can part from the serial one at a near tie, tens of tokens in. The pipelined prefill of an E4M3 trunk
(rad0020) differs from its serial step in the same way. The measurements are in VTP37.

## Detail

Files:

- `arch/common/rad_fp8.h`: `NormQuantFP8::step_plain`.
- `arch/qwen35_fp8/qwen35_fp8.cpp`: the pipelined prefill's mode and its second-lane buffer set in `declare`.

Tests: `tools/tests/patch/test_vtp37_int8_pipelined_prefill.py`: metadata and recipe, edit content, guards, and applying the
chain to the pinned radiance v1.17.0 source.
