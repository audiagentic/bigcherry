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

Untested (not promoted). Measurements are recorded in VTP37.

## Detail

Files:

- `arch/common/rad_fp8.h`: `NormQuantFP8::step_plain`.
- `arch/qwen35_fp8/qwen35_fp8.cpp`: the pipelined prefill's mode and its second-lane buffer set in `declare`.

Tests: `tools/tests/patch/test_vtp37_int8_pipelined_prefill.py`: metadata and recipe, edit content, guards, and applying the
chain to the pinned radiance v1.17.0 source.
