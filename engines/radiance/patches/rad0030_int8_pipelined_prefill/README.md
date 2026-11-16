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

### Promotion record

Model: Qwen3.8-27B w8a8 int8 (.rad), qwen27b-tp3-int8 (TP3 over two RX 7900 XTX and an R9700)
Mechanics: tools/tests/patch/test_vtp37_int8_pipelined_prefill.py passes and patch-lint is clean on the promotion PR; the lane builds carrying it build and serve.
Activation: `--pipelined-prefill on` (the layout's setting) against `off` selects the pipelined step; VTP37.md shows the serial note gone and the pipelined path taken in the on arms.
Identity: not bit-identical to the serial step at every prompt length, as on every pipelined step (rad0020 at MXFP4 too): greedy text identical at 300, 508, 4016, 6027 and 6090 tokens and different at 1092, 2023 and 2955 near ties; first-token log-probabilities bit-identical under 1024 tokens and within 0.001 on the top token above (VTP37.md). Owner reordering quality gate (VTP37.md, 2026-11-13) instead of bit identity: KLD against the same VTP36 BF16 reference, 42,761 positions, build main 4942dad1, layout 6,6,4: serial 0.00315 mean / 0.01057 p99 / 0.0893 p99.9 / 96.86% top-1 / PPL +0.040% (repeat identical to every digit), pipelined 0.00337 / 0.01019 / 0.0838 / 96.84% / +0.014%: mean +7.1%, p99.9 -6.2%, top-1 -0.02 points, PPL -0.026% (relative gate of +10% / -0.2 points / +0.1% passes; prq-rad0030, 2026-11-16). Limit: serial at 1024-token chunks scores lower still (0.00244 mean, 0.0715 p99.9), so reordering noise is the size of the difference; VTP37 recorded the serial-reference noise-floor check as passed (0.00125 / 0.021 / 99.31% against floor 0.00197 / 0.059 / 97.81%). The original gate's absolute BF16 values (VTP14 N6 reference) could not be rerun, the reference no longer being on disk.
A/B: 2x ABBA against the build without it: one-prompt time to first token -17% to -21%, four 4096-token prompts -16%, 4/6/8 streams +7.4% / +7.8% / +8.4% (VTP37.md).

Model: Qwen3.8-27B int8 (.rad), qwen27b-tp3-int8-556-long-bf16 (TP3 over two RX 7900 XTX and an R9700, bf16 KV cache, 262,144-token context)
Mechanics: tools/tests/patch/test_vtp42_prefill_governor.py passes and patch-lint is clean on the promotion PR; card-free sched_test shows each rad0034/rad0035 setting planned step for step by its spec equivalent (README, VTP42.md).
Activation: the -long configs serve with `--prefill-governor kind=budget;factor=1.5;long_tokens=16384;order=remaining;age_ms=120000` (owner decision 2026-11-16); the arms of docs/evidence/2026-11-16-vtp42-prefill-governor/README.md differ only in that spec and show different admission behaviour (15/218 s waits for B, 3-5 s for X).
Identity: one-stream greedy text sha identical under the default budget spec (2 starts) and kind=arrival at 302, 1085 and 4019 prompt tokens, 128 new tokens (prq-rad0036/ident, 2026-11-16); the governor only orders and admits prefill work, so a lone stream is unchanged by construction (this is weak evidence under contention).
A/B: hardware, eight ~100k prompts plus waves of ~450/4k/8k/16k: 100k mean first token 299 -> 227 s (-24%), last 100k 477 -> 388 s, mid-size first-token times held at 3-28 s, every request 200, no preemptions (VTP42 evidence README, two starts an arm); 4 decode streams beside the mix: 16.6 chunks/s (~4.15/stream) against 10.35 (~2.59/stream) under the cap spec, +60%, gap p50 ~44 ms both, 100k mean 279 vs 330 s (prq-rad0036/mix, two starts an arm, 0 preemptions). Not measured: fp8 -long, a steady arrival stream.
