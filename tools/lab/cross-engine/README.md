# Cross-engine quality check (MEN05)

**Question.** When two engines serve the same model in different 4-bit formats, are they equally close to a
higher-precision reference? A speed comparison between engines means little without this.

**Why.** Radiance serves Qwen3.8-27B as an MXFP4 container with fp8 KV; BigCherry llama.cpp serves a UD-Q4_K_M GGUF
with f16 KV. On one R9700 radiance prefills about 2.4x faster (2026-10-09, `tools/lab/radiance/`,
`tools/lab/reference-vllm/`). `tools/lab/flash-next/flash-fidelity.sh` compares llama.cpp builds through llama-server's
own probe output and cannot look at another engine.

## Files

- `probe-openai.py`: next-token distributions from any OpenAI-compatible server. Same prompts as
  `long-ctx-profile.sh` `probes` mode (corpus cut to a depth, then natural-text continuations); one token per probe at
  temperature 0 with its top alternatives, through `/v1/completions`. Tokens are keyed by their bytes.
- `probes-compare.py`: each arm against a reference arm: top-1 agreement, total-variation distance over the listed
  tokens, difference of the reference's top-1 probability, and the probability mass each list covers.
- `run-quality-27b.sh`: four arms for Qwen3.8-27B: llama.cpp Q8_0 on the two XTXs (reference), the same again
  (run-to-run floor), llama.cpp UD-Q4_K_M on the R9700, radiance MXFP4 on the R9700. No drafter in any arm.

## Run

```
VIS=0,1,2,3 SCRIPT xq27 tools/lab/cross-engine/run-quality-27b.sh @<llama.cpp build run> <runs>/xq27 2048 8192 24576
```

## Reading the result

The bar is the one `flash-fidelity.sh` uses: a candidate should be no further from the reference than the other
candidate is, and `ref2 vs ref` shows what "identical" looks like on this hardware. The reference is Q8_0, not full
precision, so both 4-bit arms carry its quantisation error as a common offset; the comparison between them is what
matters. Twenty alternatives do not cover the whole distribution: read the "listed mass" figure beside every distance.
This is a distribution check, not a task benchmark.

## Disposition

Experiment-only until MEN03 gives the lab one engine adapter; then the launch code in `run-quality-27b.sh` moves behind
it and only the probe client and the comparison stay here.
