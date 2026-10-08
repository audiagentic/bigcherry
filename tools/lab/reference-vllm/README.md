# Reference lane: vLLM + radiance on the R9700

**Question.** How fast is the other AMD stack on our hardware, measured our way? A public report (2026-10-08) claims
Qwen3.8-27B at 125-134 t/s decode and 2,750-2,970 t/s prefill on one Radeon AI PRO R9700 with vLLM + radiance +
libr4d, AMD's MXFP4 checkpoint and a DFlash2-FP8 drafter. Brutus already carries that stack as the `radiance-vllm`
container (image `stilldeadcode/vllm-radiance:0.9.3`, libr4d `b9e42ab-rx9`, same tuned settings).

**What this is not.** Not a like-for-like comparison with llama.cpp results: 4-bit MXFP4 weights and FP8 KV against
Q8_0 / IQ4_XS GGUF and f16 KV, one card against two or three. It is a reference point for what the hardware can do
and a source of ideas to port.

## Files

- `run-radiance.sh`: starts the existing container, waits for health, runs the benchmark, stops the container again
  (it is left running only if it was running before). Run it as a queue `SCRIPT` job so the GPU lock covers it.
- `bench-openai.py`: prefill and decode speed of any OpenAI-compatible server. One uncached, streamed, greedy request
  per depth and repeat; the prompt is the lab corpus cut to size with a unique nonce in front.
  prefill t/s = prompt tokens / time to first token; decode t/s = (completion tokens - 1) / (last - first token time).

## Run

```
VIS=0,1,2,3 SCRIPT ref-radiance tools/lab/reference-vllm/run-radiance.sh @<any build run> <runs>/ref-radiance 2048 8192 24576 98304
```

## Reading the result

Compare decode with llama.cpp on the same model family only as an upper bound, and record the acceptance figure the
server reports. Quality is not measured here; a 4-bit checkpoint needs its own check before any conclusion about
which stack to prefer.
