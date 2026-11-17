# Flash-Next with the R9700 helper: speed, capacity and retrieval of the packed KV caches (VTP44)

## Summary
The packed caches serve on the production layout, two 7900 XTX with the R9700 helper, through the real launcher. Capacity
per token matches the helper-free run (bf16_i8 1.283x, bf16_fp8 1.308x bf16). Neither packed cache is slower than the
fp8 cache it sits beside at four streams or at 48K prefill, once one outlier is repeated. Run to run spread was large
(one arm's 48K TTFT moved from 38.7 s to 26.5 s between two runs), so only differences above that spread mean anything,
and one single-stream decode difference (bf16_i8, 19% under bf16) was not repeated and stays open.

## Outcome
Layout `flashnext-tp2xtx-r9700` (two 7900 XTX TP2 exact wire, one R9700 helper owning 29.0 GiB of experts), launched with
`launch-flashnext-best.sh start --build ... --kv-dtype X`; only the cache type changes per arm. Speculation depth 3, 6
sessions of 65,536 tokens, 2,048-token chunks, one run per arm, a new measured plan per cache type. Source
`qwen3.8-next-flash-hcbf16`, build `kv-bf16-packed` (rad0037, 38, 40, 41, 42).

| | bf16 | bf16_i8 | bf16_fp8 | fp8 |
|---|---|---|---|---|
| TTFT, 8K tokens (8,076) | 4.72 s | 4.31 s | 8.02 s (repeat 4.41 s) | 4.90 s |
| TTFT, 48K tokens (47,833) | 27.3 s | 24.6 s | 38.7 s (repeat 26.5 s) | 29.5 s |
| Decode, 1 stream (tokens/s) | 94.1 | 76.3 | 89.2 (repeat 89.2) | 81.1 |
| Decode, 4 streams (tokens/s) | 126.9 | 119.1 | 121.3 (repeat 121.7) | 113.9 |
| Needle hits (of 9, 48K tokens) | 8 | 7 | 8 (repeat 8) | 8 |
| attention page (K+V, 4 tokens, 13 layers) | 52.00 KiB | 39.81 KiB | 39.00 KiB | 26.00 KiB |
| cache per token with the QSA block keys (3.25 KiB a page) | 55.25 KiB | 43.06 KiB | 42.25 KiB | 29.25 KiB |
| vs bf16 | 1x | 1.283x | 1.308x | 1.889x |

- Decode in tokens a second is 256 tokens a stream divided by the stream's duration, because speculation delivers
  tokens in bursts (70 to 85 stream chunks for 256 tokens); chunks a second would mix acceptance with speed.
- Against fp8, the cache these sit beside: four-stream decode is +4.6% (bf16_i8) and +6.5% (bf16_fp8); 48K TTFT is
  16% faster (bf16_i8) and, on the repeat, 10% faster (bf16_fp8). One-stream decode is 6% under fp8 for bf16_i8 and 10%
  over it for bf16_fp8. fp8 is the slowest arm at four streams here, so the packed caches are not the cost.
- bf16_fp8's first run was an outlier (TTFT 8.02 s and 38.7 s, +70% and +42% against bf16); the repeat gave 4.41 s and
  26.5 s, and decode and needle were unchanged (89.2 and 121.7 tokens/s, 8 of 9). The first run came right after that
  configuration's first measured plan (a 391 s start); a cold first run is a hypothesis, not something checked.
- Not resolved: bf16_i8's one-stream decode is 19% under bf16 (76.3 against 94.1) while its four-stream decode is 6%
  under. It was run once; bf16's repeat could not be run (below). Nothing in the helper-free run showed it (28.95
  against 28.86 chunks/s), so treat it as a possible single-run effect until repeated.
- The engine sized every arm's cache for 6 x 65,536 tokens (393K tokens), so extra capacity appears as a smaller cache
  in VRAM (5.83, 4.68, 4.61 and 3.39 GiB a card), not as more tokens. Page sizes are 13/12 of the helper-free ones
  because speculation adds the draft layer's cache.

## Detail
Differences from production, all forced by the host and applied to every arm alike:
- The n-gram table is on disk (`--ngram-placement disk`), not in RAM: the production placement needs 89.7 GB of host RAM
  and about 82 GB was available (other processes on the host hold the rest). The launcher then derived a 32 GiB host
  pool and no host KV tier. Absolute decode speed is below production and the arms compare with each other only.
- The route-heat file is a recorded Flash-Next ranking from an earlier run (`perf51`), the same file in every arm,
  because `flashnext-mixed.hot-first` is not on this host. It decides which experts the helper holds.
- The client's text was rebuilt from the long KLD corpus (the original text directory was deleted by a workspace cleanup
  during this work), so TTFT prompts and needle passages differ from the helper-free run's. Numbers are comparable
  within this run only, not with FLASHNEXT-AB.md. An earlier attempt of this run had its TTFT offsets past the end of
  the rebuilt text (13 and 5,418 token prompts) and was discarded; the offsets were fixed and all four arms rerun.
- One run per arm except bf16_fp8. The planned bf16 repeat failed to start because the heat file had been deleted by the
  same cleanup between the two arms, so there is no repeat of the reference arm and no spread for it.
- The needle probe and its scoring are as in FLASHNEXT-AB.md (three codes at 10, 50 and 90% depth, chat endpoint, thinking
  off, temperature 0); 7 to 8 of 9 for all arms shows no loss for the packed caches but is too coarse to rank them.
- KLD is not repeated here: on Flash-Next it sits inside the model's arithmetic-noise band (FLASHNEXT-KLD.md).

Each arm's launch log, client log, JSON result and the engine and helper logs (page sizes and cache growth in the
engine log) are in the run's result folder `vtp44-flashnext-helper` in the work root, with the repeat in
`vtp44-flashnext-helper-repeat`; the scripts are `run_helper_ab.sh` and `fnclient.py` in the task's scratch folder.
