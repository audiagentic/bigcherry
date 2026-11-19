# ENG06 step 3: long-context decode, bf16 vs bf16_i8 vs fp8 (TP3 27B)

Config `qwen27b-tp3-int8-556-long` (fp8 = default, `-bf16` variant, bf16_i8 via `--set engine.kv_cache_dtype="bf16_i8"`), 3 R9700-class
cards, stock power, one fresh start per launch, six launches in the order i8, bf16, fp8, fp8, bf16, i8 (each format twice). `stream-bench`
temperature 0, 256 new tokens; prompt lengths as the server counted them: 1 stream 247,907 tokens, 4 streams about 148,600 tokens each
(bf16 holds 624,048 tokens, so 4 x 150K is its ceiling). Step ms is the decode step the harness reports; the first-token time includes the prefill.

| format | 1 stream, ~248K: step | first token | 4 streams, ~149K each: step | first token |
|---|---|---|---|---|
| bf16_i8 | 38.82 / 38.89 ms | 136 s | 34.58 / 34.66 ms | 289-291 s |
| bf16 | 46.86 / 46.93 ms | 145 s | 39.72 / 39.71 ms | 301-303 s |
| fp8 | 52.25 / 52.31 ms | 154 s | 42.81 / 42.86 ms | 318-319 s |

bf16_i8 against bf16: 17% shorter decode step at 248K on one stream (38.9 vs 46.9 ms), 13% at 4 x 149K, 6% shorter prefill. Against fp8: 26% and 19%.
The two launches of each format agree to within 0.3% on step time. The order (bf16 reading its full K and V at this length is bandwidth-bound; int8 V halves V's
bytes; fp8 pays dequantisation on both) follows the byte counts. The 95th-percentile event gap at 4 streams (810-908 ms) is the prefill of the other streams, not decode.

Memory capacity per format is in `docs/evidence/2026-11-19-eng06-bf16i8/`. Raw `stream-bench` JSON and logs are under `runs/`; `scripts/` are the launchers.
