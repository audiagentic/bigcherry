# VTP36: two cards against three, same release and harness (2020.8.5)

Configs as shipped: `qwen27b-tp3` (XTX, XTX, R9700; exact wire; pipelined prefill; all-reduce pin) against `qwen27b-2xtx` (two XTX; wire wht6, the lossy 6-bit wire; no pipelined prefill, no all-reduce pin),
with `max-num-seqs` raised from 6 to 8 on the pair so eight streams fit, and the pair again with `tp-wire=exact`. Order T W E E W T (one engine launch per arm, untraced), seeds 1-3 x 3 repetitions,
streams 6 and 8, speculation depth 3, 512-token prompts, 192 new tokens, temperature 0. Three fixed prompts at temperature 0 give byte-identical text in all six arms.
Two arms a config, so the error bars are wider than the four-arm notes (arm spread for the pair on the exact wire at 6 streams is 0.7 ms).

| streams | config | ms/step | tokens/step | acceptance | engine decode tok/s | per stream tok/s | bench aggregate tok/s (incl. prefill) | TTFT max |
|---|---|---|---|---|---|---|---|---|
| 6 | tp3 exact | 29.92 | 15.88 | 0.725 | 529 | 91.6 | 337.5 | 1.50 s |
| 6 | tp2 wht6 | 31.86 | 15.72 | 0.702 | 492 | 84.4 | 305.2 | 1.70 s |
| 6 | tp2 exact | 33.26 | 15.24 | 0.713 | 456 | 79.3 | 258.4 | 2.28 s |
| 8 | tp3 exact | 31.86 | 18.27 | 0.716 | 574 | 79.6 | 361.2 | 2.10 s |
| 8 | tp2 wht6 | 32.61 | 18.56 | 0.709 | 569 | 77.6 | 335.7 | 2.41 s |
| 8 | tp2 exact | 33.50 | 16.99 | 0.722 | 507 | 72.8 | 275.7 | 3.33 s |

Reading: against a pair on the same exact wire the third card is worth +16% (6 streams) and +13% (8 streams) engine decode tok/s. Against the pair on the lossy wire it is worth +7.6% at 6 streams and about nothing at 8 (level tokens a step).
The lossy wire is worth +9% / +12% decode tok/s to the pair and lowers acceptance at 6 streams (0.702 against 0.713). The aggregate gap includes configuration differences (pipelined prefill and the all-reduce pin exist only in the three-card layout), so it is not the cards alone.
Not tested: long context on the pair.
