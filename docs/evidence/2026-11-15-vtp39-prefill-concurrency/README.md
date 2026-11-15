# VTP39: long prompts beside other work on the 262,144-token configs

2026-11-15. Layout `qwen27b-tp3-int8-556-long-bf16` (TP3: 7900 XTX, 7900 XTX, R9700; split 5,5,6; bf16 KV cache,
624,051 tokens; 8 sequences; 2048-token steps; DFlash2 depth 3). Engine argv from the stored manifest, one fresh start
per run, arm flags merged into it; build with rad0034 (`--max-prefill-seqs`) on main's libr3 (RAD52 + RAD53). Every
arm ran twice, in mirrored order; the two starts of an arm agree within 1%.

## Interleave: decoders beside a 100k-token prompt

Streams decoding 3000 tokens each start 4 s before one 100,366-token prompt; chunks a second and gaps measured while it
prefills.

| decoders | `--prefill-interleave` | decode chunks/s | x base | long prompt's first token | gap p50 / max |
|---|---|---|---|---|---|
| 4 | 0 | 6.0 | - | 51.3 s | 847 / 1160 ms |
| 4 | 2 | 16.0 | 2.7 | 56.3 s (+9.8%) | 41 / 1150 ms |
| 4 | 4 | 24.4 | 4.1 | 61.2 s (+19%) | 34 / 1157 ms |
| 4 | 8 | 37.5 | 6.2 | 71.1 s (+39%) | 34 / 1149 ms |
| 7 | 0 | 10.3 | - | 52.1 s | 864 / 1166 ms |
| 7 | 2 | 26.4 | 2.6 | 59.5 s (+14%) | 58 / 1162 ms |

A lone 100k prompt (nothing decoding): 44.03 s at 0 and 44.00 s at 2. Eight 100k prompts at once: mean first token
433 s at interleave 2 against 434 s at 0. The decode share bought and the prompt time it costs move together; the
longest gap stays one 2048-token step at depth (about 1.15 s), which only a smaller busy chunk (step 4) would cut.

Decision (owner, 2026-11-15): interleave 2 is the `-long` configs' default (`configs/qwen27b.toml`).

## Prefill cap: eight 100k-token prompts at once

`fullcache.py`: eight distinct ~100k-token prompts (799,760 tokens, more than the cache) sent together, four ~450-token
requests just behind them, then an over-length prompt and a burst of 80 tiny requests.

| `--max-prefill-seqs` | interleave | mean first token, long | first / last | short requests | preemptions (forced) |
|---|---|---|---|---|---|
| 0 | 0 | 434 s | 377 / 530 s | 321-338 s | 6 (4) |
| 2 | 0 | 265 s (-39%) | 82 / 440 s | 121-123 s * | 0 (0) |
| 1 | 0 | 203 s * / 208 s (-52%) | 46-50 / 360-368 s | 44-45 s * / 3.6-4.5 s | 0 (0) |
| 1 | 2 | 206 s | 49 / 363 s | 3.5-4.4 s | 0 (0) |
| 0 | 2 | 433 s | 380 / 529 s | 320-335 s | 6 (4) |

\* before the short-prompt fix. Every request completed with status 200 in every arm; the over-length prompt got a 400
and the burst past the admission queue 429s, as before.

The first rad0034 build exempted from the cap only a prompt finished in its admission step. Beside a running 100k
prefill a busy step leaves a waiting request one share (256 tokens), so the ~450-token requests counted as long and
waited for that prefill (about 45 s). The fix exempts any prompt whose remainder fits one shared step; the short
requests then start in under 5 s, and the long prompts pay 2% (203 -> 208 s).

## Against the acceptance criteria

1. 4 decoders beside a 100k prefill at 3x base with the long prompt within +10%: **not met by one setting**. Interleave 2
   gives 2.7x at +9.8%; 3x needs about +12%. Gap max recorded (1.15 s at every setting).
2. Eight 100k prompts, mean first token at least 35% lower with no forced preemption: **met** by a cap of 1 (-52%, 0
   preemptions) and of 2 (-39%).
3. A short request beside long prefills gets its first token within a few steps: **met** with the cap and the fix
   (3.5-4.5 s, a few 2048-token steps at depth); not met without the cap (5.5 minutes).
4. Defaults unchanged until the owner's choice: interleave 2 chosen for `-long`; the cap stays off pending the owner.
