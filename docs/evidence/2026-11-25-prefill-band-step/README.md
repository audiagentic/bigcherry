# First-token time steps up by 20 ms between 128 and 129 prompt tokens

Stream-bench first-token numbers at a "128-token" prompt came out in two groups, about 106 and 128 ms on `qwen27b-tp3` (IQ8R), independent of idle time, of position in the request sequence and of the release. The cause is the prompt's token count: the prefill GEMM works in bands of 128 rows, so one row past a multiple of 128 costs a whole second band.

## First token against exact prompt tokens (chat endpoint, the server's own prompt token count, median of 3, release 2020.11.2)

| prompt tokens | 92 | 112 | 122 | 128 | 129 | 130 | 140 | 152 | 172 | 212 | 252 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| first token (ms) | 92.7 | 99.7 | 102.1 | 105.7 | 126.2 | 124.3 | 129.0 | 133.5 | 141.3 | 158.3 | 171.3 |

Release 2020.8.7 gives the same curve (128 tokens 103.7 ms, 129 tokens 125.8 ms), so it is not a regression. A client that sends the same prompt gets the same mode every time; stream-bench draws random words per seed, so its "128-token" prompts land on either side of the boundary by seed (seeds 1 to 8 gave 106 106 129 106 129 130 130 106 ms, and the same seeds gave the same pattern on a second pass). Compare first-token times at a stated exact prompt token count (for example 120 and 136), not at a word count.

## The step in the GEMM itself

Cold replay of the int8 prefill tile (`gemm_iq8r_tiled`), one XTX, 60 launches per point (median):

| N x K | 120 rows | 128 rows | 129 rows | 136 rows |
|---|---|---|---|---|
| 13312 x 5120 | 238.9 us | 239.9 | 411.6 | 410.8 |
| 5120 x 6656 | 164.0 | 163.5 | 224.3 | 224.2 |
| 6144 x 5120 | 140.8 | 140.6 | 177.4 | 177.7 |
| 5120 x 2304 | 86.4 | 87.2 | 109.6 | 105.7 |

Summed over the four shapes the step is about 0.29 ms a layer set, about 19 ms over 64 layers: the observed 20 ms.

## Timing method (cold replay)

Each point is the median of 60 timed launches, each preceded by a 256 MiB read that empties the caches, measured by the test binary's own event timing in one process. The window must be long: on the R9700 the first 10 to 25 launches of a process run at a lower power state (about 2.2 times slower for IQR rows), so a median over 15 launches or fewer can land in either state, while a median over 30 or more is steady. Compare a candidate and its baseline with the points ABBA ordered in separate processes, and call a point a win or a loss only beyond its own spread.
