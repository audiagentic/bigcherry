# VTP36: decode A/B of one piece against four for the 3-rank all-reduce at decode sizes

The shipped `qwen27b-tp3` pin runs the exact 3-rank two-shot in four pieces (`ts_pieces=4`), chosen on prefill-size messages (2048 rows). A decode message is about 27 rows x 5120 = 138,000 elements
(276 KB exact), and four pieces mean nine dependent flag values per call (2*pieces+1), each about 2 us through host memory (measured flag ping-pong between card pairs: 1.90-2.09 us one way), against three flag values
with one piece. The A/B asks whether decode-size messages should use one piece.

## Setup

Release 2020.8.5, `qwen27b-tp3` (IQ8R linears, DFlash2 drafter, speculation depth 3), untraced, same ABBA protocol as the other A/B notes (two cycles A B B A A B B A, one engine launch per arm, each arm under its own
GPU lease, seeds 1-3, three repetitions per seed, streams 6 and 8, 192 new tokens, temperature 0, 512-token prompts, three fixed prompts for a text check).
A = the shipped pin `all_reduce:nblocks=4,nthreads=256,ts_pieces=4,ts_fuse=1,ts_once=1,ts_duplex=1`. B = the same pin plus `ts_pieces_min=524288` (elements), so messages under 524,288 elements (every decode step: at most about
164,000 elements at 8 streams) take one piece and larger ones keep four. The engine log of the B arms prints the changed pin; the wire is exact in both arms.

## Result (within this session only)

| streams | A shipped, ms/step mean (sd of arm means) | B one piece for decode, ms/step (sd) | B - A (s.e.) | acceptance A / B | tokens/step A / B |
|---|---|---|---|---|---|
| 6 | 30.29 (0.36) | 30.02 (0.20) | -0.27 ms, -0.9% (0.21) | 0.727 / 0.728 | 16.24 / 15.99 |
| 8 | 31.68 (0.28) | 31.76 (0.22) | +0.08 ms, +0.2% (0.18) | 0.721 / 0.717 | 18.17 / 18.18 |

Arm means, 6 streams: A 30.63 30.54 29.86 30.12, B 29.95 30.02 30.29 29.81; 8 streams: A 31.51 31.45 32.07 31.68, B 32.05 31.82 31.55 31.62 ms/step. The text of the three fixed prompts is byte-identical in all eight arms.
The harness's noise floor on this release is about 0.85 ms at 6 streams and 0.26 ms at 8 streams (see the replay-and-noise note).

## Reading

- No effect: -0.27 ms (s.e. 0.21) at 6 streams and +0.08 ms (s.e. 0.18) at 8 streams; the effect is bounded to about +-0.4 ms (two standard errors). The cut of 1.5-2.3 ms a pass predicted from six fewer dependent flag values per call
  (129 calls) is not there, so the nine flag values are not what the fixed part of an all-reduce call is made of at decode sizes, or the pieces already overlap their handshakes.
- The decision is NO-GO: the shipped four-piece pin stays at decode sizes and no size-derived piece rule is worth building on this evidence.
- The remaining candidates for the fixed part of the call are the kernel entry and exit, the fused-norm consumer launch behind it, and the R9700's own link time; they are not separable from the tape alone (the
  timeline stamps of the collective are a lab launch parameter, not reachable from a production start).
