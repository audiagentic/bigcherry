# Flash-Next KV caches scored with the routing replayed, 11 documents (VTP44 closing set)

## Summary
The two-document replay result (FLASHNEXT-REPLAY.md) held on a larger, independent set. Over 11 documents (69,108 scored positions) the
caches rank bf16_i8, then bf16_fp8, then fp8, and both packed caches beat the shipped fp8 cache: bf16_i8 in 11 of 11 documents, bf16_fp8
in 10 of 11, with document-level 95% intervals that exclude zero. The ordering holds without the two noisiest documents and on the eight
short documents alone. KLD on the long documents is dominated by positions that are fragile for any change (ids, timestamps and hashes
in the corpus), so absolute values depend heavily on the text; the paired per-document differences are what carry the result.

## Outcome
`qwen3.8-next-flash-hcbf16`, two 7900 XTX TP2 exact wire, no helper, deterministic, one sequence, no speculation, build with rad0043,
0044 and 0045. The corpus is 11 documents: eight of about 12,000 characters (3,300 to 4,400 tokens) and three of about 48,000
(10,800 to 14,500 positions), from regions of the project-document text that the first pair did not use. The reference is bf16, chunk 64,
recorded with the top 64 tokens per position. The reference configuration was run once more against it while recording its expert route
(KLD 6.6e-11, top-1 100%: zero to rounding), and every arm below replays that route.

| run | KLD mean | p99 | p99.9 | max | top-1 |
|---|---|---|---|---|---|
| bf16, chunk 64 (reference configuration, replayed) | 6.6e-11 | 0 | 0 | 0 | 100% |
| bf16, chunk 128 instead of 64 | 0.02400 | 0.381 | 3.13 | 12.4 | 95.0% |
| bf16_i8 | 0.02288 | 0.334 | 3.29 | 12.0 | 95.0% |
| bf16_fp8 | 0.02513 | 0.370 | 3.15 | 16.6 | 94.9% |
| fp8 | 0.02740 | 0.450 | 3.11 | 13.5 | 94.4% |

Paired differences (A - B), position-weighted mean, 95% interval from a bootstrap over documents (10,000 resamples; `robustness-output.txt`):

| difference | all 11 documents | without long-01 and long-02 (9) | the 8 short documents |
|---|---|---|---|
| bf16_i8 - chunk-128 reference | -0.00112 [-0.00177, -0.00023] | -0.00049 [-0.00144, +0.00011] | -0.00086 [-0.00167, -0.00021] |
| bf16_fp8 - chunk-128 reference | +0.00113 [-0.00060, +0.00310] | +0.00055 [-0.00018, +0.00117] | +0.00010 [-0.00032, +0.00044] |
| fp8 - chunk-128 reference | +0.00341 [+0.00071, +0.00676] | +0.00118 [+0.00018, +0.00195] | +0.00065 [-0.00002, +0.00126] |
| bf16_fp8 - bf16_i8 | +0.00225 [+0.00073, +0.00431] | +0.00105 [+0.00056, +0.00158] | +0.00096 [+0.00049, +0.00168] |
| fp8 - bf16_i8 | +0.00452 [+0.00151, +0.00803] | +0.00168 [+0.00125, +0.00201] | +0.00151 [+0.00115, +0.00193] |
| fp8 - bf16_fp8 | +0.00227 [+0.00051, +0.00386] | +0.00063 [+0.00018, +0.00092] | +0.00055 [+0.00003, +0.00095] |
| documents where bf16_i8 beats fp8 | 11 of 11 | 9 of 9 | 8 of 8 |
| documents where bf16_fp8 beats fp8 | 10 of 11 | 8 of 9 | 7 of 8 |

Mean KLD of the three subsets (chunk 128 / bf16_i8 / bf16_fp8 / fp8): all 0.0240 / 0.0229 / 0.0251 / 0.0274; without the two long documents
0.0062 / 0.0057 / 0.0067 / 0.0074; short documents only 0.0066 / 0.0057 / 0.0067 / 0.0072.

## Detail
- Ordering: bf16_i8 is at or below the chunk-128 reference on every cut and bf16_fp8 is about level with it, so neither packed cache adds
  error that stands out from a change in the arithmetic; fp8 sits above it. Against the shipped fp8 cache both packed types are better in
  nearly every document, which is the criterion that matters for the product.
- Long documents: `long-01` and `long-02` carry mean KLD of 0.04 to 0.06 in every arm, `long-00` and the eight short documents 0.003 to
  0.015. In `long-01` and `long-02` the elevated KLD begins at about the same place in every arm (around position 960 in `long-02`, 2,200
  to 2,450 in `long-01`) whatever the change, half of the KLD sits in the top 1% of positions, and the arms agree on which positions are
  bad (per-position correlation 0.3 to 0.8; `sensitivity-output.txt`). The text there is ledger event ids, timestamps and hash strings,
  tokens a model cannot predict, so any numeric change reorders them. The long documents therefore measure how fragile the reference
  is on that text more than the cache; the paired comparison cancels most of it, and the result holds without them.
- The chunk-128 run is a reference for arithmetic noise, not a clean one on long documents: the cache arms, at the reference's own
  chunk size, show the same long-document elevation.
- Replay itself: the reference configuration replayed against its own reference scores 6.6e-11, and every replayed run's expert counts
  equal the record's in the layers checked in the two-document set; the closing run additionally needs its route to be the same on the
  review-fix build (see below).
- Cost: about 15 minutes of GPU time per arm for 69,108 positions (the chunk-128 arm 9 minutes).

## Limits
- One corpus of 11 documents, all from the project's own planning, evidence and log text, one run per arm (the runs are deterministic, so a
  repeat is byte-identical and adds nothing; more documents are what would add information). Natural-language prose and code were not
  scored separately.
- Prefill only; TP2 on two gfx1100 cards. The R9700-engine layouts use a different model container and were not scored this way (the layout
  smoke exercised their packed-cache kernels and needle probes only).
- Positions inside a document are not independent; the intervals are over documents, which is the unit that is.

## Files
`flashnext-replay-11docs/`: the six KLD files, the run receipt, `run_kld3.sh` (reference, route record and the five replayed arms),
`make_corpus3.py` (how the 11 documents are cut), `analyze_kld3.py`, `robustness_kld3.py`, `sensitivity_kld3.py` and their outputs.
Absolute paths in the files are replaced by `<work-root>`, `<dev-root>` and `<models-root>`. The per-position rows, the stats files and the
route file (about 17 MB) were in the run folder and are not kept.
