# rad0035_prefill_order

**Status:** superseded
**Plan item:** VTP41

Scope: generic Radiance prefill ordering and long-prefill threshold on top of rad0034, off by default.

## Summary

`--long-prefill-tokens N` sets what counts as a long prefill for `--max-prefill-seqs` (default one shared step).
`--prefill-order remaining` walks and admits prefills by least prompt left (after priority), with those behind the
first keeping one share between them; `--prefill-age-ms T` puts prompts waiting longer than T back in arrival order.
Serving v2 exposes the three optional tuning fields; no configuration enables them.

## Outcome

Untested and not promoted; off by default. Card-free scheduler tests: a 1000-token prompt beside three running
4000-token prefills reaches its first token at step 5 instead of after step 12. Hardware, beside eight 100k
prompts under the shipped cap: 4k / 16k requests 317-433 s shipped, 4-8 / 20-27 s with both options, the 100k
prompts +10%; see README.md.

## Detail

See README.md.
