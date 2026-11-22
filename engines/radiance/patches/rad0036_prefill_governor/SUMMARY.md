# rad0036_prefill_governor

**Status:** validated
**Plan item:** VTP42

Scope: one generic, swappable Radiance prefill policy behind a single option; replaces rad0034 and rad0035.

## Summary

`--prefill-governor SPEC` (`prefill_governor` in the manifest, edition 3) chooses which waiting prompts may start
prefilling and in what order prefill work is walked. `kind=arrival` (the empty spec) is the engine's own order;
`kind=cap` limits concurrent long prefills above a length threshold; `kind=budget` holds the prompt left in flight
within a factor of the largest request, with a reservation for a prompt waiting past `age_ms`. `order=remaining`
and `age_ms` give the ordering, common to every kind.

## Outcome

Untested. Card-free tests show each rad0034/rad0035 setting planned step for step by its spec equivalent; on the
card the `-long` spec matches them and a 1.5x budget with mid-size prompts exempt cuts the 100k mean 24%;
the `-long` tuning carries the same policy as one spec string. See README.md.

## Detail

See README.md.
