# rad0029_rank_own_split

**Status:** untested
**Plan item:** VTP33/VTP14

Scope: tensor parallelism over cards of different size or with uneven weight shares, for dense models (no expert
plane), on discrete cards.

## Summary

In an uneven dense world, every rank gets its own VRAM split. Each rank's weights pool holds its own weights, and its KV
pool takes the free memory it has above the smallest card. Every rank then pages the least token count any rank can page
in its own pool, instead of the common split's count. Before this, the rank that limits the token count kept the common
split: the smallest card's memory and the largest rank's weights pool.

## Outcome

Untested (not promoted). Built for VTP33: at TP3 (XTX, XTX, R9700) the R9700 holds 2 of the 4 KV heads and limits the
cache, while about 8 GiB of its card stayed free.

## Detail

See README.md.
