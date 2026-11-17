# rad0043_expert_stats_protected

**Status:** untested
**Plan item:** VTP44

## Summary

`--expert-stats` counts the routed experts a layer keeps in bf16 beside its quantised table (its protected experts). Before, their histogram indices had no entry in the stats map and were tallied as `unmapped`, so a quantised layer's counts came up short by exactly its protected experts' selections.

## Outcome

Not yet run. Expected on Flash-Next (hcbf16) with the same 4,163-token run: `ranks.unmapped` 0 instead of 1,672 and `selections == tokens x top_k` in all 48 layers.

## Detail

`Engine::xstats_open` builds the map from the program's weights that sit in an expert slot. The protected experts' weights are declared under the layer group alone, so they were skipped. The router's histogram is in sort-place order: the quantised experts take places 0 to n_reg-1 and the protected ones are sorted last. The patch appends, after each layer's regular places, the model experts named in the tensor names of the layer-scope weights (their container directory entries carry no expert field), ascending and without duplicates (the order the block lists its protected experts). Nothing outside `--expert-stats` changes: the map is read only by the stats tally.
