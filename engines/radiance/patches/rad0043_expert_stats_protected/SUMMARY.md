# rad0043_expert_stats_protected

**Status:** untested
**Plan item:** VTP44

## Summary

`--expert-stats` counts the routed experts a layer keeps in bf16 beside its quantised table (its protected experts). Before, their histogram indices had no entry in the stats map and were tallied as `unmapped`, so a quantised layer's counts came up short by exactly its protected experts' selections.

## Outcome

Not yet run. Expected on Flash-Next (hcbf16) with the same 4,163-token run: `ranks.unmapped` 0 instead of 1,672 and `selections == tokens x top_k` in all 48 layers.

## Detail

`Engine::xstats_open` builds the map from the program's weights that sit in an expert slot. The protected experts' weights are declared under the layer group alone, so they were skipped. The patch adds a second pass over the weights declared at layer scope whose container entry names an expert, and places each at the local index model id less the rank's expert base, read off the regular units of the layer (and used only if they all agree). Nothing outside `--expert-stats` changes: the map is read only by the stats tally.
