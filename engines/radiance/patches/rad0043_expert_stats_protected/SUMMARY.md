# rad0043_expert_stats_protected

**Status:** untested
**Plan item:** VTP44

## Summary

`--expert-stats` counts the routed experts a layer keeps in bf16 beside its quantised table (its protected experts). Before, their histogram indices had no entry in the stats map and were tallied as `unmapped`, so a quantised layer's counts came up short by exactly its protected experts' selections.

## Outcome

Not yet run. Expected on Flash-Next (hcbf16) with the same 4,163-token run: `ranks.unmapped` 0 instead of 1,672 and `selections == tokens x top_k` in all 48 layers.

## Detail

`Engine::xstats_open` builds the map from the program's weights that sit in an expert slot. The protected experts' weights are declared under the layer group alone, so they were skipped. The router's histogram is in sort-place order: the quantised experts take places 0 to n_reg-1 and the protected ones are sorted last. The patch appends, after each layer's regular places, the model experts named in the tensor names of the layer-scope weights (their container directory entries carry no expert field), ascending and without duplicates (the order the block lists its protected experts). Nothing outside `--expert-stats` changes: the map is read only by the stats tally.

## Evidence (2026-11-17)
Build `expert-stats-protected`, qwen3.8-next-flash-hcbf16, two 7900 XTX, six runs over the 4,163-token KLD corpus
(prefill): every run has `ranks.unmapped` 0 (it was 1,672), 48 of 48 layers with selections == tokens x 10, 1,998,240
selections in all. Flips against bf16-a with the protected experts counted: chunk 128 22,529 (1.127%), bf16_i8 22,068,
bf16_fp8 22,065, fp8 22,114 (1.104-1.107%); the same-config repeat 0. Earlier figures that left out the protected
experts (22,052-22,508) were within 0.1%. The state stays untested: this was a diagnostic run, not a qualification.

## 2026-11-18 correction (found by rad0044's route record)
The first version of this patch appended the protected experts but kept the container's numbering for the quantised ones. Comparing a recorded run's per-layer histograms (rad0044, the ids the router returns) with `--expert-stats` showed 44 of 48 layers identical and layers 34, 44, 46 and 47 shifted: the container numbers the quantised experts compactly while their tensors are named by model expert, so each expert after a protected one was filed one id too low per protected expert below it, and the protected experts' counts sat at the appended indices. Totals and every flip count between runs were unaffected (both runs share the labels); per-expert counts in those four layers were wrong, and so would any `--expert-profile` made from them. The patch now labels every routed unit from its tensor name.
