"""VTP44: --expert-stats labels every routed unit with its model expert, and counts a layer's protected experts.

A quantised layer keeps a few experts in bf16 beside its table (MoeFP8::prot). The router's histogram is in sort-place
order: the quantised experts take places 0 .. n_reg-1 and the protected ones are sorted last. Engine::xstats_open labelled
a place with the expert number of its weight's container entry, which has two faults on such a layer:

- the container numbers the quantised experts compactly (0 .. n_reg-1) while their tensors are named by model expert
  (blk.L.ffn_gate_up_exps.<model id>.weight), so every expert after a protected one was filed under an id one lower per
  protected expert below it (layer 34 of Flash-Next: places 407..494 under ids 407..494 instead of 408..495);
- the protected experts' weights are declared under the layer group, not a place, so their places were never mapped
  and their selections were tallied as `unmapped` (1,672 in a 4,163-token run: layers 34, 44, 46 and 47).

The label of a place is now the model expert in its weight's tensor name, the id the router itself returns. The
protected experts' places follow the regular ones, appended in ascending order, the order MoeFP8 lists them. Layers
without protected experts are labelled as before (the two numberings agree there).
"""
from bigcherry.patcher import Edit, FilePatch

PATCHES = (
    FilePatch(
        path="core/engine.cpp",
        description="Map the protected experts' histogram indices in Engine::xstats_open.",
        edits=(
            Edit(
                id="xstats-map-protected-experts",
                anchor=r'^            for \(int32_t& v : row\) if \(v == kUnset \|\| v < 0\) v = -1;$',
                text='''
        std::vector<std::vector<int32_t>> xs_extra((size_t)n_layer);
        std::vector<std::vector<char>> xs_hit((size_t)n_layer);
        for (size_t w = 1; w < P.weights.size(); ++w) {
            const WeightInfo& wi = P.weights[w];
            const std::string& xs_name = wi.source.empty() ? wi.name : wi.source;
            int xs_l = -1, xs_e = -1;
            if ((sscanf(xs_name.c_str(), "blk.%d.ffn_gate_up_exps.%d.weight", &xs_l, &xs_e) != 2 &&
                 sscanf(xs_name.c_str(), "blk.%d.ffn_down_exps.%d.weight", &xs_l, &xs_e) != 2) ||
                xs_l < 0 || xs_l >= n_layer || xs_l != wi.decl.group.layer || xs_e < 0) continue;
            std::vector<int32_t>& row = layers[(size_t)xs_l];
            const int32_t at = wi.decl.group.expert;
            if (at < 0) {
                xs_extra[(size_t)xs_l].push_back(xs_e);
            } else if ((size_t)at < row.size()) {
                std::vector<char>& hit = xs_hit[(size_t)xs_l];
                hit.resize(row.size(), 0);
                row[(size_t)at] = hit[(size_t)at] && row[(size_t)at] != xs_e ? kClash : xs_e;   /* gate/up and down disagree */
                hit[(size_t)at] = 1;
            }
        }
        for (int32_t l = 0; l < n_layer; ++l) {
            std::vector<int32_t>& g = xs_extra[(size_t)l];
            std::vector<int32_t>& row = layers[(size_t)l];
            size_t xs_bad = 0;
            for (size_t i = 0; i < row.size(); ++i)
                xs_bad += row[i] == kClash || (!g.empty() && (i >= xs_hit[(size_t)l].size() || !xs_hit[(size_t)l][i]));
            if (xs_bad)
                RAD_WARN("--expert-stats: layer %d: %zu routed unit(s) have no consistent model-expert tensor name; "
                         "their labels may be wrong", (int)l, xs_bad);
            std::sort(g.begin(), g.end());
            row.insert(row.end(), g.begin(), std::unique(g.begin(), g.end()));
        }''',




                guard=r'xs_extra',
                expect_matches=1,
                max_span_lines=1,
                rationale="the closing statement of the per-rank map rows; the protected places follow the regular ones, so they are appended once the regular units have fixed the rows",
            ),
        ),
    ),
)
