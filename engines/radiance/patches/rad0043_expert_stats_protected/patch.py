"""VTP44: --expert-stats counts a layer's protected experts.

A quantised layer keeps a few experts in bf16 beside its table (MoeFP8::prot). The router's histogram is in sort-place order:
the quantised experts take places 0 .. n_reg-1 (each unit declared in its place's slot, named by its model expert in the
container) and the protected ones are sorted last. Their weights are declared under the layer group, not a place, so
Engine::xstats_open's weight loop never saw them: the map row stopped at n_reg and their histogram places were tallied as
`unmapped`, leaving the layer's counts short. Flash-Next (hcbf16) keeps two such experts in layers 34, 44 and 46 and four in
layer 47, and a 4,163-token run was missing 1,672 selections in those four layers.

The map entry for a protected place is its model expert, read from the tensor name of the layer-scope weights (the container directory keeps these bf16 entries outside the expert directory, so their `expert` field is -1). They are appended in ascending expert order, the order MoeFP8 lists them (`prot` is built ascending).
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
        for (int32_t l = 0; l < n_layer; ++l) {
            std::vector<int32_t> xs_protected;
            for (size_t w = 1; w < P.weights.size(); ++w) {
                const WeightInfo& wi = P.weights[w];
                if (wi.decl.group.layer != l || wi.decl.group.expert >= 0) continue;
                int xs_pl = -1, xs_pe = -1;
                const bool xs_named = sscanf(wi.source.c_str(), "blk.%d.ffn_gate_up_exps.%d.weight", &xs_pl, &xs_pe) == 2 ||
                                      sscanf(wi.source.c_str(), "blk.%d.ffn_down_exps.%d.weight", &xs_pl, &xs_pe) == 2;
                if (xs_named && xs_pl == l && xs_pe >= 0) xs_protected.push_back(xs_pe);
            }
            std::sort(xs_protected.begin(), xs_protected.end());
            xs_protected.erase(std::unique(xs_protected.begin(), xs_protected.end()), xs_protected.end());
            for (const int32_t g : xs_protected) layers[(size_t)l].push_back(g);
        }''',

                guard=r'xs_protected',
                expect_matches=1,
                max_span_lines=1,
                rationale="the closing statement of the per-rank map rows; the protected places follow the regular ones, so they are appended once the regular units have fixed the rows",
            ),
        ),
    ),
)
