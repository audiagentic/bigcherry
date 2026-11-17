"""VTP44: --expert-stats counts a layer's protected experts.

A quantised layer keeps a few experts in bf16 beside its table (MoeFP8::prot). Their weights are declared under the layer
group, not an expert slot, so Engine::xstats_open's weight loop never saw them: their histogram indices fell outside the
map and were tallied as `unmapped`, and the layer's counts came up short. Flash-Next (hcbf16) keeps two such experts in
layers 34, 44 and 46 and four in layer 47, and a 4,163-token run was missing 1,672 selections in those four layers.

The map entry for a protected expert is its model id (the container entry's expert) at the local index model id less the
rank's expert base. The base is read off the regular units of the same layer, and only when they all agree.
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
        for (size_t w = 1; w < P.weights.size(); ++w) {
            const WeightInfo& wi = P.weights[w];
            const int32_t l = wi.decl.group.layer;
            if (l < 0 || l >= n_layer || wi.decl.group.expert >= 0) continue;
            const RadFileEntry* fe = file_ ? file_->find(wi.source) : nullptr;
            if (!fe || fe->layer != l || fe->expert < 0) continue;
            std::vector<int32_t>& row = layers[(size_t)l];
            int64_t xs_protected_base = INT64_MIN;
            bool xs_protected_agree = true;
            for (size_t e = 0; e < row.size() && xs_protected_agree; ++e) {
                if (row[e] < 0) continue;
                const int64_t b = (int64_t)row[e] - (int64_t)e;
                if (xs_protected_base == INT64_MIN) xs_protected_base = b;
                else xs_protected_agree = b == xs_protected_base;
            }
            if (!xs_protected_agree || xs_protected_base == INT64_MIN) continue;
            const int64_t xs_protected_at = (int64_t)fe->expert - xs_protected_base;
            if (xs_protected_at < 0 || xs_protected_at > 65535) continue;
            if ((size_t)xs_protected_at >= row.size()) row.resize((size_t)xs_protected_at + 1, -1);
            if (row[(size_t)xs_protected_at] < 0) row[(size_t)xs_protected_at] = fe->expert;
        }''',
                guard=r'xs_protected_base',
                expect_matches=1,
                max_span_lines=1,
                rationale="the closing statement of the per-rank map rows; the protected weights are mapped once the regular units have fixed the rows and the base",
            ),
        ),
    ),
)
