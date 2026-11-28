"""Runtime plumbing for ABI44's resident B16F8 and B16I8 cache layouts."""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _replace(edit_id: str, before: str, after: str, guard: str | None = None) -> Edit:
    return Edit(
        id=edit_id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=(guard[3:] if guard and guard.startswith("re:") else re.escape(guard or after)),
        expect_matches=1,
        max_span_lines=before.count("\n") + 1,
        rationale="Carry the selected packed KV layout to the resident pool",
    )


def _create(path: str, edit_id: str, content: str, guard: str | None = None) -> FilePatch:
    return FilePatch(path=path, language="none", create=True, edits=(
        Edit(id=edit_id, anchor=r"\A", mode="insert_after", text=content,
             guard=re.escape(guard or content.splitlines()[0]),
             rationale="Define and exercise the ABI44 resident B16 runtime contract"),
    ))


_PACKED_KV = r"""/* Packed B16 selection and pre-allocation checks for the engine and builder. */
#pragma once

#include "rad_builder.h"

#include <cstdint>
#include <cstring>

namespace rad {

inline bool b16_kv_layout(RadKVLayout layout) {
    return layout == RAD_KVQ_B16F8 || layout == RAD_KVQ_B16I8;
}

inline RadKVLayout configured_b16_kv_layout(const char* name) {
    const RadKVLayout layout = rad_kvq_from_name(name);
    return b16_kv_layout(layout) ? layout : RAD_KVQ_NONE;
}

inline void set_build_kv_format(const char* name, RadBuildCtx* out) {
    out->kv_layout = configured_b16_kv_layout(name);
    out->kv_dtype = out->kv_layout != RAD_KVQ_NONE ? RAD_U8
                  : !std::strcmp(name, "i8") ? RAD_I8
                  : !std::strcmp(name, "fp8") ? RAD_F8E4M3 : RAD_BF16;
}

inline bool b16_kv_model_supported(RadKVLayout layout, const RadArchProbe& probe) {
    return b16_kv_layout(layout) && (probe.kv_layout_ok & (1u << layout)) != 0;
}

inline bool b16_kv_group_supported(const RadKVGroupDecl& d) {
    return b16_kv_layout(d.kv_layout) && d.kind == RAD_KV_FULL &&
           d.head_dim == 256 && d.n_head_kv > 0 && d.dtype == RAD_U8;
}

/* A paged group's block covers all bound layers. Validate before multiplying so invalid
 * or overflowing declarations cannot produce a usable pool plan. */
inline int64_t b16_kv_block_bytes(const RadKVGroupDecl& d, int64_t page, int64_t layers) {
    if (!b16_kv_group_supported(d) || (page != 4 && page != 16) || layers <= 0) return 0;
    const int64_t per_head = page * rad_kv_group_row_elems(d.kv_layout, d.dtype, d.head_dim);
    if (per_head <= 0 || d.n_head_kv > INT64_MAX / per_head) return 0;
    const int64_t per_layer = per_head * d.n_head_kv;
    return layers <= INT64_MAX / per_layer ? per_layer * layers : 0;
}

}  // namespace rad
"""

_CONFIG_HELP = '''    { "--kv-cache-dtype",     "T",      "bf16 (default) | i8 | fp8 -- i8 halves what a token of\\n"
                                        "context costs: K and V as int8 with one scale each per token\\n"
                                        "and head. fp8 (E4M3): GLM-5.3's latent cache; DeepSeek-V4.1's\\n"
                                        "compressed latent and index keys, exact (the bf16 cache's bits)" },
'''

_CONFIG_PARSE = '''            if (std::strcmp(v, "bf16") && std::strcmp(v, "i8") && std::strcmp(v, "fp8")) {
                RAD_ERR("--kv-cache-dtype takes bf16, i8 or fp8, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->kv_cache_dtype = v;'''

_BUILD_DTYPE = '''    bc.kv_dtype = cfg_.kv_cache_dtype == "i8" ? RAD_I8
                : cfg_.kv_cache_dtype == "fp8" ? RAD_F8E4M3 : RAD_BF16;'''

_STREAM_PROBE = '''    (void)rad_tools_probe(plugins_, meta_, &arch_probe_, cfg_.image_gen_max_pixels, cfg_.image_gen_max_refs);
    if (cfg_.kv_resident > 0 && arch_probe_.kv_stream_ok == 0) {'''
_PACKED_ADMISSION = '''    const int probe = rad_tools_probe(plugins_, meta_, &arch_probe_, cfg_.image_gen_max_pixels,
                                      cfg_.image_gen_max_refs);
    const RadKVLayout layout = configured_b16_kv_layout(cfg_.kv_cache_dtype.c_str());
    if (layout != RAD_KVQ_NONE) {
        if (probe < 0 || !b16_kv_model_supported(layout, arch_probe_)) {
            RAD_ERR("--kv-cache-dtype %s: the '%s' plugin does not serve this packed cache "
                    "for this model", cfg_.kv_cache_dtype.c_str(), plugins_.arch_plugin.c_str());
            return RAD_E_UNSUPPORTED;
        }
        if (cfg_.kv_resident > 0) {
            RAD_WARN("--kv-resident ignored with --kv-cache-dtype %s: packed KV stays resident",
                     cfg_.kv_cache_dtype.c_str());
            cfg_.kv_resident = 0;
            decide("--kv-resident", "0", "the packed KV layout has no streamed writer");
        }
    }
    if (cfg_.kv_resident > 0 && arch_probe_.kv_stream_ok == 0) {'''

_DECL = '''    KVGroupInfo g;
    g.name = n;
    g.decl = *d;
    prog_.kv_groups.push_back(std::move(g));'''
_DECL_PACKED = '''    KVGroupInfo g;
    g.name = n;
    g.decl = *d;
    /* Only the trunk attention group inherits the deployment selection. Auxiliary groups,
     * including the sparse index and recurrent state, keep their own layouts. The declaring
     * plugin still owns the physical dtype and the matching store and reader. */
    if (n == "kv_attn" && prog_.ctx.kv_layout != RAD_KVQ_NONE) {
        if (d->kv_layout != RAD_KVQ_NONE && d->kv_layout != prog_.ctx.kv_layout)
            return err(RAD_E_INVAL, "kv group '%s': layout differs from deployment", name);
        g.decl.kv_layout = prog_.ctx.kv_layout;
    }
    if (g.decl.kv_layout != RAD_KVQ_NONE && !b16_kv_group_supported(g.decl))
        return err(RAD_E_UNSUPPORTED, "kv group '%s': packed B16 needs a full H256 RAD_U8 cache",
                   name);
    prog_.kv_groups.push_back(std::move(g));'''

_LAYER_BYTES = '''    const int64_t nh = gi.decl.n_head_kv > 0 ? gi.decl.n_head_kv : 1;
    return rad_dtype_bytes(gi.decl.dtype,
                           bs * nh * rad_kv_row_elems(gi.decl.dtype, gi.decl.head_dim));'''
_LAYER_BYTES_PACKED = '''    if (gi.decl.kv_layout != RAD_KVQ_NONE)
        return b16_kv_block_bytes(gi.decl, bs, 1);
''' + _LAYER_BYTES

_PLAN_BYTES = '''                gi.bytes_per_block =
                    rad_dtype_bytes(d.dtype, gi.block_size * elem_n_head *
                                             rad_kv_row_elems(d.dtype, d.head_dim)) * n_bound;'''
_PLAN_BYTES_PACKED = '''                if (d.kv_layout != RAD_KVQ_NONE) {
                    gi.bytes_per_block = b16_kv_block_bytes(d, gi.block_size, n_bound);
                    if (gi.bytes_per_block == 0)
                        err(RAD_E_UNSUPPORTED,
                            "kv group '%s': unsupported packed B16 page geometry (block %lld)",
                            gi.name.c_str(), (long long)gi.block_size);
                } else {
''' + "\n".join("    " + line for line in _PLAN_BYTES.splitlines()) + '''
                }'''

_FALLBACK = '''            p.bytes_per_block = g.bytes_per_block > 0
                ? g.bytes_per_block
                : p.block_size * g.decl.n_head_kv * rad_kv_row_elems(g.decl.dtype, g.decl.head_dim) *
                  elt * p.n_layers;'''
_FALLBACK_PACKED = '''            if (g.decl.kv_layout != RAD_KVQ_NONE) {
                p.bytes_per_block = b16_kv_block_bytes(g.decl, p.block_size, p.n_layers);
                if (g.bytes_per_block > 0 && g.bytes_per_block != p.bytes_per_block) {
                    RAD_ERR("KV group '%s': packed B16 page bytes disagree with its declaration",
                            p.name.c_str());
                    return RAD_E_INVAL;
                }
            } else {
''' + "\n".join("    " + line for line in _FALLBACK.splitlines()) + '''
            }'''

_STREAM = '    const KVStreamOps ops = kvstream_ops(prog, &gi);'
_STREAM_GUARD = '''    if (gi.decl.kv_layout != RAD_KVQ_NONE) {
        RAD_WARN("--kv-resident ignored: group '%s' has no packed KV stream writer and stays resident",
                 gi.name.c_str());
        *out = KVStreamSetup{};
        return RAD_OK;
    }
'''

_FIXTURE = r"""/* Card-free integration of packed KV config, declaration, memory planning and binding. */
#include "rad_test.h"

#include "build/rad_build.h"
#include "fleet/config.h"
#include "kvstream/bringup.h"
#include "mem/kv.h"
#include "mem/packed_kv.h"
#include "runtime/kvbind.h"

using namespace rad;

namespace {

rad_kvgroup declare_cache(Builder& b, const RadBuildCtx& ctx, int64_t page) {
    RadKVGroupDecl d{};
    d.kind = RAD_KV_FULL;
    d.dtype = ctx.kv_dtype;
    d.head_dim = 256;
    d.n_head_kv = 2;
    const rad_kvgroup group = b.decl_kv_group("kv_attn", &d);
    const RadParam p[] = { RAD_INT("q_len", 1), RAD_INT("head_dim", 256),
                           RAD_INT("window", 0), RAD_INT("block_size", page),
                           RAD_INT("causal", 1), RAD_INT("n_head", 24),
                           RAD_STR("kv_dtype", rad_dtype_name(d.dtype)) };
    b.decl_op("t_attn_ref", p, 7, nullptr, 0);
    b.bind_layer_kv(0, group);
    b.bind_layer_kv(1, group);
    return group;
}

int parse_cli(const char* name, Config* cfg) {
    std::string words[] = {"radiance", "--model", "unused.rad", "--kv-cache-dtype", name};
    char* argv[] = {words[0].data(), words[1].data(), words[2].data(),
                    words[3].data(), words[4].data(), nullptr};
    return config_parse(5, argv, cfg);
}

}  // namespace

TEST(config_to_resident_page) {
    Registry reg;
    REQUIRE_EQ(reg.load_dir(home() + "/testkernels", {"radtest_base"}), RAD_OK);
    RadModelMeta meta{};
    meta.arch_id = "qwen4exp";
    meta.n_layers = 2;
    const struct {
        const char* name;
        RadKVLayout layout;
        uint32_t dtype;
        int64_t row, layer_bytes;
    } formats[] = {
        {"bf16_fp8", RAD_KVQ_B16F8, RAD_U8, 768, 6144},
        {"bf16_i8", RAD_KVQ_B16I8, RAD_U8, 784, 6272},
        {"fp8", RAD_KVQ_NONE, RAD_F8E4M3, 512, 4096},
        {"i8", RAD_KVQ_NONE, RAD_I8, 516, 4128},
    };
    for (const auto& f : formats) {
        Config cli;
        REQUIRE_EQ(parse_cli(f.name, &cli), RAD_OK);
        fleet::FleetConfig yaml;
        REQUIRE_EQ(fleet::parse(std::string("models:\n  flash:\n    model: unused.rad\n"
                                            "    kv-cache-dtype: ") + f.name + "\n",
                                "packed-kv-fixture", &yaml), RAD_OK);
        REQUIRE_EQ(yaml.models.size(), size_t{1});
        Config& cfg = yaml.models.front().cfg;
        CHECK_EQ(cfg.kv_cache_dtype, cli.kv_cache_dtype);
        CHECK_EQ(cfg.given.at("--kv-cache-dtype"), "models.flash.kv-cache-dtype");
        cfg.max_seqs = 1;
        cfg.max_ctx = 65536;

        RadBuildCtx ctx{};
        set_build_kv_format(cfg.kv_cache_dtype.c_str(), &ctx);
        CHECK_EQ(ctx.kv_layout, f.layout);
        CHECK_EQ(uint32_t(ctx.kv_dtype), f.dtype);
        if (f.layout != RAD_KVQ_NONE) {
            RadArchProbe probe{};
            CHECK(!b16_kv_model_supported(f.layout, probe));
            probe.kv_layout_ok = 1u << f.layout;
            CHECK(b16_kv_model_supported(f.layout, probe));
            const RadKVLayout other = f.layout == RAD_KVQ_B16F8 ? RAD_KVQ_B16I8 : RAD_KVQ_B16F8;
            CHECK(!b16_kv_model_supported(other, probe));
        }

        Builder b(reg, meta, ctx);
        const rad_kvgroup group = declare_cache(b, ctx, 4);
        REQUIRE(group);
        CHECK_EQ(b.kv_layer_bytes(group), f.layer_bytes);
        REQUIRE_EQ(b.finish(), RAD_OK);
        const Program& prog = b.program();
        const KVGroupInfo& info = prog.kv_groups[group];
        CHECK_EQ(info.decl.kv_layout, f.layout);
        CHECK_EQ(info.bytes_per_block, 2 * f.layer_bytes);
        CHECK_EQ(info.block_size, 4);

        KVManager manager;
        REQUIRE_EQ(manager.plan_groups(prog.kv_groups, cfg), RAD_OK);
        REQUIRE(manager.plan(group));
        CHECK_EQ(manager.plan(group)->layer_bytes_per_block(), f.layer_bytes);
        auto fallback = prog.kv_groups;
        fallback[group].bytes_per_block = 0;
        REQUIRE_EQ(manager.plan_groups(fallback, cfg), RAD_OK);
        REQUIRE(manager.plan(group));
        CHECK_EQ(manager.plan(group)->layer_bytes_per_block(), f.layer_bytes);

        std::vector<KVPoolBinding> pools;
        bind_kv_pools(prog, [&](int32_t g) { return g ? manager.plan(g) : nullptr; }, &pools);
        CHECK_EQ(pools[group].dtype, f.dtype);
        CHECK_EQ(pools[group].shape[1], 2);
        CHECK_EQ(pools[group].shape[2], 4);
        CHECK_EQ(pools[group].shape[3], f.row);
        CHECK_EQ(pools[group].layer_slot[1], 1);

        if (f.layout != RAD_KVQ_NONE) {
            KVStreamSetup stream;
            cfg.kv_resident = 32768;
            CHECK_EQ(kvstream_configure(cfg, cfg.max_ctx, prog.kv_groups, prog, &stream),
                     RAD_OK);
            CHECK_EQ(stream.group, -1);
            CHECK_EQ(stream.ring_blocks, 0);
        }
    }
}

TEST(invalid_layout_refused_before_pool_setup) {
    Config cfg;
    CHECK_EQ(parse_cli("bf16_typo", &cfg), RAD_E_INVAL);
    Registry reg;
    REQUIRE_EQ(reg.load_dir(home() + "/testkernels", {"radtest_base"}), RAD_OK);
    RadModelMeta meta{};
    RadBuildCtx ctx{};
    set_build_kv_format("bf16_i8", &ctx);

    RadKVGroupDecl d{};
    d.kind = RAD_KV_FULL;
    d.dtype = RAD_U8;
    d.head_dim = 128;
    d.n_head_kv = 2;
    Builder wrong_head(reg, meta, ctx);
    CHECK_EQ(wrong_head.decl_kv_group("kv_attn", &d), RAD_NULL_HANDLE);

    d.head_dim = 256;
    d.dtype = RAD_I8;
    Builder wrong_dtype(reg, meta, ctx);
    CHECK_EQ(wrong_dtype.decl_kv_group("kv_attn", &d), RAD_NULL_HANDLE);

    Builder wrong_page(reg, meta, ctx);
    const auto group = declare_cache(wrong_page, ctx, 8);
    REQUIRE(group);
    CHECK_EQ(wrong_page.kv_layer_bytes(group), 0);
    CHECK_EQ(wrong_page.finish(), RAD_E_UNSUPPORTED);
    KVManager manager;
    CHECK_EQ(manager.plan_groups(wrong_page.program().kv_groups, cfg), RAD_E_INVAL);
}

RAD_TEST_MAIN()
"""

PATCHES = (
    _create("tests/bc_b16_kv_runtime_test.cpp", "b16-core-fixture", _FIXTURE),
    FilePatch(path="tests/CMakeLists.txt", language="none", edits=(
        _replace("b16-test-registry", '  if(n IN_LIST RAD_TESTS_WITH_LEGACY)\n',
                 '  if(n STREQUAL "bc_b16_kv_runtime_test")\n'
                 '    add_dependencies(${n} radtest_base)\n'
                 '  endif()\n'
                 '  if(n IN_LIST RAD_TESTS_WITH_LEGACY)\n'),
    )),
    _create("core/mem/packed_kv.h", "b16-core-helpers", _PACKED_KV),
    FilePatch(path="core/config.cpp", language="none", edits=(
        _replace("b16-config-helper", '#include "rad_core.h"\n',
                 '#include "rad_core.h"\n#include "mem/packed_kv.h"\n'),
        _replace("b16-config-help", _CONFIG_HELP,
                 '''    { "--kv-cache-dtype",     "T",      "bf16 (default) | i8 | fp8 | bf16_fp8 | bf16_i8 -- "
                                        "i8: int8 K and V, one scale each per token and head.\\n"
                                        "fp8 (E4M3): GLM-5.3 latent; DeepSeek-V4.1 compressed latent and index keys.\\n"
                                        "packed caches require model support and stay resident" },
''', "packed caches require model support and stay resident"),
        _replace("b16-native-cli-yaml", _CONFIG_PARSE,
                 '''            if (std::strcmp(v, "bf16") && std::strcmp(v, "i8") && std::strcmp(v, "fp8") &&
                configured_b16_kv_layout(v) == RAD_KVQ_NONE) {
                RAD_ERR("--kv-cache-dtype takes bf16, i8, fp8, bf16_fp8 or bf16_i8, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->kv_cache_dtype = v;''', "re:(?:configured_b16_kv_layout|rad_kvq_from_name)\\(v\\) == RAD_KVQ_NONE"),
    )),
    FilePatch(path="core/engine_bringup.cpp", language="none", edits=(
        # guarded on its own line: rad0011 inserts its includes after vram_budget.h too
        _replace("b16-engine-helper", '#include "mem/vram_budget.h"\n',
                 '#include "mem/vram_budget.h"\n#include "mem/packed_kv.h"\n', '#include "mem/packed_kv.h"\n'),
        _replace("b16-build-ctx", _BUILD_DTYPE,
                 '    set_build_kv_format(cfg_.kv_cache_dtype.c_str(), &bc);'),
        _replace("b16-model-admission", _STREAM_PROBE, _PACKED_ADMISSION, "re:(?:b16_kv_model_supported|packed_kv_model_supported)\\(layout, arch_probe_\\)"),
    )),
    FilePatch(path="core/build/rad_builder.cpp", language="none", edits=(
        _replace("b16-builder-helper", '#include "kvstream/bringup.h"\n',
                 '#include "kvstream/bringup.h"\n#include "mem/packed_kv.h"\n'),
        _replace("b16-declare-group", _DECL, _DECL_PACKED, "re:(?:b16_kv_group_supported|packed_kv_group_supported)\\(g\\.decl\\)"),
        _replace("b16-layer-sizing", _LAYER_BYTES, _LAYER_BYTES_PACKED, "re:(?:b16_kv_block_bytes|packed_kv_block_bytes)\\(gi\\.decl, bs, 1\\)"),
        _replace("b16-planner-sizing", _PLAN_BYTES, _PLAN_BYTES_PACKED, "re:(?:b16_kv_block_bytes|packed_kv_block_bytes)\\(d, gi\\.block_size, n_bound\\)"),
    )),
    FilePatch(path="core/mem/kv.cpp", language="none", edits=(
        _replace("b16-kv-manager-helper", '#include "mem/kv.h"\n',
                 '#include "mem/kv.h"\n#include "mem/packed_kv.h"\n'),
        _replace("b16-preallocation-geometry", '        const int64_t elt = rad_dtype_bytes(g.decl.dtype, 1);',
                 '''        if (g.decl.kv_layout != RAD_KVQ_NONE && !b16_kv_group_supported(g.decl)) {
            RAD_ERR("KV group '%s': packed B16 needs a full H256 RAD_U8 cache", p.name.c_str());
            return RAD_E_UNSUPPORTED;
        }
        const int64_t elt = rad_dtype_bytes(g.decl.dtype, 1);''', "re:(?:b16_kv_group_supported|packed_kv_group_supported)\\(g\\.decl\\)"),
        _replace("b16-allocation-fallback", _FALLBACK, _FALLBACK_PACKED, "re:(?:b16_kv_block_bytes|packed_kv_block_bytes)\\(g\\.decl, p\\.block_size, p\\.n_layers\\)"),
    )),
    FilePatch(path="core/runtime/kvbind.cpp", language="none", edits=(
        _replace("b16-bind-row", '            b.shape[3] = rad_kv_row_elems(d.dtype, d.head_dim);',
                 '            b.shape[3] = rad_kv_group_row_elems(d.kv_layout, d.dtype, d.head_dim);'),
    )),
    FilePatch(path="core/kvstream/bringup.cpp", language="none", edits=(
        _replace("b16-stream-refusal", _STREAM, _STREAM_GUARD + _STREAM),
    )),
)
