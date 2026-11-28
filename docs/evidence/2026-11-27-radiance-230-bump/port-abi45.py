#!/usr/bin/env python3
"""One-time migration edits for the measured ABI45 contract; requires previous patch source."""
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--apply', action='store_true', required=True)
parser.parse_args()
p = Path(__file__).resolve().parents[3] / "engines/radiance/patches/rad0048_abi44_kv_contract/patch.py"
s = p.read_text()
assert 'Advance the local plugin ABI from 43 to 44' in s, 'requires original pre-port patch'
# Keep package/edit identifiers, change actual version/contract.
s = s.replace("CHECK(RAD_ABI_VERSION == 44u)", "CHECK(RAD_ABI_VERSION == 45u)")
s = s.replace("ABI_FIELD(RadBatch, ckpt_cu)", "ABI_FIELD(RadBatch, vl_mask)")
# Native anchors and resulting independent measured table.
a = s.index('    FilePatch(path="tests/abi_layout_test.cpp"')
b = s.index('    FilePatch(path="tests/bc_kv_layout_contract_test.c"', a)
s = s[:a] + r'''    FilePatch(path="tests/abi_layout_test.cpp", language="none", description="Check ABI45 layout", edits=(
        _replace("abi44-test-doc", "taken at RAD_ABI_VERSION 38 and RAD_FORMAT_VER 2",
                 "taken at RAD_ABI_VERSION 45 and RAD_FORMAT_VER 2",
                 "taken at RAD_ABI_VERSION 45 and RAD_FORMAT_VER 2"),
        _replace("abi44-test-version", "/* The layout at RAD_ABI_VERSION 44 / RAD_FORMAT_VER 2 (see the file header). */\n"
                 "constexpr uint32_t kTableAbi = 44, kTableFormat = 2;",
                 "/* The layout at RAD_ABI_VERSION 45 / RAD_FORMAT_VER 2 (see the file header). */\n"
                 "constexpr uint32_t kTableAbi = 45, kTableFormat = 2;", "constexpr uint32_t kTableAbi = 45,"),
        _replace("abi44-test-context-size", "CHECK_EQ(sizeof(RadBuildCtx), (size_t)224);",
                 "CHECK_EQ(sizeof(RadBuildCtx), (size_t)256);", "CHECK_EQ(sizeof(RadBuildCtx), (size_t)256);"),
        _replace("abi44-test-probe-size", "CHECK_EQ(sizeof(RadArchProbe), (size_t)128);",
                 "CHECK_EQ(sizeof(RadArchProbe), (size_t)136);", "CHECK_EQ(sizeof(RadArchProbe), (size_t)136);"),
        _replace("abi44-test-batch-size", "CHECK_EQ(sizeof(RadBatch), (size_t)504);",
                 "CHECK_EQ(sizeof(RadBatch), (size_t)512);", "CHECK_EQ(sizeof(RadBatch), (size_t)512);"),
    )),
    FilePatch(path="tests/abi_layout_table.h", language="none", description="Pin ABI45 layout extensions", edits=(
        _replace("abi44-table-doc", "at RAD_ABI_VERSION 43 and", "at RAD_ABI_VERSION 45 and",
                 "at RAD_ABI_VERSION 45 and"),
        _replace("abi44-table-probe-size", '{ "RadArchProbe", "", 0, 128 },',
                 '{ "RadArchProbe", "", 0, 136 },', '{ "RadArchProbe", "", 0, 136 },'),
        _replace("abi44-table-probe", '{ "RadArchProbe", "ckpt_snap_ok", 124, 4 },\n',
                 '{ "RadArchProbe", "ckpt_snap_ok", 124, 4 },\n'
                 '{ "RadArchProbe", "kv_layout_ok", 128, 4 },\n',
                 '{ "RadArchProbe", "kv_layout_ok", 128, 4 },'),
        _replace("abi44-table-batch-size", '{ "RadBatch", "", 0, 504 },',
                 '{ "RadBatch", "", 0, 512 },', '{ "RadBatch", "", 0, 512 },'),
        _replace("abi44-table-batch", '{ "RadBatch", "vl_mask", 496, 8 },\n',
                 '{ "RadBatch", "vl_mask", 496, 8 },\n'
                 '{ "RadBatch", "cu_seqlens_h", 504, 8 },\n',
                 '{ "RadBatch", "cu_seqlens_h", 504, 8 },'),
        _replace("abi44-table-context-size", '{ "RadBuildCtx", "", 0, 208 },',
                 '{ "RadBuildCtx", "", 0, 256 },', '{ "RadBuildCtx", "", 0, 256 },'),
        _replace("abi44-table-context", '{ "RadBuildCtx", "draft_head_off", 204, 4 },\n',
                 '{ "RadBuildCtx", "draft_head_off", 204, 4 },\n'
                 '{ "RadBuildCtx", "expert_split", 208, 8 },\n'
                 '{ "RadBuildCtx", "dspark_walk", 216, 4 },\n'
                 '{ "RadBuildCtx", "tp_ownership", 224, 8 },\n'
                 '{ "RadBuildCtx", "pipelined_prefill", 232, 4 },\n'
                 '{ "RadBuildCtx", "pipelined_prefill_min_rows", 240, 8 },\n'
                 '{ "RadBuildCtx", "kv_layout", 248, 4 },\n'
                 '{ "RadBuildCtx", "attn_prefill_fp8", 252, 4 },\n',
                 '{ "RadBuildCtx", "attn_prefill_fp8", 252, 4 },'),
        _replace("abi44-table-group-size", '{ "RadKVGroupDecl", "", 0, 56 },',
                 '{ "RadKVGroupDecl", "", 0, 64 },', '{ "RadKVGroupDecl", "", 0, 64 },'),
        _replace("abi44-table-group", '{ "RadKVGroupDecl", "conv_width", 48, 8 },\n',
                 '{ "RadKVGroupDecl", "conv_width", 48, 8 },\n'
                 '{ "RadKVGroupDecl", "kv_layout", 56, 4 },\n',
                 '{ "RadKVGroupDecl", "kv_layout", 56, 4 },'),
    )),
    FilePatch(path="abi/rad_abi.h", language="none", description="Advance composed ABI from native 44 to 45", edits=(
        _replace("abi44-version", "#define RAD_ABI_VERSION 44u\n", "#define RAD_ABI_VERSION 45u\n",
                 "#define RAD_ABI_VERSION 45u\n"),
    )),
''' + s[b:]
s = s.replace("an_abi43_plugin_is_refused_by_the_abi44_engine", "an_abi44_plugin_is_refused_by_the_abi45_engine")
s = s.replace('RADTEST_ABI43_PLUGIN', 'RADTEST_ABI44_PLUGIN')
s = s.replace('ABI version 43', 'ABI version 44').replace('this engine is version 44', 'this engine is version 45')
s = s.replace('an_abi44_plugin_without_layout_export_is_refused', 'an_abi45_plugin_without_layout_export_is_refused')
s = s.replace('ABI44 migration gates: a real ABI43 plugin and an ABI44 plugin', 'ABI45 migration gates: a real ABI44 plugin and an ABI45 plugin')
s = s.replace('retained ABI43 DSO', 'retained ABI44 DSO')
p.write_text(s)
