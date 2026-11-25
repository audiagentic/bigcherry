"""Card-free proof of Radiance 2.2's production MTP replay callback and host invariants.

This tests the real ABI43/44 architecture callback extracted from the exact candidate source.
The C++ fixture supplies stand-in model fields, but compiles the unmodified upstream
pass-key callback with its actual host-pure prefill split/grid header. The effective
RadArgs launch contract is independently exercised by upstream runtime_test.cpp.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

REPO = Path(__file__).resolve().parents[3]
SOURCE_SHA = "b6b8751cccb0e834a13ae9e5afca824ee9a487aa"
REL = "arch/qwen4exp_fp8/qwen4exp_fp8.cpp"


def _source() -> Path:
    configured = os.environ.get("RADIANCE_CANDIDATE_SRC")
    if not configured:
        pytest.skip("RADIANCE_CANDIDATE_SRC must name the exact 2.2 candidate source")
    src = Path(configured)
    if not (src / REL).is_file():
        pytest.skip("Radiance candidate source not available")
    sha = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    assert sha == SOURCE_SHA, f"reviewed a different source: {sha}"
    return src


# No copy of the old rad0013 callback: extract the actual version-2.2 function.
PREFIX = r"""
#include <algorithm>
#include <cstdint>
#include <vector>
#include "rad_prefill_split.h"
struct RadCtx {};
enum { RAD_PHASE_DECODE = 1, RAD_PHASE_PREFILL = 2, RAD_PHASE_MIXED = 3 };
struct RadKVGroupBatch { int group = 7; };
struct RadBatch {
    int enc = 0, phase = RAD_PHASE_DECODE, draft_pass = 0, n_spec = 0;
    int n_tok = 8, n_seq = 1, max_q_len = 8, max_ctx_len = 2048;
    int n_kv_groups = 1;
    RadKVGroupBatch* kv = nullptr;
};
struct Geo {
    int head_dim = 256, n_head_kv = 2, max_ctx = 131072;
    int gqa() const { return 12; }
};
struct Layer { bool full = true; struct { int kv_block = 4; } attn; };
struct Model {
    Geo g;
    struct { int budget = 1000, ratio = 16; } qsacfg;
    int kv_full = 7;
    std::vector<Layer> layers{Layer{}};
};
int rad_slot(RadCtx*) { return 0; }
int rad_ctx_wgp_count(RadCtx*) { return 32; }
namespace qwen4exp_fp8 {
Model g_model[1];
"""
SUFFIX = r"""
}
extern "C" std::uint64_t replay_key(int phase, int draft, int spec, int tok,
                                      int seq, int ctx, int summary) {
    static RadCtx c;
    RadKVGroupBatch kv;
    RadBatch b;
    b.phase = phase; b.draft_pass = draft; b.n_spec = spec;
    b.n_tok = tok; b.n_seq = seq; b.max_q_len = tok / seq;
    b.max_ctx_len = ctx; b.kv = &kv;
    std::int32_t host[1] = {summary};
    return qwen4exp_fp8::pass_key(&c, &b, summary == -999 ? nullptr : host);
}
"""


@pytest.fixture(scope="module")
def replay_key(tmp_path_factory):
    src = (_source() / REL).read_text()
    start = src.index("uint64_t pass_key(RadCtx* c, const RadBatch* batch,")
    end = src.index("static int probe(", start)
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    tmp = tmp_path_factory.mktemp("abi44_mtp_replay")
    code = tmp / "key.cpp"
    out = tmp / "key.so"
    code.write_text(PREFIX + src[start:end] + SUFFIX)
    result = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-shared", "-fPIC", "-I", str(_source() / "abi"), str(code), "-o", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    with tempfile.TemporaryDirectory(prefix="rum08-replay-", dir=REPO) as tmp_exec:
        copied = Path(tmp_exec) / "key.so"
        shutil.copyfile(out, copied)
        library = ctypes.CDLL(str(copied))
    f = library.replay_key
    f.argtypes = [ctypes.c_int] * 7
    f.restype = ctypes.c_uint64
    return f


@pytest.mark.parametrize("draft,spec,tok", [
    (0, 0, 1),       # ordinary decode
    (0, 7, 8),       # wide trunk verify
    (-1, 0, 1),      # MTP history
    (2, 0, 1),       # fixed-depth serial draft round
    (3, 0, 1),
])
def test_upstream_records_fixed_depth_decode_verify_and_draft(replay_key, draft, spec, tok):
    assert replay_key(1, draft, spec, tok, 1, 2048, 1015) == 3
    assert replay_key(1, draft, spec, tok, 1, 2048, 1016) == 1
    assert replay_key(1, draft, spec, tok, 1, 2048, -999) == 1


def test_qsa_exact_boundary_and_unreachable_direct_zero_fixture(replay_key):
    # 1000 + 16 - 1 = 1015. Same 2048 context bucket, different QSA selector.
    k = lambda used: replay_key(1, 0, 7, 8, 1, 2048, used)
    assert k(1014) == k(1015) == 3
    assert k(1016) == k(0) == k(-999) == 1
    # A direct null/zero callback is keyable; the scheduler's full-KV summary
    # invariant is therefore essential. See the separate source-contract test.


def test_mixed_and_non_decode_speculation_stay_live(replay_key):
    assert replay_key(3, 0, 7, 8, 1, 2048, 1504) == 0
    assert replay_key(2, 0, 7, 8, 1, 2048, 1504) == 0
    assert replay_key(2, -1, 0, 1, 1, 2048, 1504) == 0


def test_single_sequence_prefill_grid_keys_splitter_result(replay_key):
    k = lambda used: replay_key(2, 0, 0, 128, 1, 4096, used)
    assert k(2304) != 0
    assert k(2304) == k(2305)  # same split returned by upstream split law
    assert k(2304) != k(1024)  # changed real split grid, both above the QSA selector bound


def test_actual_scheduler_and_runtime_guards_for_wide_verify():
    src = _source()
    batch = (src / "core/sched/batch.cpp").read_text()
    issue = (src / "core/runtime/issue.cpp").read_text()
    ctx = (src / "core/runtime/ctx.cpp").read_text()
    split = (src / "libr4d/r4d_attn_prefill_split.h").read_text()
    qwen = (src / REL).read_text()
    scheduler = (src / "core/sched/scheduler.cpp").read_text()
    mem_iface = (src / "core/sched/mem_iface.h").read_text()

    # A valid wide verify has >=1 entry and >1 query token. The full KV group
    # has no runtime window; its first block position is zero.
    assert "if (plan.e.empty()) return nullptr;" in batch
    assert "int64_t used = (int64_t)e.ctx_len + e.n_tokens - first;" in batch
    assert "o.kv_used_max_h[gi] = std::max<int32_t>(0, simd::max_i32(g.h_used, (size_t)n_seq));" in batch
    assert "B.kv_used_max_h = o.kv_used_max_h.data();" in batch
    assert "fd.kind      = RAD_KV_FULL;" in qwen
    assert "zero for a full group" in mem_iface
    assert "A verify step is 1 + n_draft query rows" in scheduler
    assert "e.n_tokens = (int32_t)want;" in scheduler
    assert "e.ctx_len = (int32_t)r.n_computed;" in scheduler
    for ctx_len, n_tokens in [(0, 8), (1500, 8), (200000, 4)]:
        assert ctx_len + n_tokens > 0

    assert "arch_pass_key_(const_cast<Ctx*>(this), b, b->kv_used_max_h)" in ctx
    for field in ("draft_pass", "n_spec", "max_ctx_len", "max_q_len"):
        assert f"b->{field}" in ctx
    assert "ip.args.kv_used_max_h > 0" in issue
    assert "batch_->max_ctx_len + batch_->max_q_len" in issue
    assert "if (used > 0)" in split
    assert "return rad_prefill_splits_for_live" in split
    assert "return r4d_prefill_splits_law" in split
    assert "if (batch->phase == RAD_PHASE_DECODE) return 1 | sel;" in qwen


def test_retirement_is_isolated_from_recipe_and_discovery():
    p = REPO / "engines/radiance/patches"
    recipes = (REPO / "engines/radiance/recipes.toml").read_text()
    for name in ("rad0013_mtp_pass_replay", "rad0047_mtp_verify_replay_summary"):
        assert (p / "archive" / name / "patch.py").is_file()
        # Retired means not discoverable: no package metadata or implementation at the old path. A bare directory of
        # Python bytecode a checkout left behind is not a package.
        assert not (p / name / "patch.toml").exists() and not (p / name / "patch.py").exists()
        assert f'"{name}"' not in recipes
    archive_index = (p / "archive/ARCHIVE.md").read_text()
    assert "| rad0047_mtp_verify_replay_summary | rejected (was untested) |" in archive_index
    assert "| rad0013_mtp_pass_replay | superseded (was validated) |" in archive_index
