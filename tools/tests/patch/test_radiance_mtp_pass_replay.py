"""Card-free, compiled pass-key contract and managed patch mechanics for the MTP replay guard."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))

from bigcherry.patch.apply import apply_all

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "engines/radiance/patches/rad0013_mtp_pass_replay/patch.py"
# The pinned upstream bare mirror under the work root; absent (CI) -> the minimal fixture below.
MIRROR = Path(os.environ.get("BC_WORK_ROOT", "")) / "upstream/radiance.git" if os.environ.get("BC_WORK_ROOT") else None
UPSTREAM = "arch/qwen4exp_fp8/qwen4exp_fp8.cpp"


def load():
    spec = importlib.util.spec_from_file_location("rad0013_mtp_pass_replay", PACKAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_fixture(mod):
    # A minimal upstream fixture exercises the same unique, exact old guard even
    # on hosts that have not yet populated the pinned upstream bare mirror.
    return (
        "namespace qwen4exp_fp8 {\n"
        "uint64_t pass_key(RadCtx* c, const RadBatch* batch, const int32_t* used_max_h) {\n"
        "    if (!c || !batch) return 0;\n"
        "    if (batch->enc) return 1;\n"
        + mod.OLD + "\n"
        "    return 1;\n"
        "}\n"
        "} // namespace qwen4exp_fp8\n"
        'extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* b, const int32_t* u) {\n'
        "    return qwen4exp_fp8::pass_key(c, b, u);\n"
        "}\n"
    )


def pinned_source():
    if MIRROR is None or not MIRROR.is_dir():
        return None
    proc = subprocess.run(
        ["git", f"--git-dir={MIRROR}", "show", f"v1.17.0:{UPSTREAM}"],
        check=False, capture_output=True, text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def root_with_source(tmp_path, source):
    f = tmp_path / UPSTREAM
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(source)
    return f


def test_exact_pinned_v117_guard_and_idempotence(tmp_path):
    src = pinned_source()
    if src is None:
        pytest.skip("v1.17.0 upstream mirror unavailable on this runner")
    p = root_with_source(tmp_path, src)
    patch = load().PATCH
    results = apply_all([patch], tmp_path)
    assert len(results) == 1 and results[0].ok, results
    assert [e.status for e in results[0].results] == ["applied"]
    after = p.read_bytes()
    assert b"BC_MTP_DECODE_REPLAY_GEOMETRY" in after
    results = apply_all([patch], tmp_path)
    assert results[0].ok
    assert [e.status for e in results[0].results] == ["already-applied"]
    assert p.read_bytes() == after


def test_synthetic_mechanics_and_guard(tmp_path):
    mod = load()
    p = root_with_source(tmp_path, old_fixture(mod))
    results = apply_all([mod.PATCH], tmp_path)
    assert results[0].ok, results
    after = p.read_bytes()
    assert b"BC_MTP_DECODE_REPLAY_GEOMETRY" in after
    results = apply_all([mod.PATCH], tmp_path)
    assert results[0].ok
    assert [e.status for e in results[0].results] == ["already-applied"]
    assert after == p.read_bytes()


def test_drift_refused_without_mutation(tmp_path):
    mod = load()
    source = old_fixture(mod).replace(
        "if (batch->draft_pass != 0) return 0;",
        "if (batch->draft_pass != 0 || batch->n_spec) return 0;",
    )
    p = root_with_source(tmp_path, source)
    before = p.read_bytes()
    results = apply_all([mod.PATCH], tmp_path)
    assert any(not r.ok for r in results), results
    assert p.read_bytes() == before


HARNESS_PREFIX = r"""
#include <algorithm>
#include <cstdint>
#include <vector>
using std::int32_t;
using std::int64_t;
using std::uint64_t;
struct RadCtx {};
enum { RAD_PHASE_DECODE=1, RAD_PHASE_PREFILL=2, RAD_PHASE_MIXED=3 };
struct RadKVStream { int64_t block_bytes=0, slots=0, host_layer_stride=0; };
struct RadKVGroupBatch {
    int group=7;
    int64_t block_size=4, block_table_pitch=1024;
    RadKVStream* stream=nullptr;
};
struct RadBatch {
    int enc=0, phase=RAD_PHASE_DECODE, draft_pass=0, n_spec=0;
    int64_t n_tok=8, n_seq=2, n_seq_decode=2, n_tok_decode=8;
    int max_q_len=4, max_q_len_decode=4, max_q_len_prefill=0;
    int max_ctx_len=20000, n_kv_groups=1;
    RadKVGroupBatch* kv=nullptr;
};
struct Geom {
    int head_dim=256, n_head_kv=2, max_ctx=131072, max_tok=2048, ratio=12;
    int gqa() const { return ratio; }
};
struct AttnGatedFP8 {
    int kv_block=4;
    int qsa_sel=1, qsa_sequ=1, op_attn=1, op_attn_gq=1;
    int64_t qsa_exact_to=3000, qsa_topk=512, max_rows=256, max_tok=2048;
    int64_t qsa_rows() const { return max_rows; }
    int64_t qsa_gq_rows() const { return std::max(max_tok, max_rows); }
};
struct Layer { bool full=true; AttnGatedFP8 attn{}; };
struct Model {
    Geom g{};
    std::vector<Layer> layers{Layer{}};
    int kv_full=7; bool have_mtp=true;
    struct { AttnGatedFP8 attn{}; } mtp;
};
Model g_model[1];
int rad_rank(RadCtx*) { return 0; }
int rad_ctx_wgp_count(RadCtx*) { return 32; }
int rad_prefill_splits_for_live(int, int, int, int, int, int, int, int, int) { return 1; }
uint64_t rad_prefill_grid_key(int, int, int, int, int, int) { return 9; }
"""


def callback_from_source(source):
    start = source.index("uint64_t pass_key(RadCtx* c, const RadBatch* batch,")
    if "static int probe(" in source[start:]:
        stop = source.index("static int probe(", start)
        body = source[start:stop]
    else:
        stop = source.index("} // namespace qwen4exp_fp8", start)
        body = source[start:stop]
    return (
        HARNESS_PREFIX + "\nnamespace qwen4exp_fp8 {\n" + body +
        "\n}\n"
        'extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* b, const int32_t* u) {\n'
        "    return qwen4exp_fp8::pass_key(c, b, u);\n"
        "}\n"
    )


HARNESS_SUFFIX = r"""
extern "C" uint64_t measure(int draft, int spec, int phase, int qlen, int nseq,
                              int ntok, int ctx, int pitch, int ratio,
                              int topk, int stage, int have_mtp,
                              int full, int block, int group) {
    static RadCtx c;
    static RadKVStream stream;
    static RadKVGroupBatch kv;
    RadBatch b;
    auto& m = g_model[0];
    m.g.ratio = ratio; m.g.n_head_kv = ratio == 6 ? 1 : 2;
    m.have_mtp = have_mtp;
    m.layers[0].full = full;
    m.layers[0].attn.qsa_topk = topk;
    m.mtp.attn.qsa_topk = topk;
    m.layers[0].attn.kv_block = block;
    m.mtp.attn.kv_block = block;
    kv.group = group; kv.block_size = block; kv.block_table_pitch = pitch;
    stream.block_bytes = 128; stream.slots = 2; stream.host_layer_stride = 65536;
    kv.stream = stage ? &stream : nullptr;
    b.kv = &kv; b.draft_pass = draft; b.n_spec = spec; b.phase = phase;
    b.max_q_len = qlen; b.max_q_len_decode = qlen;
    b.n_seq = nseq; b.n_tok = ntok; b.max_ctx_len = ctx;
    b.n_seq_decode = 0; b.n_tok_decode = 0; // serial draft uses zero split counters
    int32_t used[1] = {ctx};
    return rad_arch_pass_key(&c, &b, used);
}
"""


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    import ctypes

    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("no C++ compiler")
    mod = load()
    tmp = tmp_path_factory.mktemp("rad_mtp_pass_key")
    source = pinned_source() or old_fixture(mod)
    p = root_with_source(tmp, source)
    results = apply_all([mod.PATCH], tmp)
    assert all(r.ok for r in results), results
    harness = callback_from_source(p.read_text()) + HARNESS_SUFFIX
    cpp = tmp / "harness.cpp"
    cpp.write_text(harness)
    lib = tmp / "libmtpkey.so"
    result = subprocess.run(
        [cxx, "-std=c++17", "-O0", "-shared", "-fPIC", str(cpp), "-o", str(lib)],
        check=False, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    loaded = ctypes.CDLL(str(lib))
    loaded.measure.argtypes = [ctypes.c_int] * 15
    loaded.measure.restype = ctypes.c_uint64
    return loaded


DEFAULT = (0, 3, 1, 4, 2, 8, 20000, 1024, 12, 512, 0, 1, 1, 4, 7)


def key(lib, **changes):
    args = dict(zip(
        ("draft", "spec", "phase", "qlen", "nseq", "ntok", "ctx",
         "pitch", "ratio", "topk", "stage", "have_mtp", "full", "block", "group"),
        DEFAULT,
    ))
    args.update(changes)
    return lib.measure(*args.values())


def test_real_arch_export_keys_decode_verify_and_draft(library):
    assert key(library) != 0   # 2 x 4-token verify, gqa12
    assert key(library, draft=-1, qlen=1, ntok=2) != 0   # history/round 1
    assert key(library, draft=2, qlen=1, ntok=2) != 0    # later draft round
    assert key(library, draft=3, qlen=1, ntok=2, ratio=6) != 0
    assert key(library, qlen=7, ntok=14, ratio=6) != 0  # gqa6 width 10


@pytest.mark.parametrize("changes", [
    {"qlen": 6, "ntok": 12},  # gqa12 decode width <=5; prefill split if wider
    {"qlen": 11, "ntok": 22, "ratio": 6},
    {"phase": 3},
    {"phase": 2},  # speculative prefill
    {"draft": 2, "phase": 2},
    {"draft": 1},
    {"draft": 2, "have_mtp": 0},
    {"full": 0},
    {"block": 16},
    {"group": 99},
    {"stage": 1, "ctx": 1000},  # ring staging choice below QSA exact bound
    {"ntok": 11, "qlen": 5},    # impossible max band for two sequences
    {"nseq": 0},
    {"qlen": 0},
])
def test_unprovable_geometries_stay_live(library, changes):
    assert key(library, **changes) == 0


@pytest.mark.parametrize("changes", [
    {"draft": -1},
    {"draft": 2},
    {"spec": 4},
    {"ntok": 6, "qlen": 3},
    {"nseq": 1, "ntok": 4},
    {"ctx": 24000},
    {"pitch": 18000},  # decode verify split law changes at 65,536 depth
    {"topk": 640},     # different QSA table pitch/split geometry
    {"ratio": 6},
])
def test_geometry_inputs_change_arch_key(library, changes):
    assert key(library, **changes) != 0
    assert key(library, **changes) != key(library)


def test_prefill_live_split_not_recaptured(library):
    assert key(library, draft=2, spec=3, phase=2, qlen=200, ntok=400) == 0
    assert key(library, spec=3, phase=3, qlen=2, ntok=4) == 0
