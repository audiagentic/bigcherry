"""rad0016 (VTP14 N1): per-rank kernel code eligibility and an explicit ordered rank -> card list.

Radiance v1.17.0 keeps one architecture per process: usable_devices drops every card whose arch differs
from the first, and the registry checks each kernel library's code against that first card only
(Registry::set_code_device(devs[0])). A tensor-parallel world of two RX 7900 XTX (gfx1100, libr3) and one
R9700 (gfx1201, libr4d) needs each rank to select rows its own card runs.

This patch, inert unless `--rank-devices` is given:
- registry: `set_code_devices(ordered ordinals)` before loading; each device row keeps a per-rank mask
  (`KernelRow::code_ranks`) read from the library's registered code objects per card (dev_code_runs), fixed
  at load and never mutated by rank threads; a library no rank runs loses its device rows as upstream.
- selection: `select`/`resolve`/`build_bands` take the rank (from RadBuildCtx.rank through the Builder); a
  device row the rank's card cannot run is refused by name and the next library answers. A question asked
  for no rank gets only rows every rank's card runs.
- engine: `--rank-devices BDF,BDF,...` assigns rank i to the i-th card (stable PCI address, unique, discrete,
  exactly --tp); only this list admits cards of different architectures. After declare: the reference-kernel
  gate on every rank (upstream checks rank 0 only), the collectives every rank issues must be the same
  kernels in the same order, and a per-rank receipt (card, arch, libraries, collectives) is logged.
Without `--rank-devices` the engine, registry and selection behave exactly as upstream.

Source-pinned to radiance v1.17.0 (19d4e8fe); every anchor is exact upstream text.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from bigcherry.patcher import Edit, FilePatch

SRC = Path(__file__).resolve().parent / "src"

# The build identity hashes this file's bytes only (patch_implementation_digest), so the src/ files it
# creates are pinned here: a src/ change must update these digests, which changes this file and the build.
SRC_SHA256 = {
    "bc_rank_devices.cpp": "bf47feaef6d24638809e4806a4c78dfac702012ecc33057ceadcef27283d1715",
    "bc_rank_devices.h": "af3c44a839db1d135f26d3de05eb66c7b9482032eb625946cd27cf07499c41a5",
    "bc_rank_devices_test.cpp": "9de7af026b77eb625da3a74d62f5589d3792ef5dd6d1b8c17c5706af5bae1b83",
}


def _src_text(name: str) -> str:
    data = (SRC / name).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if SRC_SHA256.get(name) != got:
        raise RuntimeError(f"{name}: sha256 {got} is not the pinned {SRC_SHA256.get(name)}; update SRC_SHA256")
    return data.decode("utf-8")


def _edit(id: str, before: str, after: str, *, mode: str = "replace", guard: str, rationale: str = "",
          max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode=mode,
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale=rationale or "VTP14 N1: per-rank kernel code eligibility",
    )


def _created(path: str, source: str, guard: str) -> FilePatch:
    text = _src_text(source)
    return FilePatch(
        path=path,
        create=True,
        description="VTP14 N1 " + source,
        language="none",
        edits=(Edit(
            id="create-" + source.replace(".", "-").replace("_", "-"),
            anchor=r"\A",
            mode="insert_after",
            text=text,
            guard=re.escape(guard),
            max_span_lines=1,
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any collision.",
        ),),
    )


# ---------------------------------------------------------------- core/rad_core.h
RAD_CORE = FilePatch(path="core/rad_core.h", language="none", description="VTP14 N1 row mask and config", edits=(
    _edit(
        "n1-rowmask",
        "    /* rad_kernel_chain: its launches can wait in the kernel instead of behind the barrier bit. */\n"
        "    bool                 chain = false;\n"
        "};",
        "    /* rad_kernel_chain: its launches can wait in the kernel instead of behind the barrier bit. */\n"
        "    bool                 chain = false;\n"
        "    /* BC_VTP14_N1_ROW_RANKS: bit r is set when rank r's card runs this device row's code\n"
        "     * (Registry::set_code_devices). All ones on the one-device path and for host rows. Fixed at\n"
        "     * load, read-only while the rank threads select. */\n"
        "    uint32_t             code_ranks = ~0u;\n"
        "};",
        guard="BC_VTP14_N1_ROW_RANKS",
    ),
    _edit(
        "n1-config",
        "    int   p2p = RAD_P2P_AUTO;         /* --p2p auto|on|off */\n",
        "    int   p2p = RAD_P2P_AUTO;         /* --p2p auto|on|off */\n"
        "    /* BC_VTP14_N1_CONFIG: --rank-devices, rank i on the i-th card (canonical PCI addresses).\n"
        "     * Empty: radiance's own device choice, one architecture. */\n"
        "    std::vector<std::string> rank_devices;\n"
        "    /* --tp-wire-unvalidated: allow the lossy wire across --rank-devices ranks it is not validated on\n"
        "     * (lab diagnosis of the TP3 tiered-int8 fault only). */\n"
        "    bool tp_wire_unvalidated = false;\n",
        guard="BC_VTP14_N1_CONFIG",
    ),
))

# ---------------------------------------------------------------- core/plugin/registry.h
REGISTRY_H = FilePatch(path="core/plugin/registry.h", language="none", description="VTP14 N1 registry API", edits=(
    _edit(
        "n1-set-devices",
        "    void set_code_device(int device) { code_device_ = device; }",
        "    void set_code_device(int device) { code_devices_.clear(); code_device_ = device; }\n"
        "    /* BC_VTP14_N1_DEVICE_LIST: rank r runs on devices[r]. Each library's device code is checked\n"
        "     * against every rank's card at load, and a device row is offered only to ranks whose card runs\n"
        "     * it (KernelRow::code_ranks). RAD_E_INVAL for an empty or over-long list (32 ranks), a negative\n"
        "     * or repeated device, or once kernel rows exist. set_code_device returns to the one-device\n"
        "     * path. */\n"
        "    int set_code_devices(const std::vector<int>& devices);\n"
        "    const std::vector<int>& code_devices() const { return code_devices_; }\n"
        "    /* Tests: answers in place of the fat binaries a dlopen registered (dev_code_runs). 1 when the\n"
        "     * plugin's device code runs on the card, 0 with `why` when not. */\n"
        "    using CodeRunsProbe = std::function<int(const std::string& plugin, int device, std::string* why)>;\n"
        "    void set_code_runs_probe(CodeRunsProbe p) { code_probe_ = std::move(p); }\n"
        "    /* The code_ranks bits a device row needs to be offered to `rank`: 0 without a rank list (no\n"
        "     * filter), every rank's bit for rank < 0, and a bit no row has for a rank outside the list. */\n"
        "    uint32_t rank_bits(int rank) const;",
        guard="BC_VTP14_N1_DEVICE_LIST",
    ),
    _edit(
        "n1-select-rank",
        "    const KernelRow* select(std::string_view op, const Geometry& g, int domain,\n"
        "                            SelectionTrace* why = nullptr, bool record = true,\n"
        "                            const RowFilter* accept = nullptr);",
        "    const KernelRow* select(std::string_view op, const Geometry& g, int domain,\n"
        "                            SelectionTrace* why = nullptr, bool record = true,\n"
        "                            const RowFilter* accept = nullptr,\n"
        "                            int rank = -1); /* BC_VTP14_N1_SELECT_RANK */",
        guard="BC_VTP14_N1_SELECT_RANK",
    ),
    _edit(
        "n1-resolve-rank",
        "    Resolved resolve(std::string_view op, const Geometry& g, int domain, SelectionTrace* why = nullptr,\n"
        "                     const RowFilter* accept = nullptr);",
        "    Resolved resolve(std::string_view op, const Geometry& g, int domain, SelectionTrace* why = nullptr,\n"
        "                     const RowFilter* accept = nullptr,\n"
        "                     int rank = -1); /* BC_VTP14_N1_RESOLVE_RANK */",
        guard="BC_VTP14_N1_RESOLVE_RANK",
    ),
    _edit(
        "n1-bands-rank",
        "    std::vector<Band> build_bands(std::string_view op, std::string_view ranged_key,\n"
        "                                  int64_t lo, int64_t hi, const Geometry& base,\n"
        "                                  const RowFilter* accept = nullptr);",
        "    std::vector<Band> build_bands(std::string_view op, std::string_view ranged_key,\n"
        "                                  int64_t lo, int64_t hi, const Geometry& base,\n"
        "                                  const RowFilter* accept = nullptr,\n"
        "                                  int rank = -1); /* BC_VTP14_N1_BANDS_RANK */",
        guard="BC_VTP14_N1_BANDS_RANK",
    ),
    _edit(
        "n1-storage",
        "    int code_device_ = 0;\n"
        "    bool no_device_code_ = false;",
        "    int code_device_ = 0;\n"
        "    /* BC_VTP14_N1_STORAGE: set before loading, by one thread; read-only afterwards. */\n"
        "    std::vector<int> code_devices_;\n"
        "    CodeRunsProbe code_probe_;\n"
        "    bool no_device_code_ = false;",
        guard="BC_VTP14_N1_STORAGE",
    ),
))

# ---------------------------------------------------------------- core/plugin/loader.cpp
_LOADER_BLOCK = (
    "        const bool no_code_here =\n"
    "            n_device > 0 &&\n"
    "            (no_device_code_ ||\n"
    "             (s.code_to > s.code_from &&\n"
    "              rad::dev_code_runs(s.code_from, s.code_to, code_device_, &why) == 0));\n"
    "        if (no_code_here) {"
)

LOADER = FilePatch(path="core/plugin/loader.cpp", language="none", description="VTP14 N1 per-rank code masks", edits=(
    _edit(
        "n1-mask-scope",
        "    std::vector<const RadQuantizerInfo*> staged_quants;\n"
        "    int n_declared_schemas = 0;",
        "    std::vector<const RadQuantizerInfo*> staged_quants;\n"
        "    /* BC_VTP14_N1_MASK_SCOPE: which ranks' cards run this library's device code. */\n"
        "    uint32_t code_mask = ~0u;\n"
        "    int n_declared_schemas = 0;",
        guard="BC_VTP14_N1_MASK_SCOPE",
    ),
    _edit(
        "n1-set-devices-impl",
        "/* ------------------------------------------------------------------ staging */\n",
        "/* BC_VTP14_N1_DEVICE_IMPL: the rank list is fixed before any row exists, so the masks rows carry\n"
        " * never change while rank threads select. */\n"
        "int Registry::set_code_devices(const std::vector<int>& devices) {\n"
        "    if (!rows_.empty() || devices.empty() || devices.size() > 32) return RAD_E_INVAL;\n"
        "    for (size_t i = 0; i < devices.size(); ++i) {\n"
        "        if (devices[i] < 0) return RAD_E_INVAL;\n"
        "        for (size_t j = 0; j < i; ++j)\n"
        "            if (devices[j] == devices[i]) return RAD_E_INVAL;\n"
        "    }\n"
        "    code_devices_ = devices;\n"
        "    code_device_ = devices[0];\n"
        "    return RAD_OK;\n"
        "}\n"
        "\n"
        "uint32_t Registry::rank_bits(int rank) const {\n"
        "    if (code_devices_.empty()) return 0;\n"
        "    const size_t n = code_devices_.size();\n"
        "    const uint32_t all = n >= 32 ? ~0u : ((1u << n) - 1u);\n"
        "    if (rank < 0) return all;\n"
        "    if ((size_t)rank >= n) return ~0u;   /* more bits than any device row carries: no match */\n"
        "    return 1u << rank;\n"
        "}\n"
        "\n"
        "/* ------------------------------------------------------------------ staging */\n",
        guard="BC_VTP14_N1_DEVICE_IMPL",
    ),
    _edit(
        "n1-loader-ranks",
        _LOADER_BLOCK,
        "        /* BC_VTP14_N1_LOADER_RANKS: with a rank list, the library's code is checked against each\n"
        "         * rank's card. Its device rows stay when any rank runs them, carrying the mask; selection\n"
        "         * offers a row only to those ranks. A library no rank runs loses its device rows below,\n"
        "         * exactly as on the one-device path. */\n"
        "        if (!code_devices_.empty() && n_device > 0) {\n"
        "            code_mask = 0;\n"
        "            std::string skipped;\n"
        "            for (size_t i = 0; i < code_devices_.size(); ++i) {\n"
        "                std::string why_i;\n"
        "                int runs = 0;\n"
        "                if (no_device_code_) why_i = why;\n"
        "                else if (code_probe_) runs = code_probe_(p.name, code_devices_[i], &why_i);\n"
        "                else runs = rad::dev_code_runs(s.code_from, s.code_to, code_devices_[i], &why_i);\n"
        "                if (runs > 0) {\n"
        "                    code_mask |= 1u << i;\n"
        "                } else {\n"
        "                    skipped += fmt(\"%srank %zu (device %d: %s)\", skipped.empty() ? \"\" : \", \", i,\n"
        "                                   code_devices_[i], why_i.empty() ? \"no code object for it\" : why_i.c_str());\n"
        "                }\n"
        "            }\n"
        "            if (code_mask != 0 && !skipped.empty())\n"
        "                RAD_INFO(\"kernel plugin '%s': its %d device kernel(s) are offered only to the ranks whose \"\n"
        "                         \"card runs them; not to %s\", p.name.c_str(), n_device, skipped.c_str());\n"
        "            if (code_mask == 0) why = skipped;\n"
        "        }\n"
        "        const bool no_code_here =\n"
        "            n_device > 0 &&\n"
        "            (!code_devices_.empty()\n"
        "                 ? code_mask == 0\n"
        "                 : (no_device_code_ ||\n"
        "                    (s.code_to > s.code_from &&\n"
        "                     rad::dev_code_runs(s.code_from, s.code_to, code_device_, &why) == 0)));\n"
        "        if (no_code_here) {",
        guard="BC_VTP14_N1_LOADER_RANKS",
        max_span=8,
    ),
    _edit(
        "n1-row-mask",
        "        row.chain        = chainable && chainable(sr.second) != 0;\n"
        "        rows_.push_back(row);",
        "        row.chain        = chainable && chainable(sr.second) != 0;\n"
        "        /* BC_VTP14_N1_ROW_MASK: published once at load, read-only during selection. */\n"
        "        if (sr.first->domain == RAD_DOMAIN_DEVICE) row.code_ranks = code_mask;\n"
        "        rows_.push_back(row);",
        guard="BC_VTP14_N1_ROW_MASK",
    ),
))

# ---------------------------------------------------------------- core/plugin/select.cpp
SELECT = FilePatch(path="core/plugin/select.cpp", language="none", description="VTP14 N1 rank-aware selection", edits=(
    _edit(
        "n1-scan-decl",
        "                           const Registry::RowFilter* accept, std::string_view only,\n"
        "                           SelectionTrace& t, std::vector<std::string>* misses);",
        "                           const Registry::RowFilter* accept, std::string_view only,\n"
        "                           SelectionTrace& t, std::vector<std::string>* misses,\n"
        "                           uint32_t need = 0); /* BC_VTP14_N1_SCAN_DECL */",
        guard="BC_VTP14_N1_SCAN_DECL",
    ),
    _edit(
        "n1-select-impl",
        "const KernelRow* Registry::select(std::string_view op, const Geometry& g, int domain,\n"
        "                                  SelectionTrace* why, bool record, const RowFilter* accept) {",
        "const KernelRow* Registry::select(std::string_view op, const Geometry& g, int domain,\n"
        "                                  SelectionTrace* why, bool record, const RowFilter* accept,\n"
        "                                  int rank) { /* BC_VTP14_N1_SELECT_IMPL */",
        guard="BC_VTP14_N1_SELECT_IMPL",
    ),
    _edit(
        "n1-select-scan",
        "    const KernelRow* best = scan_rows(rows, g, domain, device_facts_, derived_keys, accept, {},\n"
        "                                      t, &misses);",
        "    const KernelRow* best = scan_rows(rows, g, domain, device_facts_, derived_keys, accept, {},\n"
        "                                      t, &misses, rank_bits(rank)); /* BC_VTP14_N1_SELECT_SCAN */",
        guard="BC_VTP14_N1_SELECT_SCAN",
    ),
    _edit(
        "n1-select-in-scan",
        "    const KernelRow* best = scan_rows(rows_for(op), g, domain, device_facts_, derived_keys,\n"
        "                                      accept, plugin, t, &misses);",
        "    const KernelRow* best = scan_rows(rows_for(op), g, domain, device_facts_, derived_keys,\n"
        "                                      accept, plugin, t, &misses,\n"
        "                                      rank_bits(-1)); /* BC_VTP14_N1_SELECT_IN_SCAN */",
        guard="BC_VTP14_N1_SELECT_IN_SCAN",
    ),
    _edit(
        "n1-scan-def",
        "                           const Registry::RowFilter* accept, std::string_view only,\n"
        "                           SelectionTrace& t, std::vector<std::string>* misses) {",
        "                           const Registry::RowFilter* accept, std::string_view only,\n"
        "                           SelectionTrace& t, std::vector<std::string>* misses,\n"
        "                           uint32_t need) { /* BC_VTP14_N1_SCAN_DEF */",
        guard="BC_VTP14_N1_SCAN_DEF",
    ),
    _edit(
        "n1-scan-guard",
        "        std::string whynot;\n"
        "        bool ok = row_matches(*r, g, facts, derived_keys, &whynot);",
        "        /* BC_VTP14_N1_SCAN_GUARD: a device row whose code the asking rank's card (or, for a question\n"
        "         * asked for no rank, every rank's card) does not run is refused by name, and the next\n"
        "         * library answers. Host rows carry every bit. */\n"
        "        if (need && (r->code_ranks & need) != need) {\n"
        "            const std::string whynot =\n"
        "                (need & (need - 1u)) == 0u\n"
        "                    ? fmt(\"plugin '%s' has no device code for rank %d's card\", r->plugin.c_str(),\n"
        "                          __builtin_ctz(need))\n"
        "                    : fmt(\"plugin '%s' has no device code for every rank's card\", r->plugin.c_str());\n"
        "            t.candidates.push_back({ r->plugin, r->info->name, r->info->priority, false, whynot });\n"
        "            if (std::find(misses->begin(), misses->end(), whynot) == misses->end())\n"
        "                misses->push_back(whynot);\n"
        "            continue;\n"
        "        }\n"
        "        std::string whynot;\n"
        "        bool ok = row_matches(*r, g, facts, derived_keys, &whynot);",
        guard="BC_VTP14_N1_SCAN_GUARD",
    ),
    _edit(
        "n1-resolve-impl",
        "Resolved Registry::resolve(std::string_view op, const Geometry& g, int domain, SelectionTrace* why,\n"
        "                           const RowFilter* accept) {",
        "Resolved Registry::resolve(std::string_view op, const Geometry& g, int domain, SelectionTrace* why,\n"
        "                           const RowFilter* accept, int rank) { /* BC_VTP14_N1_RESOLVE_IMPL */",
        guard="BC_VTP14_N1_RESOLVE_IMPL",
    ),
    _edit(
        "n1-resolve-forward",
        "    r.row = select(op, g, domain, &t, true, accept);",
        "    r.row = select(op, g, domain, &t, true, accept, rank); /* BC_VTP14_N1_RESOLVE_FORWARD */",
        guard="BC_VTP14_N1_RESOLVE_FORWARD",
    ),
))

# ---------------------------------------------------------------- core/plugin/bands.cpp
BANDS = FilePatch(path="core/plugin/bands.cpp", language="none", description="VTP14 N1 rank through bands", edits=(
    _edit(
        "n1-bands-impl",
        "std::vector<Band> Registry::build_bands(std::string_view op, std::string_view ranged_key,\n"
        "                                        int64_t lo, int64_t hi, const Geometry& base,\n"
        "                                        const RowFilter* accept) {",
        "std::vector<Band> Registry::build_bands(std::string_view op, std::string_view ranged_key,\n"
        "                                        int64_t lo, int64_t hi, const Geometry& base,\n"
        "                                        const RowFilter* accept, int rank) { /* BC_VTP14_N1_BANDS_IMPL */",
        guard="BC_VTP14_N1_BANDS_IMPL",
    ),
    _edit(
        "n1-bands-forward",
        "            b.dom[dom] = resolve(op, g, dom, &t, accept);",
        "            b.dom[dom] = resolve(op, g, dom, &t, accept, rank); /* BC_VTP14_N1_BANDS_FORWARD */",
        guard="BC_VTP14_N1_BANDS_FORWARD",
    ),
))

# ---------------------------------------------------------------- core/build
BUILD_H = FilePatch(path="core/build/rad_build.h", language="none", description="VTP14 N1 builder rank", edits=(
    _edit(
        "n1-builder-rank",
        "    Registry& reg_;\n"
        "    Program prog_;",
        "    Registry& reg_;\n"
        "    /* BC_VTP14_N1_BUILDER_RANK: this declare's rank, from RadBuildCtx; selection is asked for it. */\n"
        "    int rank_ = -1;\n"
        "    Program prog_;",
        guard="BC_VTP14_N1_BUILDER_RANK",
    ),
))

BUILDER = FilePatch(path="core/build/rad_builder.cpp", language="none", description="VTP14 N1 builder rank", edits=(
    _edit(
        "n1-builder-ctor",
        "Builder::Builder(Registry& reg, const RadModelMeta& meta, const RadBuildCtx& ctx)\n"
        "    : reg_(reg) {",
        "Builder::Builder(Registry& reg, const RadModelMeta& meta, const RadBuildCtx& ctx)\n"
        "    : reg_(reg), rank_(ctx.rank) { /* BC_VTP14_N1_BUILDER_CTOR */",
        guard="BC_VTP14_N1_BUILDER_CTOR",
    ),
    _edit(
        "n1-builder-bands",
        "    oi.bands = reg_.build_bands(oi.op, oi.ranged_key, oi.range_lo, oi.range_hi, oi.base,\n"
        "                                oi.weights.empty() ? nullptr : &accept);",
        "    oi.bands = reg_.build_bands(oi.op, oi.ranged_key, oi.range_lo, oi.range_hi, oi.base,\n"
        "                                oi.weights.empty() ? nullptr : &accept,\n"
        "                                rank_); /* BC_VTP14_N1_BUILDER_BANDS */",
        guard="BC_VTP14_N1_BUILDER_BANDS",
    ),
))

# ---------------------------------------------------------------- core/config.cpp
CONFIG = FilePatch(path="core/config.cpp", language="none", description="VTP14 N1 --rank-devices", edits=(
    _edit(
        "n1-config-help",
        "    { \"--p2p\",                \"MODE\",   \"how the collectives reach the other cards: auto (default)\\n\"",
        "    { \"--rank-devices\",       \"BDFS\",   \"BigCherry VTP14: rank i on the i-th card, by PCI address\\n\"\n"
        "                                        \"(0000:03:00.0,...), exactly --tp of them; the only way\\n\"\n"
        "                                        \"to serve on cards of different architectures\" }, /* BC_VTP14_N1_HELP */\n"
        "    { \"--tp-wire-unvalidated\", nullptr, \"BigCherry VTP14, lab only: allow the lossy --tp-wire across\\n\"\n"
        "                                        \"--rank-devices ranks it is not validated on (fault diagnosis)\" },\n"
        "    { \"--p2p\",                \"MODE\",   \"how the collectives reach the other cards: auto (default)\\n\"",
        guard="BC_VTP14_N1_HELP",
        max_span=1,
    ),
    _edit(
        "n1-config-parse",
        "        else if (S(\"--p2p\")) {\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;",
        "        else if (S(\"--tp-wire-unvalidated\")) c->tp_wire_unvalidated = true; /* BC_VTP14_N1_PARSE */\n"
        "        else if (S(\"--rank-devices\")) {\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;\n"
        "            std::string why;\n"
        "            if (!bc_parse_bdf_list(v, &c->rank_devices, &why)) {\n"
        "                RAD_ERR(\"--rank-devices: %s\", why.c_str());\n"
        "                return RAD_E_INVAL;\n"
        "            }\n"
        "        }\n"
        "        else if (S(\"--p2p\")) {\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;",
        guard="BC_VTP14_N1_PARSE",
    ),
    _edit(
        "n1-config-explain",
        "    { \"--p2p\",                [](const Config& c) {",
        "    { \"--tp-wire-unvalidated\", [](const Config& c) { /* BC_VTP14_N1_EXPLAIN */\n"
        "          return std::string(c.tp_wire_unvalidated ? \"on (lab diagnosis)\" : \"off\"); } },\n"
        "    { \"--rank-devices\",       [](const Config& c) {\n"
        "          std::string s;\n"
        "          for (const std::string& d : c.rank_devices) s += (s.empty() ? \"\" : \",\") + d;\n"
        "          return s.empty() ? std::string(\"radiance's own choice, one architecture\") : s; } },\n"
        "    { \"--p2p\",                [](const Config& c) {",
        guard="BC_VTP14_N1_EXPLAIN",
        max_span=1,
    ),
    _edit(
        "n1-config-include",
        "#include \"place/planner.h\"\n",
        "#include \"place/planner.h\"\n"
        "#include \"plugin/bc_rank_devices.h\" /* BC_VTP14_N1_CONFIG_INCLUDE */\n",
        guard="BC_VTP14_N1_CONFIG_INCLUDE",
        max_span=2,
    ),
))

# ---------------------------------------------------------------- core/engine_bringup.cpp
ENGINE = FilePatch(path="core/engine_bringup.cpp", language="none", description="VTP14 N1 engine start", edits=(
    _edit(
        "n1-engine-include",
        "#include \"build/startup.h\"\n",
        "#include \"build/startup.h\"\n"
        "#include \"plugin/bc_rank_devices.h\" /* BC_VTP14_N1_ENGINE_INCLUDE */\n",
        guard="BC_VTP14_N1_ENGINE_INCLUDE",
        max_span=2,
    ),
    _edit(
        "n1-engine-devices",
        "    if (s < 0) { RAD_ERR(\"no usable device\"); return s; }\n",
        "    if (s < 0) { RAD_ERR(\"no usable device\"); return s; }\n"
        "    /* BC_VTP14_N1_ENGINE_DEVICES: an explicit ordered list replaces the one-architecture choice,\n"
        "     * before anything is read or loaded. */\n"
        "    if (!cfg_.rank_devices.empty()) {\n"
        "        /* The lossy wire (tiered int8 at three ranks) faulted in a mixed-library TP3 world with an invalid\n"
        "         * KV operand; the cause is unproven (VTP14.md), so it is refused here until it is instrumented and\n"
        "         * validated. The exact wire is the validated contract; it is never substituted silently. */\n"
        "        if (!cfg_.tp_wire_exact && !cfg_.tp_wire_unvalidated) {\n"
        "            RAD_ERR(\"--rank-devices: --tp-wire %s is not validated across these ranks (a 3-rank tiered int8 run \"\n"
        "                    \"faulted with an invalid KV operand); serve with --tp-wire exact\", \"wht6\");\n"
        "            return RAD_E_UNSUPPORTED;\n"
        "        }\n"
        "        if (!cfg_.tp_wire_exact)   /* --tp-wire-unvalidated: a lab run reproducing the fault, never serving */\n"
        "            RAD_WARN(\"--rank-devices: --tp-wire wht6 across these ranks is UNVALIDATED (--tp-wire-unvalidated); \"\n"
        "                     \"a lab diagnosis only, not a serving configuration\");\n"
        "        std::vector<BcDeviceInfo> inv;\n"
        "        for (int i = 0, n = rad_dev_count(); i < n; ++i) {\n"
        "            RadDeviceProps p{};\n"
        "            if (rad_dev_props(i, &p) < 0) continue;\n"
        "            inv.push_back({ i, dev_pci_address(i), p.arch, p.name, dev_integrated(i) });\n"
        "        }\n"
        "        std::vector<int> picked;\n"
        "        std::string why;\n"
        "        if (bc_rank_devices_plan(inv, cfg_.rank_devices, cfg_.tp, &picked, &why) != RAD_OK) {\n"
        "            RAD_ERR(\"--rank-devices: %s\", why.c_str());\n"
        "            return RAD_E_DEVICE;\n"
        "        }\n"
        "        devs = picked;\n"
        "        const std::vector<std::string> archs = bc_rank_archs(inv, picked);\n"
        "        std::string ranks;\n"
        "        for (size_t r = 0; r < picked.size(); ++r)\n"
        "            for (const BcDeviceInfo& d : inv)\n"
        "                if (d.ordinal == picked[r])\n"
        "                    ranks += fmt(\"%srank %zu: %s %s (device %d)\", ranks.empty() ? \"\" : \"; \", r,\n"
        "                                 d.pci.c_str(), d.arch.c_str(), d.ordinal);\n"
        "        decide(\"--rank-devices\", ranks,\n"
        "               archs.size() > 1 ? \"an explicit ordered list across architectures: each rank selects the \"\n"
        "                                  \"kernels its own card runs (BigCherry VTP14)\"\n"
        "                                : \"an explicit ordered list\");\n"
        "        if (archs.size() > 1) {\n"
        "            std::string a;\n"
        "            for (const std::string& x : archs) a += (a.empty() ? \"\" : \", \") + x;\n"
        "            RAD_WARN(\"device: the ranks span architectures (%s). Each rank selects only kernels its own \"\n"
        "                     \"card runs, and startup stops if a rank lacks one or the ranks' collectives differ. \"\n"
        "                     \"Experimental (BigCherry VTP14).\", a.c_str());\n"
        "        }\n"
        "    }\n",
        guard="BC_VTP14_N1_ENGINE_DEVICES",
    ),
    _edit(
        "n1-engine-code-devices",
        "    rad_tools_registry().set_code_device(devs[0]);\n",
        "    if (!cfg_.rank_devices.empty()) { /* BC_VTP14_N1_ENGINE_CODE_DEVICES */\n"
        "        const std::vector<int> rank_devs(devs.begin(), devs.begin() + cfg_.tp);\n"
        "        if (rad_tools_registry().set_code_devices(rank_devs) != RAD_OK) {\n"
        "            RAD_ERR(\"--rank-devices: the kernel registry already holds rows; the rank list must be set \"\n"
        "                    \"before any kernel library loads\");\n"
        "            return RAD_E_STATE;\n"
        "        }\n"
        "    } else {\n"
        "        rad_tools_registry().set_code_device(devs[0]);\n"
        "    }\n",
        guard="BC_VTP14_N1_ENGINE_CODE_DEVICES",
    ),
    _edit(
        "n1-engine-gate",
        "    RAD_TRY(rad_gate_reference_kernels(ranks_[0]->program, cfg_.accept_reference_kernels, \"serve\"));\n",
        "    if (cfg_.rank_devices.empty()) {\n"
        "        RAD_TRY(rad_gate_reference_kernels(ranks_[0]->program, cfg_.accept_reference_kernels, \"serve\"));\n"
        "    } else { /* BC_VTP14_N1_ENGINE_GATE */\n"
        "        /* Ranks of different cards select different rows, so every rank's program is checked:\n"
        "         * reference kernels, the collectives every rank issues, and a receipt of what each runs. */\n"
        "        std::vector<std::vector<BcCollStep>> colls(ranks_.size());\n"
        "        for (const auto& rp : ranks_) {\n"
        "            const Rank& r = *rp;\n"
        "            LogRank lr(r.index);\n"
        "            RAD_TRY(rad_gate_reference_kernels(r.program, cfg_.accept_reference_kernels, \"serve\"));\n"
        "            std::map<std::string, int> by_lib;\n"
        "            for (const OpInfo& op : r.program.ops)\n"
        "                for (const Band& b : op.bands) {\n"
        "                    for (int dm = 0; dm < RAD_N_DOMAINS; ++dm)\n"
        "                        if (const KernelRow* row = b.dom[dm].row)\n"
        "                            ++by_lib[row->plugin + (dm == RAD_DOMAIN_HOST ? \" (host)\" : \"\")];\n"
        "                    const KernelRow* row = b.dom[RAD_DOMAIN_DEVICE].row;\n"
        "                    if (row && row->info && row->collective)\n"
        "                        colls[(size_t)r.index].push_back({ op.op, row->info->name, row->plugin,\n"
        "                                                           b.span + \": \" + b.dom[RAD_DOMAIN_DEVICE].geom.str() });\n"
        "                }\n"
        "            std::string libs;\n"
        "            for (const auto& [lib, n] : by_lib) libs += fmt(\"%s%s %d\", libs.empty() ? \"\" : \", \", lib.c_str(), n);\n"
        "            const std::vector<BcCollStep>& c = colls[(size_t)r.index];\n"
        "            RadDeviceProps dp{};\n"
        "            (void)rad_dev_props(r.device, &dp);\n"
        "            RAD_INFO(\"rank receipt: rank %d on device %d %s %s; op bands by library: %s; %zu collective \"\n"
        "                     \"band(s)%s%s\", r.index, r.device, dev_pci_address(r.device).c_str(), dp.arch,\n"
        "                     libs.empty() ? \"none\" : libs.c_str(), c.size(), c.empty() ? \"\" : \", first \",\n"
        "                     c.empty() ? \"\" : (c[0].kernel + \" (\" + c[0].plugin + \")\").c_str());\n"
        "        }\n"
        "        std::string why;\n"
        "        if (bc_rank_collectives_agree(colls, &why) != RAD_OK) {\n"
        "            RAD_ERR(\"--rank-devices: the ranks' collectives differ: %s\", why.c_str());\n"
        "            return RAD_E_STATE;\n"
        "        }\n"
        "    }\n",
        guard="BC_VTP14_N1_ENGINE_GATE",
    ),
))

# ---------------------------------------------------------------- tests/config_test.cpp
CONFIG_TEST = FilePatch(path="tests/config_test.cpp", language="none", description="VTP14 N1 option inventory", edits=(
    _edit(
        "n1-config-test-value",
        "        { \"--max-source-pixels\", { \"1000000\" } },\n",
        "        { \"--max-source-pixels\", { \"1000000\" } },\n"
        "        { \"--rank-devices\", { \"0000:03:00.0,0000:07:00.0\", \"--tp\", \"2\" } }, /* BC_VTP14_N1_CONFIG_TEST */\n",
        guard="BC_VTP14_N1_CONFIG_TEST",
        max_span=2,
    ),
))

# ---------------------------------------------------------------- core/device/aql.cpp
# HIP's own __hipRegisterFatBinary returns no handle for a fat binary without code for device 0, even when
# another visible card runs it (measured: every gfx1201-only libr4d binary returns null with the two gfx1100
# cards visible first). The AQL backend loads code objects itself from the bundle, per device, so the handle is
# only its lookup key: it issues its own for such a binary and never hands that one to HIP.
AQL = FilePatch(path="core/device/aql.cpp", language="none", description="VTP14 N1 code past device 0", edits=(
    _edit(
        "n1-aql-include",
        "#include <unordered_map>\n",
        "#include <unordered_map>\n"
        "#include <unordered_set> /* BC_VTP14_N1_AQL_INCLUDE */\n",
        guard="BC_VTP14_N1_AQL_INCLUDE",
        max_span=2,
    ),
    _edit(
        "n1-aql-own-handles",
        "    std::unordered_map<void**, Module*> by_handle;\n"
        "    std::unordered_map<const void*, Func*> by_stub;\n",
        "    std::unordered_map<void**, Module*> by_handle;\n"
        "    std::unordered_map<const void*, Func*> by_stub;\n"
        "    /* BC_VTP14_N1_AQL_OWN_HANDLES: handles this backend issued for fat binaries HIP returned none\n"
        "     * for; never passed to HIP. */\n"
        "    std::unordered_set<void**> own_handles;\n",
        guard="BC_VTP14_N1_AQL_OWN_HANDLES",
    ),
    _edit(
        "n1-aql-register-fat",
        "        if (!h) {\n"
        "            Module* m = new Module();\n"
        "            m->bundle = w->binary;\n"
        "            r.order.push_back(m);\n"
        "            return h;\n"
        "        }\n",
        "        if (!h) {\n"
        "            /* BC_VTP14_N1_AQL_REGISTER_FAT: HIP judges a fat binary by device 0 alone, so code for a\n"
        "             * later card of another architecture (an R9700 behind two RX 7900 XTX) gets no handle\n"
        "             * either. This backend loads code objects from the bundle per device, so it issues its own\n"
        "             * handle and records the functions; a card the bundle has no code for still fails\n"
        "             * a_code_runs and the plugin loader drops the rows. */\n"
        "            Module* m = new Module();\n"
        "            m->bundle = w->binary;\n"
        "            r.order.push_back(m);\n"
        "            void** own = new void*(nullptr);\n"
        "            r.own_handles.insert(own);\n"
        "            r.by_handle[own] = m;\n"
        "            return own;\n"
        "        }\n",
        guard="BC_VTP14_N1_AQL_REGISTER_FAT",
    ),
    _edit(
        "n1-aql-register-fn",
        "    if (real().reg_fn)\n"
        "        real().reg_fn(modules, host_fn, device_fn, device_name, thread_limit, tid, bid, block_dim,\n"
        "                      grid_dim, wsize);\n"
        "    Registry& r = reg();\n"
        "    std::lock_guard<std::mutex> lk(r.mu);\n",
        "    Registry& r = reg();\n"
        "    /* BC_VTP14_N1_AQL_REGISTER_FN: a handle this backend issued is unknown to HIP. */\n"
        "    bool own = false;\n"
        "    {\n"
        "        std::lock_guard<std::mutex> lk(r.mu);\n"
        "        own = r.own_handles.count(modules) != 0;\n"
        "    }\n"
        "    if (real().reg_fn && !own)\n"
        "        real().reg_fn(modules, host_fn, device_fn, device_name, thread_limit, tid, bid, block_dim,\n"
        "                      grid_dim, wsize);\n"
        "    std::lock_guard<std::mutex> lk(r.mu);\n",
        guard="BC_VTP14_N1_AQL_REGISTER_FN",
    ),
    _edit(
        "n1-aql-unregister",
        "    if (real().unreg_fat) real().unreg_fat(modules);\n",
        "    /* BC_VTP14_N1_AQL_UNREGISTER: a handle this backend issued is unknown to HIP. */\n"
        "    bool own = false;\n"
        "    {\n"
        "        Registry& r = reg();\n"
        "        std::lock_guard<std::mutex> lk(r.mu);\n"
        "        own = r.own_handles.count(modules) != 0;\n"
        "    }\n"
        "    if (real().unreg_fat && !own) real().unreg_fat(modules);\n",
        guard="BC_VTP14_N1_AQL_UNREGISTER",
        max_span=2,
    ),
))

PATCHES = (
    RAD_CORE,
    REGISTRY_H,
    LOADER,
    SELECT,
    BANDS,
    BUILD_H,
    BUILDER,
    CONFIG,
    ENGINE,
    CONFIG_TEST,
    AQL,
    _created("core/plugin/bc_rank_devices.h", "bc_rank_devices.h", "BC_VTP14_N1: an explicit, ordered rank"),
    _created("core/plugin/bc_rank_devices.cpp", "bc_rank_devices.cpp", "BC_VTP14_N1: see bc_rank_devices.h"),
    _created("tests/bc_rank_devices_test.cpp", "bc_rank_devices_test.cpp", "BC_VTP14_N1 (BigCherry managed patch rad0016)"),
)
