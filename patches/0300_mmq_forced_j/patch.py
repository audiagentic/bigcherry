"""HI06 - MMQ forced-J variant dispatch.

Upstream picks MMQ's tile width J by scanning the config table for the value
that minimises tile count, then switching on the result. The scan and the switch
are welded together inside ``mul_mat_q_switch_J``, so J is not something a
caller can express an opinion about.

This patch separates them and threads a forced value down as an explicit,
appended, defaulted parameter -- the same shape as HI07 and HI08, so all three
families work identically:

* ``mul_mat_q_launch_forced_J`` takes J as an argument. It is the switch,
  lifted out verbatim.
* ``mul_mat_q_compute_J_best`` is upstream's scan, lifted out unchanged, so the
  tuner can ask "what would native pick here?" without launching anything.
* ``mul_mat_q_switch_J`` becomes those two calls, with a forced value replacing
  the scan's answer when one is supplied.

The native path is untouched: with no forced value the scan's answer is used
exactly as before, and native and forced go through the same launcher. Forced
J = J_best is therefore the same code, not merely equivalent code.

MMQ is the least invasive of the three families because ``launch_mul_mat_q``
already takes J as a template parameter -- the switch was always there, it just
had no way to be told a value.

Note that valid J values are **not** ``range(8, 129, 8)``, despite the switch
covering all sixteen. The config table is sparse: ``ggml_cuda_mmq_get_config``
returns ``GGML_TYPE_COUNT`` for combinations it does not define, and upstream's
scan skips those. CDNA defines 154 rows where RDNA3 defines 260.
``ggml_cuda_mmq_variant_is_eligible`` applies the same test, so a candidate for
an undefined (type, J, fallback) is rejected before launch rather than aborting
inside it.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

# J values the switch covers. Pinned by the audit (source_audit.MMQ_J_VALUES);
# if upstream changes the set, the audit fails before this patch is applied.
_J_VALUES = list(range(8, 129, 8))

_FORCED_CASES = "\n".join(
    f"        case {j:3d}:\n"
    f"            launch_mul_mat_q<type, {j:3d}, fallback>(ctx, args, stream);\n"
    f"            break;"
    for j in _J_VALUES
)

_HELPERS = f"""
// bigcherry (HI06): the J switch, lifted out so a caller can supply J. Both the
// native path and measured dispatch go through here, which is what makes a
// forced J equal to J_best the same code as native selection, not merely
// equivalent code.
template <ggml_type type, bool fallback>
static void mul_mat_q_launch_forced_J(
        ggml_backend_cuda_context & ctx, const mmq_args & args,
        cudaStream_t stream, const int J) {{
    switch (J) {{
{_FORCED_CASES}
        default:
            GGML_ABORT("bigcherry: unsupported MMQ J");
            break;
    }}
}}

// bigcherry (HI24): which J would native's own scan choose for this shape?
//
// This is upstream's scan, moved out of the template so it can be asked at
// runtime. The tuner needs it to identify the forced candidate that is *the
// same kernel* as native: `mul_mat_q_switch_J` overwrites J_best with forced_J
// and calls one launcher, so forcing J == J_best is native rather than merely
// equivalent to it (RV21). That makes the pair a free noise canary -- any
// measured difference between them is measurement error, needing no external
// reference to calibrate.
//
// Returns 0 when nothing is eligible, which callers must read as "no such
// candidate" rather than "J = 0".
// Declared here and defined once in mmq.cu, the same shape as
// ggml_cuda_mmq_variant_is_eligible. It cannot be `inline` in this header: the
// tuner's translation unit includes only hip-autotune-dispatch.cuh, so an
// inline definition here leaves it with a declaration and no body, which links
// but fails at load with an undefined symbol.
int ggml_cuda_mmq_native_j_best(ggml_type type, bool fallback, int64_t ncols_max);

// bigcherry (HI06): upstream's J scan, as a pure function. The tuner needs the
// native answer both as a fallback and as the baseline a challenger has to
// beat, and needs it without launching anything.
//
// Delegates to the runtime query above so there is one implementation rather
// than two. A second copy of this scan would be exactly the restatement that
// has caused four defects in this project already.
template <ggml_type type, bool fallback>
static int mul_mat_q_compute_J_best(const mmq_args & args) {{
    return ggml_cuda_mmq_native_j_best(type, fallback, args.ncols_max);
}}

// bigcherry (HI06): hard eligibility (standards 12.4). Answers "could this J
// run at all", never "would it be fast". A (type, J, fallback) the config table
// does not define is rejected here, not discovered at launch.
static inline bool ggml_cuda_mmq_config_is_eligible(
        const ggml_type type, const int J, const bool fallback,
        const int cc, const size_t shared_mem_limit) {{
    if (J < 8 || J > 128 || J % 8 != 0) {{
        return false;
    }}
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    if (config.type == GGML_TYPE_COUNT) {{
        return false;
    }}
    return mmq_get_nbytes_shared(config, cc) <= shared_mem_limit;
}}

template <ggml_type type, bool fallback>
void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args,
                        cudaStream_t stream, int forced_J = 0) {{
    int J_best = mul_mat_q_compute_J_best<type, fallback>(args);

    // bigcherry (HI06): a forced J replaces the scan's answer. Zero means
    // "native policy", so with no forced value this is exactly upstream.
    if (forced_J != 0) {{
        J_best = forced_J;
    }}

    mul_mat_q_launch_forced_J<type, fallback>(ctx, args, stream, J_best);
}}"""

_MMQ_SOURCE = """
// bigcherry (HI06/HI71): hard eligibility (standards 12.4), as a non-template
// entry point the dispatch layer can call.
//
// HI71 (real Brutus hardware, 2026-08-18): dense-shape-aware. upstream's
// dense ggml_cuda_mul_mat_q() pads its quantized src1_q8_1 activation buffer
// with ggml_cuda_mmq_get_J_max(type, fallback, cc, ncols_max) -- a tail sized
// for upstream's OWN J-selection search over the REAL batch width. mmq.cuh's
// Y-tile staging load reads a full compile-time J-wide tile unconditionally,
// even for the grid's last (partial) tile (ntx = ceil(ncols_max / J) allows
// ncols_max < J by design), and that read is bounded only by the buffer's
// available padding, not by anything the eligibility check used to verify.
// A forced J outside the envelope that J_max's own search over ncols_max
// would have produced can therefore read past the buffer.
//
// Confirmed on real hardware: forcing J=112 (mmq:q8_0:j112:fb0) against a
// real ncols_max=3 dispatch signature (MTP speculative-decoding draft-batch
// verification) needs 109 padding columns; ggml_cuda_mmq_get_J_max(q8_0,
// fb0, gfx1201, 3) returns 0 -- a 109-column out-of-bounds read, and the
// exact crash this check now prevents.
//
// ncols_max must be the REAL number of destination columns this launch will
// process. A caller with no real signature to test (ggml_cuda_mmq_type_is_
// supported, below) must use ggml_cuda_mmq_config_is_eligible directly
// rather than passing a sentinel here -- a sentinel would just recreate the
// ambiguity this function used to have when it silently discarded ncols_max.
bool ggml_cuda_mmq_variant_is_eligible(
        ggml_type type, int j, bool fallback, int cc, size_t shared_mem_limit,
        int64_t ncols_max) {
    const int id = ggml_cuda_get_device();
    if (cc == 0) {
        cc = ggml_cuda_info().devices[id].cc;
    }
    if (shared_mem_limit == 0) {
        shared_mem_limit = ggml_cuda_info().devices[id].smpbo;
    }
    if (!ggml_cuda_mmq_config_is_eligible(type, j, fallback, cc,
                                          shared_mem_limit)) {
        return false;
    }
    GGML_ASSERT(ncols_max > 0 &&
        "ggml_cuda_mmq_variant_is_eligible requires a real batch width -- "
        "use ggml_cuda_mmq_config_is_eligible for a config-only check with "
        "no real dispatch signature");
    const int required_tail = (int) ((j - (ncols_max % j)) % j);
    const int available_tail = ggml_cuda_mmq_get_J_max(type, fallback, cc, ncols_max);
    return required_tail <= available_tail;
}

// bigcherry (HI24): which J would native's own scan choose for this shape?
//
// Upstream's scan, lifted out of `mul_mat_q_compute_J_best` so it can be asked
// at runtime; that template now delegates here, so there is one implementation
// rather than two. A second copy would be exactly the restatement that has
// caused four defects in this project already.
//
// The tuner needs this to identify the forced candidate that is *the same
// kernel* as native: `mul_mat_q_switch_J` overwrites J_best with forced_J and
// calls one launcher, so forcing J == J_best is native rather than merely
// equivalent to it (RV21). That makes the pair a free noise canary -- any
// difference between their medians is measurement error, calibrated without
// any external reference.
//
// Returns 0 when nothing is eligible, which callers must read as "no such
// candidate" rather than "J = 0".
int ggml_cuda_mmq_native_j_best(ggml_type type, bool fallback, int64_t ncols_max) {
    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }
        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }
        const int ntiles_x = (int) ((ncols_max + config.J - 1) / config.J);
        if (ntiles_x < ntiles_J_best) {
            J_best        = J;
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

// bigcherry (RV02): does MMQ have *any* compiled configuration for this type on
// this device?
//
// Derived by asking upstream's own per-architecture config tables rather than
// by restating the supported-type switch in ggml_cuda_should_use_mmq. Three
// reasons that is the better source: a restated list silently disagrees the
// moment upstream adds a type (the failure this whole check exists to prevent);
// the tables are per architecture, so a type present in the switch but absent
// from this GPU's table is correctly answered "no" here and wrongly "yes"
// there; and should_use_mmq itself cannot be reused wholesale because it folds
// in the performance heuristics the tuner exists to override.
//
// The J sweep mirrors ggml_cuda_mmq_config_is_eligible's own bounds. It is
// 32 table lookups of a constexpr function, run once per candidate per
// signature during tuning and never on a replay path.
//
// HI71: calls ggml_cuda_mmq_config_is_eligible directly, NOT
// ggml_cuda_mmq_variant_is_eligible -- there is no real dispatch signature
// here (this asks "does any config exist for this type at all", not "is
// this config safe for this batch"), and ggml_cuda_mmq_variant_is_eligible
// now requires a real ncols_max to answer that different, shape-aware
// question.
bool ggml_cuda_mmq_type_is_supported(ggml_type type, int cc, size_t shared_mem_limit) {
    const int id = ggml_cuda_get_device();
    if (cc == 0) {
        cc = ggml_cuda_info().devices[id].cc;
    }
    if (shared_mem_limit == 0) {
        shared_mem_limit = ggml_cuda_info().devices[id].smpbo;
    }
    for (int j = 8; j <= 128; j += 8) {
        if (ggml_cuda_mmq_config_is_eligible(type, j, /*fallback =*/ false, cc,
                                             shared_mem_limit)
         || ggml_cuda_mmq_config_is_eligible(type, j, /*fallback =*/ true, cc,
                                             shared_mem_limit)) {
            return true;
        }
    }
    return false;
}

// bigcherry (HI06): forced-variant entry point for measured dispatch.
// forced_j == 0 means "native policy", so the registry's native wrapper and
// every forced variant share one path.
void ggml_cuda_mul_mat_q_variant(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        int forced_j, int forced_fallback) {
    // `fallback` is decided by row divisibility inside mul_mat_q_case and is
    // not a free choice, so it is not forced here. A candidate whose fallback
    // disagrees with the shape is rejected by eligibility instead; overriding
    // it would silently run a different kernel than the candidate names.
    GGML_UNUSED(forced_fallback);

    ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst, forced_j);
}

"""

HEADER_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq.cuh",
    description="MMQ forced-J launcher, J_best as a pure function, eligibility",
    edits=(
        Edit(
            id="mmq-helpers-and-switch",
            # Replaces mul_mat_q_switch_J entirely -- template line, signature
            # and body -- with the lifted helpers plus the two-call version.
            #
            # Deliberately one edit rather than two. The helpers contain a copy
            # of upstream's scan, so an edit anchored on the scan text would
            # match whichever copy came first and splice in the wrong place.
            # Replacing the whole function keeps the anchor unambiguous.
            anchor=r"^template <ggml_type type, bool fallback>\n"
                   r"void mul_mat_q_switch_J\(ggml_backend_cuda_context & ctx, "
                   r"const mmq_args & args, cudaStream_t stream\) \{\n"
                   r"[\s\S]*?GGML_ABORT\([^)]*\);\n"
                   r"            break;\n    \}\n\}",
            rationale="the whole of mul_mat_q_switch_J: its scan and its "
                      "sixteen-case switch",
            mode="replace",
            text=_HELPERS,
            guard=r"mul_mat_q_launch_forced_J",
            max_span_lines=130,
        ),
        Edit(
            id="mmq-case-signature",
            anchor=r"^template <ggml_type type>\n"
                   r"void mul_mat_q_case\(ggml_backend_cuda_context & ctx, "
                   r"const mmq_args & args, cudaStream_t stream\) \{$",
            rationale="the definition of mul_mat_q_case",
            mode="replace",
            text="template <ggml_type type>\n"
                 "void mul_mat_q_case(ggml_backend_cuda_context & ctx, "
                 "const mmq_args & args, cudaStream_t stream,\n"
                 "                    int forced_J = 0) {",
            guard=r"const mmq_args & args, cudaStream_t stream,\n"
                  r"                    int forced_J = 0\) \{",
        ),
        Edit(
            id="mmq-case-forward",
            anchor=r"mul_mat_q_switch_J<type, fallback>\(ctx, args, stream\);",
            rationale="both fallback branches inside mul_mat_q_case",
            mode="replace_all",
            expect_matches=2,
            text="mul_mat_q_switch_J<type, fallback>(ctx, args, stream, forced_J);",
            guard=r"mul_mat_q_switch_J<type, fallback>\(ctx, args, stream, forced_J\);",
        ),
        Edit(
            id="mmq-instantiation-macro",
            # DECL_MMQ_CASE restates mul_mat_q_case's parameter list to
            # explicitly instantiate it, so it has to gain the new parameter
            # too. Deliberately without `= 0`: repeating a default argument on
            # an explicit instantiation is ill-formed.
            anchor=r"    template void mul_mat_q_case<type>\(ggml_backend_cuda_context & ctx, "
                   r"const mmq_args & args, cudaStream_t stream\) \\",
            rationale="the explicit-instantiation macro DECL_MMQ_CASE",
            mode="replace",
            text="    template void mul_mat_q_case<type>(ggml_backend_cuda_context & ctx, "
                 "const mmq_args & args, cudaStream_t stream, int forced_J) \\",
            guard=r"cudaStream_t stream, int forced_J\) \\",
        ),
        Edit(
            id="mmq-public-decl",
            anchor=r"^void ggml_cuda_mul_mat_q\(\n"
                   r"        ggml_backend_cuda_context & ctx, const ggml_tensor \* src0, "
                   r"const ggml_tensor \* src1, const ggml_tensor \* ids, ggml_tensor \* dst\);$",
            rationale="the declaration of the public MMQ entry point",
            mode="replace",
            # Defaulted, so every existing caller keeps the native policy.
            text="void ggml_cuda_mul_mat_q(\n"
                 "        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, "
                 "const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,\n"
                 "        int forced_J = 0);",
            guard=r"int forced_J = 0\);",
        ),
    ),
)

SOURCE_PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmq.cu",
    description="MMQ type dispatcher forwards a forced J; variant entry point",
    edits=(
        Edit(
            id="mmq-type-switch-signature",
            anchor=r"^static void ggml_cuda_mul_mat_q_switch_type\("
                   r"ggml_backend_cuda_context & ctx, const mmq_args & args, "
                   r"cudaStream_t stream\) \{$",
            rationale="the per-type MMQ dispatcher",
            mode="replace",
            text="static void ggml_cuda_mul_mat_q_switch_type("
                 "ggml_backend_cuda_context & ctx, const mmq_args & args,\n"
                 "                                            cudaStream_t stream, "
                 "int forced_J = 0) {",
            guard=r"cudaStream_t stream, int forced_J = 0\) \{",
        ),
        Edit(
            id="mmq-type-switch-forward",
            anchor=r"(mul_mat_q_case<GGML_TYPE_[A-Z0-9_]+>)\(ctx, args, stream\);",
            rationale="every per-type case in the MMQ type dispatcher",
            mode="replace_all",
            expect_matches=22,
            text=r"\1(ctx, args, stream, forced_J);",
            guard=r"mul_mat_q_case<GGML_TYPE_[A-Z0-9_]+>\(ctx, args, stream, forced_J\);",
        ),
        Edit(
            id="mmq-public-param",
            anchor=r"^void ggml_cuda_mul_mat_q\(\n"
                   r"        ggml_backend_cuda_context & ctx, const ggml_tensor \* src0, "
                   r"const ggml_tensor \* src1, const ggml_tensor \* ids, ggml_tensor \* dst\) \{$",
            rationale="the public MMQ entry point",
            mode="replace",
            text="void ggml_cuda_mul_mat_q(\n"
                 "        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, "
                 "const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,\n"
                 "        int forced_J) {",
            guard=r"ggml_tensor \* dst,\n        int forced_J\) \{",
        ),
        Edit(
            id="mmq-public-forward",
            anchor=r"ggml_cuda_mul_mat_q_switch_type\(ctx, args, stream\);",
            rationale="the dense and MUL_MAT_ID calls into the type dispatcher",
            mode="replace_all",
            expect_matches=2,
            text="ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, forced_J);",
            guard=r"ggml_cuda_mul_mat_q_switch_type\(ctx, args, stream, forced_J\);",
        ),
        Edit(
            id="mmq-eligibility-and-entry",
            anchor=r"^bool ggml_cuda_should_use_mmq\(enum ggml_type type, int cc, "
                   r"int64_t ne11, int64_t n_experts\) \{$",
            rationale="the function following ggml_cuda_mul_mat_q",
            mode="insert_before",
            text=_MMQ_SOURCE,
            guard=r"void ggml_cuda_mul_mat_q_variant\(",
        ),
    ),
)

PATCHES = [HEADER_PATCH, SOURCE_PATCH]
