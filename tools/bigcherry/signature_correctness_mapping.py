"""HI80: map a real BigCherry dispatch signature into the exact op_filter
regex and target_tensor name tools/bigcherry/correctness_evidence.py's
generate_correctness_evidence() needs to drive test-backend-ops against the
SAME operation the signature actually identifies.

Getting this mapping wrong would silently produce correctness evidence for
the wrong signature -- a correctness-of-the-safety-gate risk, not just a
usability gap (HI80's own framing). So the two enum tables this module
needs (ggml_type id -> name, ggml_op id -> name) are never hand-transcribed:
they are parsed directly out of the real vendored source
(ggml/include/ggml.h's enum declarations, ggml/src/ggml.c's type_traits and
GGML_OP_NAME tables) every time this module is used, so they can never
silently drift out of sync with whatever source revision is actually
checked out.

Scope of this slice: MUL_MAT only -- the op actually blocking the two
confirmed HI80-referenced promotions (batch_for_mmvf). Every other ggml op
has its own bespoke test-backend-ops test_case subclass with different
vars() fields (see test-backend-ops.cpp), so a general op-agnostic
translator is not a small next step from here; extending to another op
means reading that op's own test_case subclass in test-backend-ops.cpp
first, the same way this module's MUL_MAT support was derived (see
_mul_mat_op_filter's docstring for the exact source lines this was verified
against).
"""

from __future__ import annotations

import re
from pathlib import Path

_GGML_H_RELATIVE = Path("ggml/include/ggml.h")
_GGML_C_RELATIVE = Path("ggml/src/ggml.c")


class SignatureMappingError(ValueError):
    """The signature cannot be mapped to an op_filter/target_tensor pair --
    either it names an op this module does not yet support, or the vendor
    source it needs to parse is missing/malformed."""


def load_ggml_type_names(vendor_root: Path) -> dict[int, str]:
    """Parse ggml.h's ``enum ggml_type`` (for the numeric ids) and ggml.c's
    ``type_traits`` table (for each id's real ``.type_name`` string) and
    return the combined {id: name} map -- exactly what
    tools/bigcherry/correctness_evidence.py's op_filter needs to spell a
    type the same way test-backend-ops' own var_to_str(ggml_type) does
    (which is a direct call to ggml_type_name(), verified against
    tests/test-backend-ops.cpp)."""
    header = _read(vendor_root / _GGML_H_RELATIVE)
    source = _read(vendor_root / _GGML_C_RELATIVE)

    enum_match = re.search(r"enum ggml_type \{(.*?)\};", header, re.S)
    if enum_match is None:
        raise SignatureMappingError(
            f"could not find 'enum ggml_type' in {vendor_root / _GGML_H_RELATIVE}"
        )
    enum_ids: dict[str, int] = {}
    for name, value in re.findall(r"GGML_TYPE_(\w+)\s*=\s*(-?\d+)", enum_match.group(1)):
        enum_ids[name] = int(value)
    if not enum_ids:
        raise SignatureMappingError("enum ggml_type parsed but yielded no entries")

    traits_match = re.search(
        r"static const struct ggml_type_traits type_traits\[GGML_TYPE_COUNT\] = \{(.*?)\n\};",
        source, re.S,
    )
    if traits_match is None:
        raise SignatureMappingError(
            f"could not find the type_traits table in {vendor_root / _GGML_C_RELATIVE}"
        )
    names: dict[str, str] = {}
    for enum_name, type_name in re.findall(
        r'\[GGML_TYPE_(\w+)\]\s*=\s*\{[^}]*?\.type_name\s*=\s*"([^"]+)"',
        traits_match.group(1), re.S,
    ):
        names[enum_name] = type_name
    if not names:
        raise SignatureMappingError("type_traits table parsed but yielded no type_name entries")

    return {enum_ids[k]: v for k, v in names.items() if k in enum_ids}


def load_ggml_op_names(vendor_root: Path) -> dict[int, str]:
    """Parse ggml.h's ``enum ggml_op`` declaration order (ids are implicit,
    sequential from 0 -- there are no explicit numeric assignments in this
    enum, verified against the real header) together with ggml.c's
    ``GGML_OP_NAME`` array (positionally indexed by that same order) and
    return the combined {id: name} map."""
    header = _read(vendor_root / _GGML_H_RELATIVE)
    source = _read(vendor_root / _GGML_C_RELATIVE)

    enum_match = re.search(r"enum ggml_op \{(.*?)\};", header, re.S)
    if enum_match is None:
        raise SignatureMappingError(
            f"could not find 'enum ggml_op' in {vendor_root / _GGML_H_RELATIVE}"
        )
    op_order = re.findall(r"GGML_OP_(\w+)", enum_match.group(1))
    if not op_order:
        raise SignatureMappingError("enum ggml_op parsed but yielded no entries")
    # GGML_OP_COUNT is a sentinel terminator, not a real op -- it has no
    # slot in GGML_OP_NAME, so it must be excluded here rather than
    # counted as a positional entry the name table is missing.
    if op_order[-1] != "COUNT":
        raise SignatureMappingError(
            f"expected enum ggml_op's last entry to be the GGML_OP_COUNT "
            f"sentinel, found GGML_OP_{op_order[-1]} instead -- the enum "
            "layout has changed in a way this parser does not recognize"
        )
    op_order = op_order[:-1]

    names_match = re.search(
        r"static const char \* GGML_OP_NAME\[GGML_OP_COUNT\] = \{(.*?)\n\};",
        source, re.S,
    )
    if names_match is None:
        raise SignatureMappingError(
            f"could not find the GGML_OP_NAME table in {vendor_root / _GGML_C_RELATIVE}"
        )
    name_strings = re.findall(r'"([^"]+)"', names_match.group(1))
    if not name_strings:
        raise SignatureMappingError("GGML_OP_NAME table parsed but yielded no strings")

    # Fail closed on a length mismatch rather than silently zipping the
    # shorter of the two together: this table is an evidence-authority
    # mapping (a wrong id->name association here would misattribute
    # correctness evidence to the wrong op), so a vendor-source format
    # drift that desyncs the enum from the name array must be a loud
    # error here, not a quietly-truncated result.
    if len(name_strings) != len(op_order):
        raise SignatureMappingError(
            f"enum ggml_op has {len(op_order)} entries but GGML_OP_NAME has "
            f"{len(name_strings)} -- these must be positionally aligned 1:1; "
            "a mismatch means the vendor source format has drifted from what "
            "this parser expects, and any id->name mapping derived from it "
            "would be unverified rather than fail closed"
        )

    return dict(enumerate(name_strings))


def _read(path: Path) -> str:
    if not path.is_file():
        raise SignatureMappingError(f"expected vendor source file not found: {path}")
    return path.read_text(encoding="utf-8")


def signature_to_op_filter(
    signature: dict[str, object], *, vendor_root: Path,
) -> tuple[str, str]:
    """Map a real ggml_hip_dispatch_signature_v1 JSON (as emitted by
    src/ggml/src/ggml-cuda/hip-autotune-signature.cpp's
    ggml_hip_signature_json(), and as recorded in a measurements.jsonl's
    canonical field) into (op_filter, target_tensor) for
    generate_correctness_evidence().

    MUL_MAT only this slice -- raises SignatureMappingError for any other
    op. See _mul_mat_op_filter for the exact test-backend-ops.cpp lines
    this was verified against."""
    op_names = load_ggml_op_names(vendor_root)
    op_id = int(signature["op"])
    op_name = op_names.get(op_id)
    if op_name is None:
        raise SignatureMappingError(f"unknown ggml_op id {op_id!r} -- not present in the parsed enum/name tables")
    if op_name != "MUL_MAT":
        raise SignatureMappingError(
            f"signature_to_op_filter only supports MUL_MAT this slice, got op={op_name!r} "
            f"(id={op_id}) -- extending to another op requires reading that op's own "
            "test_case subclass in test-backend-ops.cpp first (see this module's docstring)"
        )
    return _mul_mat_op_filter(signature, vendor_root=vendor_root)


def _mul_mat_op_filter(signature: dict[str, object], *, vendor_root: Path) -> tuple[str, str]:
    """Verified against tests/test-backend-ops.cpp's struct test_mul_mat
    (vendor/llama.cpp/tests/test-backend-ops.cpp, ~line 4548 at the source
    revision this was written against) AND against the real caller,
    tools/bigcherry/correctness_evidence.py's run_test_backend_ops(), which
    hardcodes ``-o MUL_MAT`` and passes op_filter directly as ``-p``'s
    value. The ``-p`` filter is matched via std::regex_search against
    test_case::vars() ALONE (test-backend-ops.cpp's main(), ~line 10602) --
    vars() does NOT include the op name or its surrounding parentheses,
    only the comma-joined field=value list ``-o`` already narrowed the
    candidate set to MUL_MAT for. The returned op_filter is therefore a
    vars()-shaped fragment (e.g. "^type_a=f32,type_b=f32,m=32,n=32,k=32,"),
    never a "MUL_MAT(...)"-shaped string -- an earlier draft of this
    function got this wrong before being corrected against the real caller.

    - vars() returns VARS_TO_STR10(type_a, type_b, m, n, k, bs, nr, per,
      k_v, o); var_to_str(ggml_type) is a direct call to ggml_type_name().
    - test_mul_mat's build_graph() constructs
      a = ggml_new_tensor_4d(ctx, type_a, k_physical, m, bs[0], bs[1])
      b = ggml_new_tensor_4d(ctx, type_b, k_physical, n, bs[0]*nr[0], bs[1]*nr[1])
      so for the (default, non-permuted, k_v=0) case: m = src0.ne[1],
      n = src1.ne[1], k = src0.ne[0] == src1.ne[0].
    - the output tensor is named "out" (ggml_set_name(out, "out")) for the
      default o=1 case; o>1 additionally produces "out2" tensors summed
      into "out" -- this module only targets the o=1 case.

    bs, nr, per, k_v, and o are NOT reconstructable from a BigCherry
    dispatch signature alone: ne0[2]/ne0[3] carry the PRODUCT bs[i]*nr[i]
    with no way to recover the two factors independently, and per/k_v/o
    have no signature field at all. A real BigCherry dispatch signature is
    a single, non-batched, non-permuted, non-view local operation
    (standards 5.2: signatures are built from the actual local tensor
    slice), so bs=[1,1], nr=[1,1], per=[0,1,2,3], k_v=0, o=1 is the
    expected case -- this function ASSERTS it by pinning the full vars()
    string with both a leading and trailing anchor, rather than wildcarding
    the unreconstructable fields.

    This is a hardware-verified requirement, not a convenience: leaving
    bs/nr/per wildcarded was tried first and found, on real Brutus
    hardware, to match 13 DISTINCT registered test-backend-ops cases for a
    single (type_a, type_b, m, n, k) tuple -- all producing a tensor
    literally named "out", so find_digest_for_tensor()/
    find_metric_for_tensor() (which match by tensor name alone) could
    silently take evidence from whichever of the 13 happened to run first,
    not necessarily the one the signature actually identifies. Anchoring
    the full string collapsed that same query to exactly the one intended
    case (confirmed by re-running the identical filter with bs=[1,1],
    nr=[1,1], per=[0,1,2,3], k_v=0, o=1 appended and observing a single
    matching test-case line, not 13)."""
    type_names = load_ggml_type_names(vendor_root)

    def _type_name(type_id: object) -> str:
        name = type_names.get(int(type_id))
        if name is None:
            raise SignatureMappingError(f"unknown ggml_type id {type_id!r}")
        return name

    ne0 = signature.get("ne0")
    ne1 = signature.get("ne1")
    if not isinstance(ne0, (list, tuple)) or len(ne0) != 4:
        raise SignatureMappingError(f"signature ne0 must be a 4-element array, got {ne0!r}")
    if not isinstance(ne1, (list, tuple)) or len(ne1) != 4:
        raise SignatureMappingError(f"signature ne1 must be a 4-element array, got {ne1!r}")

    k_from_a = int(ne0[0])
    k_from_b = int(ne1[0])
    if k_from_a != k_from_b:
        raise SignatureMappingError(
            f"signature's src0/src1 shared inner dimension disagrees: "
            f"ne0[0]={k_from_a}, ne1[0]={k_from_b} -- not a valid MUL_MAT signature"
        )

    # The returned op_filter unconditionally PINS bs=[1,1], nr=[1,1] --
    # that is only a valid test-case identity for a signature whose own
    # outer dimensions (ne0[2:4], ne1[2:4]) are actually (1,1). Prove that
    # from the signature itself rather than merely asserting it in prose:
    # a signature with real batch/repeat dimensions would otherwise get
    # correctness evidence for an unrelated, un-batched test case, exactly
    # the "authorized the wrong thing" failure mode HI80 exists to prevent.
    ne0_outer = (int(ne0[2]), int(ne0[3]))
    ne1_outer = (int(ne1[2]), int(ne1[3]))
    if ne0_outer != (1, 1):
        raise SignatureMappingError(
            f"signature's src0 outer dimensions are {ne0_outer}, not (1, 1) -- "
            "this function only maps the non-batched default case (bs=[1,1], "
            "nr=[1,1]); a signature with real outer/batch dimensions needs "
            "its own op_filter derivation, not this one's pinned assumption"
        )
    if ne1_outer != (1, 1):
        raise SignatureMappingError(
            f"signature's src1 outer dimensions are {ne1_outer}, not (1, 1) -- "
            "this function only maps the non-batched default case (bs=[1,1], "
            "nr=[1,1]); a signature with real outer/batch dimensions needs "
            "its own op_filter derivation, not this one's pinned assumption"
        )

    type_a = _type_name(signature["src0_type"])
    type_b = _type_name(signature["src1_type"])
    m = int(ne0[1])
    n = int(ne1[1])
    k = k_from_a

    op_filter = (
        rf"^type_a={re.escape(type_a)},type_b={re.escape(type_b)},"
        rf"m={m},n={n},k={k},"
        rf"bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1$"
    )
    return op_filter, "out"


# HI80 (2026-08-23 real-hardware finding): signature_to_op_filter() only ever
# produces evidence when a real dispatch signature happens to coincide with
# one of test-backend-ops' own fixed synthetic corpus shapes -- verified on
# Brutus that a real gpt-oss-20B MUL_MAT (m=32, n=21, k=2880) matches NONE of
# them, so 0 tests run and no evidence is emitted, for a signature that is
# otherwise perfectly valid. test-backend-ops already has an escape hatch
# for exactly this: `--test-file <path>`, read by make_test_cases_from_file()
# (vendor/llama.cpp/tests/test-backend-ops.cpp) into one test_generic_op per
# line, which builds a graph directly from the given op/type/ne/sources
# instead of going through the fixed generator -- so it works for ANY real
# signature, not just ones lucky enough to already be registered. Verified
# end to end on real Brutus hardware for the exact blocked m=32/n=21/k=2880
# signature: BIGCHERRY_CORRECTNESS_METRIC fires for tensor "out" exactly as
# it does for a registered-corpus case (test_generic_op inherits test_case's
# shared eval() path, which is where patches 1222/1223 hook in -- nothing
# op-specific about those patches, so no new patch to test-backend-ops
# itself is needed).
#


def signature_to_test_file_line(
    signature: dict[str, object], *, vendor_root: Path,
) -> tuple[str, str, str]:
    """Map a real MUL_MAT dispatch signature into one test-backend-ops
    ``--test-file`` line (see module docstring above), its target tensor
    name ("out", same as signature_to_op_filter -- test_generic_op
    hardcodes this name too), and its digest tensor name ("leaf_0" -- the
    first, unnamed source tensor test_generic_op creates; ggml auto-names
    unnamed leaf tensors "leaf_0", "leaf_1", ... in creation order, and
    this module always emits src0 first). Same scope restriction as
    signature_to_op_filter: non-batched, non-permuted, contiguous case only
    (ne0/ne1 outer dims == (1, 1)) -- this is the real shape of an actual
    BigCherry dispatch signature (standards 5.2), not an arbitrary
    restriction.

    Why a separate digest tensor (2026-08-23 real-hardware finding):
    test_generic_op's destination tensor is never pre-filled with random
    data the way a registered test case's is (a side effect of that other
    class's own build_graph(), not something test_case::eval() does
    universally), so it never gets a BIGCHERRY_REF_DIGEST line at all --
    confirmed on Brutus for this exact function's output line, even though
    its BIGCHERRY_CORRECTNESS_METRIC line fires correctly. "leaf_0" (a
    genuine random-input leaf) always gets digested, and proves the same
    thing the digest check exists for -- that both runs saw identical
    reference input -- just via a different, always-present tensor.
    correctness_evidence.py's collect_seed_evidence() takes this as its
    separate ``digest_tensor`` parameter.

    Unlike signature_to_op_filter, this always produces a matchable case
    for any signature within its supported type/shape scope -- there is no
    fixed corpus to coincide with, only test-backend-ops' own file-format
    parser (a fixed, simple line grammar, verified against
    make_test_cases_from_file() in tests/test-backend-ops.cpp).

    HI111 (2026-08-24 real-hardware finding, dense Qwen3.8-27B/Q8_0
    production campaign): src0/src1 are always serialized with nb=[0,0,0,0]
    (test-backend-ops' own "use default contiguous strides" sentinel --
    is_non_contiguous()'s `if (src.nb[0] == 0) return false;`), the same
    trick signature_to_mul_mat_id_test_file_line() already uses, rather than
    this function hand-computing byte strides from a type-size whitelist.
    ggml itself derives the correct block-aware contiguous layout for
    whatever type is given, quantized or not -- an earlier version of this
    function hand-computed nb from a `{F32, F16, BF16}`-only type-size table
    and rejected every quantized signature (Q8_0, Q4_K, Q6_K, ...) outright,
    which on a real Q8_0-quantized production model left almost the entire
    signature set without correctness evidence."""
    op_names = load_ggml_op_names(vendor_root)
    op_id = int(signature["op"])
    op_name = op_names.get(op_id)
    if op_name is None:
        raise SignatureMappingError(f"unknown ggml_op id {op_id!r} -- not present in the parsed enum/name tables")
    if op_name != "MUL_MAT":
        raise SignatureMappingError(
            f"signature_to_test_file_line only supports MUL_MAT this slice, got op={op_name!r} "
            f"(id={op_id})"
        )

    ne0 = signature.get("ne0")
    ne1 = signature.get("ne1")
    ned = signature.get("ned")
    for field_name, value in (("ne0", ne0), ("ne1", ne1), ("ned", ned)):
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise SignatureMappingError(f"signature {field_name} must be a 4-element array, got {value!r}")

    k_from_a = int(ne0[0])
    k_from_b = int(ne1[0])
    if k_from_a != k_from_b:
        raise SignatureMappingError(
            f"signature's src0/src1 shared inner dimension disagrees: "
            f"ne0[0]={k_from_a}, ne1[0]={k_from_b} -- not a valid MUL_MAT signature"
        )
    # HI112 (2026-08-24, dense Qwen3.8-27B/Q8_0 production campaign): src0
    # (the weights) must stay non-batched -- this mapper never sees a
    # genuinely batched weight matrix on real hardware -- but src1 (the
    # activations) batching over ne[2] is real, common production traffic:
    # MTP's multi-draft-token verification batches multiple candidate
    # continuations against the SAME weight matrix in one ggml_mul_mat call
    # (ggml's own src1-broadcasts-over-src0 rule). test-backend-ops'
    # build_graph() has no batching restriction at all once nb is the
    # zero-strides sentinel (see is_non_contiguous()/build_graph() in
    # tests/test-backend-ops.cpp -- ggml_new_tensor_4d takes ne[2]/ne[3]
    # unconditionally), so this was an artificial restriction in this
    # module, not something test-backend-ops itself requires. The 4th
    # dimension stays pinned to 1 -- no real signature with ne[3] != 1 has
    # been observed yet, so that case is left unverified rather than
    # silently accepted.
    ne0_outer = (int(ne0[2]), int(ne0[3]))
    ne1_outer = (int(ne1[2]), int(ne1[3]))
    ned_outer = (int(ned[2]), int(ned[3]))
    if ne0_outer != (1, 1) or ne1_outer[1] != 1 or ned_outer[1] != 1:
        raise SignatureMappingError(
            f"signature outer dimensions are src0={ne0_outer} src1={ne1_outer} "
            f"dst={ned_outer} -- this function only maps a non-batched src0 "
            "against an (possibly ne[2]-batched) src1/dst, matching "
            "ggml_mul_mat's own src1-broadcasts-over-src0 rule; a batched "
            "src0 or a non-trivial 4th dimension is out of scope"
        )
    if ne1_outer[0] != ned_outer[0]:
        raise SignatureMappingError(
            f"signature src1 batch dimension ne1[2]={ne1_outer[0]} disagrees "
            f"with dst batch dimension ned[2]={ned_outer[0]} -- not a valid "
            "MUL_MAT signature"
        )

    type_names = load_ggml_type_names(vendor_root)

    def _type_name(type_id: object) -> str:
        name = type_names.get(int(type_id))
        if name is None:
            raise SignatureMappingError(f"unknown ggml_type id {type_id!r}")
        return name.upper()

    src0_type_id = int(signature["src0_type"])
    src1_type_id = int(signature["src1_type"])
    dst_type_id = int(signature["dst_type"])
    # Validated for the enum tables' own sake (raises on an unknown id) --
    # the resolved names are otherwise unused: nb is always the zero-strides
    # sentinel below, so no per-type byte size is needed for any of the
    # three types, quantized or not.
    _type_name(src0_type_id)
    _type_name(src1_type_id)
    _type_name(dst_type_id)

    ne0_i = [int(x) for x in ne0]
    ne1_i = [int(x) for x in ne1]
    ned_i = [int(x) for x in ned]
    zero_strides = [0, 0, 0, 0]

    # GGML_OP_MUL_MAT's own id, op_params (unused by MUL_MAT -- 0 of them),
    # num_sources=2 (src0, src1), name "-" (auto-generated, matches
    # make_test_cases_from_file()'s own '-' -> empty-name convention).
    line = " ".join(str(x) for x in (
        op_id, dst_type_id, *ned_i, 0, 2,
        src0_type_id, *ne0_i, *zero_strides,
        src1_type_id, *ne1_i, *zero_strides,
    )) + " -"
    return line, "out", "leaf_0"


def _ggml_type_id_for_name(type_names: dict[int, str], name: str) -> int:
    """Reverse load_ggml_type_names()'s {id: name} map to find a specific
    type's numeric id -- used for GGML_TYPE_I32 (the ids tensor's real,
    hardcoded type per mmvq.cu's own `GGML_ASSERT(!ids || ids->type ==
    GGML_TYPE_I32)`) so its enum value is derived from the same
    source-parsed table every other type id in this module comes from,
    never a hand-transcribed numeric literal that could silently drift."""
    for type_id, type_name in type_names.items():
        if type_name.upper() == name:
            return type_id
    raise SignatureMappingError(
        f"could not find ggml_type id for {name!r} in the parsed type_traits table"
    )


def signature_to_mul_mat_id_test_file_line(
    signature: dict[str, object], *, vendor_root: Path,
) -> tuple[str, str, str]:
    """HI105: MUL_MAT_ID sibling of signature_to_test_file_line(), derived
    from ggml_mul_mat_id's real shape identity (src/llama-graph.cpp's
    build_moe_ffn, verified 2026-08-24 against RD54's own two real
    signatures) rather than test-backend-ops' registered test_mul_mat_id
    class, whose fixed synthetic corpus a real production signature
    generally does not coincide with (the same real-hardware finding that
    motivated the plain-MUL_MAT test-file path).

    Real shape identity (ggml_mul_mat_id(as, b, ids)):
        as  (src0, weights):     [k, m, n_expert]                (ne0)
        b   (src1, activations): [k, ne1[1], n_tokens], ne1[1]
                                  broadcastable up to n_expert_used
                                  (ne1[1]==1: up/gate's real shape;
                                  ne1[1]==n_expert_used: down's) -- taken
                                  LITERALLY from the recorded signature, no
                                  broadcast-flag branching needed the way
                                  test_mul_mat_id's synthetic `b` bool
                                  requires; the generic test-file path just
                                  takes whatever explicit shape is given.
        ids:                     [n_expert_used, n_tokens], GGML_TYPE_I32
                                  -- not itself a recorded signature field;
                                  derived from ned[1]/ned[2] (confirmed via
                                  `selected_experts = ggml_argsort_top_k(...)
                                  // [n_expert_used, n_tokens]`).
        out (dst):                [m, n_expert_used, n_tokens]    (ned)

    Every source is serialized with nb=[0,0,0,0] (test-backend-ops' own
    input_tensor convention for "use default contiguous strides", per
    is_non_contiguous()'s `if (src.nb[0] == 0) return false;`) rather than
    this module hand-computing byte strides the way the plain-MUL_MAT path
    does -- this sidesteps needing Q8_0's block-quantized layout math in
    Python entirely; ggml itself derives the correct contiguous layout for
    whatever type is given, quantized or not.

    Requires patches/1236_hi105_deterministic_mul_mat_id_ids.py applied
    (REQUIRES patches/1222) so two independent process invocations
    (forced-native, forced-candidate) see identical, full-expert-range
    routing for the same BIGCHERRY_TEST_DETERMINISTIC_SEED -- without it,
    test_generic_op's own MUL_MAT_ID initializer is both non-deterministic
    across processes and confined to a {0..n_expert_used-1} range rather
    than the real full expert pool (see that patch's own docstring)."""
    op_names = load_ggml_op_names(vendor_root)
    op_id = int(signature["op"])
    op_name = op_names.get(op_id)
    if op_name is None:
        raise SignatureMappingError(f"unknown ggml_op id {op_id!r} -- not present in the parsed enum/name tables")
    if op_name != "MUL_MAT_ID":
        raise SignatureMappingError(
            f"signature_to_mul_mat_id_test_file_line only supports MUL_MAT_ID, "
            f"got op={op_name!r} (id={op_id})"
        )

    flags = int(signature.get("flags", 0))
    _HAS_IDS = 1 << 3
    if not (flags & _HAS_IDS):
        raise SignatureMappingError(
            f"signature op is MUL_MAT_ID but flags={flags!r} does not have "
            "GGML_HIP_SIG_HAS_IDS (bit 3) set -- not a genuine MoE-routed signature"
        )
    _CONTIGUOUS = (1 << 0) | (1 << 1) | (1 << 2)
    if flags & _CONTIGUOUS != _CONTIGUOUS:
        raise SignatureMappingError(
            f"signature flags={flags!r} indicate a non-contiguous src0/src1/dst -- "
            "this function only maps the contiguous default case"
        )

    ne0 = signature.get("ne0")
    ne1 = signature.get("ne1")
    ned = signature.get("ned")
    for field_name, value in (("ne0", ne0), ("ne1", ne1), ("ned", ned)):
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise SignatureMappingError(f"signature {field_name} must be a 4-element array, got {value!r}")
    ne0_i = [int(x) for x in ne0]
    ne1_i = [int(x) for x in ne1]
    ned_i = [int(x) for x in ned]

    n_expert = int(signature.get("n_expert", 0))
    n_expert_used = int(signature.get("n_expert_used", 0))
    if n_expert <= 0 or n_expert_used <= 0:
        raise SignatureMappingError(
            f"signature op is MUL_MAT_ID but n_expert={n_expert!r}/"
            f"n_expert_used={n_expert_used!r} are not both positive"
        )
    if n_expert_used > n_expert:
        raise SignatureMappingError(
            f"signature n_expert_used={n_expert_used} exceeds n_expert={n_expert} -- "
            "not a valid MUL_MAT_ID signature (patch 1236's C++ shuffle does "
            "pool.begin() + t->ne[0], which would be invalid iterator arithmetic "
            "for a pool smaller than the requested slice)"
        )
    if n_expert != ne0_i[2]:
        raise SignatureMappingError(
            f"signature n_expert={n_expert} disagrees with src0's own expert-count "
            f"dimension ne0[2]={ne0_i[2]} -- not a valid MUL_MAT_ID signature"
        )
    if n_expert_used != ned_i[1]:
        raise SignatureMappingError(
            f"signature n_expert_used={n_expert_used} disagrees with dst's own "
            f"n_expert_used dimension ned[1]={ned_i[1]} -- not a valid MUL_MAT_ID signature"
        )
    if ne0_i[0] != ne1_i[0]:
        raise SignatureMappingError(
            f"signature's src0/src1 shared inner (K) dimension disagrees: "
            f"ne0[0]={ne0_i[0]}, ne1[0]={ne1_i[0]} -- not a valid MUL_MAT_ID signature"
        )
    if ned_i[0] != ne0_i[1]:
        raise SignatureMappingError(
            f"signature dst rows ned[0]={ned_i[0]} disagree with src0's own output-row "
            f"dimension ne0[1]={ne0_i[1]} -- not a valid MUL_MAT_ID signature"
        )
    if ned_i[2] != ne1_i[2]:
        raise SignatureMappingError(
            f"signature dst token dimension ned[2]={ned_i[2]} disagrees with src1's "
            f"own token dimension ne1[2]={ne1_i[2]} -- not a valid MUL_MAT_ID signature"
        )
    if ne0_i[3] != 1 or ne1_i[3] != 1 or ned_i[3] != 1:
        raise SignatureMappingError(
            "signature has a non-trivial 4th (batch) dimension -- this function "
            "only maps the non-batched default case"
        )
    if ne1_i[1] <= 0 or n_expert_used % ne1_i[1] != 0:
        raise SignatureMappingError(
            f"signature's src1 middle dimension ne1[1]={ne1_i[1]} does not evenly "
            f"divide n_expert_used={n_expert_used} -- not a valid ggml_mul_mat_id "
            "broadcast shape"
        )

    type_names = load_ggml_type_names(vendor_root)

    def _type_name(type_id: object) -> str:
        name = type_names.get(int(type_id))
        if name is None:
            raise SignatureMappingError(f"unknown ggml_type id {type_id!r}")
        return name.upper()

    src0_type_id = int(signature["src0_type"])
    src1_type_id = int(signature["src1_type"])
    dst_type_id = int(signature["dst_type"])
    _type_name(src0_type_id)
    _type_name(src1_type_id)
    _type_name(dst_type_id)

    ids_type_id = _ggml_type_id_for_name(type_names, "I32")
    ids_ne = [n_expert_used, ned_i[2], 1, 1]
    zero_strides = [0, 0, 0, 0]

    # GGML_OP_MUL_MAT_ID's own id, op_params (unused -- 0 of them),
    # num_sources=3 (as, b, ids -- ggml_mul_mat_id's real source order,
    # matching test_mul_mat_id::build_graph()), name "-" (auto-generated).
    line = " ".join(str(x) for x in (
        op_id, dst_type_id, *ned_i, 0, 3,
        src0_type_id, *ne0_i, *zero_strides,
        src1_type_id, *ne1_i, *zero_strides,
        ids_type_id, *ids_ne, *zero_strides,
    )) + " -"
    # HI105 (dev-gpt-agent review, 2026-08-24): the digest tensor for
    # MUL_MAT_ID is the IDS tensor ("leaf_2", the 3rd of 3 sources in
    # creation order -- ggml_format_name(node, "leaf_%d", cgraph->n_leafs)
    # names unnamed leaves; confirmed empirically for source 0 of a
    # 2-source MUL_MAT_ID case, but NOT yet independently confirmed for a
    # 3-source case on real hardware -- verify the real BIGCHERRY_REF_DIGEST
    # `name=` output on the first real run before trusting this), not
    # "leaf_0" (the weights) the way the plain-MUL_MAT path uses. Patch
    # 1222's deterministic init_tensor_uniform already gives leaf_0/leaf_1
    # (weights/activations) a proven, verified determinism guarantee; the
    # NEW risk MUL_MAT_ID introduces is expert routing (patch 1236's own
    # subject), so checking the ids digest is what actually closes the gap
    # patch 1236 exists to close -- native and candidate could otherwise
    # each compare successfully against a different CPU reference with
    # different routing, and nothing would catch it. Checking every
    # digest tensor (weights AND routing) is the more complete fix and is
    # a documented follow-up, not implemented in this slice.
    return line, "out", "leaf_2"


def signature_to_any_test_file_line(
    signature: dict[str, object], *, vendor_root: Path,
) -> tuple[str, str, str]:
    """HI105: the one authoritative dispatcher hi80_generate_correctness_
    evidence.py's CLI must call, so adding a new op's mapper here is the
    only place that needs touching to make the CLI pick it up -- a caller
    that hand-picks signature_to_test_file_line() (or the new MUL_MAT_ID
    sibling) directly risks silently routing a real signature through the
    wrong mapper, which for a shape that HAPPENS to satisfy both mappers'
    structural checks (implausible today given the different num_src /
    ne-derivation each expects, but not something to rely on) could
    produce plausible-looking evidence for the wrong operation.

    Routes purely on the signature's own real ggml_op id (never on
    ancillary fields like n_expert/flags, which the wrong op could also
    carry) so an unsupported op fails closed here with a clear message,
    the same SignatureMappingError callers already handle for either
    individual mapper."""
    op_names = load_ggml_op_names(vendor_root)
    op_id = int(signature["op"])
    op_name = op_names.get(op_id)
    if op_name is None:
        raise SignatureMappingError(f"unknown ggml_op id {op_id!r} -- not present in the parsed enum/name tables")
    if op_name == "MUL_MAT":
        return signature_to_test_file_line(signature, vendor_root=vendor_root)
    if op_name == "MUL_MAT_ID":
        return signature_to_mul_mat_id_test_file_line(signature, vendor_root=vendor_root)
    raise SignatureMappingError(
        f"signature_to_any_test_file_line has no mapper for op={op_name!r} (id={op_id}) -- "
        "supported ops this slice: MUL_MAT, MUL_MAT_ID"
    )
