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
