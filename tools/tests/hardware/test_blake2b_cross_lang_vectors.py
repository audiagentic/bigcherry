# bigcherry: cross-language digest agreement (HI16).
#
# The dispatch/signature/hardware digests are computed in C++ at runtime
# (hip-autotune-blake2b.cpp) and in Python by the offline tools
# (hashlib.blake2b). A divergence between the two implementations would not
# fail loudly -- every replay lookup would simply miss and fall back to
# native, so the system looks healthy while doing nothing.
#
# The C++ side pins known-answer vectors in
# src/ggml/src/ggml-cuda/hip-autotune-inspect.cpp (the --selftest table) and
# runs them against the real C++ implementation in the campaign build.
# THIS test re-derives every row of that table from Python's hashlib and
# fails if any embedded C++ expectation goes stale, so the table cannot
# silently drift from either implementation.

import hashlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CPP_PATH = REPO_ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-inspect.cpp"

# {"label", "data_hex", "person_hex", person_len, "expected_hex"},
ROW_RE = re.compile(
    r'\{"([A-Za-z0-9_\-]+)",\s*"([0-9a-fA-F]*)",\s*"([0-9a-fA-F]*)",\s*(\d+),\s*"([0-9a-fA-F]+)"\},'
)

MIN_VECTORS = 10


def _table_rows():
    text = CPP_PATH.read_text(encoding="utf-8")
    start = text.find("kDigestVectors")
    assert start >= 0, "kDigestVectors table not found in hip-autotune-inspect.cpp"
    section = text[start:]
    end = section.find("];")
    assert end >= 0, "kDigestVectors table terminator not found"
    rows = ROW_RE.findall(section[:end])
    return [
        {
            "label": label,
            "data": bytes.fromhex(data_hex),
            "person": bytes.fromhex(person_hex),
            "person_len": int(person_len),
            "expected": expected_hex.lower(),
        }
        for label, data_hex, person_hex, person_len, expected_hex in rows
    ]


def test_table_is_parsed_and_sufficient():
    rows = _table_rows()
    assert len(rows) >= MIN_VECTORS, (
        f"expected at least {MIN_VECTORS} digest vectors, parsed {len(rows)} "
        "-- the C++ table changed shape or the parser is broken"
    )
    labels = [r["label"] for r in rows]
    assert len(set(labels)) == len(labels), "duplicate vector labels"


def test_every_c_embedded_expectation_matches_python_hashlib():
    failures = []
    for row in _table_rows():
        person = row["person"] if row["person_len"] > 0 else b""
        actual = hashlib.blake2b(
            row["data"], digest_size=16, person=person
        ).hexdigest()
        if actual != row["expected"]:
            failures.append(f"{row['label']}: C++={row['expected']} python={actual}")
    assert not failures, "C++ known-answer table has drifted from hashlib:\n" + "\n".join(failures)


def test_table_covers_the_production_person_prefixes():
    rows = {r["label"]: r["person"] for r in _table_rows()}
    # These are the person prefixes the runtime actually uses
    # (hip-autotune-signature.h). A table that never exercises them proves
    # nothing about production digests.
    assert rows["sig-person-real"] == b"llama-hip-tune"
    assert rows["dispatch-person-real"] == b"llama-dispatch"
    assert rows["hardware-person-real"] == b"llama-hardware"


def test_table_covers_the_structural_edge_cases():
    rows = {r["label"]: r for r in _table_rows()}
    assert rows["abc-no-person"]["person_len"] == 0, "no-person vector must pass nullptr"
    assert rows["empty-data"]["data"] == b"", "empty-data vector lost its empty payload"
    assert len(rows["one-full-block-128"]["data"]) == 128, "block-boundary vector is no longer exactly one 128-byte block"
    assert len(rows["one-block-plus-one"]["data"]) == 129, "one-block-plus-one vector is no longer 129 bytes"
    assert len(rows["multi-block-300"]["data"]) > 256, "multi-block vector is no longer multi-block"
    assert rows["short-person-zero-pad"]["person"] == b"sig", "short-person vector must exercise zero padding"
    # The exact-16 person is the ASCII string "0123456789abcdef" -- 16 raw
    # bytes, the person parameter's maximum, with no zero padding applied.
    assert rows["person-exact-16"]["person"] == b"0123456789abcdef", "exact-16 person vector changed"


def test_cpp_selftest_flag_exists():
    text = CPP_PATH.read_text(encoding="utf-8")
    assert 'arg == "--selftest"' in text, "the --selftest entry point is gone from hip-autotune-inspect.cpp"
    assert "run_digest_selftest" in text, "run_digest_selftest is gone from hip-autotune-inspect.cpp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
