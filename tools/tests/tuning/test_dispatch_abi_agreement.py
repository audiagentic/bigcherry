"""HI119 review follow-up: the dispatch signature/hardware schema version is
independently expressed in three places -- the real C++ #define
(hip-autotune-types.h), tuning/dispatch_abi.py's Python constant, and
sql/dispatch-db.sql's schema_meta seed row -- with nothing that previously
checked they agreed. This test parses all three directly out of the real
source/SQL files (never hand-transcribed) and fails fast if a future bump
touches one and misses another."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import dispatch_abi  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TYPES_H = ROOT / "src" / "ggml" / "src" / "ggml-cuda" / "hip-autotune-types.h"
DISPATCH_DB_SQL = ROOT / "sql" / "dispatch-db.sql"


def _parse_cpp_define(text: str, name: str) -> int:
    match = re.search(rf"#define\s+{re.escape(name)}\s+(\d+)", text)
    if match is None:
        raise AssertionError(f"could not find #define {name} in {TYPES_H}")
    return int(match.group(1))


def _parse_sql_schema_meta(text: str, key: str) -> int:
    match = re.search(rf"\('{re.escape(key)}',\s*'(\d+)'\)", text)
    if match is None:
        raise AssertionError(f"could not find schema_meta seed row for {key!r} in {DISPATCH_DB_SQL}")
    return int(match.group(1))


class DispatchAbiAgreementTests(unittest.TestCase):
    def test_signature_schema_version_is_the_frozen_identity_epoch_alias(self):
        # HI121 (round 9): SIGNATURE_SCHEMA_VERSION is now a compatibility
        # alias of SIGNATURE_IDENTITY_EPOCH, never assigned independently --
        # existing serialized-artifact vocabulary ("signature_schema") keeps
        # working without a second, separately hand-maintained constant.
        self.assertEqual(dispatch_abi.SIGNATURE_SCHEMA_VERSION, dispatch_abi.SIGNATURE_IDENTITY_EPOCH)

    def test_signature_schema_version_agrees_across_cpp_python_and_sql(self):
        cpp_text = TYPES_H.read_text(encoding="utf-8")
        sql_text = DISPATCH_DB_SQL.read_text(encoding="utf-8")
        cpp_value = _parse_cpp_define(cpp_text, "GGML_HIP_SIGNATURE_SCHEMA_VERSION")
        sql_value = _parse_sql_schema_meta(sql_text, "signature_schema")
        self.assertEqual(
            cpp_value, dispatch_abi.SIGNATURE_SCHEMA_VERSION,
            "GGML_HIP_SIGNATURE_SCHEMA_VERSION (C++) and dispatch_abi."
            "SIGNATURE_SCHEMA_VERSION (Python) have drifted apart -- bump both "
            "together, in the same commit as any schema reinterpretation.",
        )
        self.assertEqual(
            sql_value, dispatch_abi.SIGNATURE_SCHEMA_VERSION,
            "sql/dispatch-db.sql's schema_meta 'signature_schema' seed row has "
            "drifted from dispatch_abi.SIGNATURE_SCHEMA_VERSION -- a freshly "
            "created dispatch_db would then bootstrap with a stale default.",
        )

    def test_hardware_schema_version_agrees_across_cpp_python_and_sql(self):
        cpp_text = TYPES_H.read_text(encoding="utf-8")
        sql_text = DISPATCH_DB_SQL.read_text(encoding="utf-8")
        cpp_value = _parse_cpp_define(cpp_text, "GGML_HIP_HARDWARE_SCHEMA_VERSION")
        sql_value = _parse_sql_schema_meta(sql_text, "hardware_schema")
        self.assertEqual(cpp_value, dispatch_abi.HARDWARE_SCHEMA_VERSION)
        self.assertEqual(sql_value, dispatch_abi.HARDWARE_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
