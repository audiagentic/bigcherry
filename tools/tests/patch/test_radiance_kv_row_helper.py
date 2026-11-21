"""RUM02: frozen-pin row/constraint/selection equivalence for the owned KV registry.

The immutable baseline was captured from the exact 2.2 source after the original
rad0049, rad0051, rad0052 and rad0053 r4d_rows.cpp FilePatch edits. The fixture
stores every row and constraint digest, all row names in declaration order, and
the complete chained-name membership. This test rebuilds the same full registry
from the *new* package definitions and proves semantic equivalence.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _vtp_promotion_support import load_patch_files
from bigcherry.patcher import apply_all
from test_vtp14_n1_rank_code_selection import radiance_checkout

BASELINE = Path(__file__).parent / "fixtures/rum02-kv-row-baseline.json"
PACKAGES = (
    "rad0049_qwen4exp_e4m3_kv",
    "rad0051_b16_kv_kernels",
    "rad0052_b16_kv_geometries",
    "rad0053_v4_kv_layouts",
)
ROWS = "libr4d/r4d_rows.cpp"
OWNED = "libr4d/bc_kv_rows.h"


def canonical_cpp(source: str) -> str:
    """Ignore only insignificant whitespace: quoted C++ strings retain bytes."""
    pieces = re.split(r'("(?:\\.|[^"\\])*")', source)
    return "".join(
        part if i % 2 else re.sub(r"\s+", "", part)
        for i, part in enumerate(pieces)
    )


def projection(expanded: str) -> dict:
    """Read the actual preprocessed C++ declarations (not a second row author)."""
    head = "static const RadKernelInfo kKernels[] = {"
    body = expanded.split(head, 1)[1]
    rows: dict[str, str] = {}
    order: list[str] = []
    depth = 1
    start = None
    quoted = escaped = False
    for index, char in enumerate(body):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            if depth == 1:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 1 and start is not None:
                row = canonical_cpp(body[start:index + 1])
                name = "".join(re.findall(r'"([^"]*)"', row.split(",", 1)[0]))
                if not name or name in rows:
                    raise AssertionError(f"missing/duplicate row: {row[:100]}")
                rows[name] = row
                order.append(name)
            elif depth == 0:
                break
    if depth != 0:
        raise AssertionError("unterminated kKernels")
    constraints = {}
    for match in re.finditer(
        r"\bstatic const RadConstraint ([A-Za-z_0-9]+)\[\] = \{(.*?)\};",
        expanded, re.DOTALL,
    ):
        name = match.group(1)
        if name in constraints:
            raise AssertionError(f"duplicate constraint {name}")
        constraints[name] = canonical_cpp(match.group(2)).rstrip(",")

    def chain_names(name: str) -> list[str]:
        fragment = expanded.split(f"static const char* const {name}[] = {{", 1)[1]
        entries = fragment.split("};", 1)[0].split(",")
        return ["".join(re.findall(r'"([^"]*)"', entry)) for entry in entries
                if '"' in entry]

    chained = chain_names("kChained")
    if "static const char* const kBcChained[]" in expanded:
        chained.extend(chain_names("kBcChained"))
    return {
        "rows": rows,
        "row_order": order,
        "constraints": constraints,
        "chain_names": sorted(set(chained)),
    }


def _row_selection(row: str) -> tuple[str, str, str | None]:
    fields = [
        "".join(re.findall(r'"([^"]*)"', field))
        for field in row[1:].split(",", 2)[:2]
    ]
    if len(fields) != 2 or not all(fields):
        raise AssertionError(f"row lacks an op: {row[:90]}")
    rule = re.search(r"RAD_DOMAIN_DEVICE,([^,]+),\((\w+)\)", row)
    if rule:
        return fields[1], rule.group(1), rule.group(2)
    priority = re.search(r"RAD_DOMAIN_DEVICE,([^,]+),", row)
    if priority:
        return fields[1], priority.group(1), None
    raise AssertionError(f"cannot inspect priority for {fields[0]}")


def _exclusion_rules(definition: str) -> dict[str, tuple[str, str]]:
    rules = {}
    for match in re.finditer(
        r'\{\("([^"]+)"\),RAD_C_(EQ|IN),([^,]+),([^}]+)\}', definition
    ):
        key, kind, value, literal = match.groups()
        # These are derived, not supplied, in the source op schemas; their
        # constraint values cannot disambiguate two rows at selection time.
        if key in {"block_size", "chunk"}:
            continue
        if kind == "IN":
            found = re.fullmatch(r'\("([^"]*)"\)', literal)
            if found:
                rules[key] = ("IN", found.group(1))
        else:
            number = re.fullmatch(r"\(longlong\)\((-?\d+)\)", value)
            if number:
                rules[key] = ("EQ", number.group(1))
    return rules


def definitely_exclusive(a: str, b: str) -> bool:
    """Fail closed unless one non-derived required key cannot match both."""
    left = _exclusion_rules(a)
    right = _exclusion_rules(b)
    for key in left.keys() & right.keys():
        lk, lv = left[key]
        rk, rv = right[key]
        if lk == rk == "EQ" and int(lv) != int(rv):
            return True
        if lk == rk == "IN" and not (
            set(lv.split("|")) & set(rv.split("|"))
        ):
            return True
        if lk == "EQ" and rk == "IN" and lv not in rv.split("|"):
            return True
        if lk == "IN" and rk == "EQ" and rv not in lv.split("|"):
            return True
    return False


class KvRowHelperTest(unittest.TestCase):
    def test_only_one_upstream_kv_attachment_and_explicit_shared_owner(self):
        for name in PACKAGES:
            file_patches = load_patch_files(name)
            source = [fp for fp in file_patches if fp.path == ROWS]
            if name == PACKAGES[0]:
                self.assertEqual(len(source), 1)
                self.assertEqual(len(source[0].edits), 1)
                self.assertEqual(source[0].edits[0].id, "bc-kv-row-registry-hook")
                created = [fp for fp in file_patches if fp.path == OWNED]
                self.assertEqual(len(created), 1)
                self.assertTrue(created[0].create)
            else:
                self.assertEqual(source, [], name)
                table = [fp for fp in file_patches if fp.path == OWNED]
                self.assertEqual(len(table), 1)
                self.assertFalse(table[0].create)

    def test_frozen_pin_same_rows_constraints_chains_and_tie_winners(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("Radiance frozen source not available")
        before = json.loads(BASELINE.read_text(encoding="utf-8"))
        pin = before["upstream_pin"]
        ref = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "--verify", f"{pin}^{{commit}}"],
            text=True,
        ).strip()
        self.assertEqual(pin, ref)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "libr4d").mkdir()
            source = subprocess.check_output(
                ["git", "-C", str(checkout), "show", f"{pin}:{ROWS}"]
            )
            (root / ROWS).write_bytes(source)
            for name in PACKAGES:
                parts = [
                    p for p in load_patch_files(name) if p.path in {ROWS, OWNED}
                ]
                self.assertTrue(parts)
                first = apply_all(parts, root)
                self.assertTrue(first and all(result.ok for result in first), (name, first))
                content = {p.path: (root / p.path).read_bytes() for p in parts}
                second = apply_all(parts, root)
                self.assertTrue(second and all(result.ok for result in second), (name, second))
                self.assertTrue(
                    all(edit.status == "already-applied"
                        for part in second for edit in part.results),
                    (name, second),
                )
                self.assertEqual(
                    content, {p.path: (root / p.path).read_bytes() for p in parts},
                    name,
                )
            command = [
                "c++", "-std=c++17", "-E", "-P",
                "-I", str(checkout / "libr4d"),
                "-I", str(checkout / "abi"), str(root / ROWS),
            ]
            expanded = subprocess.check_output(command, text=True)
        actual = projection(expanded)
        sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(
            {name: sha(row) for name, row in actual["rows"].items()},
            before["row_sha256"],
        )
        self.assertEqual(
            {name: sha(row) for name, row in actual["constraints"].items()},
            before["constraints_sha256"],
        )
        self.assertEqual(actual["chain_names"], before["chain_names"])
        self.assertCountEqual(actual["row_order"], before["row_order"])

        new_index = {name: i for i, name in enumerate(actual["row_order"])}
        metadata = {name: _row_selection(row) for name, row in actual["rows"].items()}
        changed_ties = 0
        for i, earlier in enumerate(before["row_order"]):
            for later in before["row_order"][i + 1:]:
                if new_index[earlier] < new_index[later]:
                    continue
                earlier_op, earlier_priority, earlier_constraint = metadata[earlier]
                later_op, later_priority, later_constraint = metadata[later]
                if earlier_op != later_op or earlier_priority != later_priority:
                    continue  # declaration position cannot affect this pair's winner
                self.assertIsNotNone(earlier_constraint, earlier)
                self.assertIsNotNone(later_constraint, later)
                self.assertTrue(
                    definitely_exclusive(
                        actual["constraints"][earlier_constraint],
                        actual["constraints"][later_constraint],
                    ),
                    f"reordering could change tied selection: {earlier} -> {later}",
                )
                changed_ties += 1
        self.assertGreater(changed_ties, 0)

    def test_hook_refuses_missing_or_ambiguous_upstream_anchor(self):
        import tempfile

        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("Radiance frozen source not available")
        hook = next(p for p in load_patch_files(PACKAGES[0]) if p.path == ROWS)
        pin = json.loads(BASELINE.read_text())["upstream_pin"]
        source = subprocess.check_output(
            ["git", "-C", str(checkout), "show", f"{pin}:{ROWS}"], text=True
        )
        anchor = "static const RadKernelInfo kKernels[] = {"
        self.assertEqual(source.count(anchor), 1)
        for corrupted in (
            source.replace(anchor, "static const RadKernelInfo kOther[] = {"),
            source + "\n" + anchor + "\n};\n",
        ):
            with (
                self.subTest(mode="missing" if anchor not in corrupted else "ambiguous"),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                (root / "libr4d").mkdir()
                target = root / ROWS
                target.write_text(corrupted)
                outcome = apply_all([hook], root)
                self.assertFalse(all(result.ok for result in outcome), outcome)
                self.assertEqual(target.read_text(), corrupted)


if __name__ == "__main__":
    unittest.main()
