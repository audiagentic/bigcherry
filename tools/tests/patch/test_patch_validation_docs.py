"""Documentation-contract tests for the sole patch-validation policy."""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli.main import build_parser  # noqa: E402
from bigcherry.patch.gates import GateId, GateIntent, gate_applies  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "docs/reference/testing/PATCH_VALIDATION.md"
POINTER = ROOT / "docs/reference/patches/PATCH_VALIDATION.md"
REFERENCE_CHAIN = (
    CANONICAL,
    POINTER,
    ROOT / "docs/reference/patches/PATCH_AUTHORING.md",
    ROOT / "docs/reference/patches/PATCH_SYSTEM.md",
    ROOT / "docs/reference/README.md",
    ROOT / "docs/reference/testing/README.md",
    ROOT / "docs/reference/patches/PATCH_REFACTOR_RUNBOOK.md",
)


def _strip_fenced_markdown(source: str) -> str:
    """Remove fenced examples before interpreting headings, tables, or links."""
    output: list[str] = []
    fence_char: str | None = None
    for line in source.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_char is not None:
            if match and match.group(1)[0] == fence_char:
                fence_char = None
            continue
        if match:
            fence_char = match.group(1)[0]
            continue
        output.append(line)
    return "\n".join(output)


def _split_table_row(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return tuple(cell.strip() for cell in stripped.split("|"))


def _parse_gate_matrix(source: str) -> dict[str, tuple[str, ...]]:
    """Parse the one live gate matrix, ignoring examples in code fences."""
    clean = _strip_fenced_markdown(source)
    header = "| Gate | Applies to | Requirement | Authoritative implementation |"
    starts = [index for index, line in enumerate(clean.splitlines()) if line.strip() == header]
    if len(starts) != 1:
        raise AssertionError(f"expected one gate matrix header, found {len(starts)}")

    lines = clean.splitlines()
    rows: list[tuple[str, ...]] = []
    for line in lines[starts[0] + 1 :]:
        if not line.strip().startswith("|"):
            break
        cells = _split_table_row(line)
        if cells and set(cells) == {"---"}:
            continue
        if len(cells) == 4 and re.match(r"^G[0-7]\b", cells[0]):
            rows.append(cells)
    parsed: dict[str, tuple[str, ...]] = {}
    for row in rows:
        gate_id = re.match(r"^(G[0-7])\b", row[0])
        assert gate_id is not None
        if gate_id.group(1) in parsed:
            raise AssertionError(f"duplicate matrix row for {gate_id.group(1)}")
        parsed[gate_id.group(1)] = row
    return parsed


def _gate_section_ids(source: str) -> list[str]:
    clean = _strip_fenced_markdown(source)
    return [
        match.group(1)
        for line in clean.splitlines()
        if (match := re.match(r"^##\s+(G[0-7])\b", line))
    ]


def _patch_gates_intent_choices() -> tuple[str, ...]:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    patch_gates = subparsers.choices["patch-gates"]
    intent = next(action for action in patch_gates._actions if action.dest == "intent")
    return tuple(str(choice) for choice in intent.choices)


def _resolve_documented_symbol(reference: str) -> object:
    symbol = reference.removesuffix("()")
    module_name, attribute = symbol.rsplit(".", 1)
    module = importlib.import_module("bigcherry." + module_name)
    return getattr(module, attribute)


def _authority_references(authority_cell: str) -> tuple[str, ...]:
    """Extract callable references from the canonical matrix itself."""
    return tuple(
        reference
        for reference in re.findall(r"`([^`]+\(\))`", authority_cell)
        if reference.startswith(("patch.", "patch_admission."))
    )


def _relative_links(source: str) -> list[str]:
    clean = _strip_fenced_markdown(source)
    links: list[str] = []
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", clean):
        target = match.group(1).strip().strip("<>")
        target = target.split(None, 1)[0]
        parsed = urlsplit(target)
        if (
            not parsed.path
            or parsed.scheme
            or target.startswith(("//", "#"))
        ):
            continue
        links.append(unquote(parsed.path))
    return links


class PatchValidationDocumentationTests(unittest.TestCase):
    def test_gate_matrix_matches_live_applicability(self) -> None:
        matrix = _parse_gate_matrix(CANONICAL.read_text(encoding="utf-8"))
        self.assertEqual(set(matrix), {gate.value for gate in GateId})
        self.assertEqual(len(matrix), len(tuple(GateId)))
        for gate in GateId:
            applies = {
                intent.value
                for intent in GateIntent
                if gate_applies(gate, intent)
            }
            documented = {
                token.strip().lower()
                for token in matrix[gate.value][1].split(",")
            }
            self.assertEqual(documented, applies, gate.value)

    def test_gate_sections_are_unique_and_ordered(self) -> None:
        sections = _gate_section_ids(CANONICAL.read_text(encoding="utf-8"))
        self.assertEqual(sections, [f"G{index}" for index in range(8)])

    def test_matrix_names_resolvable_live_authorities(self) -> None:
        matrix = _parse_gate_matrix(CANONICAL.read_text(encoding="utf-8"))
        for gate_id, row in matrix.items():
            references = _authority_references(row[3])
            self.assertTrue(references, gate_id)
            for reference in references:
                self.assertTrue(callable(_resolve_documented_symbol(reference)), reference)
        self.assertIn(
            "patch.gates.evaluate_repository_lint_gates()",
            _authority_references(matrix["G1"][3]),
        )
        self.assertIn(
            "patch.gates._evaluate_lint_summary()",
            _authority_references(matrix["G1"][3]),
        )
        self.assertIn(
            "patch.gates._evaluate_lint_package()",
            _authority_references(matrix["G3"][3]),
        )

    def test_public_intents_match_the_real_cli_parser(self) -> None:
        document = _strip_fenced_markdown(CANONICAL.read_text(encoding="utf-8"))
        match = re.search(
            r"The public\s+`patch-gates`\s+intents are exactly\s+(.+?)\.",
            document,
        )
        self.assertIsNotNone(match)
        assert match is not None
        documented = tuple(re.findall(r"`([^`]+)`", match.group(1)))
        self.assertEqual(documented, _patch_gates_intent_choices())

    def test_fenced_examples_do_not_create_matrix_or_gate_sections(self) -> None:
        fixture = """```markdown
| Gate | Applies to | Requirement | Authoritative implementation |
| --- | --- | --- | --- |
| G0 fake | lint | fake | `missing.symbol()` |
## G0 fake
```
| Gate | Applies to | Requirement | Authoritative implementation |
| --- | --- | --- | --- |
| G0 Exact | author | exact | `patch.gates.evaluate_composition_gate()` |
"""
        self.assertEqual(set(_parse_gate_matrix(fixture)), {"G0"})
        self.assertEqual(_gate_section_ids(fixture), [])

    def test_relative_link_fixture_ignores_external_and_fragment_only(self) -> None:
        fixture = """```markdown
[missing](not-real.md)
```
[external](https://example.test/not-real.md)
[anchor](#local)
[canonical](../testing/PATCH_VALIDATION.md#G0)
"""
        self.assertEqual(_relative_links(fixture), ["../testing/PATCH_VALIDATION.md"])

    def test_reference_chain_links_resolve(self) -> None:
        for source_path in REFERENCE_CHAIN:
            self.assertTrue(source_path.exists(), source_path)
            for link in _relative_links(source_path.read_text(encoding="utf-8")):
                target = (source_path.parent / link).resolve()
                self.assertTrue(target.exists(), f"{source_path}: {link}")

        authoring = (ROOT / "docs/reference/patches/PATCH_AUTHORING.md").read_text(encoding="utf-8")
        self.assertIn("[`PATCH_VALIDATION.md`](PATCH_VALIDATION.md)", authoring)
        pointer = POINTER.read_text(encoding="utf-8")
        self.assertIn("[testing/PATCH_VALIDATION.md](../testing/PATCH_VALIDATION.md)", pointer)
        self.assertNotIn("| Gate | Applies to |", pointer)
        self.assertEqual(_gate_section_ids(pointer), [])


if __name__ == "__main__":
    unittest.main()
