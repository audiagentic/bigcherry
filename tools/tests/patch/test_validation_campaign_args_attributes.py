"""PA43 GPT review (req_44b45662bd314ab4) blocker regression.

The RD08 CLI flags were retired but the staged legacy run() flow still read
``args.run_rd08_contract`` / ``args.run_rd08_lanes``; parsed main() args lack
those attributes, so a standard campaign raised AttributeError. This pins the
invariant structurally: every ``args.<name>`` read anywhere in the shared
campaign code must be a dest the real argparse parser defines.
"""

from __future__ import annotations

import argparse
import ast
import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from bigcherry.patch import validation_campaign as vc  # noqa: E402

_PATCH_DIR = TOOLS_DIR / "bigcherry" / "patch"
_SOURCES = (
    _PATCH_DIR / "validation_campaign.py",
    *sorted((_PATCH_DIR / "campaign").glob("*.py")),
)


def _real_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bigcherry patch-validation-campaign")
    vc._add_core_arguments(parser)
    vc._add_benchmark_and_producer_arguments(parser)
    return parser


def _args_attribute_reads(path: Path) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    reads: dict[str, int] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and isinstance(node.ctx, ast.Load)
        ):
            reads.setdefault(node.attr, node.lineno)
    return reads


class ArgsAttributesAreParserDestsTests(unittest.TestCase):
    def test_every_args_attribute_read_is_a_parser_dest(self) -> None:
        dests = {action.dest for action in _real_parser()._actions}
        undefined = {
            f"{path.name}:{line} args.{name}"
            for path in _SOURCES
            for name, line in _args_attribute_reads(path).items()
            if name not in dests
        }
        self.assertEqual(undefined, set())

    def test_minimal_standard_campaign_parses_without_retired_rd_flags(self) -> None:
        args = _real_parser().parse_args(
            [
                "--patch", "1202_rd04_bf16_flash_attn_tile",
                "--model", "m.gguf",
                "--hip-path", "hip",
                "--amdgpu-targets", "gfx1100",
                "--manifest", "manifest.json",
                "--workdir", "wd",
            ]
        )
        for retired in ("run_rd08_contract", "run_rd08_lanes", "run_rd73_contract", "rd73_corpus"):
            self.assertFalse(hasattr(args, retired), retired)

    def test_retired_rd_flags_are_rejected(self) -> None:
        for flag in ("--run-rd08-contract", "--run-rd08-lanes", "--run-rd73-contract"):
            with self.assertRaises(SystemExit):
                _real_parser().parse_args(
                    ["--patch", "p", "--hip-path", "h", "--workdir", "w", flag]
                )


if __name__ == "__main__":
    unittest.main()
