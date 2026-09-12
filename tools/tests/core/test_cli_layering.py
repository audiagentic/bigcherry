from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import check  # noqa: E402


class CliMainBackedgeTests(unittest.TestCase):
    def _write_module(self, source: str, *, relative: str = "cli/patch.py") -> tuple[Path, Path]:
        root = Path(tempfile.mkdtemp(prefix="bigcherry-cli-layering-"))
        self.addCleanup(lambda: _remove_tree(root))
        product_root = root / "bigcherry"
        path = product_root / relative
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return path, product_root

    def test_static_relative_and_dynamic_forms_are_detected(self) -> None:
        path, product_root = self._write_module(
            "\n".join(
                (
                    "import bigcherry.__main__",
                    "import bigcherry.__main__ as entrypoint",
                    "from bigcherry import __main__",
                    "from bigcherry.__main__ import cmd_apply",
                    "from .. import __main__",
                    "from ..__main__ import cmd_apply",
                    "import importlib; importlib.import_module('bigcherry.__main__')",
                    "from importlib import import_module; import_module('bigcherry.__main__')",
                    "__import__('bigcherry.__main__')",
                )
            )
            + "\n",
        )

        self.assertEqual(
            check._cli_main_backedge_lines(path, product_root),
            tuple(range(1, 10)),
        )

    def test_dynamic_aliases_relative_names_and_shadowing(self) -> None:
        path, product_root = self._write_module(
            "\n".join(
                (
                    "import importlib as il",
                    "from importlib import import_module as load",
                    "il.import_module('bigcherry.__main__')",
                    "load('bigcherry.__main__')",
                    "import importlib",
                    "importlib.import_module('.__main__', package='bigcherry')",
                    "from importlib import import_module as relative_load",
                    "relative_load('..__main__', package='bigcherry.cli')",
                    "from unrelated import import_module",
                    "import_module('bigcherry.__main__')",
                    "def local(import_module):",
                    "    return import_module('bigcherry.__main__')",
                )
            )
            + "\n",
        )

        self.assertEqual(
            check._cli_main_backedge_lines(path, product_root),
            (3, 4, 6, 8),
        )

    def test_nested_function_and_class_scope_resolution(self) -> None:
        path, product_root = self._write_module(
            "\n".join(
                (
                    "import importlib",
                    "def outer_shadow():",
                    "    importlib = object()",
                    "    def inner():",
                    "        return importlib.import_module('bigcherry.__main__')",
                    "def outer_alias():",
                    "    import importlib as il",
                    "    def inner():",
                    "        return il.import_module('bigcherry.__main__')",
                    "def outer_global():",
                    "    importlib = object()",
                    "    def inner():",
                    "        global importlib",
                    "        return importlib.import_module('bigcherry.__main__')",
                    "def outer_nonlocal():",
                    "    import importlib",
                    "    def inner():",
                    "        nonlocal importlib",
                    "        return importlib.import_module('bigcherry.__main__')",
                    "class UsesModuleGlobal:",
                    "    importlib.import_module('bigcherry.__main__')",
                    "class DoesNotLeakClassLocal:",
                    "    import importlib as il",
                    "    def method(self):",
                    "        return il.import_module('bigcherry.__main__')",
                )
            )
            + "\n",
        )

        self.assertEqual(
            check._cli_main_backedge_lines(path, product_root),
            (9, 14, 19, 21),
        )

    def test_comments_docstrings_and_unrelated_modules_are_ignored(self) -> None:
        path, product_root = self._write_module(
            '"""from .. import __main__"""\n'
            "# import bigcherry.__main__\n"
            "text = 'importlib.import_module(\\\"bigcherry.__main__\\\")'\n"
            "import some_package.__main__\n"
        )

        self.assertEqual(check._cli_main_backedge_lines(path, product_root), ())

    def test_hygiene_reports_backedge_and_exempts_entrypoint(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="bigcherry-cli-hygiene-"))
        self.addCleanup(lambda: _remove_tree(root))
        product_root = root / "tools" / "bigcherry"
        cli_path = product_root / "cli" / "patch.py"
        cli_path.parent.mkdir(parents=True)
        cli_path.write_text("from .. import __main__\n", encoding="utf-8")
        (product_root / "__main__.py").write_text(
            "from .. import __main__\n", encoding="utf-8"
        )

        findings = check.tooling_hygiene(root)
        backedges = [item for item in findings if item.code == "TR14.CLI_MAIN_BACKEDGE"]
        self.assertEqual(len(backedges), 1)
        self.assertEqual(backedges[0].path, "tools/bigcherry/cli/patch.py")

    def test_current_production_tree_has_no_backedges(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        findings = check.tooling_hygiene(repo_root)
        self.assertEqual(
            [item for item in findings if item.code == "TR14.CLI_MAIN_BACKEDGE"],
            [],
        )


def _remove_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()


if __name__ == "__main__":
    unittest.main()
