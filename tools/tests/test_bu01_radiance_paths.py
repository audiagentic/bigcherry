"""Regression coverage for BU01 radiance path selection."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("changed_paths", ROOT / ".github/scripts/changed_paths.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def selected(paths):
    patterns = [
        "engines/radiance/**",
        "!engines/radiance/**/*.md",
        "!engines/radiance/**/*.mdx",
        "!engines/radiance/**/docs/**",
        "tools/bigcherry/build/**",
    ]
    return mod.first_match(paths, patterns) is not None


def test_radiance_markdown_and_docs_skip():
    assert not selected(["engines/radiance/README.md"])
    assert not selected(["engines/radiance/components/libr3/README.md"])
    assert not selected(["engines/radiance/components/docs/example.cpp"])
    assert not selected(["engines/radiance/docs/planning/BU01.md"])


def test_buildable_changes_still_run():
    for name in (
        "engines/radiance/components/libr3/CMakeLists.txt",
        "engines/radiance/components/libr3/src/kernel.hip",
        "engines/radiance/components/libr3/component.toml",
        "engines/radiance/patches/fix.patch",
        "tools/bigcherry/build/recipe.py",
    ):
        assert selected([name])
    assert selected(["engines/radiance/README.md", "engines/radiance/patches/fix.patch"])


def test_exclusions_win_regardless_of_rule_order():
    rules = ["x/**", "!x/**/*.md", "x/important.md"]
    assert mod.first_match(["x/important.md"], rules) is None
    assert mod.first_match(["x/other.md"], rules) is None
    assert mod.first_match(["x/src.cpp"], rules) == "x/src.cpp"
