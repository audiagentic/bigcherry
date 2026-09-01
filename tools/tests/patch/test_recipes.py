"""Pin management (recipes.py) and patch-module metadata.

recipes.py is now minimal: it only reads/rewrites the top-level
``pinned = "..."`` line. Patch selection lives entirely in the v2
``[source.*]``/``[patch-set.*]`` machinery, tested elsewhere.
"""

from __future__ import annotations


import pytest

from bigcherry.patch import patchset
from bigcherry import recipes


def write(tmp_path, body: str):
    path = tmp_path / "recipes.toml"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------- pinned

def test_pinned_reads_the_top_level_value(tmp_path):
    path = write(tmp_path, 'pinned = "b200"\n')
    assert recipes.pinned(path) == "b200"


@pytest.mark.parametrize("body, match", [
    ("", "pinned"),
    ('pinned = ""\n', "pinned"),
    ("not = [valid", "recipes.toml"),
])
def test_malformed_files_fail_loudly(tmp_path, body, match):
    with pytest.raises(recipes.RecipeError, match=match):
        recipes.pinned(write(tmp_path, body))


def test_missing_file_is_a_recipe_error_not_an_oserror(tmp_path):
    with pytest.raises(recipes.RecipeError):
        recipes.pinned(tmp_path / "absent.toml")


def test_shipped_pin_is_readable():
    assert recipes.pinned()


# ----------------------------------------------------------------- repin

def test_repin_rewrites_one_line_and_keeps_the_rest(tmp_path):
    path = write(tmp_path, """
# a comment worth keeping
pinned = "b100"
""")
    assert recipes.repin("b200", path) == "b100"
    text = path.read_text(encoding="utf-8")
    assert 'pinned = "b200"' in text
    assert "# a comment worth keeping" in text, "comments must survive"
    assert recipes.pinned(path) == "b200"


def test_repin_to_the_current_value_is_a_no_op(tmp_path):
    path = write(tmp_path, '\npinned = "b100"\n')
    before = path.read_text(encoding="utf-8")
    assert recipes.repin("b100", path) == "b100"
    assert path.read_text(encoding="utf-8") == before


# -------------------------------------------------------------- patchset

def test_metadata_is_read_without_importing(tmp_path):
    (tmp_path / "0001_x.py").write_text(
        'raise SystemExit("must not import")\n'
        'GROUP = "upstream-fixes"\n'
        'STATE = "rejected"\n'
        'UPSTREAM = "abc1234def"\n',
        encoding="utf-8")
    info, = patchset.describe(tmp_path)
    assert (info.group, info.state, info.upstream) == (
        "upstream-fixes", "rejected", "abc1234def")


def test_declared_defaults(tmp_path):
    (tmp_path / "0001_bare.py").write_text("PATCHES = []\n", encoding="utf-8")
    info, = patchset.describe(tmp_path)
    assert info.group == patchset.DEFAULT_GROUP
    assert info.state == patchset.DEFAULT_STATE == "untested"
    assert info.upstream is None
    assert info.state_valid


def test_unrecognised_state_is_reported_not_coerced(tmp_path):
    (tmp_path / "0001_typo.py").write_text('STATE = "validted"\n', encoding="utf-8")
    info, = patchset.describe(tmp_path)
    assert info.state == "validted"
    assert not info.state_valid


def test_underscore_modules_are_ignored(tmp_path):
    (tmp_path / "_helper.py").write_text('STATE = "validated"\n', encoding="utf-8")
    assert patchset.describe(tmp_path) == []


@pytest.mark.parametrize("arg, expected", [
    (None, None),
    ("", frozenset()),
    ("core", frozenset({"core"})),
    (" core , upstream-fixes ", frozenset({"core", "upstream-fixes"})),
])
def test_parse_filter(arg, expected):
    assert patchset.parse_filter(arg) == expected


def test_upstream_landed_is_unknown_outside_a_checkout(tmp_path):
    assert patchset.upstream_landed("abc1234", tmp_path) is None
