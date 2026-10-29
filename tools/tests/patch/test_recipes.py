"""Pin management (recipes.py) and patch-module metadata.

recipes.py is now minimal: it only reads/rewrites the top-level
``pinned = "..."`` line. Patch selection lives entirely in the v2
``[source.*]``/``[patch-set.*]`` machinery, tested elsewhere.
"""

from __future__ import annotations


import pytest

from bigcherry.patch import patchset
from bigcherry import recipes
from bigcherry.core import engines, paths


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
    # every engine pins in the recipes file its own declaration names
    for engine in engines.names(paths.REPO_ROOT):
        assert recipes.pinned(engines.load(paths.REPO_ROOT, engine).recipes_path(paths.REPO_ROOT)), engine


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
        'STATE = "rejected"\n'
        'UPSTREAM = "abc1234def"\n',
        encoding="utf-8")
    info, = patchset.describe("llamacpp", tmp_path)
    assert (info.state, info.upstream) == ("rejected", "abc1234def")


def test_declared_defaults(tmp_path):
    (tmp_path / "0001_bare.py").write_text("PATCHES = []\n", encoding="utf-8")
    info, = patchset.describe("llamacpp", tmp_path)
    assert info.state == patchset.DEFAULT_STATE == "untested"
    assert info.upstream is None
    assert info.state_valid


def test_unrecognised_state_is_reported_not_coerced(tmp_path):
    (tmp_path / "0001_typo.py").write_text('STATE = "validted"\n', encoding="utf-8")
    info, = patchset.describe("llamacpp", tmp_path)
    assert info.state == "validted"
    assert not info.state_valid


def test_underscore_modules_are_ignored(tmp_path):
    (tmp_path / "_helper.py").write_text('STATE = "validated"\n', encoding="utf-8")
    assert patchset.describe("llamacpp", tmp_path) == []


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


def test_set_pinned_version_rewrites_only_that_line(tmp_path):
    path = write(tmp_path, '# radiance\npinned = "v1.13.0"\npinned-version = "1.9.0"\n')
    assert recipes.set_pinned_version("1.13.0", path) == "1.9.0"
    assert path.read_text(encoding="utf-8") == '# radiance\npinned = "v1.13.0"\npinned-version = "1.13.0"\n'


def test_set_pinned_version_refuses_a_file_without_it(tmp_path):
    with pytest.raises(recipes.RecipeError):
        recipes.set_pinned_version("1.13.0", write(tmp_path, 'pinned = "v1.13.0"\n'))


@pytest.mark.parametrize(("tag", "version"), [("v1.13.0", "1.13.0"), ("2.0", "2.0")])
def test_version_of_tag(tag, version):
    assert recipes.version_of_tag(tag) == version


@pytest.mark.parametrize("tag", ["b11552", "v1.13.0-rc1", "main"])
def test_version_of_tag_refuses_non_release_tags(tag):
    with pytest.raises(recipes.RecipeError):
        recipes.version_of_tag(tag)
