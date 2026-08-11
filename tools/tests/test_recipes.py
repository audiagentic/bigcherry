"""Recipe resolution and patch-module metadata.

The cases here are the ones where a wrong answer is silent: a recipe that
selects more than it says, or a state filter that reads a typo as a valid
value. A build that takes the wrong patches still builds.
"""

from __future__ import annotations


import pytest

from bigcherry import patchset, recipes


def write(tmp_path, body: str):
    path = tmp_path / "recipes.toml"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------- recipes

def test_omitted_axis_means_all_and_empty_list_means_none(tmp_path):
    """The distinction `upstream` depends on.

    Reading `groups = []` as "no filter" would produce a fully patched tree
    under the one recipe whose whole purpose is to be unpatched.
    """
    path = write(tmp_path, """
        pinned = "b1"
        [recipe.everything]
        [recipe.nothing]
        groups = []
    """)
    loaded, _ = recipes.load(path)
    assert loaded["everything"].groups is None
    assert loaded["everything"].states is None
    assert loaded["nothing"].groups == frozenset()


def test_pinned_is_followed_or_frozen(tmp_path):
    path = write(tmp_path, """
        pinned = "b200"
        [recipe.current]
        ref = "pinned"
        [recipe.historical]
        ref = "b100"
        [recipe.implicit]
    """)
    loaded, pinned = recipes.load(path)
    assert pinned == "b200"
    assert loaded["current"].ref == "b200"
    assert loaded["current"].follows_pin
    assert loaded["historical"].ref == "b100"
    assert not loaded["historical"].follows_pin
    assert loaded["implicit"].ref == "b200"


def test_unknown_recipe_names_the_alternatives(tmp_path):
    path = write(tmp_path, '\npinned = "b1"\n[recipe.dev]\n')
    with pytest.raises(recipes.RecipeError, match="dev"):
        recipes.get("nope", path)


@pytest.mark.parametrize("body, match", [
    ('[recipe.dev]\n', "pinned"),
    ('pinned = ""\n[recipe.dev]\n', "pinned"),
    ('pinned = "b1"\n', "no recipes"),
    ('pinned = "b1"\n[recipe.dev]\ngroups = "core"\n', "list of strings"),
])
def test_malformed_files_fail_loudly(tmp_path, body, match):
    with pytest.raises(recipes.RecipeError, match=match):
        recipes.load(write(tmp_path, body))


def test_missing_file_is_a_recipe_error_not_an_oserror(tmp_path):
    with pytest.raises(recipes.RecipeError):
        recipes.load(tmp_path / "absent.toml")


def test_names_survives_an_unusable_file(tmp_path):
    """--help must render even when the recipe file is broken."""
    assert recipes.names(write(tmp_path, "not = [valid")) == []


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


# ------------------------------------------------- builds and platforms

BUILDS_AND_PLATFORMS = """
pinned = "b1"
[build.tune]
description = "measures"
options = { GGML_HIP_AUTOTUNE = "ON" }
[build.stock]
options = {}
[platform.linux]
targets = "gfx1100;gfx1201"
options = { GGML_HIP = "ON" }
c-compiler = "/opt/rocm/llvm/bin/clang"
[recipe.dev]
builds = ["tune", "stock"]
platform = "linux"
[recipe.bare]
"""


def test_builds_and_platforms_resolve(tmp_path):
    config = recipes.load_config(write(tmp_path, BUILDS_AND_PLATFORMS))
    assert config.build("tune").options == {"GGML_HIP_AUTOTUNE": "ON"}
    assert config.build("stock").options == {}
    dev = config.recipe("dev")
    assert dev.builds == ("tune", "stock")
    platform = config.platform_for(dev)
    assert platform.targets == "gfx1100;gfx1201"
    assert platform.c_compiler == "/opt/rocm/llvm/bin/clang"
    assert platform.cxx_compiler is None


def test_recipe_without_a_platform_says_so(tmp_path):
    config = recipes.load_config(write(tmp_path, BUILDS_AND_PLATFORMS))
    bare = config.recipe("bare")
    assert bare.builds == ()
    with pytest.raises(recipes.RecipeError, match="does not name a platform"):
        config.platform_for(bare)


@pytest.mark.parametrize("body, match", [
    ('pinned = "b1"\n[recipe.d]\nbuilds = ["nope"]\n', "unknown build"),
    ('pinned = "b1"\n[recipe.d]\nplatform = "nope"\n', "unknown platform"),
    ('pinned = "b1"\n[platform.p]\n[recipe.d]\n', "targets"),
])
def test_cross_references_are_checked_at_load(tmp_path, body, match):
    with pytest.raises(recipes.RecipeError, match=match):
        recipes.load_config(write(tmp_path, body))


def test_bool_options_are_rejected_with_the_right_spelling(tmp_path):
    body = 'pinned = "b1"\n[build.b]\noptions = { GGML_HIP = true }\n[recipe.d]\n'
    with pytest.raises(recipes.RecipeError, match='"ON"/"OFF"'):
        recipes.load_config(write(tmp_path, body))


# ----------------------------------------------------------------- repin

def test_repin_rewrites_one_line_and_keeps_the_rest(tmp_path):
    path = write(tmp_path, """
# a comment worth keeping
pinned = "b100"

[recipe.dev]
ref = "pinned"
[recipe.frozen]
ref = "b050"
""")
    assert recipes.repin("b200", path) == "b100"
    text = path.read_text(encoding="utf-8")
    assert 'pinned = "b200"' in text
    assert "# a comment worth keeping" in text, "comments must survive"

    loaded, pinned = recipes.load(path)
    assert pinned == "b200"
    assert loaded["dev"].ref == "b200", "followers move"
    assert loaded["frozen"].ref == "b050", "frozen recipes do not"


def test_repin_to_the_current_value_is_a_no_op(tmp_path):
    path = write(tmp_path, '\npinned = "b100"\n[recipe.dev]\n')
    before = path.read_text(encoding="utf-8")
    assert recipes.repin("b100", path) == "b100"
    assert path.read_text(encoding="utf-8") == before


# ------------------------------------------------- the comparison set

def test_default_recipes_separate_the_three_questions():
    """upstream / bigcherry-native / bigcherry answer different questions.
    Measuring only upstream against a tuned build conflates layer overhead
    with tuning win, so all three must be in the default set."""
    config = recipes.load_config()
    default = {n for n, r in config.recipes.items() if r.default}
    assert default == {"upstream", "bigcherry-native", "bigcherry"}
    assert config.recipe("upstream").groups == frozenset()
    assert config.build("native").variant_set == "inventory"
    assert config.recipe("bigcherry-native").builds == ("native",)


def test_default_flag_parses(tmp_path):
    config = recipes.load_config(write(tmp_path, """
pinned = "b1"
[recipe.a]
default = true
[recipe.b]
"""))
    assert config.recipe("a").default is True
    assert config.recipe("b").default is False


# ------------------------------------------------------- tree state key

def _recipe(**kw):
    base = dict(name="r", ref="b1", groups=None, states=None,
                follows_pin=False)
    return recipes.Recipe(**{**base, **kw})


def test_same_selection_same_key_regardless_of_build_or_platform():
    a = _recipe(name="a", states=frozenset({"validated"}),
                builds=("tune",), platform="linux")
    b = _recipe(name="b", states=frozenset({"validated"}),
                builds=("record", "replay"), platform="windows")
    key = recipes.tree_state_key
    assert key(a.ref, a.groups, a.states) == key(b.ref, b.groups, b.states)


def test_key_separates_the_things_that_do_change_the_tree():
    key = recipes.tree_state_key
    base = key("b1", None, frozenset({"validated"}))
    assert key("b2", None, frozenset({"validated"})) != base, "ref matters"
    assert key("b1", frozenset(), frozenset({"validated"})) != base, "groups matter"
    assert key("b1", None, frozenset({"untested"})) != base, "states matter"
    assert key("b1", None, None) != key("b1", frozenset(), frozenset())


def test_key_is_order_independent():
    key = recipes.tree_state_key
    assert (key("b1", frozenset({"a", "b"}), None)
            == key("b1", frozenset({"b", "a"}), None))


def test_default_set_needs_two_tree_states_not_three():
    config = recipes.load_config()
    default = [r for r in config.recipes.values() if r.default]
    keys = {recipes.tree_state_key(r.ref, r.groups, r.states) for r in default}
    assert len(default) == 3
    assert len(keys) == 2
    patched = recipes.tree_state_key(
        config.recipe("bigcherry").ref,
        config.recipe("bigcherry").groups,
        config.recipe("bigcherry").states)
    assert recipes.tree_state_key(
        config.recipe("bigcherry-native").ref,
        config.recipe("bigcherry-native").groups,
        config.recipe("bigcherry-native").states) == patched


# ------------------------------------------------------- pipeline builds

def test_variant_set_and_needs_are_read(tmp_path):
    config = recipes.load_config(write(tmp_path, """
pinned = "b1"
[build.record]
options = { GGML_HIP_AUTOTUNE_RECORD = "ON" }
variant-set = "inventory"
[build.tune]
variant-set = "workload-max"
needs = "inventory"
[build.plain]
[recipe.d]
"""))
    assert config.build("record").variant_set == "inventory"
    assert config.build("record").needs == "none"
    assert config.build("tune").needs == "inventory"
    assert config.build("plain").variant_set is None


def test_unknown_needs_is_rejected(tmp_path):
    body = 'pinned = "b1"\n[build.b]\nneeds = "magic"\n[recipe.d]\n'
    with pytest.raises(recipes.RecipeError, match="needs must be one of"):
        recipes.load_config(write(tmp_path, body))


def test_shipped_pipeline_stages_declare_their_inputs():
    config = recipes.load_config()
    assert config.build("record").needs == "none"
    assert config.build("record").variant_set == "inventory"
    assert config.build("tune").needs == "inventory"
    assert config.build("tune").variant_set == "workload-max"
    assert config.build("replay").needs == "inventory"
    assert set(config.recipe("release").builds) == {"record", "tune", "replay"}


# ----------------------------------------------------- the shipped config

def test_shipped_recipes_are_valid_and_states_are_recognised():
    loaded, pinned = recipes.load()
    assert pinned
    assert {"dev", "release", "upstream"} <= set(loaded)
    assert loaded["upstream"].groups == frozenset(), \
        "upstream must select no patches"
    assert loaded["release"].states == frozenset({"validated"})

    for info in patchset.describe():
        assert info.state_valid, f"{info.name} declares unknown state {info.state!r}"
