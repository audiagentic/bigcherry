"""Every named serving config has its native YAML, and the release ships exactly those."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from bigcherry.release import runtime_assets as assets

CONFIGS = Path(__file__).resolve().parents[3] / "engines/radiance/configs"


def config_names() -> set[str]:
    names: set[str] = set()
    for toml in ("flashnext.toml", "qwen27b.toml"):
        names |= set(tomllib.loads((CONFIGS / toml).read_text(encoding="utf-8"))["config"])
    return names


def test_every_named_config_has_a_native_yaml():
    yaml = {p.stem for p in CONFIGS.glob("*.yaml")}
    assert yaml == config_names()


def test_the_release_ships_exactly_the_config_yaml_files():
    assert set(assets.NATIVE_RELEASE_PROFILES) == {p.name for p in CONFIGS.glob("*.yaml")}
    assert list(assets.NATIVE_RELEASE_PROFILES) == sorted(assets.NATIVE_RELEASE_PROFILES)
    assert all(re.fullmatch(r"[a-z0-9-]+\.yaml", name) for name in assets.NATIVE_RELEASE_PROFILES)


def test_each_yaml_names_one_model_and_a_card_per_rank():
    for path in CONFIGS.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        tp = int(re.search(r"^    tp: (\d+)$", text, re.M).group(1))
        cards = re.search(r"^cards: \[(.*)\]$", text, re.M).group(1).split(", ")
        assert cards == [str(i) for i in range(tp)], path.name
        assert len(re.findall(r"^    model: ", text, re.M)) == 1, path.name


def test_the_install_side_allowlist_equals_the_packer_list():
    cpp = (CONFIGS.parents[0] / "components/serving/release_archive.cpp").read_text(encoding="utf-8")
    block = re.search(r"kNativeProfiles = \{(.*?)\};", cpp, re.S).group(1)
    assert re.findall(r'"([^"]+)"', block) == [n.removesuffix(".yaml") for n in assets.NATIVE_RELEASE_PROFILES]
