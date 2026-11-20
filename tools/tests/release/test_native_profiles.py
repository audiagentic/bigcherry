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


# ---- x-bigcherry: every config is one file; a variant declares what differs from its base ----------------------------

import pytest
import yaml


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def paths(document: dict) -> dict:
    """Every leaf of a config as {yaml path: value}; description, variant-of and differs are provenance, not settings."""
    leaves: dict = {}

    def walk(node, prefix):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else key)
        else:
            leaves[prefix] = node

    body = {k: v for k, v in document.items() if k != "x-bigcherry"}
    walk(body, "")
    walk({k: v for k, v in document["x-bigcherry"].items()
          if k not in ("description", "variant-of", "differs")}, "x-bigcherry")
    return leaves


@pytest.mark.parametrize("name", sorted(p.stem for p in CONFIGS.glob("*.yaml")))
def test_each_config_has_a_closed_x_bigcherry_section(name):
    section = load(name)["x-bigcherry"]
    assert section["schema"] == 1 and section["description"] and section["ranks"]
    assert set(section) <= {"schema", "description", "variant-of", "differs", "ranks", "helpers", "service", "memory",
                            "handoff", "helper", "kernels"}
    assert sum(r["count"] for r in section["ranks"]) == next(iter(load(name)["models"].values()))["tp"]
    assert ("variant-of" in section) == ("differs" in section)


@pytest.mark.parametrize("name", sorted(p.stem for p in CONFIGS.glob("*.yaml")))
def test_a_variant_differs_from_its_base_exactly_where_it_says(name):
    section = load(name)["x-bigcherry"]
    if "variant-of" not in section:
        return
    mine, base = paths(load(name)), paths(load(section["variant-of"]))
    actual = {k for k in mine.keys() | base.keys() if mine.get(k) != base.get(k) and k != "server.port"}
    declared = set(section["differs"])
    assert actual == declared, (name, sorted(actual ^ declared))
