"""RSC02 step 2: independently rerun Python RSC03 on the same config/hardware fixture the native C++ suite uses.

Golden file has every checked-in Flash-Next/Qwen27B config, with actual BDF/VRAM topology
from tools/tests/campaign/test_rsc11_inventory.py and explicitly synthetic model measurements.
The C++ bc_resources_parity_test checks the *same* complete ledger, refusing drift.
No Python runs in bc-serve's production start path.
"""
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

from bigcherry.engine_adapters import radiance_layout
from bigcherry.engine_adapters import radiance_resources as rr

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "engines/radiance/components/serving/golden/rsc02_resource_parity.json"
CONFIGS = ROOT / "engines/radiance/configs"


def _layout(helpers):
    archs = radiance_layout.load_archs()
    devices = {h["bdf"]: SimpleNamespace(bdf=h["bdf"], arch=h["arch"],
                                        ordinal=i, vram_mib=h["vram_mib"])
               for i, h in enumerate(helpers)}
    helper_cards = [SimpleNamespace(index=i, arch=archs[h["arch"]])
                    for i, h in enumerate(helpers)]

    class Inventory:
        def by_bdf(self, bdf):
            return devices[bdf]

    return SimpleNamespace(helper_bdfs=tuple(h["bdf"] for h in helpers),
                           resolved=SimpleNamespace(helpers=tuple(helper_cards)),
                           inventory=Inventory())


def test_native_python_parity_all_checked_in_configs():
    fixture = json.loads(GOLDEN.read_text())
    configs = {p.stem for p in CONFIGS.glob("*.yaml")}   # one native YAML per named config
    assert {c["name"] for c in fixture["cases"]} == set(configs)
    assert all(sum(c["name"] == name for c in fixture["cases"]) == 2 for name in configs)
    assert len(fixture["cases"]) == 40
    assert {c["hardware_profile"] for c in fixture["cases"]} == {
        "test_rsc11_inventory.FACTS", "test_power_plan.layout.FACTS"
    }
    for case in fixture["cases"]:
        i = case["input"]
        helpers = i["helpers"]
        model = rr.ModelMemory(i["owner_map_digest"], i["embedding_bytes"],
                               i["unowned_expert_bytes"], i["staging_bytes"],
                               i["ngram_ram_bytes"], i["checkpoint_bytes"],
                               i["other_engine_bytes"], i["kv_bytes_per_token"])
        target = rr.Targets(i["context_tokens"], i["concurrent_sessions"], i["retained_tokens"],
                            i["margin_mib"] * rr.MIB, i["reserve_mib"] * rr.MIB)
        host = rr.HostLimits(i["available_mib"] * rr.MIB,
                             i["cgroup_room_mib"] * rr.MIB, i["pin_room_mib"] * rr.MIB)
        out = rr.derive(_layout(helpers), model, target, host,
                        worker_footprints_bytes={h["bdf"]: h["worker_bytes"] for h in helpers})
        assert out.explain() == case["expected"], case["name"]
