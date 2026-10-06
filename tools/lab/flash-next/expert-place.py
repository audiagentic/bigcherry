#!/usr/bin/env python3
"""MET04: reorder each layer's routed experts by usage, so an index-range split places them by usage.

1283 (BIGCHERRY_MOE_EP=1) gives each card a contiguous range of expert indices, the same in every layer. This tool
writes a copy of a GGUF (all its shards) in which every layer's experts are permuted so that those ranges hold what
is wanted: the layer's hot experts dealt across all cards in the --traffic ratio (hottest first, weighted round
robin, until a card's --caps are used), and the rest, coldest last, on the cards with room left.

With --caps 128,128,256 --traffic 1,1,1 and BIGCHERRY_MOE_EP_TS=128,128,256: the two XTXs hold a quarter of the
experts each, all of them hot, the R9700 holds its share of the hot ones and every cold one, and each card sees about
a third of the routed experts.

The permutation is lossless and changes no result beyond summation order: per layer the same permutation is applied
to the expert axis of ffn_gate_exps / ffn_up_exps / ffn_down_exps and to the router rows (ffn_gate_inp.weight), as
whole byte blocks inside the copied files; every other byte is untouched. Usage comes from a Strata-format profile
(`STRP`, as 1338's BIGCHERRY_MOE_CACHE_PROFILE_OUT writes): (layer, expert) pairs ranked by routing frequency.

The placement rule, which a load-time implementation (MET11) has to follow exactly: per layer, walk the experts from
most to least used (profile order; experts the profile does not rank follow in index order); each one goes to the card
with the highest credit among the cards that still have room, where every open card first gains its --traffic share
and the chosen card then loses the sum of the open cards' shares (ties to the lower card index). The new order is card
0's experts, then card 1's, ..., each hottest first.

Usage: expert-place.py --profile P --caps 128,128,256 --traffic 1,1,1 --out-dir DIR SHARD1.gguf [SHARD2.gguf ...]
       expert-place.py --profile P --caps 128,128,256 --traffic 1,1,1 --plan-only plan.json
"""
import argparse
import json
import re
import shutil
import struct
import sys
import types
from pathlib import Path

sys.modules.setdefault("yaml", types.ModuleType("yaml"))   # gguf-py imports it for metadata we do not use
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vendor/llama.cpp/gguf-py"))
from gguf.gguf_reader import GGUFReader  # noqa: E402

_EXPERT = re.compile(r"^blk\.(\d+)\.(ffn_(?:gate|up|down|gate_up)_exps\.weight|ffn_gate_inp\.weight)$")


def read_profile(path):
    blob = Path(path).read_bytes()
    if blob[:4] != b"STRP":
        raise SystemExit(f"{path}: not a Strata profile")
    _, n_layer, n_expert, _, n = struct.unpack_from("<5I", blob, 4)
    per_layer = [[] for _ in range(n_layer)]
    for i in range(n):
        layer, e = struct.unpack_from("<HH", blob, 24 + 4 * i)
        per_layer[layer].append(e)
    for layer, row in enumerate(per_layer):   # a profile may rank fewer pairs than exist: the rest are cold
        seen = set(row)
        row.extend(e for e in range(n_expert) if e not in seen)
        assert len(row) == n_expert and len(set(row)) == n_expert, f"layer {layer}: bad ranking"
    return n_layer, n_expert, per_layer


def place(ranked, caps, traffic):
    """new order of one layer's experts (old ids): card 0's experts, then card 1's, ..., hottest first on each card"""
    room, credit, cards = list(caps), [0.0] * len(caps), [[] for _ in caps]
    for e in ranked:
        open_cards = [d for d in range(len(caps)) if room[d] > 0]
        for d in open_cards:
            credit[d] += traffic[d]
        d = max(open_cards, key=lambda c: (credit[c], -c))   # smooth weighted round robin among cards with room
        credit[d] -= sum(traffic[c] for c in open_cards)
        cards[d].append(e)
        room[d] -= 1
    return [e for row in cards for e in row]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shards", nargs="*")
    ap.add_argument("--plan-only", metavar="JSON", help="write only the placement map (no model copy): the reference a load-time placement (MET11) must reproduce")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--caps", required=True, help="experts per card, in device order; must add up to the expert count")
    ap.add_argument("--traffic", required=True, help="share of the hot experts per card")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    caps = [int(x) for x in a.caps.split(",")]
    traffic = [float(x) for x in a.traffic.split(",")]
    n_layer, n_expert, ranked = read_profile(a.profile)
    if sum(caps) != n_expert or len(caps) != len(traffic) or min(traffic) <= 0:
        raise SystemExit(f"--caps must add up to {n_expert} and --traffic needs one positive share per card")
    order = [place(ranked[layer], caps, traffic) for layer in range(n_layer)]
    plan = {"profile": str(a.profile), "caps": caps, "traffic": traffic, "n_layer": n_layer, "n_expert": n_expert, "order": order}
    if a.plan_only:
        Path(a.plan_only).write_text(json.dumps(plan), encoding="utf-8")
        print(f"plan for {n_layer} layers x {n_expert} experts: caps {caps}, traffic {traffic}; map in {a.plan_only}")
        return
    if not a.shards or not a.out_dir:
        raise SystemExit("give the model shards and --out-dir, or --plan-only")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    for shard in a.shards:
        dst = out / Path(shard).name
        shutil.copyfile(shard, dst)
        reader = GGUFReader(str(dst))
        jobs = [(t.name, int(t.data_offset), int(t.n_bytes), int(t.shape[-1])) for t in reader.tensors if _EXPERT.match(t.name)]
        del reader
        with open(dst, "r+b") as f:
            for name, offset, n_bytes, last in jobs:
                layer = int(_EXPERT.match(name).group(1))
                if last != n_expert or n_bytes % n_expert:
                    raise SystemExit(f"{name}: last dimension {last} / {n_bytes} bytes is not {n_expert} equal expert blocks")
                block = n_bytes // n_expert
                f.seek(offset)
                old = f.read(n_bytes)
                new = b"".join(old[e * block:(e + 1) * block] for e in order[layer])
                f.seek(offset)
                f.write(new)
                f.seek(offset)
                assert f.read(n_bytes) == new, f"{name}: write did not read back"
                done.setdefault(layer, []).append(name.split(".", 2)[2])
        print(f"{dst.name}: {len(jobs)} tensors permuted", flush=True)
    missing = [layer for layer in range(n_layer) if len(done.get(layer, [])) < 4 and "ffn_gate_up_exps.weight" not in done.get(layer, [])]
    if missing:
        raise SystemExit(f"layers without the four expert tensors (gate, up, down, router): {missing[:8]}")
    (out / "expert-placement.json").write_text(json.dumps(plan), encoding="utf-8")
    print(f"placed {n_layer} layers x {n_expert} experts: caps {caps}, traffic {traffic}; map in {out / 'expert-placement.json'}")


if __name__ == "__main__":
    main()
