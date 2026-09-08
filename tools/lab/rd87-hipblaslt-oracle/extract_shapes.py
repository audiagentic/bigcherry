"""Extract deduped real GEMM/MMVQ dispatch shapes + native timing from a
BigCherry tuning-campaign measurements.jsonl file, for RD87's offline
hipBLASLt oracle comparison.

Not a library -- ad hoc lab driver, run directly on the box holding the
campaign file (Brutus). See README.md.
"""
import json
import sys
from pathlib import Path

# ggml op ids observed in this campaign: 29 = GGML_OP_MUL_MAT, 100 = GGML_OP_MUL_MAT_ID
OP_NAMES = {29: "mul_mat", 100: "mul_mat_id"}
# ggml type ids observed: 0 = F32, 8 = Q8_0
TYPE_NAMES = {0: "f32", 8: "q8_0"}


def load_rows(path: Path):
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") == "result":
                yield row


def candidate_timing(row, name):
    for c in row.get("candidates", []):
        if c.get("name") == name and c.get("status") == "ok":
            return c.get("median_us")
    return None


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <measurements.jsonl> <out.json>", file=sys.stderr)
        raise SystemExit(2)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])

    seen = {}
    for row in load_rows(src):
        c = row["canonical"]
        op = c["op"]
        # ne0 = src0 dims [k, m, ...]; ne1 = src1 dims [k, n, ...]
        k = c["ne0"][0]
        m = c["ne0"][1]
        n = c["ne1"][1]
        src0_type = c["src0_type"]
        src1_type = c["src1_type"]
        dst_type = c["dst_type"]
        key = (op, k, m, n, src0_type, src1_type, dst_type)
        native_us = candidate_timing(row, row["native"])
        winner_us = candidate_timing(row, row["winner"])
        if native_us is None:
            continue
        entry = {
            "op": OP_NAMES.get(op, str(op)),
            "k": k,
            "m": m,
            "n": n,
            "src0_type": TYPE_NAMES.get(src0_type, str(src0_type)),
            "src1_type": TYPE_NAMES.get(src1_type, str(src1_type)),
            "dst_type": TYPE_NAMES.get(dst_type, str(dst_type)),
            "n_expert": c["n_expert"],
            "n_expert_used": c["n_expert_used"],
            "native_name": row["native"],
            "native_median_us": native_us,
            "bigcherry_winner_name": row["winner"],
            "bigcherry_winner_median_us": winner_us,
            "bigcherry_improvement_pct": row.get("improvement_pct"),
        }
        # keep the fastest-measured native sample if the same shape recurs
        if key not in seen or native_us < seen[key]["native_median_us"]:
            seen[key] = entry

    shapes = sorted(seen.values(), key=lambda e: (e["op"], -e["k"] * e["m"] * max(e["n"], 1)))
    out.write_text(json.dumps(shapes, indent=2))
    print(f"extracted {len(shapes)} unique shapes -> {out}")


if __name__ == "__main__":
    main()
