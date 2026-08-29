"""HI141 diagnostic: dump raw f1/f2 output arrays for native vs nw8:rpb1
(guilty) at the exact real production signature (ne1=[5120,4,1,1]) using the
throwaway BIGCHERRY_DUMP_ELEMENTS_DIR instrumentation, then compute
elementwise deltas to discriminate uniform FP reassociation (A2) from a
sparse/structured lane defect (A1), per GPT's explicit requirement before
closing A1/A2 on aggregate NMSE alone."""
import os
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/vault/development/llmhosts/bigcherry-hi143-e2e-20260829/tools")
from bigcherry.tuning import signature_mapping as sm

BINARY = Path("/mnt/vault/experiments/bigcherry-hi143-e2e-20260829/work/builds/f3e14c088e667bfc1c9d533422150c5d/2ed73c7aabfa96ef55b09e48d748bbfe/bin/test-backend-ops")
VENDOR_ROOT = Path("/mnt/vault/experiments/bigcherry-hi143-e2e-20260829/work/sources/7ed939cded44ad21629d167eeb4613a0")
DUMP_ROOT = Path("/mnt/vault/experiments/bigcherry-hi143-e2e-20260829/elementwise-dump")

SIGNATURE = {
    "dst_type": 0, "flags": 7, "fusion": 0, "glu_op": 0, "n_expert": 0, "n_expert_used": 0,
    "nb0": [32, 5120, 635699200, 635699200], "nb1": [1, 5120, 20480, 20480],
    "nbd": [1, 124160, 496640, 496640], "ne0": [5120, 124160, 1, 1], "ne1": [5120, 4, 1, 1],
    "ned": [124160, 4, 1, 1], "op": 29, "prec": 0, "schema_version": 2, "src0_type": 8, "src1_type": 0,
}

CANDIDATES = {
    "native": None,
    "nw1_rpb1_clean": "mmvq:q8_0:w4:nw1:rpb1:sk0:v1",
    "nw8_rpb1_guilty": "mmvq:q8_0:w4:nw8:rpb1:sk0:v1",
}
SEED = 1


def run_one(label, candidate_name, test_file_path):
    dump_dir = DUMP_ROOT / label
    dump_dir.mkdir(parents=True, exist_ok=True)
    for f in dump_dir.glob("*.bin"):
        f.unlink()
    env = dict(os.environ)
    env["BIGCHERRY_TEST_DETERMINISTIC_SEED"] = str(SEED)
    env["BIGCHERRY_DUMP_ELEMENTS_DIR"] = str(dump_dir)
    if candidate_name is None:
        env["GGML_HIP_DISPATCH_MODE"] = "native"
        env.pop("GGML_HIP_FORCE_CANDIDATE", None)
        env.pop("GGML_HIP_FORCE_CANDIDATE_STRICT", None)
    else:
        env["GGML_HIP_DISPATCH_MODE"] = "replay"
        env["GGML_HIP_FORCE_CANDIDATE"] = candidate_name
        env["GGML_HIP_FORCE_CANDIDATE_STRICT"] = "1"
    proc = subprocess.run(
        [str(BINARY), "test", "--test-file", str(test_file_path)],
        capture_output=True, text=True, env=env,
    )
    print(f"=== {label} === rc={proc.returncode}")
    for line in proc.stderr.splitlines():
        if "BIGCHERRY_CORRECTNESS_METRIC" in line or "BIGCHERRY_REF_DIGEST" in line:
            print(f"  {line}")
    f1_files = sorted(dump_dir.glob("*.f1.bin"))
    if not f1_files:
        print("  !! no dump file produced -- check patch/env wiring")
        return None
    return f1_files[0]


def load_floats(path):
    data = path.read_bytes()
    n = len(data) // 4
    return struct.unpack(f"<{n}f", data)


def main():
    test_file_line, target_tensor, digest_tensor = sm.signature_to_test_file_line(
        SIGNATURE, vendor_root=VENDOR_ROOT,
    )
    print(f"test-file line: {test_file_line}")
    print(f"target_tensor={target_tensor} digest_tensor={digest_tensor}\n")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(test_file_line + "\n")
        test_file_path = Path(handle.name)

    dumps = {}
    for label, candidate in CANDIDATES.items():
        f1_path = run_one(label, candidate, test_file_path)
        if f1_path is not None:
            dumps[label] = load_floats(f1_path)

    if "native" not in dumps:
        print("native dump missing, aborting comparison")
        return

    native = dumps["native"]
    for label, values in dumps.items():
        if label == "native":
            continue
        if len(values) != len(native):
            print(f"{label}: LENGTH MISMATCH native={len(native)} candidate={len(values)}")
            continue
        deltas = [abs(a - b) for a, b in zip(native, values)]
        nz = [(i, d) for i, d in enumerate(deltas) if d != 0.0]
        print(f"\n=== {label} vs native ===")
        print(f"  total elements: {len(deltas)}")
        print(f"  nonzero-delta elements: {len(nz)} ({100.0*len(nz)/len(deltas):.2f}%)")
        if nz:
            ds = sorted(d for _, d in nz)
            print(f"  delta min/median/max: {ds[0]:.3e} / {ds[len(ds)//2]:.3e} / {ds[-1]:.3e}")
            # index clustering check: are nonzero indices spread evenly or clumped?
            idxs = [i for i, _ in nz]
            print(f"  first 10 nonzero indices: {idxs[:10]}")
            print(f"  last 10 nonzero indices: {idxs[-10:]}")
            # row = index % 5120 (ne0[0]), col = index // 5120 (which of the 4 batch columns)
            rows = sorted(set(i % 5120 for i in idxs))
            cols = sorted(set(i // 5120 for i in idxs))
            print(f"  distinct columns (of 4) touched: {cols}")
            print(f"  distinct rows touched: {len(rows)} of 5120")


if __name__ == "__main__":
    main()
