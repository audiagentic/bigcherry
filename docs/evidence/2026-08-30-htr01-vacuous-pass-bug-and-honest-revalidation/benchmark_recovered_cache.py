"""Real tg128 throughput comparison: native dispatch vs HTR01's recovered
cache (recovery-candidate.cache, the artifact from the successful third
real-hardware validation run), using the exact same HI141 regression
scenario (4096-token prefill then MTP-speculative decode) and production
launch flags already used throughout this session -- not a correctness
check this time, a real speed measurement.
"""
import sys
sys.path.insert(0, "/mnt/vault/development/llmhosts/bigcherry-hi143-e2e-20260829/tools")

import json
import time
import subprocess
import urllib.request
from pathlib import Path

BINARY_PATH = Path("/home/audumla/.cache/bigcherry/artifacts-store/builds/b6048afbd7787b0fc34982c237d81743/69f0be9b60f65f42cf172bb29b183c1e/llama-server")
RECOVERED_CACHE = Path("/mnt/vault/experiments/hi141-recovery-validation/recovery-candidate.cache")
MODEL = "/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf"
DEVICES = "0,1"
PROMPT_PATH = Path("/mnt/vault/development/llmhosts/bigcherry-hi143-e2e-20260829/tools/bigcherry/tuning/fixtures/hi141_qwen38_27b_mtp_4096_v1.txt")
PORT = 44029
REPS = 5

COMMON_ARGS = [
    "-ngl", "99", "-c", "64000",
    "-sm", "tensor", "--flash-attn", "on",
    "--ubatch-size", "512", "--batch-size", "2048",
    "--threads", "8", "--parallel", "1",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
    "-ctkd", "q8_0", "-ctvd", "q8_0",
]


def wait_ready(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server did not become ready")


def run_leg(label, dispatch_mode, cache_path=None):
    import os
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = DEVICES
    env["GGML_HIP_DISPATCH_MODE"] = dispatch_mode
    if cache_path is not None:
        env["GGML_HIP_DISPATCH_CACHE"] = str(cache_path)
    else:
        env.pop("GGML_HIP_DISPATCH_CACHE", None)
    cmd = [str(BINARY_PATH), "-m", MODEL, "--port", str(PORT), *COMMON_ARGS]
    log_path = Path(f"/mnt/vault/experiments/bench-{label}.log")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    tps_values = []
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        wait_ready()
        for rep in range(REPS):
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion",
                data=json.dumps({
                    "prompt": prompt, "n_predict": 128, "seed": 42, "temperature": 0.0,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.loads(r.read())
            timings = body.get("timings", {})
            tps = timings.get("predicted_per_second")
            draft_n = timings.get("draft_n")
            draft_n_accepted = timings.get("draft_n_accepted")
            print(f"  [{label}] rep={rep+1} predicted_per_second={tps} "
                  f"draft_n={draft_n} draft_n_accepted={draft_n_accepted} "
                  f"tokens_predicted={body.get('tokens_predicted')}", flush=True)
            if tps is not None:
                tps_values.append(tps)
    finally:
        shutdown_req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/shutdown", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(shutdown_req, timeout=10)
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    return tps_values


NATIVE_MEDIAN = 101.4586883388816  # already measured earlier this session, real, reproducible twice


def main():
    print("=== recovered cache (post-fix, honest run) ===")
    recovered_tps = run_leg("recovered-honest", "replay", cache_path=RECOVERED_CACHE)
    print(f"recovered median tps: {sorted(recovered_tps)[len(recovered_tps)//2] if recovered_tps else 'N/A'}")

    print("\n########## SUMMARY ##########")
    if recovered_tps:
        r_med = sorted(recovered_tps)[len(recovered_tps)//2]
        delta_pct = (r_med - NATIVE_MEDIAN) / NATIVE_MEDIAN * 100
        print(f"native:    {NATIVE_MEDIAN:.2f} tok/s")
        print(f"recovered: {r_med:.2f} tok/s")
        print(f"delta:     {delta_pct:+.2f}%")


if __name__ == "__main__":
    main()
