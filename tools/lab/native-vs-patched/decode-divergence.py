#!/usr/bin/env python3
"""Fixed-work plain-decode (no MTP) greedy divergence + speed check between two llama-server builds.

usage: decode-divergence.py OUTDIR MODEL NAME=BINARY [NAME=BINARY ...]
Runs each build sequentially on GPUs 0,1 (-sm tensor), temp 0, fixed prompts, n_predict fixed, n_probs=1.
Reports first divergent token index per prompt, max |logprob| delta over the common prefix, and predicted tok/s.
"""
import json, os, signal, subprocess, sys, time, urllib.request

PROMPTS = [
    "Write a detailed explanation of how a B-tree differs from a binary search tree, with examples.",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Summarise the causes and consequences of the French Revolution in five paragraphs.",
    "Translate to German and explain each grammatical choice: 'The quick brown fox jumps over the lazy dog.'",
]
N_PREDICT = 256
PORT = 18091


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def run_arm(binary, model, outdir, name):
    env = dict(os.environ, HIP_VISIBLE_DEVICES="0,1", ROCR_VISIBLE_DEVICES="0,1")
    log = open(os.path.join(outdir, f"{name}.server.log"), "w")
    proc = subprocess.Popen(
        [binary, "-m", model, "-sm", "tensor", "-ngl", "99", "--fit", "off", "-c", "8192",
         "--flash-attn", "on", "--ubatch-size", "512", "--batch-size", "2048", "--threads", "8",
         "--parallel", "1", "--port", str(PORT), "--host", "127.0.0.1"],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    raise SystemExit(f"{name}: server exited early")
                time.sleep(2)
        rows = []
        for rep in range(2):
            for i, p in enumerate(PROMPTS):
                r = post("/completion", {"prompt": p, "n_predict": N_PREDICT, "temperature": 0,
                                         "top_k": 1, "n_probs": 1, "cache_prompt": False, "seed": 1})
                toks = [(t["token"], t["logprob"] if "logprob" in t else t.get("logprob", 0.0))
                        for t in r["completion_probabilities"]]
                rows.append({"rep": rep, "prompt": i, "tokens": toks,
                             "tg_tps": r["timings"]["predicted_per_second"]})
        return rows
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    outdir, model, *arms = sys.argv[1:]
    os.makedirs(outdir, exist_ok=True)
    res = {}
    for a in arms:
        name, binary = a.split("=", 1)
        res[name] = run_arm(binary, model, outdir, name)
        json.dump(res[name], open(os.path.join(outdir, f"{name}.json"), "w"))
    names = list(res)
    base = names[0]
    for other in names[1:]:
        print(f"== {other} vs {base}")
        for b, o in zip(res[base], res[other]):
            n = min(len(b["tokens"]), len(o["tokens"]))
            first = next((k for k in range(n) if b["tokens"][k][0] != o["tokens"][k][0]), None)
            m = first if first is not None else n
            dmax = max((abs(b["tokens"][k][1] - o["tokens"][k][1]) for k in range(m)), default=0.0)
            print(f"rep{b['rep']} p{b['prompt']}: first_divergent_token={first} n={n} "
                  f"max|dlogprob| over common prefix={dmax:.5f} tps {b['tg_tps']:.2f} -> {o['tg_tps']:.2f}")
    for nm in names:
        v = [r["tg_tps"] for r in res[nm]]
        print(f"{nm}: mean tg tps {sum(v)/len(v):.2f}")


if __name__ == "__main__":
    main()
