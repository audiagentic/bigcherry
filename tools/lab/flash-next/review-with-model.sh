#!/bin/bash
# Ask the local Flash-Next model to review code: start llama-server in the production layout (245760 context, f16 KV,
# three-card tensor split, MTP draft on the 6900 XT), send every prompt file in <prompt-dir> as one chat request, save
# the answer and the timings next to it, stop the server. Doubles as a real-workload measurement: each request is a
# long uncached prompt followed by a long generation.
# Usage: review-with-model.sh <llama-server> <out-dir> <prompt-dir>      env: MAX_TOKENS (6000), CTX (245760), UB (512)
#   <prompt-dir>/*.txt  : one review request each (plain text: instructions, then the code or diff).
#   <out-dir>/<name>.answer.md, <name>.timings.json, server.log
set -u
bin=$1 out=$2 prompts=$3
mkdir -p "$out"
model=${BC_MODEL:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
draft=${DRAFT:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf}
port=$((46000 + RANDOM % 1000))
export HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3
export BIGCHERRY_FEATURES=flashnext BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_DRAFT_VOCAB_N=65536
"$bin" -m "$model" -ngl 99 --fit off -c "${CTX:-245760}" -ub "${UB:-512}" -b "${UB:-512}" --flash-attn on --parallel 1 \
    --threads 16 -ot '^per_layer_token_embd\.weight$=CPU' -ot '^token_embd\.weight$=CPU' \
    -dev ROCm0,ROCm1,ROCm2 -devd ROCm3 -sm tensor -ts 0.31,0.27,0.42 \
    -md "$draft" --no-spec-draft-backend-sampling --spec-type draft-mtp --spec-draft-n-max 3 \
    -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 --allreduce cpu-root --port "$port" --host 127.0.0.1 > "$out/server.log" 2>&1 &
pid=$!
ok=0
for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && { ok=1; break; }
    kill -0 "$pid" 2> /dev/null || break
    sleep 2
done
if [ "$ok" != 1 ]; then
    echo "SERVER_FAILED"; grep -E " E |out of memory" "$out/server.log" | head -3
    kill "$pid" 2> /dev/null; wait "$pid" 2> /dev/null
    exit 1
fi
python3 - "$port" "$out" "$prompts" "${MAX_TOKENS:-6000}" <<'PY'
import glob, json, os, sys, urllib.request
port, out, prompts, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
system = ("You are reviewing C++ and Python changes to llama.cpp patches. Be concrete: cite the lines you mean, "
          "say what is wrong or risky and why, and say what you checked and could not check. "
          "Do not restate the code. Finish with a short verdict.")
for path in sorted(glob.glob(os.path.join(prompts, "*.txt"))):
    name = os.path.splitext(os.path.basename(path))[0]
    body = {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": open(path, errors="replace").read()}],
            "max_tokens": max_tokens, "temperature": 0.2, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    except Exception as exc:  # keep going: one failed request must not lose the others
        print(f"{name}: REQUEST_FAILED {exc}", flush=True)
        continue
    msg = r["choices"][0]["message"]
    open(os.path.join(out, name + ".answer.md"), "w").write(msg.get("content") or "")
    if msg.get("reasoning_content"):
        open(os.path.join(out, name + ".reasoning.md"), "w").write(msg["reasoning_content"])
    t = r.get("timings", {})
    json.dump({"timings": t, "usage": r.get("usage", {})}, open(os.path.join(out, name + ".timings.json"), "w"), indent=1)
    print(f"{name}: prompt {t.get('prompt_n')} tok at {t.get('prompt_per_second', 0):.1f} t/s, "
          f"generated {t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):.1f} t/s, "
          f"accepted {t.get('draft_n_accepted')}/{t.get('draft_n')}", flush=True)
PY
kill -INT "$pid" 2> /dev/null
for _ in $(seq 60); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
kill -9 "$pid" 2> /dev/null
exit 0
