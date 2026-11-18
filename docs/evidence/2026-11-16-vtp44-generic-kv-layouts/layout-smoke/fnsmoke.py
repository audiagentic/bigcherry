"""fnsmoke.py URL OUT (smoke version of fnclient.py: 8k TTFT, one-stream decode, two 12k needle probes): Flash-Next KV-cache A/B on one running engine: TTFT at 8k and 48k, decode at 1 and 4 streams,
and a needle probe (a code word placed at 10/50/90% depth of a ~48k-token context, three codes per depth)."""
import json, sys, time, threading, urllib.request, statistics as st
URL, OUT = sys.argv[1], sys.argv[2]
text = open('<dev-root>/tmp/bigcherry/vtp44-fn/kld-docs.txt', encoding='utf-8', errors='ignore').read()
def post(path, body, t=600):
    r = urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=t)
    return json.load(r)
def tok(s): return post('/tokenize', {'prompt': s})['count']
def sized(n, head, off, tail=''):
    lo, hi = n, n * 8
    for _ in range(30):
        mid = (lo + hi) // 2; t = tok(head + text[off:off + mid] + tail)
        if abs(t - n) <= n * 0.01: break
        if t < n: lo = mid
        else: hi = mid
    return text[off:off + mid], t
def stream(prompt, max_tokens, ign=False):
    t0 = time.time(); times = []; first = []; out = []
    body = {'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0, 'stream': True}
    if ign: body['ignore_eos'] = True
    q = urllib.request.Request(URL + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(q, timeout=3600) as s:
        for line in s:
            line = line.decode().strip()
            if line.startswith('data:') and line != 'data: [DONE]':
                d = json.loads(line[5:])
                if d.get('choices') and not first: first.append(time.time())
                if d.get('choices') and d['choices'][0].get('text'): times.append(time.time()); out.append(d['choices'][0]['text'])
    return t0, (times or first), ''.join(out)
res = {}
def ttft(tag, n, off):
    body, t = sized(n, f'Request {tag}. Summarise the following.\n', off)
    t0, ts, _ = stream(f'Request {tag}. Summarise the following.\n' + body, 1)
    res[tag] = {'tokens': t, 'ttft_s': ts[0] - t0, 'prefill_tok_s': t / (ts[0] - t0)}; print(tag, res[tag], flush=True)
def decode(streams, max_tokens=256):
    out = [None] * streams
    def one(i): out[i] = stream(f'Request dec{streams}-{i}. ' + text[50000 + i * 3000:50000 + i * 3000 + 1800], max_tokens, True)
    th = [threading.Thread(target=one, args=(i,)) for i in range(streams)]
    for t in th: t.start()
    for t in th: t.join()
    first = min(o[0] for o in out); last = max(o[1][-1] for o in out)
    ntok = sum(len(o[1]) for o in out)
    gaps = [b - a for o in out for a, b in zip(o[1], o[1][1:])]
    res[f'decode{streams}'] = {'chunks': ntok, 'chunks_per_s': ntok / (last - first), 'gap_p50_ms': 1000 * st.median(gaps)}
    print(f'decode{streams}', res[f'decode{streams}'], flush=True)
def needle(ctx=48000):
    words = ['amber-falcon-3917', 'violet-harbor-5204', 'copper-meadow-8846', 'jade-lantern-2671', 'silver-orchid-7435',
             'crimson-anchor-1198', 'ivory-compass-6052', 'cobalt-thistle-9313', 'saffron-glacier-4780']
    ok = 0; rows = []; k = 0
    for depth in (0.1, 0.9):
        for rep in range(1):
            code = words[k]; k += 1
            fact = f'\nNote for the archive: the vault passphrase for room {k} is {code}. Remember it.\n'
            q = f'\n\nQuestion: What is the vault passphrase for room {k}? Answer with the passphrase only.'
            body, t = sized(ctx, '', 200000 + 90000 * rep, fact + q)
            cut = int(len(body) * depth)
            prompt = body[:cut] + fact + body[cut:] + q
            r = post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 48,
                     'temperature': 0, 'reasoning_effort': 'none'}, 1800)
            ans = r['choices'][0]['message'].get('content') or ''
            hit = code in ans; ok += hit
            rows.append({'depth': depth, 'rep': rep, 'tokens': t, 'hit': hit, 'answer': ans.strip()[:60]}); print(rows[-1], flush=True)
    res['needle'] = {'hits': ok, 'of': len(rows), 'rows': rows}
ttft('ttft8k', 8000, 800000)
decode(1)
needle(12000)
json.dump(res, open(OUT, 'w'), indent=1)
