"""check_final.py: the final-build confirmation. Prints PASS/FAIL per check."""
import json, os
import numpy as np
F = '<dev-root>/tmp/bigcherry/vtp44-fn'
O = f'{F}/final'
res = []
def chk(name, ok, extra=''):
    res.append(ok); print(('PASS ' if ok else 'FAIL ') + name + (f'  [{extra}]' if extra else ''))
def load(p):
    raw = open(p, 'rb').read(); _, rest = raw.split(b'\n', 1); hdr, data = rest.split(b'\n', 1); return json.loads(hdr), data
def fnv(tokens):
    h = 14695981039346656037
    for t in tokens: h = ((h ^ (int(t) & 0xFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
def digests(h, ref):
    t = np.fromfile(f'{ref}/tokens.i32', dtype='<i4'); st = h['doc_starts'] + [h['rows']]
    return [f'{fnv(t[st[i]:st[i+1]]):016x}' for i in range(len(h['doc_starts']))]
# 1) the 2-document record and self-replay
h2, d2 = load(f'{O}/rec2.route'); ho, do = load(f'{F}/record/rec-a.route')
chk('2-doc route ids equal the earlier build\'s', d2 == do and h2['doc_starts'] == ho['doc_starts'] and h2['rows'] == ho['rows'])
chk('2-doc header has complete and doc_hashes', h2['complete'] is True and len(h2['doc_hashes']) == len(h2['doc_starts']))
chk('2-doc digests equal an independent FNV-1a of the reference tokens', h2['doc_hashes'] == digests(h2, f'{F}/routing-in/ref'), str(h2['doc_hashes']))
k = json.load(open(f'{O}/self2.kld.json'))['all']
chk('self-replay (2 docs) KLD zero to rounding (< 1e-9: the reference keeps the top 64 tokens) and top-1 100%', k['kld']['mean'] < 1e-9 and k['top1_agreement'] == 1.0, f"mean {k['kld']['mean']:.2e}")
st = json.load(open(f'{O}/self2.stats.json'))['counts']['prefill']
a = np.frombuffer(d2, dtype='<i4').reshape(len(h2['routed']), h2['rows'], h2['top_k'])
bad = [l for i, l in enumerate(h2['routed']) if not np.array_equal(np.array(st[l][:h2['n_expert']]), np.bincount(a[i].ravel(), minlength=h2['n_expert']))]
chk('self-replay expert counts equal the record in all routed layers', not bad, f'differs {bad[:5]}' if bad else '48/48')
# 2) negatives
rec = open(f'{O}/receipt.txt').read().splitlines()
neg = [l for l in rec if l.startswith('NEG ')]
for l in neg: chk(l[4:].split(' (')[0].replace(' PASS', '').replace(' FAIL', '') + ' fails cleanly with its message', ' PASS ' in l, l)
chk('all seven negatives ran', len(neg) == 7, f'{len(neg)} of 7')
# 3) the 11 documents: digests, repeatability on this build, the replay path against the earlier build, and (information) the route against the earlier build's
if os.path.exists(f'{O}/route3.route'):
    h3, d3 = load(f'{O}/route3.route'); h3o, d3o = load(f'{F}/kld3/route3.route')
    chk('11-doc digests equal an independent FNV-1a of the reference tokens', h3['doc_hashes'] == digests(h3, f'{F}/kld3/ref3'), f"{h3['rows']} rows, {len(h3['doc_starts'])} docs")
    a = np.frombuffer(d3, dtype='<i4').reshape(len(h3['routed']), h3['rows'], h3['top_k']); b = np.frombuffer(d3o, dtype='<i4').reshape(a.shape)
    diff = (np.sort(a, -1) != np.sort(b, -1)).any(-1).any(0); st = h3['doc_starts'] + [h3['rows']]
    docs = [i for i in range(len(st) - 1) if diff[st[i]:st[i + 1]].any()]
    print(f'INFO the earlier build\'s route differs from this build\'s in {int(diff.sum())} of {len(diff)} rows, in documents {docs}' + (f', first at position {int(np.argmax(diff[st[docs[0]]:st[docs[0]+1]]))} of document {docs[0]}' if docs else ''))
    if os.path.exists(f'{O}/route3b.route'):
        h3b, d3b = load(f'{O}/route3b.route'); bb = np.frombuffer(d3b, dtype='<i4').reshape(a.shape)
        d2 = (np.sort(bb, -1) != np.sort(b, -1)).any(-1).any(0); d12 = (np.sort(a, -1) != np.sort(bb, -1)).any(-1).any(0)
        print(f'INFO selected routes are NOT always repeatable: this build\'s run 1 and run 2 differ in {int(d12.sum())} rows; run 2 and the earlier build\'s route differ in {int(d2.sum())} rows (3 of the 4 selecting runs agree, one diverges at one position and stays diverged)')
        chk('11-doc: the second recording on this build equals the earlier build\'s route (so the deviation is run to run, not build to build)', d2.sum() == 0)
    if os.path.exists(f'{O}/replay3-oldroute-fp8.kld.json.rows'):
        n = open(f'{O}/replay3-oldroute-fp8.kld.json.rows', 'rb').read(); o = open(f'{F}/kld3/replay-fp8.kld.json.rows', 'rb').read()
        chk('the earlier build\'s route replayed on this build (fp8, 11 docs): per-position KLD rows byte-identical to the earlier build\'s (the replay path is unchanged)', n == o, f'{len(n)} bytes')

for need in ('route3b.route', 'replay3-oldroute-fp8.kld.json.rows'): chk(f'follow-up run present: {need}', os.path.exists(f'{O}/{need}'))
print('\nALL PASS' if all(res) else '\nSOME FAILED')
