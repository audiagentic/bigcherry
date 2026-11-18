"""verify_record.py DIR: check each .route against its .stats.json (histogram per layer), and compare runs."""
import json, os, sys
import numpy as np
D = sys.argv[1]

def load(tag):
    raw = open(os.path.join(D, tag + '.route'), 'rb').read()
    magic, rest = raw.split(b'\n', 1)
    hdr, data = rest.split(b'\n', 1)
    assert magic == b'RADROUTE1', magic
    h = json.loads(hdr)
    a = np.frombuffer(data, dtype='<i4')
    if h['complete']:
        a = a.reshape(len(h['routed']), h['rows'], h['top_k'])
    return h, a

res = {}
for tag in ('rec-a', 'rec-b', 'rec-mbt128'):
    if not os.path.exists(os.path.join(D, tag + '.route')):
        print(tag, 'no route file'); continue
    h, a = load(tag)
    st = json.load(open(os.path.join(D, tag + '.stats.json')))
    print(tag, {k: (h[k] if k != 'routed' else len(h[k])) for k in ('layers', 'routed', 'top_k', 'n_expert', 'rows', 'tp', 'max_tok', 'complete', 'why', 'doc_starts')})
    if not h['complete']:
        continue
    res[tag] = a
    cnt = st['counts']['prefill']
    bad = []
    for i, l in enumerate(h['routed']):
        hist = np.bincount(a[i].ravel(), minlength=h['n_expert'])
        want = np.array(cnt[l][:h['n_expert']] + [0] * max(0, h['n_expert'] - len(cnt[l])))
        if len(cnt[l]) != h['n_expert'] or not np.array_equal(hist, want):
            bad.append(l)
    rng = (a >= 0).all() and (a < h['n_expert']).all()
    dup = sum(len(set(r)) != len(r) for r in a[0])
    print('  layers whose recorded histogram != stats counts:', bad, '| ids in range', bool(rng), '| duplicate ids in a layer-0 row', dup,
          '| tokens', st['tokens']['prefill'])
if 'rec-a' in res and 'rec-b' in res:
    print('control pair identical:', np.array_equal(res['rec-a'], res['rec-b']))
if 'rec-a' in res and 'rec-mbt128' in res:
    a, b = res['rec-a'], res['rec-mbt128']
    sa, sb = np.sort(a, -1), np.sort(b, -1)
    rowdiff = (sa != sb).any(-1)
    print('chunk-128 vs chunk-64: token-layer rows with a different expert set: %d of %d (%.2f%%)' % (rowdiff.sum(), rowdiff.size, 100 * rowdiff.mean()))
