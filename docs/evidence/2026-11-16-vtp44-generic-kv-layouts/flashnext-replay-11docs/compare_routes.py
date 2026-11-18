"""compare_routes.py: the 11-document route recorded three times (the earlier build's run, and this build's run 1 and run 2). Layer-rows are
compared as expert SETS; the first differing position and layers are printed per document. Usage: compare_routes.py RUN1 RUN2 EARLIER"""
import json, sys
import numpy as np
def load(p):
    raw = open(p, 'rb').read(); _, rest = raw.split(b'\n', 1); h, data = rest.split(b'\n', 1); h = json.loads(h)
    return h, np.frombuffer(data, dtype='<i4').reshape(len(h['routed']), h['rows'], h['top_k'])
h1, a = load(sys.argv[1]); _, b = load(sys.argv[2]); _, o = load(sys.argv[3]); st = h1['doc_starts'] + [h1['rows']]
def cmp(x, y, name):
    d = (np.sort(x, -1) != np.sort(y, -1)).any(-1); rows = d.any(0)
    print(f'\n{name}: {int(d.sum())} of {d.size} layer-rows differ ({100 * d.mean():.4f}%), {int(rows.sum())} of {rows.size} token rows')
    for i in range(len(st) - 1):
        r = rows[st[i]:st[i + 1]]
        if r.any(): print(f'  doc {i:2d}: {int(r.sum()):6d} rows differ, first at position {int(np.argmax(r))}, in layers {np.nonzero(d[:, st[i] + int(np.argmax(r))])[0].tolist()}')
cmp(a, b, 'this build, run 1 vs run 2')
cmp(a, o, 'this build run 1 vs the earlier build')
cmp(b, o, 'this build run 2 vs the earlier build')
