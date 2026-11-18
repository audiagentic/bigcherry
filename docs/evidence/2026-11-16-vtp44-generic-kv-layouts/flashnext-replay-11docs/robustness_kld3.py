"""robustness_kld3.py DIR [CORPUS]: the replayed arms' paired differences over documents, three ways: all documents, without the two
noisiest long ones (long-01, long-02: their KLD is dominated by ids, timestamps and hashes), and the short documents only.
Bootstrap over DOCUMENTS (10,000 resamples), position-weighted. Reads DIR/replay-<arm>.kld.json and its .rows."""
import json, sys
import numpy as np
D = sys.argv[1]; CORPUS = sys.argv[2] if len(sys.argv) > 2 else 'corpus3.jsonl'
arms = ['mbt128', 'bf16_i8', 'bf16_fp8', 'fp8']
src = [json.loads(l)['source'] for l in open(CORPUS)]
J = {a: json.load(open(f'{D}/replay-{a}.kld.json')) for a in arms}
pos = np.array([J['fp8']['by_source'][s]['positions'] for s in src], float); e = np.concatenate([[0], np.cumsum(pos)]).astype(int)
R = {a: np.fromfile(f'{D}/replay-{a}.kld.json.rows', dtype='<f4').reshape(-1, 4)[:, 0].astype(float) for a in arms}
doc = {a: np.array([R[a][e[i]:e[i + 1]].sum() for i in range(len(src))]) for a in arms}
rng = np.random.default_rng(1)
def run(keep, label):
    k = np.array([i for i, s in enumerate(src) if keep(s)]); n = pos[k]; idx = rng.integers(0, len(k), (10000, len(k)))
    print(f'\n{label}: {len(k)} documents, {int(n.sum())} positions; mean KLD', {a: round(float(doc[a][k].sum() / n.sum()), 5) for a in arms})
    for a, b in (('bf16_i8', 'mbt128'), ('bf16_fp8', 'mbt128'), ('fp8', 'mbt128'), ('bf16_fp8', 'bf16_i8'), ('fp8', 'bf16_i8'), ('fp8', 'bf16_fp8')):
        x = doc[a][k] - doc[b][k]; m = x.sum() / n.sum(); c = np.percentile(x[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
        print(f'  {a:9s}-{b:9s} {m:+.5f} [{c[0]:+.5f},{c[1]:+.5f}] {"excl 0" if c[0] > 0 or c[1] < 0 else "incl 0"}')
    for a in ('bf16_i8', 'bf16_fp8'): print(f'  {a} beats fp8 in {int((doc[a][k] / n < doc["fp8"][k] / n).sum())} of {len(k)} documents')
run(lambda s: True, 'all documents')
run(lambda s: s not in ('set3-long-01', 'set3-long-02'), 'without the two noisiest long documents')
run(lambda s: 'short' in s, 'the eight short documents only')
