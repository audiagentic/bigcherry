"""analyze_kld3.py DIR [CORPUS]: the replayed arms' KLD over documents.

For each arm: the all-position mean, tail and top-1; each document's mean; paired differences against the chunk-128 noise
reference and between the caches, with 95% intervals bootstrapped over DOCUMENTS (the unit that is independent) and, for
comparison, over positions (optimistic: neighbouring tokens share context); and how many documents each cache beats fp8 in.
Reads DIR/replay-<arm>.kld.json and its .rows (float32 x 4 per scored position, KLD first)."""
import json, sys
import numpy as np
D = sys.argv[1]
arms = [a for a in ('mbt128', 'bf16_i8', 'bf16_fp8', 'fp8') if __import__('os').path.exists(f'{D}/replay-{a}.kld.json')]
J = {a: json.load(open(f'{D}/replay-{a}.kld.json')) for a in arms}
# The result file lists documents alphabetically; the per-position rows follow the corpus's order.
CORPUS = sys.argv[2] if len(sys.argv) > 2 else '<dev-root>/tmp/bigcherry/vtp44-fn/corpus3.jsonl'
src = [json.loads(l)['source'] for l in open(CORPUS)]
assert set(src) == set(J[arms[0]]['by_source']), 'the corpus and the result disagree on the documents'
pos = [J[arms[0]]['by_source'][s]['positions'] for s in src]
edges = np.concatenate([[0], np.cumsum(pos)])
R = {a: np.fromfile(f'{D}/replay-{a}.kld.json.rows', dtype='<f4').reshape(-1, 4)[:, 0].astype(np.float64) for a in arms}
assert all(len(R[a]) == edges[-1] for a in arms), 'rows do not match the documents'
doc = {a: np.array([R[a][edges[i]:edges[i + 1]].sum() for i in range(len(src))]) for a in arms}   # per-document KLD sums
n = np.array(pos, dtype=np.float64)
print(f'{len(src)} documents, {int(n.sum())} scored positions; documents: ' + ', '.join(f'{s.replace("set3-","")}={p}' for s, p in zip(src, pos)))
print(f"\n{'arm':9s} {'mean':>8s} {'p99':>8s} {'p99.9':>8s} {'max':>7s} {'top1':>7s}")
for a in arms:
    k = J[a]['all']; q = k['kld']
    print(f"{a:9s} {q['mean']:8.5f} {q['p99']:8.4f} {q['p99_9']:8.4f} {q['max']:7.3f} {k['top1_agreement']:7.4f}")
print('\nper-document mean KLD:')
print(f"{'document':14s}{'pos':>7s} " + ' '.join(f'{a:>9s}' for a in arms))
for i, s in enumerate(src):
    print(f'{s.replace("set3-",""):14s}{int(n[i]):7d} ' + ' '.join(f'{doc[a][i] / n[i]:9.5f}' for a in arms))
rng = np.random.default_rng(0); B = 10000
didx = rng.integers(0, len(src), (B, len(src)))
def dci(x_a, x_b):   # paired, position-weighted mean over resampled documents
    num = (x_a[didx] - x_b[didx]).sum(1); den = n[didx].sum(1); m = num / den; return (x_a - x_b).sum() / n.sum(), np.percentile(m, [2.5, 97.5])
pidx = rng.integers(0, int(n.sum()), (2000, int(n.sum())))
def pci(a, b): x = R[a] - R[b]; return np.percentile(x[pidx].mean(1), [2.5, 97.5])
pairs = [(a, b) for a, b in (('bf16_i8', 'mbt128'), ('bf16_fp8', 'mbt128'), ('fp8', 'mbt128'), ('bf16_fp8', 'bf16_i8'), ('fp8', 'bf16_i8'), ('fp8', 'bf16_fp8')) if a in arms and b in arms]
print('\npaired differences (A - B): mean, 95% over documents, 95% over positions (optimistic)')
for a, b in pairs:
    m, (lo, hi) = dci(doc[a], doc[b]); plo, phi = pci(a, b)
    print(f'  {a:9s} - {b:9s} {m:+.5f}  docs [{lo:+.5f}, {hi:+.5f}] {"excl 0" if lo > 0 or hi < 0 else "incl 0"}   pos [{plo:+.5f}, {phi:+.5f}]')
if 'fp8' in arms:
    print('\ndocuments where the cache beats fp8 (lower mean KLD):')
    for a in arms:
        if a != 'fp8': print(f"  {a:9s} {int((doc[a] / n < doc['fp8'] / n).sum())} of {len(src)}")
