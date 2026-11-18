"""sensitivity_kld3.py DIR [CORPUS]: where the long documents' KLD comes from. For each long document and replayed arm: the first
256-token window whose mean KLD exceeds 0.02, the share of the document's KLD in its top 1% of positions, and the per-position
correlation between arms. Reads DIR/replay-<arm>.kld.json.rows (float32 x 4 per scored position, KLD first)."""
import json, sys
import numpy as np
D = sys.argv[1]; CORPUS = sys.argv[2] if len(sys.argv) > 2 else 'corpus3.jsonl'
src = [json.loads(l)['source'] for l in open(CORPUS)]
arms = ['mbt128', 'bf16_i8', 'bf16_fp8']
d = json.load(open(f'{D}/replay-mbt128.kld.json')); pos = [d['by_source'][s]['positions'] for s in src]; e = np.concatenate([[0], np.cumsum(pos)])
R = {a: np.fromfile(f'{D}/replay-{a}.kld.json.rows', dtype='<f4').reshape(-1, 4)[:, 0].astype(float) for a in arms}
for name in [s for s in src if 'long' in s]:
    i = src.index(name); sl = slice(e[i], e[i + 1]); x = {a: R[a][sl] for a in arms}
    print(f'\n{name}: {pos[i]} positions')
    for a in arms:
        c = np.convolve(x[a], np.ones(256) / 256, mode='valid'); w = np.nonzero(c > 0.02)[0]
        print(f'  {a:9s} first 256-token window above 0.02 starts at position {w[0] if len(w) else None}; share of KLD mass in the top 1% of positions: {np.sort(x[a])[::-1][:max(1, len(x[a]) // 100)].sum() / x[a].sum():.2f}')
    for a, b in (('mbt128', 'bf16_i8'), ('bf16_i8', 'bf16_fp8'), ('mbt128', 'bf16_fp8')):
        print(f'  per-position KLD correlation {a} vs {b}: {np.corrcoef(x[a], x[b])[0, 1]:.2f}')
