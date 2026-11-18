"""bootstrap_replay.py DIR: paired bootstrap over token positions of the replayed arms' mean KLD.

Reads DIR/replay-<arm>.kld.json.rows (float32 x 4 per scored position, KLD first; the run folder, not kept here) and
resamples positions 2,000 times. Positions are not independent (neighbouring tokens share context), so the intervals are
optimistic; one document pair, one run per arm."""
import sys
import numpy as np
d = sys.argv[1]
arms = ['mbt128', 'bf16_i8', 'bf16_fp8', 'fp8']
R = {t: np.fromfile(f'{d}/replay-{t}.kld.json.rows', dtype='<f4').reshape(-1, 4)[:, 0].astype(np.float64) for t in arms}
n = len(R['fp8'])
idx = np.random.default_rng(0).integers(0, n, (2000, n))
ci = lambda x: np.percentile(x[idx].mean(1), [2.5, 97.5])
for t in arms:
    lo, hi = ci(R[t]); print(f'{t:9s} {R[t].mean():.5f}  [{lo:.5f}, {hi:.5f}]')
for a, b in (('bf16_i8', 'mbt128'), ('bf16_fp8', 'mbt128'), ('fp8', 'mbt128'), ('bf16_fp8', 'bf16_i8'), ('fp8', 'bf16_i8'), ('fp8', 'bf16_fp8')):
    x = R[a] - R[b]; lo, hi = ci(x)
    print(f'{a:9s} - {b:9s} {x.mean():+.5f}  [{lo:+.5f}, {hi:+.5f}]', 'excludes 0' if lo > 0 or hi < 0 else 'includes 0')
