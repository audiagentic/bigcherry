"""verify_replay.py: each replayed run's expert counts vs the record's histogram, and its KLD vs the same arm without replay."""
import json, os
import numpy as np
F = '<dev-root>/tmp/bigcherry/vtp44-fn'
raw = open(f'{F}/route-ref.route', 'rb').read(); _, rest = raw.split(b'\n', 1); hdr, data = rest.split(b'\n', 1); h = json.loads(hdr)
a = np.frombuffer(data, dtype='<i4').reshape(len(h['routed']), h['rows'], h['top_k'])
rec = [np.bincount(a[i].ravel(), minlength=h['n_expert']) for i in range(len(h['routed']))]
arms = [('replay-self', None, 'bf16 chunk 64 (reference config)'), ('replay-mbt128', 'routing2-keep/bf16-mbt128.kld.json', 'bf16 chunk 128'),
        ('replay-bf16_i8', 'routing2-keep/bf16_i8.kld.json', 'bf16_i8'), ('replay-bf16_fp8', 'routing2-keep/bf16_fp8.kld.json', 'bf16_fp8'),
        ('replay-fp8', 'routing2-keep/fp8.kld.json', 'fp8')]
print(f"{'arm':34s} {'counts==record':>15s} {'KLD replay':>11s} {'KLD without':>12s} {'top1 replay':>12s}")
for tag, base, label in arms:
    p = f'{F}/replay/{tag}.kld.json'
    if not os.path.exists(p) or not os.path.exists(f'{F}/replay/{tag}.stats.json'):
        print(f'{label:34s} (no result)'); continue
    k = json.load(open(p))['all']; st = json.load(open(f'{F}/replay/{tag}.stats.json'))
    cnt = st['counts']['prefill']
    bad = [l for i, l in enumerate(h['routed']) if not np.array_equal(np.array(cnt[l][:h['n_expert']]), rec[i])]
    kb = json.load(open(f'{F}/{base}'))['all']['kld']['mean'] if base else 0.0
    print(f"{label:34s} {'48/48' if not bad else 'differs in ' + str(bad[:6]):>15s} {k['kld']['mean']:11.5f} {kb:12.5f} {k['top1_agreement']:12.4f}")
