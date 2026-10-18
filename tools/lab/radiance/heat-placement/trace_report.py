#!/usr/bin/env python3
"""RAD06/RHP: critical-path report from the helpers' TRACE lines (R3_MOE_HELPER_TRACE=N), no profiler.

Each finished request a helper lane prints
    TRACE,lane,L,seq,S,layer,Y,rows,R,routes,K,local_us,A,wait_us,B,helper_gpu_us,C,enter,T
where, for one layer of one step,
    local_us      engine clock: publish kernel -> wait kernel entered = the engine's local work (and anything else
                  queued on its stream) between publishing the routes and needing the helpers' answer
    wait_us       engine clock: wait kernel entered -> this helper's completion seen. The wait kernel looks at the
                  helpers in order, so the second helper's figure includes the first's.
    helper_gpu_us helper clock: its lane's first kernel -> its done word. Different device, so never compared with the
                  engine figures except as a duration.
    enter         the wait kernel's entry tick; both helpers' lines of one layer carry the same value.

Lines are grouped by `enter`: one group is one layer of one step. exposed wait = the largest wait_us in the group. A
layer whose exposed wait is ~0 had its helpers hidden behind local work; a large one is on the critical path.

  python3 trace_report.py helper0.log helper1.log [...]      (all the logs of one cell)
"""
import collections
import re
import sys


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')


def main():
    groups = collections.defaultdict(list)
    for path in sys.argv[1:]:
        for line in open(path, errors='replace'):
            if not line.startswith('TRACE,'):
                continue
            f = line.strip().split(',')
            d = {f[i]: f[i + 1] for i in range(1, len(f) - 1, 2)}
            groups[int(d['enter'])].append(d)
    if not groups:
        sys.exit('no TRACE lines (set R3_MOE_HELPER_TRACE=N on the helpers)')
    rows = []
    for enter, ds in groups.items():
        large = any(int(d['rows']) > 8 for d in ds)
        rows.append({
            'large': large,
            'layer': int(ds[0]['layer']),
            'rows': int(ds[0]['rows']),
            'local': float(ds[0]['local_us']),
            'wait': max(float(d['wait_us']) for d in ds),
            'helper_max': max(float(d['helper_gpu_us']) for d in ds),
            'helpers': len(ds),
        })
    print('%d layer-steps (%d with both helpers in the group)' % (len(rows), sum(1 for r in rows if r['helpers'] > 1)))
    for name, sel in (('large lane (prefill)', [r for r in rows if r['large']]), ('small lane (decode)', [r for r in rows if not r['large']])):
        if not sel:
            continue
        w = [r['wait'] for r in sel]
        l = [r['local'] for r in sel]
        h = [r['helper_max'] for r in sel]
        tot = sum(w) + sum(l)
        print('%s: %d steps' % (name, len(sel)))
        print('   local_us         p50 %8.1f  p95 %8.1f  mean %8.1f' % (pct(l, .5), pct(l, .95), sum(l) / len(l)))
        print('   exposed wait_us  p50 %8.1f  p95 %8.1f  mean %8.1f' % (pct(w, .5), pct(w, .95), sum(w) / len(w)))
        print('   helper_gpu_us    p50 %8.1f  p95 %8.1f  mean %8.1f   (slower helper)' % (pct(h, .5), pct(h, .95), sum(h) / len(h)))
        print('   wait / (local + wait) = %.1f%%;  layers with wait < 5 us: %.1f%%' % (100.0 * sum(w) / tot, 100.0 * sum(1 for x in w if x < 5.0) / len(w)))
    by_layer = collections.defaultdict(list)
    for r in rows:
        if r['large']:
            by_layer[r['layer']].append(r['wait'])
    if by_layer:
        worst = sorted(by_layer.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))[:6]
        print('largest mean exposed wait, large lane: ' + ', '.join('L%d %.0f us' % (k, sum(v) / len(v)) for k, v in worst))


if __name__ == '__main__':
    main()
