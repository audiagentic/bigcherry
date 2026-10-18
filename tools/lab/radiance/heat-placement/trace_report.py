#!/usr/bin/env python3
"""RAD06/RHP: critical-path report from the helpers' TRACE lines (R3_MOE_HELPER_TRACE=N), no profiler.

Each finished request a helper lane prints
    TRACE,lane,L,seq,S,layer,Y,rows,R,routes,K,local_us,A,wait_us,B,helper_gpu_us,C,enter,T
where, for one layer of one step,
    local_us      engine clock: publish kernel -> wait kernel entered. This is the publish-to-wait window, not isolated
                  local-expert time: it also holds the rest of the publish path and anything else queued on the engine
                  stream before the wait. The helpers are published one after the other, so their windows differ.
    wait_us       engine clock: wait kernel entered -> this helper's completion seen. The wait kernel looks at the
                  helpers in order, so the second helper's figure includes the first's.
    helper_gpu_us helper clock: its lane's first kernel -> its done word (a duration; never compared with engine ticks).
    enter         the wait kernel's entry tick; both helpers' lines of one layer-step carry the same value.

Lines are grouped by `enter`: one group is one layer of one step. A PAIRED group has both helpers' lines, and its exposed
wait is the larger wait_us. A SINGLE group has one line: the other helper either had no routes that step (a genuine zero)
or its line was not printed (a sampling stride above 1 counts each helper by its own sequence). The two are reported
apart and never mixed; use R3_MOE_HELPER_TRACE=1 so a single group means an idle helper.

  python3 trace_report.py [--small-rows 8] helper0.log helper1.log [...]      (all the logs of one cell)
"""
import argparse
import collections
import sys


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')


def stats(name, v):
    return '   %-18s p50 %8.1f  p95 %8.1f  mean %8.1f  sum %12.0f  n %d' % (name, pct(v, .5), pct(v, .95), sum(v) / len(v), sum(v), len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--small-rows', type=int, default=8, help='rows at or below this take the small (decode) lane')
    ap.add_argument('logs', nargs='+')
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for path in a.logs:
        for line in open(path, errors='replace'):
            if not line.startswith('TRACE,'):
                continue
            f = line.strip().split(',')
            d = {f[i]: f[i + 1] for i in range(1, len(f) - 1, 2)}
            groups[int(d['enter'])].append(d)
    if not groups:
        sys.exit('no TRACE lines (set R3_MOE_HELPER_TRACE=N on the helpers)')
    steps = []
    for enter, ds in groups.items():
        rows = int(ds[0]['rows'])
        steps.append({
            'large': rows > a.small_rows,
            'paired': len(ds) > 1,
            'layer': int(ds[0]['layer']),
            'local_min': min(float(d['local_us']) for d in ds),
            'local_max': max(float(d['local_us']) for d in ds),
            'wait': max(float(d['wait_us']) for d in ds),
            'helper_max': max(float(d['helper_gpu_us']) for d in ds),
        })
    n_pair = sum(1 for s in steps if s['paired'])
    print('%d layer-steps: %d paired, %d single-helper (idle partner when the stride is 1)' % (len(steps), n_pair, len(steps) - n_pair))
    for lane, sel_all in (('large lane (prefill)', [s for s in steps if s['large']]),
                          ('small lane (decode)', [s for s in steps if not s['large']])):
        if not sel_all:
            continue
        print('%s: %d steps, %d paired, %d single' % (lane, len(sel_all), sum(1 for s in sel_all if s['paired']), sum(1 for s in sel_all if not s['paired'])))
        for kind, sel in (('paired', [s for s in sel_all if s['paired']]), ('single', [s for s in sel_all if not s['paired']])):
            if not sel:
                continue
            print('  [%s]' % kind)
            print(stats('publish->wait min', [s['local_min'] for s in sel]))
            print(stats('publish->wait max', [s['local_max'] for s in sel]))
            print(stats('exposed wait_us', [s['wait'] for s in sel]))
            print(stats('helper_gpu_us', [s['helper_max'] for s in sel]))
            w = [s['wait'] for s in sel]
            print('   waits under 10 us: %.1f%%' % (100.0 * sum(1 for x in w if x < 10.0) / len(w)))
        by_layer = collections.defaultdict(list)
        for s in sel_all:
            if s['paired']:
                by_layer[s['layer']].append(s['wait'])
        rank = sorted(by_layer.items(), key=lambda kv: -sum(kv[1]))[:8]
        print('  layers by TOTAL exposed wait (paired): ' + '; '.join(
            'L%d sum %.0f us (n %d, p50 %.0f, p95 %.0f)' % (k, sum(v), len(v), pct(v, .5), pct(v, .95)) for k, v in rank))
    print('Note: sum(exposed wait) is time on the critical path, not a share of the run; divide it by the run\'s wall time for that.')


if __name__ == '__main__':
    main()
