#!/usr/bin/env python3
"""RAD06/RHP/RAD10: critical-path report from the helpers' TRACE lines (R3_MOE_HELPER_TRACE=N), no profiler.

Each finished request a helper lane prints
    TRACE,lane,L,seq,S,layer,Y,rows,R,routes,K,local_us,A,wait_us,B,helper_gpu_us,C,enter,T,wall_ns,W
where, for one layer of one step,
    local_us      engine clock: publish kernel -> wait kernel entered. This is the publish-to-wait window, not isolated
                  local-expert time: it also holds the rest of the publish path and anything else queued on the engine
                  stream before the wait. The helpers are published one after the other, so their windows differ.
    wait_us       engine clock: wait kernel entered -> this helper's completion seen. The wait kernel looks at the
                  helpers in order, so the second helper's figure includes the first's.
    helper_gpu_us helper clock: its lane's first kernel -> its done word (a duration; never compared with engine ticks).
    enter         the wait kernel's entry tick; both helpers' lines of one layer-step carry the same value.
    wall_ns       the helper host's epoch clock when the line was printed; it places a line in a benchmark round.

Lines are grouped by `enter`: one group is one layer of one step. A PAIRED group has both helpers' lines, and its exposed
wait is the larger wait_us. A SINGLE group has one line: the other helper either had no routes that step (a genuine zero)
or its line was not printed (a sampling stride above 1 counts each helper by its own sequence). The two are reported
apart and never mixed; use R3_MOE_HELPER_TRACE=1 so a single group means an idle helper.

  python3 trace_report.py [--small-rows 8] helper0.log helper1.log [...]      (all the logs of one cell)
  python3 trace_report.py --windows pair-001-x/stream-bench.json helper0.log helper1.log
      the same report per stream count: each round of a bigcherry stream-bench record is a window [wall_start_ns,
      wall_end_ns + slack] and a layer-step belongs to the window holding its newest line's wall_ns
"""
import argparse
import collections
import json
import sys


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')


def stats(name, v):
    return '   %-18s p50 %8.1f  p95 %8.1f  mean %8.1f  sum %12.0f  n %d' % (name, pct(v, .5), pct(v, .95), sum(v) / len(v), sum(v), len(v))


def read_steps(paths, small_rows):
    groups = collections.defaultdict(list)
    for path in paths:
        for line in open(path, errors='replace'):
            if not line.startswith('TRACE,'):
                continue
            f = line.strip().split(',')
            d = {f[i]: f[i + 1] for i in range(1, len(f) - 1, 2)}
            groups[int(d['enter'])].append(d)
    steps = []
    for enter, ds in groups.items():
        rows = int(ds[0]['rows'])
        steps.append({
            'large': rows > small_rows,
            'rows': rows,
            'paired': len(ds) > 1,
            'layer': int(ds[0]['layer']),
            'local_min': min(float(d['local_us']) for d in ds),
            'local_max': max(float(d['local_us']) for d in ds),
            'wait': max(float(d['wait_us']) for d in ds),
            'helper_max': max(float(d['helper_gpu_us']) for d in ds),
            'helper_min': min(float(d['helper_gpu_us']) for d in ds),
            'wall_ns': max((int(d['wall_ns']) for d in ds if 'wall_ns' in d), default=None),
        })
    return steps


def lane_report(steps, small_rows):
    n_pair = sum(1 for s in steps if s['paired'])
    print('%d layer-steps: %d paired, %d single-helper (idle partner when the stride is 1)' % (len(steps), n_pair, len(steps) - n_pair))
    for lane, sel_all in (('large lane (prefill)', [s for s in steps if s['large']]),
                          ('small lane (decode)', [s for s in steps if not s['large']])):
        if not sel_all:
            continue
        rows = [s['rows'] for s in sel_all]
        print('%s: %d steps, %d paired, %d single; rows a step p50 %d  p95 %d  max %d'
              % (lane, len(sel_all), sum(1 for s in sel_all if s['paired']), sum(1 for s in sel_all if not s['paired']),
                 pct(rows, .5), pct(rows, .95), max(rows)))
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


def window_summary(steps, rounds, slack_ns):
    """Per stream count: what the helpers saw during that count's rounds. Returns {streams: summary dict}."""
    out = {}
    for r in rounds:
        lo, hi = r['wall_start_ns'], r['wall_end_ns'] + slack_ns
        mine = [s for s in steps if s['wall_ns'] is not None and lo <= s['wall_ns'] <= hi]
        acc = out.setdefault(r['streams'], {'steps': [], 'rounds': 0})
        acc['steps'] += mine
        acc['rounds'] += 1
    result = {}
    for n, acc in sorted(out.items()):
        st = acc['steps']
        paired = [s for s in st if s['paired']]
        small = [s for s in st if not s['large']]
        result[n] = {
            'rounds': acc['rounds'], 'layer_steps': len(st),
            'rows_p50': pct([s['rows'] for s in st], .5), 'rows_p95': pct([s['rows'] for s in st], .95),
            'rows_max': max((s['rows'] for s in st), default=0),
            'small_lane_share': (len(small) / len(st)) if st else float('nan'),
            'exposed_wait_us_p50': pct([s['wait'] for s in paired], .5), 'exposed_wait_us_p95': pct([s['wait'] for s in paired], .95),
            'exposed_wait_us_sum_per_round': sum(s['wait'] for s in paired) / acc['rounds'],
            'helper_gpu_us_p50': pct([s['helper_max'] for s in paired], .5),
            'helper_imbalance_p50': pct([(s['helper_max'] - s['helper_min']) / s['helper_max'] for s in paired if s['helper_max'] > 0], .5),
            'single_helper_share': (len(st) - len(paired)) / len(st) if st else float('nan'),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--small-rows', type=int, default=8, help='rows at or below this take the small (decode) lane')
    ap.add_argument('--windows', help="a stream-bench record: report per stream count by its rounds' wall-clock windows")
    ap.add_argument('--slack-ms', type=float, default=300.0, help='trace lines are printed after the request; keep lines this long past a round')
    ap.add_argument('--json', action='store_true', help='print the --windows summary as JSON')
    ap.add_argument('logs', nargs='+')
    a = ap.parse_args()
    steps = read_steps(a.logs, a.small_rows)
    if not steps:
        sys.exit('no TRACE lines (set R3_MOE_HELPER_TRACE=N on the helpers)')
    if a.windows:
        rounds = json.load(open(a.windows))['rounds']
        if not any(s['wall_ns'] is not None for s in steps):
            sys.exit('the TRACE lines carry no wall_ns (a helper built before RAD10-02)')
        summary = window_summary(steps, rounds, int(a.slack_ms * 1e6))
        if a.json:
            json.dump(summary, sys.stdout, indent=1, sort_keys=True)
            print()
            return
        print('%-8s %6s %10s %8s %8s %8s %12s %12s %12s %10s' % (
            'streams', 'rounds', 'layersteps', 'rows p50', 'rows p95', 'small %', 'wait p50 us', 'wait p95 us', 'wait/round', 'imbal p50'))
        for n, s in summary.items():
            print('%-8d %6d %10d %8.0f %8.0f %8.1f %12.1f %12.1f %12.0f %10.2f' % (
                n, s['rounds'], s['layer_steps'], s['rows_p50'], s['rows_p95'], 100 * s['small_lane_share'],
                s['exposed_wait_us_p50'], s['exposed_wait_us_p95'], s['exposed_wait_us_sum_per_round'], s['helper_imbalance_p50']))
        print("Note: wait/round is the exposed helper wait summed over the round (prefill and decode together); compare it with the round's wall time.")
        return
    lane_report(steps, a.small_rows)
    print("Note: sum(exposed wait) is time on the critical path, not a share of the run; divide it by the run's wall time for that.")


if __name__ == '__main__':
    main()
