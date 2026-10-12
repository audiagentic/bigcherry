#!/usr/bin/env python3
"""Summarise ROCprofiler-SDK CSV around Radiance's two-rank all-reduce.

The raw CSV remains authoritative. D2H->next-H2D is only an upper bound on host
fold/wait/wake; mapped pinned-memory relay can have no HIP copy records at all.
Usage: ar_phase_summary.py <rocprof-dir>
"""
import csv, glob, os, statistics, sys
from collections import defaultdict

def rows(root, suffix):
    out=[]
    for path in glob.glob(os.path.join(root, '**', '*'+suffix), recursive=True):
        try:
            with open(path, newline='', errors='replace') as f: out += list(csv.DictReader(f))
        except Exception as e: print('WARN', path, e)
    return out

def dur_ns(r):
    try: return int(r.get('End_Timestamp','0'))-int(r.get('Start_Timestamp','0'))
    except Exception: return 0

def values(rows_): return [dur_ns(r)/1000.0 for r in rows_ if dur_ns(r)>0]

def stat(name, rows_):
    v=sorted(values(rows_))
    if not v:
        print('%-28s calls=0' % name); return
    p50=v[len(v)//2]; p90=v[min(len(v)-1,int(len(v)*0.9))]
    print('%-28s calls=%7d total_ms=%9.3f p50_us=%8.2f p90_us=%8.2f' %
          (name,len(v),sum(v)/1000,p50,p90))

def main():
    if len(sys.argv)!=2: raise SystemExit(__doc__)
    root=sys.argv[1]
    kr=rows(root,'kernel_trace.csv'); cp=rows(root,'memory_copy_trace.csv'); hp=rows(root,'hip_api_trace.csv')
    ar=[r for r in kr if any(x in r.get('Kernel_Name','').lower() for x in
        ('ar_','all_reduce','allreduce','wht','had','rmsnorm_quant'))]
    d2h=[r for r in cp if 'DEVICE_TO_HOST' in r.get('Direction','')]
    h2d=[r for r in cp if 'HOST_TO_DEVICE' in r.get('Direction','')]
    waits=[r for r in hp if any(x in r.get('Function','') for x in
        ('Synchronize','Wait','Query','EventRecord','StreamWait'))]
    print('== raw event classes')
    stat('AR/quant kernels',ar); stat('device -> host copies',d2h); stat('host -> device copies',h2d); stat('HIP wait/sync API',waits)

    by=defaultdict(list)
    for r in ar: by[r.get('Kernel_Name','?')].append(r)
    print('\n== AR/quant kernel names')
    for name, rr in sorted(by.items(), key=lambda kv:-sum(dur_ns(x) for x in kv[1]))[:20]:
        v=values(rr)
        print('%7d %9.3f ms %8.2f us  %s' % (len(rr),sum(v)/1000,statistics.median(v) if v else 0,name[:180]))

    hs=sorted((int(r.get('Start_Timestamp','0') or 0),r) for r in h2d)
    gaps=[]
    for r in d2h:
        end=int(r.get('End_Timestamp','0') or 0)
        for start,_ in hs:
            if start < end: continue
            if start-end > 2_000_000: break
            gaps.append((start-end)/1000.0); break
    print('\n== host fold/wait upper-bound from D2H -> next H2D')
    if gaps:
        gaps.sort()
        print('pairs=%d p50_us=%.2f p90_us=%.2f mean_us=%.2f' %
              (len(gaps),gaps[len(gaps)//2],gaps[min(len(gaps)-1,int(len(gaps)*.9))],sum(gaps)/len(gaps)))
    else:
        print('NO_COPY_PAIRS: relay may use mapped pinned memory; inspect ar-source-map output')

    print('\n== interpretation')
    print("Compare p50 components with the engine's ~145 us ar_rmsnorm_quant_fp8 call.")
    print('copies dominate -> bytes/codec; D2H->H2D gap dominates -> host relay/wait;')
    print('AR/quant kernels dominate -> codec/fusion. Do not label an unmatched gap host fold')
    print('without the source map.')

if __name__=='__main__': main()
