import json, glob, os, statistics, sys
D='<dev-root>/tmp/bigcherry/eng06-hw'
SIZES=(512,2048,8003,30000,60000,100366)
def launch(path):
    out={'ttft':{}, 'decode':{}}
    for n in SIZES:
        f=f'{path}/ttft-{n}.json'
        if not os.path.exists(f): continue
        rd=json.load(open(f))['rounds'][0]; ps=rd['per_stream'][0]
        out['ttft'][n]=(ps['ttft_ms']/1000.0, ps['prompt_tokens'])
    f=f'{path}/decode.json'
    if os.path.exists(f):
        d=json.load(open(f))
        for s in (1,4,6,8):
            rs=[r for r in d['rounds'] if r['streams']==s and r['ok']]
            if not rs: continue
            ch=[]; ms=[]; tps=[]; tok=[]
            for r in rs:
                ev=sum(p['events'] for p in r['per_stream'])
                win=r['wall_s']-max(p['ttft_ms'] for p in r['per_stream'])/1000.0
                ch.append(ev/win)
                e=r.get('engine') or {}
                ms.append(e.get('decode_ms_per_step')); tok.append(e.get('decode_tokens_per_step'))
                tps.append(r['aggregate']['aggregate_tps'])
            out['decode'][s]={'chunks_per_s':statistics.mean(ch),'ms_per_step':statistics.mean(m for m in ms if m),'tok_per_step':statistics.mean(t for t in tok if t),'agg_tps':statistics.mean(tps),'reps':len(rs)}
    return out
if __name__=='__main__':
    for p in sorted(glob.glob(f'{D}/t*-*/')):
        r=launch(p.rstrip('/')); print(os.path.basename(p.rstrip('/')))
        print('  ttft s:', {n:round(v[0],3) for n,v in r['ttft'].items()}, ' tokens:', {n:v[1] for n,v in r['ttft'].items()})
        print('  decode:', {s:{k:round(x,2) for k,x in v.items() if k!='reps'} for s,v in r['decode'].items()})
