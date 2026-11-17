import json,time,sys,glob,statistics,urllib.request,re
U="http://127.0.0.1:18410"
C=("radiance:engine_steps_decode_total","radiance:engine_step_seconds_decode_total","radiance:draft_tokens_total","radiance:draft_accepted_total")
def metrics():
    t=urllib.request.urlopen(U+"/metrics",timeout=60).read().decode()
    out={}
    for line in t.splitlines():
        for c in C:
            if line.startswith(c+" ") or line.startswith(c+"{"):
                out[c]=out.get(c,0.0)+float(line.rsplit(" ",1)[1])
    return out
def call(p,n):
    b=json.dumps({"model":"x","prompt":p,"max_tokens":n,"temperature":0,"ignore_eos":True}).encode()
    t=time.time();d=json.load(urllib.request.urlopen(urllib.request.Request(U+"/v1/completions",b,{"Content-Type":"application/json"}),timeout=7200));return time.time()-t,d["usage"]
root="<repo>/docs"
blob="\n\n".join(open(f,errors="ignore").read() for f in sorted(glob.glob(root+"/**/*.md",recursive=True)))
call("Hello",8)
for L in [int(x) for x in sys.argv[1].split(",")]:
    chars=int(L*3.9); off=(L*7919)%max(1,len(blob)-chars) if len(blob)>chars else 0
    seg=blob[off:off+chars]
    if len(seg)<chars: seg=(seg+"\n\n"+blob)[:chars]
    seg=f"[doc {L}] "+seg
    tp,u=call(seg,1)
    ms=[];acc=[]
    for _ in range(2):
        m0=metrics(); call(seg,257); m1=metrics()
        d={c:m1.get(c,0)-m0.get(c,0) for c in C}
        ms.append(1000*d[C[1]]/max(1,d[C[0]])); acc.append(d[C[3]]/max(1,d[C[2]]))
    print("ctx %7d  prefill %6.2f s (%5.0f tok/s)  decode ms/step %6.2f (%s)  accept %.3f"%(u["prompt_tokens"],tp,u["prompt_tokens"]/tp,statistics.mean(ms),", ".join("%.2f"%x for x in ms),statistics.mean(acc)),flush=True)
