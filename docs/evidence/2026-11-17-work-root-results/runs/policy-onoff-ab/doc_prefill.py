import json,time,sys,glob,urllib.request
port=sys.argv[1]; L=int(sys.argv[2]); offs=[int(x) for x in sys.argv[3].split(",")]
def call(p,n):
    b=json.dumps({"model":"x","prompt":p,"max_tokens":n,"temperature":0,"ignore_eos":True}).encode()
    t=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",b,{"Content-Type":"application/json"}),timeout=3600)); return time.time()-t,d["usage"]
blob="\n\n".join(open(f,errors="ignore").read() for f in sorted(glob.glob("<repo>/docs/**/*.md",recursive=True)))
chars=int(L*3.9)
for k in offs:
    off=(k*7919*L)%max(1,len(blob)-chars); seg=f"[doc {L} {k}] "+blob[off:off+chars]
    tp,u=call(seg,1); print(f"doc slice {k}: {u['prompt_tokens']} prompt tokens in {tp:.1f}s = {u['prompt_tokens']/tp:.0f} tok/s",flush=True)
