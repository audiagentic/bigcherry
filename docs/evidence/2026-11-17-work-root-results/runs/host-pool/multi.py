import json,time,glob,threading,urllib.request,re,sys
r="<work-root>/sources/9f62d6758917777787fb5e86fd85b4d5"
code="\n\n".join(open(f,errors="ignore").read() for f in sorted(glob.glob(r+"/**/*.cpp",recursive=True)+glob.glob(r+"/**/*.h",recursive=True)))
docs="\n\n".join(open(f,errors="ignore").read() for f in sorted(glob.glob("<repo>/docs/**/*.md",recursive=True)))
def call(p,n):
    b=json.dumps({"model":"x","prompt":p,"max_tokens":n,"temperature":0,"ignore_eos":True}).encode()
    t=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18410/v1/completions",b,{"Content-Type":"application/json"}),timeout=7200)); return time.time()-t,d["usage"]
call("Hi",4)
# 8 sessions: 4 code + 4 docs, ~150K tokens each (~1.2M tokens in all, 80% of the cache), all at once
segs=[]
for k in range(8):
    blob=code if k%2==0 else docs; L=150000; c=int(L*(1.9 if k%2==0 else 3.9)); off=((k+1)*7919*L)%(len(blob)-c); segs.append(f"[s{k}] "+blob[off:off+c])
res=[None]*8
def run(i): res[i]=call(segs[i],200)
t=time.time(); th=[threading.Thread(target=run,args=(i,)) for i in range(8)]; [x.start() for x in th]; [x.join() for x in th]; dt=time.time()-t
pt=sum(x[1]["prompt_tokens"] for x in res); ct=sum(x[1]["completion_tokens"] for x in res)
print("8 sessions at once: %d prompt + %d new tokens in %.0f s = %.0f tok/s overall (prefill-dominated)"%(pt,ct,dt,(pt+ct)/dt),flush=True)
for i,x in enumerate(res): print(i,"%.0f s"%x[0],x[1]["prompt_tokens"])
