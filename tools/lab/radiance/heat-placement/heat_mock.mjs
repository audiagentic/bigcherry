// RHP01-06 deterministic CPU-only design simulator. NOT Radiance GPU code.
// Run: node tools/lab/radiance/heat-placement/heat_mock.mjs
function need(ok, msg) { if (!ok) throw Error(msg); }
class BankRing {
  constructor(size=4) { need(size>0,'size'); this.size=size;this.rows=new Map();this.completed=new Set();this.done=new Set();this.produced=0;this.credited=0;this.dropped=0; }
  key(r){return [r.rank,r.layer,r.step].join(':');}
  slot(r){return [r.rank,r.layer,r.step%this.size].join(':');}
  issue(r) {
    need(r.step>=0&&r.layer>=0&&r.rank>=0&&['prefill','decode','mixed'].includes(r.phase),'id/phase');
    need(Array.isArray(r.counts)&&r.counts.length&&r.counts.every(n=>Number.isInteger(n)&&n>=0)&&r.reports>0,'counts');
    const k=this.key(r), s=this.slot(r), old=this.rows.get(s);
    need(!this.done.has(k),'already retired');
    if(old){need(old.step!==r.step,'duplicate report needs reports multiplicity');this.completed.delete(this.key(old));this.done.add(this.key(old));this.dropped++;}
    this.rows.set(s,{...r,counts:[...r.counts]});this.produced++;
  }
  complete(rank,layer,step){const r={rank,layer,step}, v=this.rows.get(this.slot(r));if(!v||v.step!==step)return false;this.completed.add(this.key(r));return true;}
  drain(beforeStep) {
    const out=[...this.rows.values()].filter(r=>r.step<beforeStep&&this.completed.has(this.key(r))).sort((a,b)=>a.step-b.step||a.rank-b.rank||a.layer-b.layer);
    for(const r of out){this.rows.delete(this.slot(r));this.completed.delete(this.key(r));this.done.add(this.key(r));this.credited++;}
    return out;
  }
  stats(){const pending=this.rows.size;need(this.produced===this.credited+this.dropped+pending,'accounting');return {produced:this.produced,credited:this.credited,dropped:this.dropped,pending};}
}
class Heat {
  constructor(decay=.98,rank=0){need(decay>0&&decay<=1,'decay');this.decay=decay;this.rank=rank;this.scores=new Map();this.totals=new Map();this.last=new Map();this.observed=0;this.ignored=0;}
  observe(r){
    if(r.rank!==this.rank){this.ignored++;return;}
    const group=r.phase+':'+r.layer,last=this.last.get(group)??-1;
    need(r.step>last,'reorder/duplicate');
    for(let e=0;e<r.counts.length;e++){const k=group+':'+e,n=r.counts[e]*r.reports;this.scores.set(k,(this.scores.get(k)||0)*this.decay+n);this.totals.set(k,(this.totals.get(k)||0)+n);}
    this.last.set(group,r.step);this.observed++;
  }
  sufficient(ring,threshold=.99){const s=ring.stats(),settled=s.credited+s.dropped;return this.observed>0&&s.pending===0&&settled>0&&s.credited/settled>=threshold;}
}
class Planner {
  constructor(experts,budgets,cost,setup={},phaseWeights={prefill:.5,decode:.5},churn=0){
    this.experts=experts;this.budgets=budgets;this.cost=cost;this.setup=setup;this.phaseWeights=phaseWeights;this.churn=churn;
    need(new Set(experts.map(x=>x.id)).size===experts.length,'duplicate expert');
  }
  valid(map){
    const used=Object.fromEntries(Object.keys(this.budgets).map(k=>[k,0]));
    if(Object.keys(map).length!==this.experts.length)return false;
    for(const e of this.experts){const d=map[e.id];if(!(d in used)||!e.allowed.includes(d))return false;used[d]+=e.bytes;}
    return Object.keys(used).every(d=>used[d]<=this.budgets[d]);
  }
  costOf(map,counts) {
    need(this.valid(map),'invalid map');let total=0;
    for(const [phase,weight] of Object.entries(this.phaseWeights))for(const layer of new Set(this.experts.map(e=>e.layer))){
      const duration=Object.fromEntries(Object.keys(this.budgets).map(k=>[k,0]));
      for(const e of this.experts){if(e.layer!==layer)continue;const d=map[e.id],n=counts[phase+':'+e.id]||0;
        if(n){if(duration[d]===0)duration[d]+=this.setup[d]||0;duration[d]+=n*this.cost[phase][d];}}
      total+=weight*Math.max(...Object.values(duration));
    }
    return total;
  }
  choose(current,counts,minGain=.05){
    const keys=this.experts.map(e=>e.id).sort(),base=this.costOf(current,counts);let best={...current},score=base;
    const candidates=[];
    for(const k of keys)for(const d of Object.keys(this.budgets).sort())if(d!==current[k])candidates.push({...current,[k]:d});
    for(let i=0;i<keys.length;i++)for(let j=i+1;j<keys.length;j++)if(current[keys[i]]!==current[keys[j]])candidates.push({...current,[keys[i]]:current[keys[j]],[keys[j]]:current[keys[i]]});
    for(const m of candidates){if(!this.valid(m))continue;const moves=keys.filter(k=>m[k]!==current[k]).length,price=this.costOf(m,counts)+moves*this.churn;
      if(price<score-1e-10){best=m;score=price;}}
    const gain=base? (base-score)/base:0;return gain>=minGain?{map:best,gain}:{map:{...current},gain:0};
  }
}
class Trigger {
  constructor(n=1000,seconds=1800){need(n>0&&seconds>0,'interval');this.n=n;this.seconds=seconds;this.lastCount=0;this.lastTime=0;this.busy=false;}
  due(count,time){need(count>=this.lastCount&&time>=this.lastTime,'clock/counter regression');if(this.busy)return null;return count-this.lastCount>=this.n?'requests':time-this.lastTime>=this.seconds?'time':null;}
  begin(count,time){const reason=this.due(count,time);if(reason)this.busy=true;return reason;}
  finish(count,time){need(this.busy,'not evaluating');need(count>=this.lastCount&&time>=this.lastTime,'regression');this.lastCount=count;this.lastTime=time;this.busy=false;}
}
class Epoch {
  constructor(map,participants=['engine','xtx0','xtx1']){this.active={...map};this.participants=new Set(participants);this.acks=new Set();this.gen=0;this.pending=null;this.inflight=0;this.rollbacks=0;}
  propose(map){need(this.pending===null,'epoch busy');this.pending={...map};this.acks.clear();}
  ack(p){need(this.pending!==null&&this.participants.has(p),'invalid ack');this.acks.add(p);}
  activate(){need(this.pending!==null&&this.inflight===0&&this.acks.size===this.participants.size,'not ready/drained');this.active=this.pending;this.pending=null;this.gen++;}
  abort(){need(this.pending!==null,'nothing to abort');this.pending=null;this.acks.clear();this.rollbacks++;}
}
// Hash is deliberately noncryptographic: tests only. RHP03 production requires SHA-256.
function checksum(o){let h=2166136261;for(const c of JSON.stringify(o))h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h.toString(16).padStart(8,'0');}
function makeSnapshot(map,gen,model,cohort){const body={schema:1,gen,model,cohort,map:Object.keys(map).sort().map(id=>[id,map[id]])};return {...body,check:checksum(body)};}
function readSnapshot(doc,model,cohort){const {check,...body}=doc;need(body.schema===1&&body.model===model&&body.cohort===cohort,'identity');need(check===checksum(body),'checksum');need(new Set(body.map.map(v=>v[0])).size===body.map.length,'duplicates');return Object.fromEntries(body.map);}
function runMockTests(){
  let count=0;const test=(name,fn)=>{try{fn();count++;}catch(e){throw Error(name+': '+e.message);}},eq=(a,b)=>need(JSON.stringify(a)===JSON.stringify(b),'unequal '+JSON.stringify(a)+' vs '+JSON.stringify(b)),err=fn=>{let failed=false;try{fn();}catch{failed=true;}need(failed,'expected rejection');};
  const r=(step,phase='prefill',rank=0,reports=1)=>({step,layer:20,rank,phase,counts:[0,3,0,1],reports});
  test('late event is not a stall',()=>{const c=new BankRing();c.issue(r(0));eq(c.drain(1),[]);c.complete(0,20,0);eq(c.drain(1).length,1);eq(c.stats().credited,1);});
  test('ring overwrite counted',()=>{const c=new BankRing(2);c.issue(r(0));c.complete(0,20,0);c.issue(r(2));eq(c.stats().dropped,1);need(!c.complete(0,20,0),'old completion');c.complete(0,20,2);eq(c.drain(3).map(x=>x.step),[2]);});
  test('out of order completion ordered',()=>{const c=new BankRing(4);for(let k=0;k<3;k++)c.issue(r(k));for(const k of [2,0,1])c.complete(0,20,k);eq(c.drain(3).map(x=>x.step),[0,1,2]);});
  test('unfinished current step excluded',()=>{const c=new BankRing();c.issue(r(7));c.complete(0,20,7);eq(c.drain(7).length,0);eq(c.drain(8).length,1);});
  test('double issue rejected',()=>{const c=new BankRing();c.issue(r(0));err(()=>c.issue(r(0)));});
  test('negative count rejected',()=>{const c=new BankRing();err(()=>c.issue({...r(0),counts:[-1]}));});
  test('decay and MTP weighting',()=>{const h=new Heat(.5);h.observe({...r(0),layer:2,counts:[2,0]});h.observe({...r(1),layer:2,counts:[0,3],reports:3});eq(h.scores.get('prefill:2:0'),1);eq(h.totals.get('prefill:2:1'),9);});
  test('phase separation',()=>{const h=new Heat();h.observe(r(0));h.observe(r(1,'decode'));need(h.scores.has('decode:20:1'),'decode');});
  test('TP mirror not doubled',()=>{const h=new Heat();h.observe(r(0));h.observe(r(0,'prefill',1));eq(h.observed,1);eq(h.ignored,1);});
  test('out of order heat fails closed',()=>{const h=new Heat();h.observe(r(3));err(()=>h.observe(r(2)));});
  test('low coverage fails closed',()=>{const c=new BankRing(1),h=new Heat();c.issue(r(0));c.issue(r(1));c.complete(0,20,1);for(const x of c.drain(2))h.observe(x);need(!h.sufficient(c),'should reject');});
  const experts=[{id:'0:0',layer:0,bytes:1,allowed:['engine','xtx0','xtx1']},{id:'0:1',layer:0,bytes:1,allowed:['engine','xtx0','xtx1']},{id:'0:2',layer:0,bytes:1,allowed:['engine']}],cur={'0:0':'engine','0:1':'xtx0','0:2':'engine'},cost={prefill:{engine:.5,xtx0:1,xtx1:1},decode:{engine:.5,xtx0:1,xtx1:1}},p=new Planner(experts,{engine:3,xtx0:2,xtx1:2},cost,{},{prefill:1,decode:0});
  test('critical path max not sum',()=>eq(p.costOf(cur,{'prefill:0:0':2,'prefill:0:1':2}),2));
  test('shadow rebalance overloaded helper',()=>{const v=p.choose(cur,{'prefill:0:0':1,'prefill:0:1':10});need(v.gain>.05,'gain');need(p.valid(v.map),'budget');});
  test('protected stays engine',()=>eq(p.choose(cur,{'prefill:0:2':100}).map['0:2'],'engine'));
  test('budget guard',()=>{const q=new Planner(experts,{engine:1,xtx0:1,xtx1:1},cost);need(!q.valid(cur),'invalid');err(()=>q.choose(cur,{}));});
  test('no heat no move',()=>eq(p.choose(cur,{}).gain,0));
  test('churn penalty',()=>{const q=new Planner(experts,{engine:3,xtx0:2,xtx1:2},cost,{},{prefill:1},999);eq(q.choose(cur,{'prefill:0:1':10}).gain,0);});
  test('request trigger and busy debounce',()=>{const t=new Trigger();eq(t.begin(999,100),null);eq(t.begin(1000,100),'requests');eq(t.begin(2000,200),null);t.finish(1000,100);eq(t.begin(1001,1900),'time');});
  test('trigger rejects clock regression',()=>{const t=new Trigger();err(()=>t.due(1,-1));});
  test('epoch waits all ACK',()=>{const e=new Epoch(cur);e.propose({'0:0':'xtx0'});e.ack('engine');err(()=>e.activate());e.ack('xtx0');e.ack('xtx1');e.inflight=1;err(()=>e.activate());e.inflight=0;e.activate();eq(e.gen,1);});
  test('epoch rollback leaves owner intact',()=>{const e=new Epoch(cur);e.propose({});e.abort();eq(e.active,cur);eq(e.rollbacks,1);});
  test('snapshot validates checksum and identity',()=>{const s=makeSnapshot(cur,2,'model','cohort');eq(readSnapshot(s,'model','cohort'),cur);err(()=>readSnapshot(s,'bad','cohort'));s.gen++;err(()=>readSnapshot(s,'model','cohort'));});
  test('snapshot duplicate owner rejected',()=>{const s=makeSnapshot(cur,1,'m','h');s.map.push(s.map[0]);s.check=checksum({schema:s.schema,gen:s.gen,model:s.model,cohort:s.cohort,map:s.map});err(()=>readSnapshot(s,'m','h'));});
  return {tests:count,failures:0,kind:'cpu-only deterministic design mock',scope:'NOT GPU or Radiance runtime validation'};
}
if(typeof process!=='undefined' && process?.argv?.[1]?.endsWith('heat_mock.mjs')) {
  process.stdout.write(JSON.stringify(runMockTests(),null,2)+'\n');
}
