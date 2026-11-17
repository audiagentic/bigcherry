#!/bin/bash
# rad0013 (MTP draft/verify recorded-pass replay) validation: current main vs main+rad0013, four-GPU flashnext-4gpu, no live
# policy. Per arm: accept-bench five prompts (ms/step, acceptance, text_sha256) and the engine log's "recorded pass" count. ABAB.
D=<work-root>/runs/rad0013-validate
MAIN=<repo>
PATCHWT=<repo>/worktrees/patch/rad-mtp-pass-replay
ENVF=$MAIN/config/environment.local.toml
echo $$ > $D/run.pid; rm -f $D/done $D/status; LEASED=0; STARTED=""
until [ -f $D/go ]; do sleep 20; done   # set once bigcherry-2's timing build is done
build() { (cd $1 && nice -n 15 env BIGCHERRY_ENVIRONMENT=$ENVF PYTHONPATH=tools python3 -m bigcherry build --lane radiance:stock:linux-radiance > $D/$2.build.log 2>&1)
  local id=$(grep -oE 'ok build_plan_id=[0-9a-f]+' $D/$2.build.log | tail -1 | cut -d= -f2); ls -d <work-root>/builds/*/$id 2>/dev/null | head -1; }
BM=$(build $MAIN main); BP=$(build $PATCHWT patch)
echo "main build $BM" >> $D/status; echo "patch build $BP" >> $D/status
[ -x "$BM/bin/radiance" ] && [ -x "$BP/bin/radiance" ] || { echo "a build failed" >> $D/status; touch $D/done; exit 1; }
cd $PATCHWT; set --; export BIGCHERRY_ENVIRONMENT=$ENVF; source tools/env/bigcherry-env.sh >/dev/null 2>&1
LM="bash $MAIN/tools/lab/radiance/launch-flashnext-best.sh"; LP="bash $PATCHWT/tools/lab/radiance/launch-flashnext-best.sh"; L=$LP; STORE=$BC_WORK_ROOT/live-policy-calibration.toml
fin() { [ -n "$STARTED" ] && $L stop >> $D/final-stop.log 2>&1; [ -f $STORE.v13 ] && mv $STORE.v13 $STORE; [ $LEASED = 1 ] && agent-lease release gpu; touch $D/done; }
trap fin EXIT; trap 'exit 130' INT TERM HUP
until agent-lease acquire gpu --ttl 1h --purpose "rad0013 validation: MTP pass replay, main vs patch, decode step + identity"; do sleep 20; done; LEASED=1
[ -f $STORE ] && mv $STORE $STORE.v13
arm() { local tag=$1 b=$2; L=$3; STARTED=1
  $L start --build $b --port 18448 > $D/$tag.launch.log 2>&1 || { echo "$tag start failed" >> $D/status; $L stop >> $D/$tag.stop.log 2>&1; STARTED=""; return; }
  E=$(grep -oE "log [^ )]*engine-[^ )]*\.log" $D/$tag.launch.log | tail -1 | cut -d' ' -f2)
  PYTHONPATH=tools timeout 900 python3 -m bigcherry accept-bench --url http://127.0.0.1:18448 --output $D/$tag.json > $D/$tag.log 2>&1 || echo "$tag accept-bench failed" >> $D/status
  $L stop > $D/$tag.stop.log 2>&1; STARTED=""; cp "$E" $D/$tag.engine.log 2>/dev/null
  echo "$tag recorded_draft=$(grep -c 'recorded pass: draft pass' $D/$tag.engine.log 2>/dev/null) $(grep -oE '[0-9.]+ ms/step' $D/$tag.log | tr '\n' ' ')" >> $D/status; }
arm main1 $BM "$LM"; arm patch1 $BP "$LP"; arm main2 $BM "$LM"; arm patch2 $BP "$LP"
python3 - $D <<'PY' >> $D/status
import json,sys
d=sys.argv[1]
def k(t): return json.load(open(f"{d}/{t}.json"))["kinds"]
m,p=k("main1"),k("patch1")
same=all(m[x]["text_sha256"]==p[x]["text_sha256"] for x in m)
acc=all(m[x]["accepted"]==p[x]["accepted"] for x in m)
import subprocess,re
g=open("<repo>/worktrees/patch/rad-mtp-pass-replay/tools/lab/radiance/launch-flashnext-best.sh").read()
fn=g[g.index("mtp_replay_recorded() {"):g.index("mtp_replay_check || exit 95")]
for t in ("main1","patch1"):
    r=subprocess.run(["bash","-c",f'LOG="{d}/{t}.engine.log"\nstart_canary() {{ return 0; }}\n{fn}mtp_replay_check'],capture_output=True,text=True)
    print(f"guard on {t} engine log: rc={r.returncode}")
print("identity: text", "IDENTICAL" if same else "DIFFERENT", "| accepted", "IDENTICAL" if acc else "DIFFERENT")
PY
