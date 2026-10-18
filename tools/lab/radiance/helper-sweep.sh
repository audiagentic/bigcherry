#!/bin/bash
# RAD06: replay sweep - build the branch once, then ONE r3_moe_helper process (one model load, one expert allocation)
# replays every captured request once per variant; rocprofv3's kernel trace gives per-call grouped-GEMM times and the
# helper's engine_join line gives rel_l2 against the CPU engine reference (R3_MOE_SWEEP_FAST=0, ~2 min a run) or, by
# default, each variant's output is compared bit for bit with the first variant's (sweep_cmp mismatches=N). No server, no per-variant load stage.
# CAPDIR=<dir> replays another capture set (e.g. small_capture.py output) instead of $W/captures.
# Usage: helper-sweep.sh <branch|-> "<variant>;<variant>..." [captures=4] [fixed env, e.g. R3_MOE_FP16=1 ...]
#   variant = "NAME=v,NAME=v" (an empty variant = environment as launched); anything libr3 reads at launch can vary
#   (R3_MOE_FP16, R3_MOE_PITCH16, R3_MOE_PROBE). `-` as branch reuses the newest build.
set -u
BR=$1; VARIANTS=$2; N=${3:-4}
W=/mnt/data/bigcherry-work
if [ "$BR" != "-" ]; then
  cd $W/wt-pr8 && git fetch -q origin && git checkout -q --detach $BR && git log --oneline -1
  export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH BIGCHERRY_ENVIRONMENT=/mnt/vault/development/projects/bigcherry/workspaces/main/config/environment.local.toml
  PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff > $W/sweep-build.log 2>&1
  grep -E 'ok build|FAILED' $W/sweep-build.log | tail -1; grep -E ' error:|r3_rewrite.py failed' $W/sweep-build.log | sort -u | head
fi
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
B=$(ls -td /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/*/*/ | head -1)
M=/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad
FILES=$(ls -S ${CAPDIR:-$W/captures}/*.bin | head -$N | tr '\n' ' ')
out=$W/sweep-out; rm -rf $out
env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 R3_MOE_HELPER_MIB=22528 R3_MOE_HELPER_SMALL_ROWS=${SMALL_ROWS:-8} R3_MOE_HELPER_MAX_ROWS=4096 \
  R3_MOE_NATIVE=1 R3_MOE_HELPER_MOE_DOWN_TILE=1 R3_MOE_HELPER_HEAT=$W/runs/perf51/heat.rank0.hot-first RADIANCE_HOME=$B/radiance_home \
  R3_MOE_SWEEP_FAST=${R3_MOE_SWEEP_FAST:-1} timeout 3600 rocprofv3 --kernel-trace --output-format csv -d $out -o run -- \
  $B/radiance_home/helpers/r3_moe_helper --replay-sweep $M "$VARIANTS" $FILES > $out.log 2>&1
echo "rc=$?"
python3 - $out.log $(find $out -name '*kernel_trace.csv' | head -1) <<'PY'
import csv, re, sys, collections
log, trace = sys.argv[1], sys.argv[2]
runs = []   # (idx, capture, layer, variant, rel_l2)
for line in open(log):
    m = re.match(r'SWEEP run (\d+) capture (\d+) layer (\d+) variant \[(.*)\]', line)
    if m: runs.append([int(m[1]), int(m[2]), int(m[3]), m[4], None]); continue
    m = re.search(r'engine_join .*rel_l2=(\S+) max_abs=(\S+)', line)
    if m and runs and runs[-1][4] is None: runs[-1][4] = 'rel_l2=' + m[1]
    m = re.search(r'sweep_cmp .*rank (\d+) (?:baseline )?mismatches=(\d+)', line)
    if m and runs and runs[-1][4] is None: runs[-1][4] = 'mismatches=' + m[2]
    m = re.search(r'sweep_sum .*rank 0 sum_abs=(\S+) nonzero=(\d+)', line)
    if m and runs: runs[-1].append('sum_abs=%s nonzero=%s' % (m[1], m[2]))
allk = []
for r in csv.DictReader(open(trace)):
    st, en = int(r['Start_Timestamp']), int(r['End_Timestamp'])
    allk.append((st, en, r['Kernel_Name']))
allk.sort()
# only the request's own kernels: the setup and table kernels around a run differ run to run
allk = [x for x in allk if 'r3_moe_gpu::lane_' in x[2] or 'moe_gemm_q_kernel' in x[2] or
        'moe_blockmap_kernel' in x[2] or 'gated_quant_fp8_kernel' in x[2]]
k = [(st, en - st) for st, en, n in allk if 'moe_gemm_q_kernel' in n]
# A run is a burst of request kernels; the host work between two runs (building the layer tables, the next request) leaves
# a gap of many milliseconds. Cut on gaps, not on a fixed kernel count: a variant that fuses kernels has fewer a run.
GAP_NS = 5000000
windows = []
for st, en, n in allk:
    if windows and st - windows[-1][1] <= GAP_NS:
        windows[-1][1] = max(windows[-1][1], en)
        windows[-1][2] += 1
    else:
        windows.append([st, en, 1])
if not runs or len(windows) != len(runs):
    print('could not group', len(allk), 'request kernels into', len(runs), 'runs: found', len(windows), 'bursts'); sys.exit(1)
print('%d runs, kernels a run %s' % (len(runs), sorted(set(w[2] for w in windows))))
rows = collections.defaultdict(dict)
for i, (idx, cap, layer, var, rel, *ex) in enumerate(runs):
    w0, w1, _ = windows[i]
    ds = [dur for st, dur in k if w0 <= st <= w1]
    chunk = [c for c in allk if w0 <= c[0] <= w1]
    span = (w1 - w0) / 1e3
    lane = sum(c[1] - c[0] for c in chunk if 'r3_moe_gpu::lane_' in c[2]) / 1e3
    rows[(cap, layer)]['%d %s' % (idx % (len(runs) // (max(r[1] for r in runs) + 1)), var or '(default)')] = ((sum(ds) / len(ds) / 1e3 if ds else 0.0), str(rel) + ' ' + (ex[0] if ex else '') + ' span=%.0fus lane_kernels=%.0fus' % (span, lane))
for key, vs in sorted(rows.items()):
    print('capture %d layer %d' % key)
    for var, (us, rel) in vs.items(): print('   %-44s %8.1f us/call  %s' % (var, us, rel))
print('SWEEP_DONE')
PY
