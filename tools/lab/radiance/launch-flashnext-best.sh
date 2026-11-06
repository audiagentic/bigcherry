#!/bin/bash
# Layout-driven Radiance launch. Flash-Next helper defaults below apply only to
# layouts with helpers; helper-free dense TP uses model, rank count and serving
# arguments resolved from engines/radiance/layouts.toml.
# Launch Qwen3.8-Flash-Next on dev in the current best layout, per
# docs/reference/engines/radiance/models/qwen3.8-flash-next.md ("Current best layout and numbers") and the harness config
# tools/lab/radiance/helper-ab.json, which this mirrors outside the harness.
#
# Offload: the engine card runs the model (--tp 1); one r3_moe_worker helper per helper card holds full-width experts (the hottest by route
# heat, up to the layout's budget), helper-owned experts skipped on the engine card; n-gram table in host
# RAM; fp8 KV; MTP depth 3; 2048-token chunks; up to 8 sessions of up to 262,144 tokens each (the context the model was
# trained on, so no rope stretch). The fp8 cache on the R9700 holds about 1.48M tokens, shared by the sessions (5.7 full-length ones);
# --max-model-len 1048576 --rope-scaling yarn:4 --max-num-seqs 2 serves the stretched 1M context instead. A prefix hit resumes at the nearest cached token (--prefix-resume nearest) rather
# than on the cold run's chunk grid: re-sent prefixes cost 0.2-0.3 s a turn instead of 0.6-0.8 s, at the price of greedy text
# that can differ from a cold run's near ties. The cache takes the VRAM the experts leave, which costs no decode or
# 4K-prefill speed (RAD37 measurements, qwen3.8-flash-next.md); one sequence is 30% slower at decode, so keep two.
# Helper kernel settings are in the manifest. The libr3 native/FP16/activation-half
# switches still need a typed plugin interface before their settings can take effect.
#
# Which cards play which part is a layout (engines/radiance/layouts.toml, `bigcherry layout list|show NAME`), resolved against this host's device
# inventory; card types (plugin, weight formats, few-rows kernel) are engines/radiance/archs.toml. --layout NAME (or FLASHNEXT_LAYOUT) picks one,
# default flashnext-4gpu (all four cards; flashnext-2xtx leaves the RX 6900 XT out); --budget ARCH=MIB overrides the VRAM budget of the group using that architecture (e.g. --layout flashnext-2xtx-rdna2-tail
# --budget gfx1030=4096). A layout that needs a card the host does not have, or a plugin the build lacks, stops here with the reason.
#
# Production (verified bundle and the tag's scripts/manifest):
#   bash tools/lab/radiance/launch-flashnext-best.sh start --release latest
# Lab: bigcherry build --lane radiance:stock:linux-radiance
#
# Usage:
#   bash tools/lab/radiance/launch-flashnext-best.sh [start] [--release TAG|latest | --build DIR] [--port N] [--heat FILE] [--host-pool-mib N] [--host-kv-mib N] [--chunk TOKENS] [--ngram-placement ram|disk|vram] [--kv-dtype fp8|fp8_v4|k6_v4] [--ngram-device INDEX] [--max-model-len TOKENS] [--record-heat PREFIX|off [--record-period S]] [--live-policy FILE] [--experiment FILE] [--calibrate --calibration-prompts FILE] [--gpu-power] [--layout NAME] [--budget ARCH=MIB] [--canary-prompt TEXT --canary-expect FACT]

#     (every start ends with a canary completion that must contain the fact, default 'The capital of France is' -> Paris; a failure stops the launch, exit 96)
#   bash tools/lab/radiance/launch-flashnext-best.sh stop
# --release re-execs from the verified release-tag worktree with --build. --release and --build
# cannot be combined. Without either, the lab default still uses the newest local build.
# --heat FILE: the route-heat file that ranks the experts, hottest first (default $WORK/heat/flashnext-mixed.hot-first, C++/HIP source and
# documentation recorded in RAD23 at equal weight; the RX 6900 XT tail only pays when it matches the workload, so record your own with --record-heat) (uint32[128][512] = max - count of a recorded histogram; heat-placement/heat-invert.py
# makes one). --record-heat PREFIX (default $WORK/flashnext/heat.live; off to disable): the engine counts every routed expert and rewrites
# PREFIX.rank0 every --record-period seconds (default 60) while serving. The counts sit in pinned host memory and cost no measurable
# speed (RHP02: 8 sessions 2,473 against 2,513 tok/s, inside the run-to-run spread), so it is on by default, and
# tools/lab/radiance/heat-placement/heat-watch.py (the start message prints the command) says when the active --heat file stops fitting. Invert the file, restart with --heat and a larger
# --budget (RHP10: workload-recorded heat plus gfx1100=23500 gave +34% prefill at 675K used context on a corpus it had not seen).
# --host-pool-mib N: pinned host memory for the engine's expert tier (default 9216). With the helpers owning almost every
#   expert the engine uses ~7.1 GiB of it; the old 22528 pinned ~16 GiB unused (RSC03 will derive this from ownership).
# --host-kv-mib N: host KV tier (--prefix-cache-mib) for finished sessions (default 12288, 0 = off). Without it finished
#   sessions keep their KV in VRAM, the KV cache stops lending VRAM to the engine's experts and a long-running server
#   loses ~half its prefill speed (49K prefill after an ~1M-token session fill: 1,901 -> 3,165 tok/s with the tier).
#   With the 47.7 GiB n-gram table in RAM, 9 GiB pool + 12 GiB tier is about the most that fits; a 24 GiB tier is refused.
# reports how much of the worst case it backs).
# --chunk TOKENS: the largest step (default 2048): the activation arena and the prefill staging are sized from it, so a smaller chunk leaves the KV cache more VRAM.
# --ngram-placement ram|disk|vram: where the 47.7 GiB n-gram table lives (default ram: pinned host memory the cards read). disk reads its rows from the
# container and frees that RAM for --host-pool-mib.
# --kv-dtype T: the KV cache type (default fp8; fp8_v4 costs 0.78 of it per token and k6_v4 0.69, both lossy). There is no raw engine argument: the engine's
# other flags are manifest fields (spec.experiment.engine: profile_ops, kv_resident, debug_selection, deterministic; give them in the
# --calibrate: LAB ONLY. Requires --experiment FILE and --live-policy FILE (explicit
# observe-or-move policy), --record-heat enabled and --calibration-prompts FILE
# (one prompt per line, repeated until the helper's measurement horizon settles).
# The helper prints one RHP16_CALIBRATION_JSON line; the launcher stops serving,
# validates it against this run's strict v1 manifest, atomically ingests it into
# $BC_WORK_ROOT/live-policy-calibration.toml and reports that path. No production
# start can use --calibrate. A missing measured quantity stays absent with a
# reason and will NOT activate an uncalibrated production policy.
# --experiment FILE object, e.g. {"engine": {"profile_ops": "on", "kv_resident": 65536}}).
# --ngram-device INDEX: an inventory card (not the engine, not a helper) that holds as much of the n-gram table as fits and gathers its rows (libr3ngram, R3_NGRAM_DEV);
# the rows past its VRAM stay in a host copy. The card then sits in the engine process beside the engine card.
# --max-model-len TOKENS: the longest context a sequence may have (default 262144, the trained context).
# --rope-scaling S: default none (contexts within the trained 262144); yarn:4 for the 1M stretch.
# --max-num-seqs N: sessions served at once (default 8; each costs per-sequence state in VRAM and checkpoint slots in pinned RAM).
# --prefix-resume MODE: default nearest; off leaves the flag out (engines before 1.14.0 do not have it).
# --gpu-power (or GPU_POWER=1): apply the cards' configured power settings ([host.*.devices.power]) for this run and
# put back what the cards ran before at stop. Without write access it lists the setup steps still missing and runs
# with the cards as they are.
# Serving manifest and plan check (RSC03 / RSC08 S3, mandatory, no way to skip it): with helpers, every start asks bc-serve for
#   the measured v1 manifest of the exact configuration it launches; the engine and the helpers start from that one file
#   (--manifest M --role ...), and the engine checks its live memory plan against the plan the manifest carries before it
#   allocates anything. The first start of a configuration that has no measured plan (a cache miss, logged once with its reason)
#   first MEASURES it: the helpers come up on their cards and the engine stops after its memory plan, about 1-2 minutes, then
#   the real start goes ahead. That measuring start uses the GPUs exactly as a launch does, so the first launch of a new
#   configuration needs the gpu lease like any launch (agent-lease acquire gpu). Any setting above (layout, sessions, context,
#   pool, KV tier, chunk, n-gram placement, KV dtype, rope, resume) changes the configuration. Plans live in $RUN/plan-cache.
#   With helpers, --ngram-device is refused (it is not a manifest field yet).
# HELPER_WRAP=<command>: run each helper under it (e.g. a profiler wrapper; its arguments --role helperN tell it which helper).
# ENGINE_WRAP=<command>: run the engine under it (a debugger, numactl, strace; the stop command signals the engine through the
#   wrapper, so it must forward SIGINT/SIGTERM). Not rocprofv3: radiance drives the card through its own AQL backend and fails
#   at start under rocprof's interception (2026-11-01); for the engine's per-op device times put {"engine": {"profile_ops": "on"}} in the --experiment FILE.
set -u

WORK=${BC_WORK_ROOT:?set BC_WORK_ROOT (source tools/env/bigcherry-env.sh)}
RUN=$WORK/flashnext
PIDS=$RUN/pids
SHM=/r3-moe-helper
PORT=36080
HEAT=$WORK/heat/flashnext-mixed.hot-first
BUILD=""
GPU_POWER=${GPU_POWER:-0}
LAYOUT=${FLASHNEXT_LAYOUT:-flashnext-4gpu}
BUDGETS=()
RECORD=$RUN/heat.live   # RHP02: on by default; --record-heat off disables
HOST_POOL_MIB=9216        # interim, measured (RSC03 derives it): the engine uses ~7.1 GiB of it with the helpers owning ~all experts
HOST_KV_MIB=12288         # interim, measured (RSC03 derives it): host KV tier for finished sessions, funded by the smaller pool
CHUNK=2048
CKPT_INTERVAL=4096   # the engine's --checkpoint-interval; the plan check's configuration (plan-check.sh) names it too
NUM_SPEC=3                # the engine's --num-speculative-tokens (MTP depth); likewise
NGRAM=ram
KV_DTYPE=fp8
CUSTOM_SERVE_OPTIONS=()
NGRAM_DEV=""
MAX_LEN=262144
ROPE=none
SEQS=8
PREFIX_RESUME=nearest
RECORD_PERIOD=60
LIVE_POLICY_FILE=""   # a JSON object with every spec.handoff.placement.live_policy member (docs/design/serving-manifest-v1)
EXPERIMENT_FILE=""    # a JSON object for spec.experiment (lab profile)
CALIBRATE=0           # lab-only measurement run, never a serving hand-off
CALIBRATION_PROMPTS=""
# The start canary: after /health, every start (cold, measured or cached) sends this one deterministic completion and the answer
# must contain the expected fact (case-insensitive, semantic, not a hash), or everything this launch started is stopped and the
# launch fails (exit 96). It catches an engine that is up but answering garbage before any client does. The prompt is a fact the
# served model answers; a model that cannot answer it gets its own with --canary-prompt/--canary-expect.
CANARY_PROMPT="The capital of France is"
CANARY_EXPECT="Paris"
ORIGINAL_ARGS=("$@")
RELEASE_TAG=""
CMD=start
[ $# -gt 0 ] && case "$1" in start|stop) CMD=$1; shift ;; esac
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=$2; shift 2 ;;
    --release) [ $# -ge 2 ] || { echo "--release needs a tag or latest" >&2; exit 2; }; RELEASE_TAG=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --heat) CUSTOM_SERVE_OPTIONS+=(--heat); HEAT=$2; shift 2 ;;
    --gpu-power) GPU_POWER=1; shift ;;
    --layout) LAYOUT=$2; shift 2 ;;
    --budget) BUDGETS+=("$2"); shift 2 ;;
    --host-pool-mib) CUSTOM_SERVE_OPTIONS+=(--host-pool-mib); HOST_POOL_MIB=$2; shift 2 ;;
    --host-kv-mib) CUSTOM_SERVE_OPTIONS+=(--host-kv-mib); HOST_KV_MIB=$2; shift 2 ;;
    --chunk) CUSTOM_SERVE_OPTIONS+=(--chunk); CHUNK=$2; shift 2 ;;
    --kv-dtype) CUSTOM_SERVE_OPTIONS+=(--kv-dtype); KV_DTYPE=$2; shift 2 ;;
    --ngram-device) CUSTOM_SERVE_OPTIONS+=(--ngram-device); NGRAM_DEV=$2; shift 2 ;;
    --max-model-len) CUSTOM_SERVE_OPTIONS+=(--max-model-len); MAX_LEN=$2; shift 2 ;;
    --rope-scaling) CUSTOM_SERVE_OPTIONS+=(--rope-scaling); ROPE=$2; shift 2 ;;
    --max-num-seqs) CUSTOM_SERVE_OPTIONS+=(--max-num-seqs); SEQS=$2; shift 2 ;;
    --prefix-resume) CUSTOM_SERVE_OPTIONS+=(--prefix-resume); PREFIX_RESUME=$2; shift 2 ;;
    --ngram-placement) CUSTOM_SERVE_OPTIONS+=(--ngram-placement); NGRAM=$2; shift 2 ;;
    --record-heat) CUSTOM_SERVE_OPTIONS+=(--record-heat); RECORD=$2; shift 2 ;;
    --record-period) CUSTOM_SERVE_OPTIONS+=(--record-period); RECORD_PERIOD=$2; shift 2 ;;
    --live-policy) CUSTOM_SERVE_OPTIONS+=(--live-policy); LIVE_POLICY_FILE=$2; shift 2 ;;
    --experiment) CUSTOM_SERVE_OPTIONS+=(--experiment); EXPERIMENT_FILE=$2; shift 2 ;;
    --calibrate) CUSTOM_SERVE_OPTIONS+=(--calibrate); CALIBRATE=1; shift ;;
    --calibration-prompts) CALIBRATION_PROMPTS=$2; shift 2 ;;
    --canary-prompt) CANARY_PROMPT=$2; shift 2 ;;
    --canary-expect) CANARY_EXPECT=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ "$CALIBRATE" = 1 ]; then
  if [ "$CMD" != start ] || [ -z "$EXPERIMENT_FILE" ] || [ -z "$LIVE_POLICY_FILE" ]; then
    echo "RADIANCE_EXIT=2 --calibrate is lab-only: --experiment and explicit --live-policy are required" >&2
    exit 2
  fi
  if [ -z "$CALIBRATION_PROMPTS" ] || [ ! -s "$CALIBRATION_PROMPTS" ]; then
    echo "RADIANCE_EXIT=2 --calibrate requires a nonempty --calibration-prompts FILE" >&2
    exit 2
  fi
  if ! grep -q "[^[:space:]]" "$CALIBRATION_PROMPTS"; then
    echo "RADIANCE_EXIT=2 --calibrate needs at least one nonblank workload prompt" >&2
    exit 2
  fi
  if [ "$RECORD" = off ]; then
    echo "RADIANCE_EXIT=2 --calibrate needs the live route histogram (--record-heat cannot be off)" >&2
    exit 2
  fi
  # The lab plan key must be the same as the later production plan key.
  # An experiment that changes engine diagnostics/memory could invalidate it.
  # An explicit empty lab experiment selects the lab profile without any such
  # changes; plan-check.sh deliberately excludes its otherwise hashed identity.
  if ! python3 - "$EXPERIMENT_FILE" "$LIVE_POLICY_FILE" <<'RHP16_VALIDATE'
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        experiment = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        policy = json.load(f)
    if experiment != {}:
        raise ValueError("the calibration experiment must be an empty JSON object (same plan as production)")
    # v1 has no live_policy.enabled: a PRESENT policy is an active one (its thread and measurement hooks run), an absent
    # one is no policy. So the lab policy must be a non-empty object of v1 live_policy members; bc-serve then checks that
    # every member is there and valid.
    if not isinstance(policy, dict) or not policy:
        raise ValueError("the explicit lab live policy must be a non-empty object of spec.handoff.placement.live_policy members (present = active)")
    if "enabled" in policy:
        raise ValueError("live_policy has no 'enabled' member in v1: a present policy is active; remove the key")
except (OSError, ValueError) as e:
    print("calibration: " + str(e), file=sys.stderr)
    sys.exit(2)
RHP16_VALIDATE
  then
    echo "RADIANCE_EXIT=2 calibration must not alter the production plan or disable measurement hooks" >&2
    exit 2
  fi
fi

if [ -n "$RELEASE_TAG" ]; then
  if [ -n "$BUILD" ]; then
    echo "RADIANCE_EXIT=2 --release and --build cannot be combined" >&2
    exit 2
  fi
  # Verify the receipt and binary digest on every use. An existing build is
  # reused only when it still matches the release record.
  RELEASE_INFO=$(PYTHONPATH=tools python3 -m bigcherry --engine radiance release-install "$RELEASE_TAG" --json) || {
    echo "RADIANCE_EXIT=99 cannot install/verify Radiance release $RELEASE_TAG" >&2
    exit 99
  }
  RELEASE_PATHS=$(printf '%s' "$RELEASE_INFO" | python3 -c 'import json,sys
r=json.load(sys.stdin)
for key in ("worktree", "build_dir"):
    value=r[key]
    if not isinstance(value,str) or not value.startswith("/") or "\n" in value:
        sys.exit("invalid release path in receipt")
    print(value)') || { echo "RADIANCE_EXIT=99 invalid release installation receipt" >&2; exit 99; }
  mapfile -t RELEASE_FIELDS <<< "$RELEASE_PATHS"
  [ "${#RELEASE_FIELDS[@]}" -eq 2 ] || { echo "RADIANCE_EXIT=99 incomplete release installation receipt" >&2; exit 99; }
  RELEASE_TREE=${RELEASE_FIELDS[0]}
  RELEASE_BUILD=${RELEASE_FIELDS[1]}
  RELEASE_ARGS=()
  for ((i = 0; i < ${#ORIGINAL_ARGS[@]}; i++)); do
    if [ "${ORIGINAL_ARGS[i]}" = --release ]; then
      ((i+=1))
    else
      RELEASE_ARGS+=("${ORIGINAL_ARGS[i]}")
    fi
  done
  cd "$RELEASE_TREE" || { echo "RADIANCE_EXIT=99 release worktree missing: $RELEASE_TREE" >&2; exit 99; }
  exec bash "$RELEASE_TREE/tools/lab/radiance/launch-flashnext-best.sh" "${RELEASE_ARGS[@]}" --build "$RELEASE_BUILD"
fi

stop_procs() {
  local stuck=0
  # engine first (SIGINT is its clean shutdown), then the helpers; each gets 60 s before SIGKILL
  for name in engine $(cd "$PIDS" 2>/dev/null && ls -r helper* 2>/dev/null); do
    [ -f "$PIDS/$name" ] || continue
    pid=$(cat "$PIDS/$name")
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid"
      for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && { echo "$name ($pid) did not stop; killing" >&2; kill -KILL "$pid"; }
      # Fail closed: a process of this launch that is still alive (its mappings, its GPU context) would share the cards and
      # a recreated /dev/shm segment name with whatever starts next, so nothing proceeds while it lives.
      for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      if kill -0 "$pid" 2>/dev/null; then
        echo "stop: $name ($pid) is still alive after SIGKILL; not continuing while it holds the cards" >&2
        stuck=1
        continue   # keep its pid file so a later stop sees it
      fi
    fi
    rm -f "$PIDS/$name"
  done
  return $stuck
}

stop_all() {
  local rc=0
  stop_procs || rc=1
  # power settings this launcher applied go back to what the cards ran before; attempted even when a process survived,
  # and the failure of either is the result
  if [ -f "$RUN/gpu-power" ]; then
    if PYTHONPATH=tools python3 -m bigcherry gpu-power restore >&2; then rm -f "$RUN/gpu-power"; else rc=1; fi
  fi
  return $rc
}

# One launch at a time: the already-running check, the measuring start and the plan store share the run folder (the PID files,
# the helper segment, the measured-plan file), so two concurrent launches could store one's plan under the other's key.
# `stop` takes the same lock, so it cannot slip in between a launch's power marker and its apply, or report "stopped" while a
# launch goes on. The lock is released at the hand-off; the engine and helpers never inherit it.
mkdir -p "$RUN"
exec 9> "$RUN/launch.lock"
flock -n 9 || { echo "RADIANCE_EXIT=98 another launch (or stop) is in progress ($RUN/launch.lock); wait for it, or interrupt it with ^C"; exit 98; }

if [ "$CMD" = stop ]; then
  if stop_all; then echo "stopped"; exit 0; fi
  echo "stop failed: a process or the power settings could not be put back (above); nothing is reported stopped" >&2
  exit 1
fi

for name in engine $(cd "$PIDS" 2>/dev/null && ls helper* 2>/dev/null); do
  if [ -f "$PIDS/$name" ] && kill -0 "$(cat "$PIDS/$name")" 2>/dev/null; then
    echo "RADIANCE_EXIT=98 already running ($name pid $(cat "$PIDS/$name")); run '$0 stop' first"; exit 98
  fi
done

# Resolve the layout before choosing binaries and the model container.
BUDGET_ARGS=(); for b in "${BUDGETS[@]:-}"; do [ -n "$b" ] && BUDGET_ARGS+=(--budget "$b"); done
LAYOUT_OUT=$(PYTHONPATH=tools python3 -m bigcherry layout env "$LAYOUT" "${BUDGET_ARGS[@]}") || { echo "RADIANCE_EXIT=99 layout '$LAYOUT' does not resolve on this host (above)"; exit 99; }
eval "$LAYOUT_OUT"
if [ "$LAYOUT_HELPER_COUNT" = 0 ] && [ "${#CUSTOM_SERVE_OPTIONS[@]}" -ne 0 ]; then
  echo "RADIANCE_EXIT=2 these options configure helper-offload serving, not the helper-free layout $LAYOUT: ${CUSTOM_SERVE_OPTIONS[*]}; use its server-args in layouts.toml" >&2
  exit 2
fi
if [ "$CALIBRATE" = 1 ] && [ "$LAYOUT_HELPER_COUNT" -eq 0 ]; then
  echo "RADIANCE_EXIT=2 --calibrate requires a helper layout" >&2
  exit 2
fi

if [ -z "$BUILD" ]; then
  for d in $(ls -td "$WORK"/builds/*/*/ 2>/dev/null); do
    [ -x "${d}bin/radiance" ] && { [ "$LAYOUT_HELPER_COUNT" = 0 ] || [ -x "${d}radiance_home/helpers/r3_moe_worker" ]; } && { BUILD=${d%/}; break; }
  done
fi
RAD_BIN=$BUILD/bin/radiance
RAD_HOME=$BUILD/radiance_home
HELPER=$RAD_HOME/helpers/r3_moe_worker
[ -x "$RAD_BIN" ] && { [ "$LAYOUT_HELPER_COUNT" = 0 ] || [ -x "$HELPER" ]; } || { echo "RADIANCE_EXIT=99 no build for layout $LAYOUT (--build '$BUILD')"; exit 99; }

# The engine runs against the ROCm SDK its build recorded; an environment pointing at another SDK refuses the start.
source "$(dirname "${BASH_SOURCE[0]}")/lib/sdk-check.sh"
check_build_sdk "$BUILD" || { echo "RADIANCE_EXIT=98 build $BUILD does not match the environment's ROCm SDK"; exit 98; }

# Warn (never block) when the host lacks settings the tooling expects: configured per-card power settings not applied,
# CPU energy counters unreadable (`bigcherry host-check`, PWR04).
PYTHONPATH=tools python3 -m bigcherry host-check 2>/dev/null | grep '^WARN' >&2 || true

# From here until the engine is up, anything that ends this launcher (a failure, ^C, SIGTERM, SIGHUP, an early exit) stops
# what it started and puts the power settings back. Only the hand-off at the end (LAUNCH_HANDED_OFF=1) keeps them.
LAUNCH_HANDED_OFF=0
launch_cleanup() {
  [ "$LAUNCH_HANDED_OFF" = 1 ] && return 0
  trap - EXIT
  echo "launch did not complete; stopping what was started" >&2
  stop_all || echo "launch cleanup: a process or the power settings could not be put back (above)" >&2
}
trap launch_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Optional per-run power settings (PWR04); when they cannot be written, gpu-power lists the missing setup steps and the run goes
# ahead with the cards as they are.
if [ "$GPU_POWER" = 1 ]; then
  mkdir -p "$RUN"
  # The marker comes first: apply records what ran before and then writes setting by setting, so an apply that fails part way
  # may have changed a card. Restoring with nothing recorded is a no-op.
  touch "$RUN/gpu-power"
  if ! PYTHONPATH=tools python3 -m bigcherry gpu-power apply >&2; then
    echo "gpu-power: the settings were not (fully) applied; putting back what ran before, then running with the cards as they are" >&2
    PYTHONPATH=tools python3 -m bigcherry gpu-power restore >&2 && rm -f "$RUN/gpu-power"
  fi
fi

if [ "$LAYOUT_HELPER_COUNT" = 0 ] && [ -n "${MODEL:-}" ]; then
  echo "RADIANCE_EXIT=2 the helper-free model is selected by layouts.toml; unset MODEL" >&2
  exit 2
fi
MODEL=${MODEL:-${BC_MODEL_ROOT:?set BC_MODEL_ROOT (source tools/env/bigcherry-env.sh)}/$LAYOUT_MODEL}
[ -f "$MODEL" ] || { echo "RADIANCE_EXIT=99 model not found: $MODEL"; exit 99; }
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then
  [ -f "$HEAT" ] || { echo "RADIANCE_EXIT=99 route-heat file not found: $HEAT"; exit 99; }
fi

source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
[ -f "$RAD_HOME/kernels/$LAYOUT_ENGINE_PLUGIN.so" ] || { echo "RADIANCE_EXIT=99 layout $LAYOUT requires kernels/$LAYOUT_ENGINE_PLUGIN.so"; exit 99; }
HELPERS=(); for ((i = 0; i < LAYOUT_HELPER_COUNT; i++)); do HELPERS+=("$i"); done
for ((i = 0; i < LAYOUT_HELPER_COUNT; i++)); do
  declare -n h="LAYOUT_HELPER_$i"
  plugin=$(printf '%s\n' "${h[@]}" | sed -n 's/^R3_MOE_WORKER_PLUGIN_FILE=//p')
  [ -f "$RAD_HOME/kernels/$plugin" ] || { echo "RADIANCE_EXIT=99 layout '$LAYOUT' needs kernels/$plugin in the build ($RAD_HOME); rebuild with the radiance lane"; exit 99; }
  unset -n h
done
echo "layout $LAYOUT: engine cards $LAYOUT_ENGINE_DEVICES; $LAYOUT_SUMMARY" >&2

# shared by the engine and the helpers (helper-ab.json "environment" + arm settings), then what the layout says
COMMON=(RADIANCE_HOME="$RAD_HOME")
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then
  COMMON+=(
    R3_MOE_WORKER_SHM="$SHM"
    R3_MOE_WORKER_MAX_ROWS=4096
    R3_MOE_WORKER_SMALL_ROWS=8
    R3_MOE_WORKER_GEMV_LARGE_ROWS=32
    R3_MOE_WORKER_MOE_DOWN_TILE=1
    R3_MOE_WORKER_HEAT="$HEAT"
    R3_MOE_WORKER_SKIP_OWNED=1
    "${LAYOUT_COMMON[@]}"
  )
  [ "$RECORD" = off ] && RECORD=""
  if [ -n "$RECORD" ]; then
    COMMON+=(R3_MOE_WORKER_HISTOGRAM="$RECORD" R3_MOE_WORKER_HISTOGRAM_PERIOD_S="$RECORD_PERIOD")
    echo "recording the route histogram to $RECORD.rank0 every ${RECORD_PERIOD}s" >&2
  fi
fi

mkdir -p "$PIDS"
STAMP=$(date +%m%d-%H%M%S)

# helpers in order: helper 0 creates the segment, helper 1 attaches; each must report ready before the next starts. A helper
# reads its whole configuration from the manifest bc-serve wrote (start_helpers MANIFEST [--measuring]): its environment is
# only which card it sees (and an optional profiler wrapper). --measuring is the measuring start's unmeasured manifest.
start_helpers() {
  local manifest=$1; shift
  for i in "${!HELPERS[@]}"; do
    log=$RUN/helper$i-$STAMP.log
    # A fresh log, made here before the helper exists: the measuring start used this same name moments ago, and its
    # "ready" line must never be mistaken for the new helper's.
    : > "$log"
    declare -n hv="LAYOUT_HELPER_$i"
    helper_visibility=()
    for kv in "${hv[@]}"; do
      case "$kv" in ROCR_VISIBLE_DEVICES=*|HIP_VISIBLE_DEVICES=*) helper_visibility+=("$kv") ;; esac
    done
    env "${helper_visibility[@]}" \
      setsid ${HELPER_WRAP:-} "$HELPER" --manifest "$manifest" --role "helper$i" "$@" > "$log" 2>&1 9>&- &
    echo $! > "$PIDS/helper$i"
    unset -n hv
    echo "helper$i (pid $!, log $log): waiting for ready..." >&2
    ready=0
    for _ in $(seq 900); do
      grep -qE '^r3_moe_worker: (ready |waiting for engine runtime expert maps)' "$log" && { ready=1; break; }
      kill -0 "$(cat "$PIDS/helper$i")" 2>/dev/null || break
      sleep 1
    done
    [ $ready = 1 ] || { echo "RADIANCE_EXIT=1 helper$i not ready; see $log"; tail -n 20 "$log"; return 1; }
  done
}

# Rank-to-card order comes from the resolved inventory. The engine process
# sees consecutive HIP ordinals; TP traffic is forced through host memory.
ENGINE_VISIBLE="$LAYOUT_ENGINE_DEVICES"
ENGINE_HIP=""
for ((i = 0; i < LAYOUT_ENGINE_RANKS; i++)); do
  [ -z "$ENGINE_HIP" ] && ENGINE_HIP="$i" || ENGINE_HIP="$ENGINE_HIP,$i"
done
ENGINE_KERNELS="$LAYOUT_ENGINE_PLUGIN,libref"
ENGINE_TOPO_ARGS=(--tp "$LAYOUT_ENGINE_RANKS")
ENGINE_ENV=("${COMMON[@]}")
if [ "$LAYOUT_ENGINE_RANKS" -gt 1 ]; then
  [ -n "$LAYOUT_TP_WIRE" ] || { echo "RADIANCE_EXIT=99 multi-rank layout missing tp-wire" >&2; exit 99; }
  ENGINE_TOPO_ARGS+=(--tp-wire "$LAYOUT_TP_WIRE" --p2p off)
fi
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then
  ENGINE_ENV+=(R3_MOE_WORKER=1)
  # RSC08 S3: with helpers the engine starts from the measured manifest alone (--manifest M --role engine). The n-gram card is
  # therefore not passable here until it is a manifest field.
  if [ -n "$NGRAM_DEV" ]; then
    echo "RADIANCE_EXIT=99 --ngram-device cannot be combined with a helper layout: the engine starts from its serving manifest and the n-gram card is not a manifest field yet" >&2
    exit 99
  fi
else
  ENGINE_CONFIG_ARGS=("${LAYOUT_SERVER_ARGS[@]}")
fi
start_engine() {
  local log=$1; shift
  env "${ENGINE_ENV[@]}" ROCR_VISIBLE_DEVICES="$ENGINE_VISIBLE" HIP_VISIBLE_DEVICES="$ENGINE_HIP"     setsid ${ENGINE_WRAP:-} "$RAD_BIN" --model "$MODEL" --radiance-home "$RAD_HOME" --port "$PORT"       --kernels "$ENGINE_KERNELS" "${ENGINE_TOPO_ARGS[@]}" "${ENGINE_CONFIG_ARGS[@]}" "$@" > "$log" 2>&1 9>&- &
}

# helpers_resident_mib: the sum of the helpers' RssAnon plus the largest RssShmem (the shared segment, once), in MiB, rounded up.
helpers_resident_mib() {
  local f pid anon=0 shm=0 a s
  for f in "$PIDS"/helper*; do
    pid=$(cat "$f") && [ -r "/proc/$pid/status" ] || { echo "helpers resident memory: no helper process at $f" >&2; return 1; }
    a=$(awk '/^RssAnon:/ {print $2}' "/proc/$pid/status"); s=$(awk '/^RssShmem:/ {print $2}' "/proc/$pid/status")
    [ -n "$a" ] && [ -n "$s" ] || { echo "helpers resident memory: cannot read /proc/$pid/status" >&2; return 1; }
    anon=$((anon + a)); [ "$s" -gt "$shm" ] && shm=$s
  done
  echo $(( (anon + shm + 1023) / 1024 ))
}

# The engine of a helper layout starts from its serving manifest alone: the manifest is its whole configuration, and a measured
# one also carries the plan it must admit before any weight loads (radiance --manifest M --role engine [--plan-memory FILE]).
# Only the card it sees and the engine-side environment the manifest does not yet carry (RSC08 S5) come from here.
start_engine_manifest() {
  local log=$1 manifest=$2; shift 2
  env "${ENGINE_ENV[@]}" ROCR_VISIBLE_DEVICES="$ENGINE_VISIBLE" HIP_VISIBLE_DEVICES="$ENGINE_HIP" \
    setsid ${ENGINE_WRAP:-} "$RAD_BIN" --manifest "$manifest" --role engine "$@" > "$log" 2>&1 9>&- &
}

# The measuring start of the plan check (plan-check.sh): helpers up, the engine runs to the end of its memory plan and
# exits by itself (--plan-memory, with the unmeasured manifest), then the helpers stop. It never serves and never runs a request.
plan_check_start_engine_measure() {
  rm -f "/dev/shm${SHM}"
  start_helpers "$PLAN_MEASURING_MANIFEST" --measuring || { stop_procs; return 1; }
  # what the helpers hold of the host while they are up: anonymous memory of each plus the shared segment once (file pages are
  # reclaimable and not counted); bc-serve plans the checkpoint slots around it (plan-check.sh passes it with the plan)
  PLAN_HELPERS_RESIDENT_MIB=$(helpers_resident_mib) || { stop_procs; return 1; }
  start_engine_manifest "$RUN/engine-measure.log" "$PLAN_MEASURING_MANIFEST" --plan-memory "$PLAN_MEASURE_FILE"
  local pid=$! rc=0
  echo $pid > "$PIDS/engine"
  wait $pid || rc=$?
  stop_procs || { echo "plan check: the measuring start's processes did not exit; not starting for real" >&2; return 1; }   # processes only: the power settings the real start still needs stay applied
  [ $rc = 0 ] && [ -s "$PLAN_MEASURE_FILE" ]
}

if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then
  source "$(dirname "${BASH_SOURCE[0]}")/lib/plan-check.sh"
  plan_manifest_prepare || { echo "RADIANCE_EXIT=97 the measured serving manifest could not be made (above); nothing was started for serving"; exit 97; }
else
  # The interim bc-serve recipes only understand the single-engine-card
  # remote-MoE layout. Require a fresh engine memory plan for dense TP.
  # This is not a calibrated plan check; do not reuse a MoE plan here.
  PLAN_MEASURE_FILE="$RUN/plan-dense.json"
  rm -f "$PLAN_MEASURE_FILE"
  start_engine "$RUN/engine-measure.log" --plan-memory "$PLAN_MEASURE_FILE"
  measure_pid=$!
  echo "$measure_pid" > "$PIDS/engine"
  measure_rc=0
  wait "$measure_pid" || measure_rc=$?
  stop_procs || { echo "RADIANCE_EXIT=97 dense plan measurement did not stop cleanly" >&2; exit 97; }
  if [ "$measure_rc" != 0 ] || [ ! -s "$PLAN_MEASURE_FILE" ] || ! python3 -m json.tool "$PLAN_MEASURE_FILE" >/dev/null 2>&1; then
    echo "RADIANCE_EXIT=97 dense engine plan measurement failed; see $RUN/engine-measure.log" >&2
    exit 97
  fi
  echo "dense memory plan measured; calibrated bc-serve plan-check is not yet available for TP layouts" >&2
fi

trap 'exit 1' ERR
rm -f "/dev/shm${SHM}"
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then start_helpers "$PLAN_MANIFEST"; fi
LOG=$RUN/engine-$STAMP.log
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then start_engine_manifest "$LOG" "$PLAN_MANIFEST"; else start_engine "$LOG"; fi
echo $! > "$PIDS/engine"
echo "engine TP=$LAYOUT_ENGINE_RANKS cards $LAYOUT_ENGINE_DEVICES (pid $!, log $LOG): waiting for /health..." >&2
up=0
for _ in $(seq 180); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { up=1; break; }
  kill -0 "$(cat "$PIDS/engine")" 2>/dev/null || break
  sleep 5
done
[ $up = 1 ] || { echo "RADIANCE_EXIT=1 engine did not come up; see $LOG"; tail -n 30 "$LOG"; false; }
start_canary() {
  local body answer reply
  body=$(CANARY_PROMPT="$CANARY_PROMPT" python3 -c 'import json,os;print(json.dumps({"model":"default","prompt":os.environ["CANARY_PROMPT"],"max_tokens":24,"temperature":0}))')
  # Transport and parsing must each succeed on their own: a failed or non-2xx request, or a reply that is not a completion, is a
  # canary failure whatever text it happens to contain.
  if ! reply=$(curl -fsS --max-time 120 "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' -d "$body" 2>&1); then
    answer="<request failed: $(printf %s "$reply" | head -c 80)>"
  elif ! answer=$(printf %s "$reply" | python3 -c 'import json,sys;t=json.load(sys.stdin)["choices"][0]["text"];sys.exit(1) if not isinstance(t,str) else print(t)' 2>/dev/null); then
    answer="<not a completion: $(printf %s "$reply" | head -c 60)>"
  else
    case "${answer,,}" in *"${CANARY_EXPECT,,}"*) echo "start canary: ok (answered '$(printf %s "$answer" | head -c 60)')" >&2; return 0 ;; esac
  fi
  echo "RADIANCE_EXIT=96 start canary failed: the engine is up but answered '$(printf %s "$answer" | head -c 80)' to '$CANARY_PROMPT' (expected it to contain '$CANARY_EXPECT'); stopping everything this launch started; engine log $LOG" >&2
  return 1
}
start_canary || exit 96
if [ "$CALIBRATE" = 1 ]; then
  # No fixed number of requests or time-based guess: this loop stops on the
  # helper's convergence condition (the retained drift-track horizon).
  helper_log="$RUN/helper0-$STAMP.log"
  [ -f "$helper_log" ] || { echo "calibration: helper 0 log absent" >&2; exit 1; }
  echo "calibration: replaying supplied prompts until RHP16_CALIBRATION_JSON appears" >&2
  while ! grep -q '^RHP16_CALIBRATION_JSON ' "$helper_log"; do
    while IFS= read -r calibration_prompt || [ -n "$calibration_prompt" ]; do
      [ -n "$calibration_prompt" ] || continue
      body=$(CALIBRATION_PROMPT="$calibration_prompt" python3 -c 'import json,os;print(json.dumps({"model":"default","prompt":os.environ["CALIBRATION_PROMPT"],"max_tokens":24,"temperature":0}))')
      curl -fsS --max-time 120 "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' -d "$body" >/dev/null || {
        echo "calibration: workload request failed" >&2; exit 1;
      }
      grep -q '^RHP16_CALIBRATION_JSON ' "$helper_log" && break
      for proc in engine helper0; do
        [ -f "$PIDS/$proc" ] && kill -0 "$(cat "$PIDS/$proc")" 2>/dev/null || {
          echo "calibration: $proc exited before convergence" >&2; exit 1;
        }
      done
    done < "$CALIBRATION_PROMPTS"
  done
  calibration_store="$WORK/live-policy-calibration.toml"
  "$RAD_HOME/bin/bc-serve" calibration-ingest --from-log "$helper_log" \
    --calibration "$calibration_store" --manifest "$PLAN_MANIFEST" \
    --plan-cache "$RUN/plan-cache" --engine-binary "$RAD_BIN" \
    "${PLAN_SITE[@]}" "${PLAN_EXTRA[@]}" "${PLAN_SET[@]}" || exit 1
  echo "calibration: recorded in $calibration_store" >&2
  stop_all || { echo "calibration: failed to stop all run processes" >&2; exit 1; }
  trap - ERR
  trap - EXIT INT TERM HUP
  exec 9>&-
  echo "RADIANCE_EXIT=0 calibration=$calibration_store"
  exit 0
fi
trap - ERR
LAUNCH_HANDED_OFF=1
trap - EXIT INT TERM HUP
exec 9>&-

cat <<INFO
RADIANCE_EXIT=0
Radiance layout $LAYOUT is up: http://127.0.0.1:$PORT (OpenAI-compatible /v1/chat/completions, /v1/completions)
  build    $BUILD
  engine   TP=$LAYOUT_ENGINE_RANKS cards $LAYOUT_ENGINE_DEVICES pid $(cat "$PIDS/engine"), log $LOG
  layout   $LAYOUT: $LAYOUT_SUMMARY\n  helpers  $(if [ "$LAYOUT_HELPER_COUNT" -eq 0 ]; then echo none; else for p in "$PIDS"/helper*; do printf "%s " "$(cat "$p")"; done; fi)
  stop     bash $0 stop
INFO
