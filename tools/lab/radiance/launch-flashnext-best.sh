#!/bin/bash
# Layout-driven Radiance launch. Flash-Next helper defaults below apply only to
# layouts with helpers; helper-free dense TP uses model, rank count and serving
# arguments resolved from engines/radiance/layouts.toml.
# Launch Qwen3.8-Flash-Next on dev in the current best layout, per
# docs/reference/engines/radiance/models/qwen3.8-flash-next.md ("Current best layout and numbers") and the harness config
# tools/lab/radiance/helper-ab.json, which this mirrors outside the harness.
#
# Offload: the engine card(s) run the model (--tp = the layout's engine card count; TP traffic through host memory, tp-wire from the
# layout); one r3_moe_worker helper per helper card holds experts (the hottest by route heat, up to the layout's budget, always full
# width; behind TP one rank a layer adds them, RAD49), helper-owned experts skipped on the engine cards; n-gram table in host
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
#   bash tools/lab/radiance/launch-flashnext-best.sh [start] [--release TAG|latest | --build DIR] [--port N] [--heat FILE] [--host-pool-mib N] [--host-kv-mib N] [--chunk TOKENS] [--ngram-placement ram|disk|vram] [--kv-dtype fp8|fp8_v4|k6_v4] [--ngram-device INDEX] [--max-model-len TOKENS] [--record-heat PREFIX|off [--record-period S]] [--live-policy FILE] [--experiment FILE] [--calibrate --calibration-prompts-a FILE --calibration-prompts-b FILE] [--gpu-power] [--layout NAME] [--budget ARCH=MIB] [--canary-prompt TEXT --canary-expect FACT]

#     (every start ends with a canary completion that must contain the fact, default 'The capital of France is' -> Paris; a failure stops the launch, exit 96)
#   bash tools/lab/radiance/launch-flashnext-best.sh stop
# --release re-execs from the verified release-tag worktree with --build. --release and --build
# cannot be combined. Without either, the lab default still uses the newest local build.
# --heat FILE: the route-heat file that ranks the experts, hottest first (default $WORK/heat/flashnext-mixed.hot-first, C++/HIP source and
# documentation recorded in RAD23 at equal weight; the RX 6900 XT tail only pays when it matches the workload, so record your own with --record-heat) (uint32[128][512] = max - count of a recorded histogram; heat-placement/heat-invert.py
# makes one). --record-heat PREFIX (default $WORK/flashnext/heat.live; off to disable): the engine counts every routed expert and rewrites
# PREFIX.rank0 every --record-period seconds (default 1: the live placement policy needs fresh heat) while serving. The counts sit in pinned host memory and cost no measurable
# speed (RHP02: 8 sessions 2,473 against 2,513 tok/s, inside the run-to-run spread), so it is on by default, and
# tools/lab/radiance/heat-placement/heat-watch.py (the start message prints the command) says when the active --heat file stops fitting. Invert the file, restart with --heat and a larger
# --budget (RHP10: workload-recorded heat plus gfx1100=23500 gave +34% prefill at 675K used context on a corpus it had not seen).
# --host-pool-mib N: pinned host memory for the engine's expert tier (explicit value required until v2). With the helpers owning almost every
#   expert the engine uses ~7.1 GiB of it; the old 22528 pinned ~16 GiB unused (RSC03 will derive this from ownership).
# --host-kv-mib N: host KV tier (--prefix-cache-mib) for finished sessions (explicit value required until v2; 0 = off). Without it finished
#   sessions keep their KV in VRAM, the KV cache stops lending VRAM to the engine's experts and a long-running server
#   loses ~half its prefill speed (49K prefill after an ~1M-token session fill: 1,901 -> 3,165 tok/s with the tier).
#   With the 47.7 GiB n-gram table in RAM, 9 GiB pool + 12 GiB tier is about the most that fits; a 24 GiB tier is refused.
# reports how much of the worst case it backs).
# --chunk TOKENS: the largest step (default 2048): the activation arena and the prefill staging are sized from it, so a smaller chunk leaves the KV cache more VRAM.
# --ngram-placement ram|disk|vram: where the 47.7 GiB n-gram table lives (default ram: pinned host memory the cards read). disk reads its rows from the
# container and frees that RAM for --host-pool-mib.
# --num-spec N: the MTP speculative depth (default 3; 0 turns speculation off). It is part of the plan and calibration key, so a depth other than the
# calibrated one starts without the live policy (the launcher's live placement line says so); a depth sweep states its policy state per arm.
# --kv-dtype T: the KV cache type (default fp8; fp8_v4 costs 0.78 of it per token and k6_v4 0.69, both lossy). There is no raw engine argument: the engine's
# other flags are manifest fields (spec.experiment.engine: profile_ops, kv_resident, debug_selection, deterministic; give them in the
# --calibrate: LAB ONLY. Requires --live-policy FILE (explicit
# observe-or-move policy), --record-heat enabled and two distinct prompt
# files (--calibration-prompts-a and --calibration-prompts-b). One prompt per
# line; both workload orders (A,B and B,A) are repeated until convergence.
# The helper prints one RHP16_CALIBRATION_JSON line; the launcher stops serving,
# validates it against this run's strict v1 manifest, atomically ingests it into
# $BC_WORK_ROOT/live-policy-calibration.toml and reports that path. No production
# start can use --calibrate. A missing measured quantity stays absent with a
# reason and will NOT activate an uncalibrated production policy.
# --experiment FILE object, e.g. {"engine": {"profile_ops": "on", "kv_resident": 65536}}).
# --ngram-device INDEX: an inventory card (not the engine, not a helper) that holds as much of the n-gram table as fits and gathers its rows (radiance's own --ngram-device; the old libr3ngram plugin and its R3_NGRAM_DEV are gone);
# the rows past its VRAM stay in a host copy. The card then sits in the engine process beside the engine card.
# --max-model-len TOKENS: the longest context a sequence may have (default 262144, the trained context).
# --rope-scaling S: default none (contexts within the trained 262144); yarn:4 for the 1M stretch.
# --max-num-seqs N: sessions served at once (default 8; each costs per-sequence state in VRAM and checkpoint slots in pinned RAM).
# --prefix-resume MODE: default nearest; off leaves the flag out (engines before 1.14.0 do not have it).
# --gpu-power (or GPU_POWER=1): apply the cards' configured power settings ([host.*.devices.power]) for this run and
# put back what the cards ran before at stop. It is all or nothing and only for this layout's cards: a setting the cards
# cannot take (outside the range the card reports, no write access, no gpu lease held) stops the launch with exit 97 before
# anything starts, listing the setup steps still missing; the original values are journalled and restored after every
# process has exited.
# Serving manifest and plan check (RSC03 / RSC08 S3 / RSC11, mandatory, no way to skip it): every start of every layout asks bc-serve
#   for the measured v1 manifest of the exact configuration it launches; the engine (and the helpers, with helpers) start from that
#   one file (--manifest M --role ...), and the engine checks its live memory plan against the plan the manifest carries before it
#   allocates anything. This script exports no setting of its own: the process environment is only where the runtime lives. A
#   helper-free layout takes its options from its server-args in layouts.toml (each one a manifest setting) and has no hand-off.
#   The first start of a configuration that has no measured plan (a cache miss, logged once with its reason)
#   first MEASURES it: the helpers (if any) come up on their cards and the engine stops after its memory plan, about 1-2 minutes, then
#   the real start goes ahead. That measuring start uses the GPUs exactly as a launch does, so the first launch of a new
#   configuration needs the gpu lease like any launch (agent-lease acquire gpu). Any setting above (layout, sessions, context,
#   pool, KV tier, chunk, n-gram placement, KV dtype, rope, resume) changes the configuration. Plans live in $RUN/plan-cache.
#   --ngram-device is refused (it is not a manifest field yet).
# HELPER_WRAP=<command>: run each helper under it (e.g. a profiler wrapper; its arguments --role helperN tell it which helper).
# ENGINE_WRAP=<command>: run the engine under it (a debugger, numactl, strace; the stop command signals the engine through the
#   wrapper, so it must forward SIGINT/SIGTERM). Not rocprofv3: radiance drives the card through its own AQL backend and fails
#   at start under rocprof's interception (2026-11-01); for the engine's per-op device times put {"engine": {"profile_ops": "on"}} in the --experiment FILE.
set -u

WORK=${BC_WORK_ROOT:?set BC_WORK_ROOT (source tools/env/bigcherry-env.sh)}
RUN=$WORK/flashnext
PIDS=$RUN/pids
SHM=/r3-moe-helper
SHM_DIR=${SHM_DIR:-/dev/shm}   # where POSIX shared memory lives; a test points it at a scratch directory
PORT=36080
HEAT=$WORK/heat/flashnext-mixed.hot-first
BUILD=""
GPU_POWER=${GPU_POWER:-0}
LAYOUT=${FLASHNEXT_LAYOUT:-flashnext-4gpu}
BUDGETS=()
RECORD=$RUN/heat.live   # RHP02: on by default; --record-heat off disables
HOST_POOL_MIB=             # no fallback: a v1 lab launch must pass an explicit measured --host-pool-mib
HOST_KV_MIB=               # no fallback: a v1 lab launch must pass an explicit measured --host-kv-mib
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
# The live placement policy (RHP16) judges heat younger than its derived max_heat_age_ns (about 10 s on flashnext-4gpu) and evaluates every
# cadence (about 5 s); a 60 s histogram period left it with a feed it called stale (26 'no_live_feed' lines, 6 evaluations and no move in 15
# minutes of traffic, runs/histogram-period-default). Every calibration and on/off A/B ran at 1 s.
RECORD_PERIOD=1
LIVE_POLICY_FILE=""   # a JSON object with every spec.handoff.placement.live_policy member (docs/design/serving-manifest-v1)
EXPERIMENT_FILE=""    # a JSON object for spec.experiment (lab profile)
CALIBRATE=0           # lab-only measurement run, never a serving hand-off
CALIBRATION_PROMPTS_A=""
CALIBRATION_PROMPTS_B=""
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
    --num-spec) CUSTOM_SERVE_OPTIONS+=(--num-spec); NUM_SPEC=$2; shift 2 ;;   # MTP depth (0 = no speculation); another depth is another plan key
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
    --calibration-prompts-a) CALIBRATION_PROMPTS_A=$2; shift 2 ;;
    --calibration-prompts-b) CALIBRATION_PROMPTS_B=$2; shift 2 ;;
    --canary-prompt) CANARY_PROMPT=$2; shift 2 ;;
    --canary-expect) CANARY_EXPECT=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ "$CALIBRATE" = 1 ]; then
  if [ "$CMD" != start ] || [ -z "$LIVE_POLICY_FILE" ]; then
    echo "RADIANCE_EXIT=2 --calibrate is lab-only: an explicit --live-policy is required" >&2
    exit 2
  fi
  if [ -n "$EXPERIMENT_FILE" ]; then
    echo "RADIANCE_EXIT=2 --calibrate measures the production plan: it takes no --experiment" >&2
    exit 2
  fi
  for prompts in "$CALIBRATION_PROMPTS_A" "$CALIBRATION_PROMPTS_B"; do
    if [ -z "$prompts" ] || [ ! -s "$prompts" ]; then
      echo "RADIANCE_EXIT=2 --calibrate requires nonempty --calibration-prompts-a and --calibration-prompts-b files" >&2
      exit 2
    fi
    if ! grep -q "[^[:space:]]" "$prompts"; then
      echo "RADIANCE_EXIT=2 --calibrate needs nonblank prompts in both workload files" >&2
      exit 2
    fi
  done
  if cmp -s "$CALIBRATION_PROMPTS_A" "$CALIBRATION_PROMPTS_B"; then
    echo "RADIANCE_EXIT=2 --calibrate requires distinct heat-shifting workloads A and B" >&2
    exit 2
  fi
  if [ "$RECORD" = off ]; then
    echo "RADIANCE_EXIT=2 --calibrate needs the live route histogram (--record-heat cannot be off)" >&2
    exit 2
  fi
  # The manifest is a lab-profile one (bc-serve --calibration-run) with the explicit policy and no experiment, so the plan it
  # measures is the production plan and the record applies to the next production start.
  if ! python3 - "$LIVE_POLICY_FILE" <<'RHP16_VALIDATE'
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        policy = json.load(f)
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
  # power settings this launcher applied go back to what the cards ran before, but only once every process that used the
  # cards has exited (a survivor keeps the settings and the journal; that is reported, not papered over). The journal
  # names the run; a failed restore keeps it.
  if [ -f "$RUN/gpu-power" ]; then
    if [ "$rc" = 0 ]; then
      if PYTHONPATH=tools python3 -m bigcherry gpu-power restore --run-id "$(cat "$RUN/gpu-power")" --processes-exited >&2; then
        rm -f "$RUN/gpu-power"
      else
        rc=1
      fi
    else
      echo "gpu-power: a serving process survived; power settings NOT restored (journal $(cat "$RUN/gpu-power"))" >&2
    fi
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

# The C++ resolver's KFD/PCI BDF assignments are authoritative. Select its
# build before asking Python for layout values; never use a separate host-card
# list or a resolver from a different build.
if [ -z "$BUILD" ]; then
  for d in $(ls -td "$WORK"/builds/*/*/ 2>/dev/null); do
    [ -x "${d}bin/radiance" ] && [ -x "${d}radiance_home/bin/bc-serve" ] && [ -x "${d}radiance_home/helpers/r3_moe_worker" ] && { BUILD=${d%/}; break; }
  done
fi
export BC_SERVE_BINARY="$BUILD/radiance_home/bin/bc-serve"
export RADIANCE_HOME="$BUILD/radiance_home"
BUDGET_ARGS=(); for b in "${BUDGETS[@]:-}"; do [ -n "$b" ] && BUDGET_ARGS+=(--budget "$b"); done
LAYOUT_OUT=$(PYTHONPATH=tools python3 -m bigcherry layout env "$LAYOUT" "${BUDGET_ARGS[@]}") || { echo "RADIANCE_EXIT=99 layout '$LAYOUT' does not resolve on this host (above)"; exit 99; }
eval "$LAYOUT_OUT"
# A helper layout's server-args are its defaults for the settings below (a TP engine with a helper must hold the experts the helper
# cannot, so it serves fewer or shorter sessions than the one-rank defaults); an option given on the command line still wins.
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then
  layout_default() {   # layout_default CLI_OPTION VARIABLE VALUE
    local o
    for o in ${CUSTOM_SERVE_OPTIONS[@]+"${CUSTOM_SERVE_OPTIONS[@]}"}; do [ "$o" = "$1" ] && return 0; done
    printf -v "$2" '%s' "$3"
  }
  for ((i = 0; i < ${#LAYOUT_SERVER_ARGS[@]}; i += 2)); do
    v=${LAYOUT_SERVER_ARGS[i+1]}
    case "${LAYOUT_SERVER_ARGS[i]}" in
      --max-model-len)          layout_default --max-model-len MAX_LEN "$v" ;;
      --max-num-seqs)           layout_default --max-num-seqs SEQS "$v" ;;
      --max-num-batched-tokens) layout_default --chunk CHUNK "$v" ;;
      --kv-cache-dtype)         layout_default --kv-dtype KV_DTYPE "$v" ;;
      --num-speculative-tokens) NUM_SPEC=$v ;;
      --checkpoint-interval)    CKPT_INTERVAL=$v ;;
      --host-pool-mib)          layout_default --host-pool-mib HOST_POOL_MIB "$v" ;;
      --host-kv-mib)            layout_default --host-kv-mib HOST_KV_MIB "$v" ;;
      --ngram-placement)        layout_default --ngram-placement NGRAM "$v" ;;
      *) echo "RADIANCE_EXIT=2 layout $LAYOUT server-args has ${LAYOUT_SERVER_ARGS[i]}, which the launcher does not set" >&2; exit 2 ;;
    esac
  done
  unset -f layout_default
fi
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

# Optional per-run power settings (PWR04/PWR09): an explicit request is all or nothing. gpu-power preflights every setting on the
# cards of this layout only, verifies the gpu lease, journals the originals, applies and reads back; on any failure it rolls
# back and this launch stops before an engine starts (never serves on half-applied or defaulted cards after being asked
# for settings).
if [ "$GPU_POWER" = 1 ]; then
  mkdir -p "$RUN"
  POWER_RUN_ID="launch-$(date +%Y%m%d-%H%M%S)-$$"
  # The marker comes first so the exit cleanup restores from the journal even when apply fails part way (a restore with no
  # journal is a no-op).
  echo "$POWER_RUN_ID" > "$RUN/gpu-power"
  if ! PYTHONPATH=tools python3 -m bigcherry gpu-power apply --layout "$LAYOUT" --run-id "$POWER_RUN_ID" >&2; then
    echo "RADIANCE_EXIT=97 gpu-power: the requested power settings were not applied (nothing was started)" >&2
    exit 97
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
# S7: derive the budgets at every start from the bc-serve BDF/owner-map
# resolution, model RAD directory, RAM and calibrated retention target.
# The CLI writes shell assignments to stdout and its complete breakdown
# to stderr. There is no implicit fixed-size memory fallback.
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then
  requested_pool=$HOST_POOL_MIB requested_kv=$HOST_KV_MIB
  RESOURCE_ASSIGNMENTS=$(PYTHONPATH=tools python3 -m bigcherry resources derive-launch \
    --layout "$LAYOUT" --model "$MODEL" --bc-serve "$BC_SERVE_BINARY" \
    --radiance-home "$RAD_HOME" --heat "$HEAT" --histogram "$RECORD" \
    --sessions "$SEQS" --context "$MAX_LEN" --ngram "$NGRAM") || {
      echo "RADIANCE_EXIT=97 RSC03 resource derivation refused the start" >&2
      exit 97
    }
  eval "$RESOURCE_ASSIGNMENTS"
  [ -z "$requested_pool" ] || HOST_POOL_MIB=$requested_pool
  [ -z "$requested_kv" ] || HOST_KV_MIB=$requested_kv
  echo "RSC03 resolved host pool $HOST_POOL_MIB MiB, host KV $HOST_KV_MIB MiB" >&2
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

# The process environment is only where the runtime lives. Every setting (the hand-off, the kernel switches, the options) is in the
# serving manifest bc-serve writes, and the engine and the helpers take it from there.
ENGINE_ENV=(RADIANCE_HOME="$RAD_HOME")
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then
  [ "$RECORD" = off ] && RECORD=""
  [ -z "$RECORD" ] || echo "recording the route histogram to $RECORD.rank0 every ${RECORD_PERIOD}s" >&2
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
if [ "$LAYOUT_ENGINE_RANKS" -gt 1 ] && [ -z "$LAYOUT_TP_WIRE" ]; then
  echo "RADIANCE_EXIT=99 multi-rank layout missing tp-wire" >&2
  exit 99
fi
# RSC08 S3 / RSC11: every layout starts from its measured manifest alone (--manifest M --role engine). The n-gram card is therefore
# not passable until it is a manifest field.
if [ -n "$NGRAM_DEV" ]; then
  echo "RADIANCE_EXIT=99 --ngram-device cannot be used: the engine starts from its serving manifest and the n-gram card is not a manifest field yet" >&2
  exit 99
fi

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

# The engine of a layout starts from its serving manifest alone: the manifest is its whole configuration, and a measured
# one also carries the plan it must admit before any weight loads (radiance --manifest M --role engine [--plan-memory FILE]).
# Only the card it sees and where the runtime lives come from here.
start_engine_manifest() {
  local log=$1 manifest=$2; shift 2
  env "${ENGINE_ENV[@]}" ROCR_VISIBLE_DEVICES="$ENGINE_VISIBLE" HIP_VISIBLE_DEVICES="$ENGINE_HIP" \
    setsid ${ENGINE_WRAP:-} "$RAD_BIN" --manifest "$manifest" --role engine "$@" > "$log" 2>&1 9>&- &
}

# The measuring start of the plan check (plan-check.sh): helpers up, the engine runs to the end of its memory plan and
# exits by itself (--plan-memory, with the unmeasured manifest), then the helpers stop. It never serves and never runs a request.
plan_check_start_engine_measure() {
  if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then
    rm -f "$SHM_DIR${SHM}"
    start_helpers "$PLAN_MEASURING_MANIFEST" --measuring || { stop_procs; return 1; }
    # what the helpers hold of the host while they are up: anonymous memory of each plus the shared segment once (file pages are
    # reclaimable and not counted); bc-serve plans the checkpoint slots around it (plan-check.sh passes it with the plan)
    PLAN_HELPERS_RESIDENT_MIB=$(helpers_resident_mib) || { stop_procs; return 1; }
  else
    PLAN_HELPERS_RESIDENT_MIB=0   # no helpers: the engine is alone
  fi
  start_engine_manifest "$RUN/engine-measure.log" "$PLAN_MEASURING_MANIFEST" --plan-memory "$PLAN_MEASURE_FILE"
  local pid=$! rc=0
  echo $pid > "$PIDS/engine"
  wait $pid || rc=$?
  stop_procs || { echo "plan check: the measuring start's processes did not exit; not starting for real" >&2; return 1; }   # processes only: the power settings the real start still needs stay applied
  [ $rc = 0 ] && [ -s "$PLAN_MEASURE_FILE" ]
}

source "$(dirname "${BASH_SOURCE[0]}")/lib/plan-check.sh"
plan_manifest_prepare || { echo "RADIANCE_EXIT=97 the measured serving manifest could not be made (above); nothing was started for serving"; exit 97; }

trap 'exit 1' ERR
rm -f "$SHM_DIR${SHM}"
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ]; then start_helpers "$PLAN_MANIFEST"; fi
if [ "$LAYOUT_HELPER_COUNT" -gt 0 ] && [ ! -e "$SHM_DIR${SHM}" ]; then
  # The helpers reported ready, so helper 0 made the segment; if it is gone now something removed it, and an engine that cannot
  # open it runs without the hand-off and is refused by the plan check with a misleading memory mismatch (seen 2026-11-09).
  echo "RADIANCE_EXIT=1 the helpers are up but $SHM_DIR${SHM} does not exist; not starting the engine" >&2
  false
fi
LOG=$RUN/engine-$STAMP.log
start_engine_manifest "$LOG" "$PLAN_MANIFEST"
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
# rad0013 runtime guard: with MTP speculation the engine records its draft/verify passes and replays them. v1.17 without rad0013
# still records draft pass 0 but refuses every other pass (the serial rounds and the head history), so each speculation round
# issues them live (the decode-step regression rad0013 removes): with a depth above 1, a recorded draft pass other than 0 is
# required. One more canary request covers a log line not yet written before deciding.
mtp_replay_recorded() { grep -qE 'recorded pass: draft pass (-[0-9]+|[1-9][0-9]*) ' "$LOG"; }
mtp_replay_check() {
  local depth
  depth=$(sed -n "s/^I speculation: 'mtp' drafts \([0-9][0-9]*\) token.*/\1/p" "$LOG" | tail -1)
  [ -n "$depth" ] && [ "$depth" -gt 1 ] || return 0
  # --profile-ops times every op and so records no pass at all: nothing to check, and the profiled run is not a measurement.
  grep -q '^W\[0\] per-op timing is on' "$LOG" && return 0
  mtp_replay_recorded && return 0
  start_canary >/dev/null 2>&1 || return 1
  mtp_replay_recorded && return 0
  echo "RADIANCE_EXIT=95 MTP speculation is on (depth $depth) but the engine recorded no draft pass beyond pass 0: its serial draft and verify passes are issued live every round instead of replayed (the v1.17 regression; is rad0013_mtp_pass_replay in this build?); stopping everything this launch started; engine log $LOG" >&2
  return 1
}
mtp_replay_check || exit 95
if [ "$CALIBRATE" = 1 ]; then
  # Four measured-policy workload phases give a finite envelope. The helper
  # still controls convergence; a partial record never activates production.
  CAL_MONITOR=tools/lab/radiance/calibration_watchdog.py
  calibration_limits=$(python3 "$CAL_MONITOR" limits "$LIVE_POLICY_FILE") || {
    echo "calibration: cannot derive finite horizon from selected policy" >&2; exit 2;
  }
  read -r calibration_limit_s calibration_report_s <<< "$calibration_limits"
  calibration_start_s=$SECONDS
  calibration_next_report_s=$calibration_report_s
  calibration_drift_bound_locked=0
  calibration_watch() {
    local elapsed=$((SECONDS - calibration_start_s))
    # Lock the first helper-measured drift-horizon extension. A growing ring
    # must not create an endlessly moving deadline.
    if [ "$calibration_drift_bound_locked" = 0 ]; then
      local derived_limits derived_limit_s derived_report_s
      derived_limits=$(python3 "$CAL_MONITOR" limits "$LIVE_POLICY_FILE" "$helper_log") || return 1
      read -r derived_limit_s derived_report_s <<< "$derived_limits"
      if [ "$derived_limit_s" -gt "$calibration_limit_s" ]; then
        calibration_limit_s=$derived_limit_s
        calibration_drift_bound_locked=1
        echo "calibration: measured helper drift horizon included; finite bound $calibration_limit_s s" >&2
      fi
    fi
    if [ "$elapsed" -ge "$calibration_next_report_s" ]; then
      echo "calibration: COLLECTING elapsed=$elapsed bound=$calibration_limit_s; missing:" >&2
      python3 "$CAL_MONITOR" waiting "$helper_log" >&2 || return 1
      calibration_next_report_s=$((elapsed + calibration_report_s))
    fi
    if [ "$elapsed" -ge "$calibration_limit_s" ]; then
      echo "calibration: FAILED policy-derived bound exhausted; missing:" >&2
      python3 "$CAL_MONITOR" waiting "$helper_log" >&2 || true
      return 1
    fi
  }
  echo "calibration: finite bound $calibration_limit_s s; waiting report period $calibration_report_s s" >&2
  helper_log="$RUN/helper0-$STAMP.log"
  [ -f "$helper_log" ] || { echo "calibration: helper 0 log absent" >&2; exit 1; }
  echo "calibration: running A->B then B->A until convergence" >&2
  # A marker is accepted only after both heat-shift orders completed.
  while :; do
    for prompts in "$CALIBRATION_PROMPTS_A" "$CALIBRATION_PROMPTS_B" \
                   "$CALIBRATION_PROMPTS_B" "$CALIBRATION_PROMPTS_A"; do
      while IFS= read -r calibration_prompt || [ -n "$calibration_prompt" ]; do
        [[ "$calibration_prompt" =~ [^[:space:]] ]] || continue
        body=$(CALIBRATION_PROMPT="$calibration_prompt" python3 -c 'import json,os;print(json.dumps({"model":"default","prompt":os.environ["CALIBRATION_PROMPT"],"max_tokens":24,"temperature":0}))')
        curl -fsS --max-time 120 "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' -d "$body" >/dev/null || {
          echo "calibration: workload request failed" >&2; exit 1;
        }
        for proc in engine helper0; do
          [ -f "$PIDS/$proc" ] && kill -0 "$(cat "$PIDS/$proc")" 2>/dev/null || {
            echo "calibration: $proc exited before convergence" >&2; exit 1;
          }
        done
        calibration_watch || exit 1
      done < "$prompts"
    done
    if python3 "$CAL_MONITOR" marker "$helper_log"; then
      break
    else
      marker_rc=$?
      [ "$marker_rc" = 10 ] || exit 1
    fi
    calibration_watch || exit 1
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
