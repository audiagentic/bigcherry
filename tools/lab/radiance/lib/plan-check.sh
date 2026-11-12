# The launcher's serving manifest for a helper layout (RSC03 / RSC08 S3), sourced by launch-flashnext-best.sh.
#
# Every start asks bc-serve for the measured v1 manifest of the exact configuration it is about to launch: the engine and the
# helpers start from that one file, and the engine admits the plan the manifest carries (engine_view) before any weight loads.
# bc-serve finds the measured plan in its cache or says it has none (exit 10); with none, the launcher measures it first (a
# start with the helpers up and the engine's --plan-memory against the unmeasured manifest, which never serves) and bc-serve
# writes the measured manifest from that plan, storing it in the cache. There is no switch that starts the engine without the
# check, and the floor comes only from bc-serve's selected calibration, never a literal here.
#
# A helper-free layout takes the same path with no helpers: the manifest has no hand-off, the measuring start is the engine alone,
# and the helpers' resident host memory is zero.
#
# Retired by RSC02 with the launcher: bc-serve then supervises the engine and helpers itself.

# plan_check_describe: sets PLAN_SET (the --set arguments) from the launcher's own configuration, so the measuring start and
# the real start describe exactly the same thing to bc-serve. One source, so they cannot differ.
plan_check_describe() {
  if [ "${LAYOUT_HELPER_COUNT:-1}" -eq 0 ]; then plan_check_describe_engine_only "$1"; return; fi
  PLAN_SET=(
    --set "layout.name=\"$LAYOUT\""
    --set "service.max_sessions=$SEQS"
    --set "service.max_context=$MAX_LEN"
    --set "memory.host_pool_mib=$HOST_POOL_MIB"
    --set "memory.host_kv_mib=$HOST_KV_MIB"
    --set "engine.max_num_batched_tokens=$CHUNK"
    --set "engine.ngram_placement=\"$NGRAM\""
    --set "engine.kv_cache_dtype=\"$KV_DTYPE\""
    --set "engine.rope_scaling=\"$ROPE\""
    --set "engine.port=$PORT"
    --set "engine.checkpoint_interval=$CKPT_INTERVAL"
    --set "engine.num_speculative_tokens=$NUM_SPEC"
    --set "handoff.shm=\"$SHM\""
    --set "handoff.histogram_period_s=$RECORD_PERIOD"
  )
  [ "$PREFIX_RESUME" = off ] || PLAN_SET+=(--set "engine.prefix_resume=\"$PREFIX_RESUME\"")
  # --budget ARCH=MIB: the per-helper budgets `bigcherry layout env` resolved with the overrides applied (R3_MOE_WORKER_MIBS, helper
  # order). Without this --set bc-serve derives them from the layout again and the override would reach only the plan key.
  if [ "${#BUDGETS[@]}" -gt 0 ] && [ -n "${BUDGETS[0]}" ]; then
    local kv mibs=""
    for kv in "${LAYOUT_COMMON[@]}"; do case "$kv" in R3_MOE_WORKER_MIBS=*) mibs=${kv#R3_MOE_WORKER_MIBS=} ;; esac; done
    [ -n "$mibs" ] || { echo "plan check: --budget given but the layout resolved no helper budgets" >&2; return 1; }
    PLAN_SET+=(--set "memory.helper_budgets_mib=[$mibs]")
  fi
  # A TP engine's ranks exchange through host memory, never P2P (this host has none), as in the helper-free path below.
  [ "${LAYOUT_ENGINE_RANKS:-1}" -le 1 ] || PLAN_SET+=(--set 'tuning.p2p="off"')
  # What else decides the plan but is not a bc-serve setting: the route-heat file's contents (they decide which experts the
  # helpers own), the per-card budget overrides and the lab experiment. Passed to every bc-serve call as --plan-extra and
  # hashed into the key. (No raw engine flag and no n-gram card: a helper layout refuses both, they are not manifest fields.)
  local heat_id experiment_id=none
  heat_id=$(sha256sum < "$HEAT" | cut -d' ' -f1)
  [ -z "$EXPERIMENT_FILE" ] || experiment_id=$(sha256sum < "$EXPERIMENT_FILE" | cut -d' ' -f1)
  PLAN_SITE=(--site "$1")
  PLAN_EXTRA=(--plan-extra "heat-sha256: $heat_id; budgets: ${BUDGETS[*]:-default}; experiment-sha256: $experiment_id")
  # What the helpers read besides the layout and the settings above: the live policy and the lab experiment objects.
  PLAN_OBJECTS=()
  # The cache is durable across runs, and optional before the first calibration.
  [ -z "${WORK:-}" ] || [ ! -f "$WORK/live-policy-calibration.toml" ] || PLAN_OBJECTS+=(--calibration "$WORK/live-policy-calibration.toml")
  [ -z "$LIVE_POLICY_FILE" ] || PLAN_OBJECTS+=(--live-policy "$LIVE_POLICY_FILE")
  [ "${CALIBRATE:-0}" != 1 ] || PLAN_OBJECTS+=(--calibration-run)
  [ -z "$EXPERIMENT_FILE" ] || PLAN_OBJECTS+=(--experiment "$EXPERIMENT_FILE")
}

# plan_check_describe_engine_only SITE: a helper-free layout. It has no hand-off and no route heat; what it is configured with is
# the layout (cards, tp-wire, model) and its allowed engine options (its server-args), each one a manifest setting. The
# pool, the KV tier and the rest are what bc-serve derives. Tensor-parallel traffic goes through host memory, no P2P.
plan_check_describe_engine_only() {
  # The pool and the KV tier default to 0 (the engine sizes them) and the n-gram table to disk: bc-serve's constants (a 9 GiB pool,
  # a 12 GiB KV tier, the table in RAM) are measured for helper layouts, where the helpers own the experts and the model has an
  # n-gram table; a dense model has none (the plan then refuses RAM), and a single-card Flash-Next needs its whole expert tier in
  # RAM. A layout says otherwise with these server-args (--host-pool-mib, --host-kv-mib, --ngram-placement).
  local pool=0 kv=0 ngram=disk
  PLAN_SET=(--set "layout.name=\"$LAYOUT\"" --set "engine.port=$PORT")
  local args=(${LAYOUT_SERVER_ARGS[@]+"${LAYOUT_SERVER_ARGS[@]}"}) i
  for ((i = 0; i < ${#args[@]}; i += 2)); do
    case "${args[i]}" in
      --max-model-len)           PLAN_SET+=(--set "service.max_context=${args[i+1]}") ;;
      --max-num-seqs)            PLAN_SET+=(--set "service.max_sessions=${args[i+1]}") ;;
      --max-num-batched-tokens)  PLAN_SET+=(--set "engine.max_num_batched_tokens=${args[i+1]}") ;;
      --kv-cache-dtype)          PLAN_SET+=(--set "engine.kv_cache_dtype=\"${args[i+1]}\"") ;;
      --num-speculative-tokens)  PLAN_SET+=(--set "engine.num_speculative_tokens=${args[i+1]}") ;;
      --checkpoint-interval)     PLAN_SET+=(--set "engine.checkpoint_interval=${args[i+1]}") ;;
      --host-pool-mib)           pool=${args[i+1]} ;;
      --host-kv-mib)             kv=${args[i+1]} ;;
      --ngram-placement)         ngram=${args[i+1]} ;;
      *) echo "plan check: layout $LAYOUT server-args has ${args[i]}, which is not a manifest setting" >&2; return 1 ;;
    esac
  done
  PLAN_SET+=(--set "memory.host_pool_mib=$pool" --set "memory.host_kv_mib=$kv" --set "engine.ngram_placement=\"$ngram\"")
  [ "${LAYOUT_ENGINE_RANKS:-1}" -le 1 ] || PLAN_SET+=(--set 'tuning.p2p="off"')
  PLAN_SITE=(--site "$1")
  PLAN_EXTRA=(--plan-extra "budgets: ${BUDGETS[*]:-default}")
  PLAN_OBJECTS=()
}

# plan_manifest_write OUT [unmeasured | measured-plan FILE RANKS HELPERS_MIB ENGINE_MIB]: the v2 serving manifest the engine and the helpers start from,
# written once (bc-serve refuses an existing OUT). With `unmeasured`: the measuring start's manifest (owner map and settings,
# no plan). Otherwise the measured one, from the plan the cache holds for this exact configuration (exit 10 when it holds none),
# or, with `measured-plan FILE ...`, from the plan a measuring start just produced and the host memory the helpers and the engine
# held in it (checked against the configuration, then stored in the cache under this configuration's key).
plan_manifest_write() {
  # the stock power profile observes the cards; a named profile is chosen explicitly, never implied
  local out=$1 serve="$RAD_HOME/bin/bc-serve" extra=() power=(--power-profile stock)
  case "${2:-}" in
    unmeasured)
      "$serve" --write-manifest "$out" --unmeasured "${power[@]}" "${PLAN_SITE[@]}" "${PLAN_SET[@]}" "${PLAN_OBJECTS[@]}" >&2
      return ;;
    measured-plan) extra=(--measured-plan "$3" --rank-demand "$4" --helpers-resident-mib "$5" --engine-resident-mib "$6") ;;
  esac
  "$serve" --write-manifest "$out" --plan-cache "$RUN/plan-cache" --engine-binary "$RAD_BIN" "${power[@]}" "${PLAN_SITE[@]}" "${PLAN_EXTRA[@]}" \
    "${PLAN_SET[@]}" "${PLAN_OBJECTS[@]}" "${extra[@]}" >&2
}

# plan_check_site FILE: the site file bc-serve resolves against (the model, runtime home and heat paths this launcher uses).
plan_check_site() {
  cat > "$1" <<SITE
[engine]
model = "$MODEL"
radiance_home = "$RAD_HOME"
SITE
  if [ "${LAYOUT_HELPER_COUNT:-1}" -gt 0 ]; then
    cat >> "$1" <<SITE
[handoff]
heat = "$HEAT"
histogram = "${RECORD:-/dev/null}"
SITE
  fi
}

# plan_live_placement_report: one line, from the manifest this start will use, saying whether the live expert placement policy is on and why not.
# (bc-serve's own "live policy calibration skipped" message goes to plan-check.err, which a successful start never shows.)
plan_live_placement_report() {
  python3 - "$PLAN_MANIFEST" "${LAYOUT_HELPER_COUNT:-0}" <<'PY' >&2
import json
import os
import sys

manifest_path, helper_count = sys.argv[1], int(sys.argv[2])
expected = os.environ.get("BC_EXPECT_LIVE_POLICY", "")
if expected not in ("", "on", "off", "not_applicable"):
    print("live placement: ERROR -- invalid BC_EXPECT_LIVE_POLICY (on|off|not_applicable)", file=sys.stderr)
    sys.exit(2)
try:
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict) or not isinstance(m.get("spec"), dict):
        raise ValueError("missing spec object")
    spec = m["spec"]
    if helper_count:
        placement = spec["handoff"]["placement"]
        if not isinstance(placement, dict):
            raise ValueError("placement is not an object")
except (OSError, ValueError, KeyError, TypeError) as e:
    print(f"live placement: ERROR -- unreadable or invalid manifest: {e}", file=sys.stderr)
    sys.exit(2)

if not helper_count:
    state, detail = "not_applicable", "layout has no helper-to-helper placement"
elif "live_policy" in placement:
    p = placement["live_policy"]
    try:
        cadence = p["cadence_ns"]
        max_age = p["max_heat_age_ns"]
        period = spec["handoff"]["telemetry"]["period_s"]
        if (not isinstance(period, int) or isinstance(period, bool) or period <= 0 or
            not isinstance(cadence, int) or isinstance(cadence, bool) or cadence <= 0 or
            not isinstance(max_age, int) or isinstance(max_age, bool) or max_age <= 0):
            raise ValueError("non-positive or non-integer heat timing")
        if period * 1_000_000_000 > min(cadence, max_age):
            raise ValueError(
                f"histogram period {period}s exceeds policy cadence {cadence}ns "
                f"or max heat age {max_age}ns"
            )
        detail = ("cadence %.1f s, margin %d ppm, min samples %d, "
                  "effect gate %s, heat period %s s" %
                  (cadence / 1e9, p["margin_ppm"], p["min_samples"],
                   p["effect_gate"], period))
    except (KeyError, TypeError, ValueError) as e:
        print(f"live placement: ERROR -- policy/heat preflight failed: {e}", file=sys.stderr)
        sys.exit(2)
    state = "on"
elif "experiment" in spec:
    # A lab experiment (forced churn) is written under the lab profile, which never derives a live policy, so the
    # writer records no reason; the experiment itself is the explicit decision.
    state, detail = "off", "lab experiment: the lab profile never derives a live policy"
else:
    why = m.get("provenance", {}).get("/spec/handoff/placement", {}).get("source", "")
    if not isinstance(why, str) or not why.startswith("live policy disabled: "):
        print("live placement: ERROR -- policy absent without a manifest reason", file=sys.stderr)
        sys.exit(2)
    state, detail = "off", why.removeprefix("live policy disabled: ")

print(f"live placement: {state.upper()} -- {detail}", file=sys.stderr)
if expected and state != expected:
    print(f"live placement: ERROR -- expected {expected}, manifest says {state}", file=sys.stderr)
    sys.exit(2)
PY
}

# plan_manifest_prepare: sets PLAN_MANIFEST to the measured manifest this start uses. Returns non-zero (after saying why) if it
# cannot be made; the launcher then does not start the engine.
plan_manifest_prepare() {
  local serve="$RAD_HOME/bin/bc-serve" cache="$RUN/plan-cache" site="$RUN/plan-site.toml" rc
  [ -x "$serve" ] || { echo "plan check: no bc-serve in the build ($serve); rebuild with the radiance lane" >&2; return 1; }
  mkdir -p "$cache"
  plan_check_site "$site"
  plan_check_describe "$site" || return 1
  PLAN_MANIFEST="$RUN/manifest-$STAMP-$$.json"
  # 10: no measured plan for this configuration: measure it. 11: the plan was measured with another host pool than its
  # measured inputs need: take the derived pool (a configuration of its own in the cache) and go round again. Each pool
  # is measured once per configuration; two changes of pool without converging is an error.
  local round pool
  for round in 1 2 3 4 5; do
    plan_manifest_write "$PLAN_MANIFEST" > "$RUN/plan-check.out" 2> "$RUN/plan-check.err"
    rc=$?
    case $rc in
      0) break ;;
      10) : > "$RUN/plan-check.err"
          plan_check_measure 2> >(tee "$RUN/plan-check.err" >&2); rc=$?   # shown and kept (the derived pool is read from it)
          [ $rc = 0 ] && break
          [ $rc = 11 ] || return 1 ;&
      11) pool=""
          for _ in $(seq 1 50); do   # the tee behind the measuring start's stderr may lag the exit by a moment
            pool=$(cat "$RUN/plan-check.out" "$RUN/plan-check.err" 2>/dev/null | sed -n 's/^derived_host_pool_mib=\([0-9][0-9]*\)$/\1/p' | tail -1)
            [ -n "$pool" ] && break
            sleep 0.1
          done
          [ -n "$pool" ] || { echo "plan check: bc-serve asked for another host pool without naming it:" >&2; cat "$RUN/plan-check.err" >&2; return 1; }
          [ "${PLAN_POOL_CHANGES:-0}" -lt 2 ] || { echo "plan check: the host pool did not converge (last asked for ${pool} MiB)" >&2; return 1; }
          PLAN_POOL_CHANGES=$(( ${PLAN_POOL_CHANGES:-0} + 1 ))
          echo "plan check: the measured inputs need a ${pool} MiB host pool; measuring with it" >&2
          PLAN_SET+=(--set "memory.host_pool_mib=$pool")
          rm -f "$PLAN_MANIFEST" ;;
      *) echo "plan check: bc-serve --write-manifest failed (exit $rc):" >&2
         cat "$RUN/plan-check.err" >&2
         return 1 ;;
    esac
  done
  [ $rc = 0 ] || { echo "plan check: no measured manifest after $round rounds" >&2; return 1; }
  plan_live_placement_report
}

# plan_check_measure: no measured plan exists for this configuration. Say so once, measure it with a start that brings the
# helpers up and stops the engine after plan(), and let bc-serve write the measured manifest from that plan (and store it).
plan_check_measure() {
  local plan="$RUN/plan-measured.json" ranks="$RUN/plan-ranks.json" t0=$SECONDS
  echo "plan check: no measured plan for this configuration (cache miss); measuring it first (the helpers come up and the engine stops after its memory plan; about 1-2 minutes, once per configuration)" >&2
  rm -f "$plan" "$ranks" "$PLAN_MANIFEST"
  PLAN_MEASURING_MANIFEST="$RUN/manifest-measuring-$STAMP-$$-${PLAN_POOL_CHANGES:-0}.json"   # one per measuring start: manifests are immutable
  plan_manifest_write "$PLAN_MEASURING_MANIFEST" unmeasured || { echo "plan check: the unmeasured manifest was not written (above)" >&2; return 1; }
  PLAN_MEASURE_FILE="$plan" PLAN_RANKS_FILE="$ranks" plan_check_start_engine_measure || { echo "plan check: the measuring start failed; see $RUN/engine-measure.log" >&2; return 1; }
  local engine_mib
  engine_mib=$(sed -n 's/.*--plan-memory: wrote .*; engine start-up took \([0-9][0-9]*\) MiB of MemAvailable;.*/\1/p' "$RUN/engine-measure.log" | tail -1)
  [ -n "$engine_mib" ] && [ -n "${PLAN_HELPERS_RESIDENT_MIB:-}" ] \
    || { echo "plan check: the measuring start did not report the resident host memory (helpers '${PLAN_HELPERS_RESIDENT_MIB:-}' MiB, engine '$engine_mib' MiB); see $RUN/engine-measure.log" >&2; return 1; }
  local rc=0
  plan_manifest_write "$PLAN_MANIFEST" measured-plan "$plan" "$ranks" "$PLAN_HELPERS_RESIDENT_MIB" "$engine_mib" || rc=$?
  [ $rc = 11 ] && return 11   # stored; the caller measures again with the derived host pool
  [ $rc = 0 ] || { echo "plan check: the measured manifest was not written (above)" >&2; return 1; }
  echo "plan check: measured in $((SECONDS - t0)) s; plan stored in $RUN/plan-cache and the manifest written" >&2
}
