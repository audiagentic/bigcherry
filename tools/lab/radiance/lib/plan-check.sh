# The launcher's serving manifest for a helper layout (RSC03 / RSC08 S3), sourced by launch-flashnext-best.sh.
#
# Every start asks bc-serve for the measured v1 manifest of the exact configuration it is about to launch: the engine and the
# helpers start from that one file, and the engine admits the plan the manifest carries (engine_view) before any weight loads.
# bc-serve finds the measured plan in its cache or says it has none (exit 10); with none, the launcher measures it first (a
# start with the helpers up and the engine's --plan-memory against the unmeasured manifest, which never serves) and bc-serve
# writes the measured manifest from that plan, storing it in the cache. There is no switch that starts the engine without the
# check, and the floor comes only from bc-serve's selected calibration, never a literal here.
#
# Retired by RSC02 with the launcher: bc-serve then supervises the engine and helpers itself.

# plan_check_describe: sets PLAN_SET (the --set arguments) from the launcher's own configuration, so the measuring start and
# the real start describe exactly the same thing to bc-serve. One source, so they cannot differ.
plan_check_describe() {
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
  [ ! -f "$WORK/live-policy-calibration.toml" ] || PLAN_OBJECTS+=(--calibration "$WORK/live-policy-calibration.toml")
  [ -z "$LIVE_POLICY_FILE" ] || PLAN_OBJECTS+=(--live-policy "$LIVE_POLICY_FILE")
  [ -z "$EXPERIMENT_FILE" ] || PLAN_OBJECTS+=(--experiment "$EXPERIMENT_FILE")
}

# plan_manifest_write OUT [unmeasured | measured-plan FILE HELPERS_MIB ENGINE_MIB]: the v1 serving manifest the engine and the helpers start from,
# written once (bc-serve refuses an existing OUT). With `unmeasured`: the measuring start's manifest (owner map and settings,
# no plan). Otherwise the measured one, from the plan the cache holds for this exact configuration (exit 10 when it holds none),
# or, with `measured-plan FILE ...`, from the plan a measuring start just produced and the host memory the helpers and the engine
# held in it (checked against the configuration, then stored in the cache under this configuration's key).
plan_manifest_write() {
  local out=$1 serve="$RAD_HOME/bin/bc-serve" extra=()
  case "${2:-}" in
    unmeasured)
      "$serve" --write-manifest "$out" --unmeasured "${PLAN_SITE[@]}" "${PLAN_SET[@]}" "${PLAN_OBJECTS[@]}" >&2
      return ;;
    measured-plan) extra=(--measured-plan "$3" --helpers-resident-mib "$4" --engine-resident-mib "$5") ;;
  esac
  "$serve" --write-manifest "$out" --plan-cache "$RUN/plan-cache" --engine-binary "$RAD_BIN" "${PLAN_SITE[@]}" "${PLAN_EXTRA[@]}" \
    "${PLAN_SET[@]}" "${PLAN_OBJECTS[@]}" "${extra[@]}" >&2
}

# plan_check_site FILE: the site file bc-serve resolves against (the model, runtime home and heat paths this launcher uses).
plan_check_site() {
  cat > "$1" <<SITE
[engine]
model = "$MODEL"
radiance_home = "$RAD_HOME"
[handoff]
heat = "$HEAT"
histogram = "${RECORD:-/dev/null}"
SITE
}

# plan_manifest_prepare: sets PLAN_MANIFEST to the measured manifest this start uses. Returns non-zero (after saying why) if it
# cannot be made; the launcher then does not start the engine.
plan_manifest_prepare() {
  local serve="$RAD_HOME/bin/bc-serve" cache="$RUN/plan-cache" site="$RUN/plan-site.toml" rc
  [ -x "$serve" ] || { echo "plan check: no bc-serve in the build ($serve); rebuild with the radiance lane" >&2; return 1; }
  mkdir -p "$cache"
  plan_check_site "$site"
  plan_check_describe "$site"
  PLAN_MANIFEST="$RUN/manifest-$STAMP-$$.json"
  plan_manifest_write "$PLAN_MANIFEST" 2> "$RUN/plan-check.err"
  rc=$?
  if [ $rc = 10 ]; then
    plan_check_measure || return 1
  elif [ $rc != 0 ]; then
    echo "plan check: bc-serve --write-manifest failed (exit $rc):" >&2
    cat "$RUN/plan-check.err" >&2
    return 1
  fi
}

# plan_check_measure: no measured plan exists for this configuration. Say so once, measure it with a start that brings the
# helpers up and stops the engine after plan(), and let bc-serve write the measured manifest from that plan (and store it).
plan_check_measure() {
  local plan="$RUN/plan-measured.json" t0=$SECONDS
  echo "plan check: no measured plan for this configuration (cache miss); measuring it first (the helpers come up and the engine stops after its memory plan; about 1-2 minutes, once per configuration)" >&2
  rm -f "$plan" "$PLAN_MANIFEST"
  PLAN_MEASURING_MANIFEST="$RUN/manifest-measuring-$STAMP-$$.json"
  plan_manifest_write "$PLAN_MEASURING_MANIFEST" unmeasured || { echo "plan check: the unmeasured manifest was not written (above)" >&2; return 1; }
  PLAN_MEASURE_FILE="$plan" plan_check_start_engine_measure || { echo "plan check: the measuring start failed; see $RUN/engine-measure.log" >&2; return 1; }
  local engine_mib
  engine_mib=$(sed -n 's/.*--plan-memory: wrote .*; engine start-up took \([0-9][0-9]*\) MiB of MemAvailable;.*/\1/p' "$RUN/engine-measure.log" | tail -1)
  [ -n "$engine_mib" ] && [ -n "${PLAN_HELPERS_RESIDENT_MIB:-}" ] \
    || { echo "plan check: the measuring start did not report the resident host memory (helpers '${PLAN_HELPERS_RESIDENT_MIB:-}' MiB, engine '$engine_mib' MiB); see $RUN/engine-measure.log" >&2; return 1; }
  plan_manifest_write "$PLAN_MANIFEST" measured-plan "$plan" "$PLAN_HELPERS_RESIDENT_MIB" "$engine_mib" \
    || { echo "plan check: the measured manifest was not written (above)" >&2; return 1; }
  echo "plan check: measured in $((SECONDS - t0)) s; plan stored in $RUN/plan-cache and the manifest written" >&2
}
