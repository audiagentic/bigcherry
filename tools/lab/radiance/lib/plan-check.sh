# The launcher's engine start-up plan check (RSC03 / RSC08 A3), sourced by launch-flashnext-best.sh.
#
# Every start asks bc-serve for the engine's check flags (--plan-check FILE --plan-check-floor-ppm N) for the exact
# configuration it is about to launch. bc-serve finds the measured plan in the cache or says it has none; with none, the
# launcher measures it first (a start with the helpers up and the engine's --plan-memory, which never serves), stores it
# under the key it asked for, then starts for real with the flags. There is no switch that starts the engine without the
# check, and the floor comes only from bc-serve's selected calibration, never a literal here.
#
# Interim: serving manifest v1's engine_view replaces the flags at the RSC08 cutover, and this file is deleted with the
# launcher (RSC02).

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
  )
  [ "$PREFIX_RESUME" = off ] || PLAN_SET+=(--set "engine.prefix_resume=\"$PREFIX_RESUME\"")
  PLAN_SITE=(--site "$1")
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

# plan_check_flags: sets PLAN_CHECK_ARGS to the flags the engine must start with. Returns non-zero (after saying why) if
# the check cannot be armed; the launcher then does not start the engine.
plan_check_flags() {
  local serve="$RAD_HOME/bin/bc-serve" cache="$RUN/plan-cache" site="$RUN/plan-site.toml" out rc
  [ -x "$serve" ] || { echo "plan check: no bc-serve in the build ($serve); rebuild with the radiance lane" >&2; return 1; }
  mkdir -p "$cache"
  plan_check_site "$site"
  plan_check_describe "$site"
  out=$("$serve" --plan-check-args --plan-cache "$cache" --engine-binary "$RAD_BIN" "${PLAN_SITE[@]}" "${PLAN_SET[@]}" 2> "$RUN/plan-check.err")
  rc=$?
  if [ $rc = 10 ]; then
    plan_check_measure "$serve" "$cache" || return 1
    out=$("$serve" --plan-check-args --plan-cache "$cache" --engine-binary "$RAD_BIN" "${PLAN_SITE[@]}" "${PLAN_SET[@]}" 2> "$RUN/plan-check.err")
    rc=$?
  fi
  if [ $rc != 0 ]; then
    echo "plan check: bc-serve --plan-check-args failed (exit $rc):" >&2
    cat "$RUN/plan-check.err" >&2
    return 1
  fi
  mapfile -t PLAN_CHECK_ARGS <<< "$out"
  [ ${#PLAN_CHECK_ARGS[@]} = 4 ] && [ "${PLAN_CHECK_ARGS[0]}" = --plan-check ] && [ "${PLAN_CHECK_ARGS[2]}" = --plan-check-floor-ppm ] \
    || { echo "plan check: unexpected bc-serve output: $out" >&2; return 1; }
  grep -h '^plan:' "$RUN/plan-check.err" >&2
}

# plan_check_measure SERVE CACHE: no measured plan exists for this configuration. Say so once, measure it with a start that
# brings the helpers up and stops the engine after plan(), and store it under the key asked for.
plan_check_measure() {
  local serve=$1 cache=$2 key plan="$RUN/plan-measured.json" t0=$SECONDS
  key=$("$serve" --plan-key --plan-cache "$cache" --engine-binary "$RAD_BIN" "${PLAN_SITE[@]}" "${PLAN_SET[@]}") || return 1
  echo "plan check: no measured plan for this configuration (cache miss, key ${key:0:16}); measuring it first (the helpers come up and the engine stops after its memory plan; about 1-2 minutes, once per configuration)" >&2
  rm -f "$plan"
  PLAN_MEASURE_FILE="$plan" plan_check_start_engine_measure || { echo "plan check: the measuring start failed; see $RUN/engine-measure.log" >&2; return 1; }
  "$serve" --plan-store --expect-key "$key" --plan-file "$plan" --plan-cache "$cache" --engine-binary "$RAD_BIN" "${PLAN_SITE[@]}" "${PLAN_SET[@]}" >&2 \
    || { echo "plan check: the measured plan was not stored (above)" >&2; return 1; }
  echo "plan check: measured in $((SECONDS - t0)) s; plan stored in $cache (key ${key:0:16})" >&2
}
