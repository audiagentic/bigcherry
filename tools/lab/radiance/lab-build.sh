# Sourced by radiance lab scripts. rad_build: prints the build directory of the moe-remote-handoff experiment for
# this checkout. A build is reused when its recorded source trees (git tree ids of engines/radiance -- components,
# patches, recipes -- and config) and its ROCm SDK (root and version) equal this checkout's, so a lab-script-only change
# does not rebuild (~5 min); any source or SDK change builds. The key is written to <build>/.lab-source-tree after a build.
rad_tree() {
    local sdk
    git rev-parse HEAD:engines/radiance HEAD:config 2>/dev/null
    sdk=$(readlink -f "${BC_ROCM_ROOT:?no ROCm SDK selected (rocm-root in the environment file)}") \
        && echo "$sdk $(cat "$sdk/.info/version")"
}
rad_build() {
    local tree b run_id log_dir
    tree=$(rad_tree | tr '\n' ' ')
    [ -n "$(git status --porcelain --untracked-files=no -- engines/radiance config 2>/dev/null)" ] && tree=""   # dirty: always build
    if [ -n "$tree" ]; then
        for b in $(ls -dt "$BC_WORK_ROOT"/builds/*/*/ 2>/dev/null); do
            if [ -f "$b/.lab-source-tree" ] && [ "$(cat "$b/.lab-source-tree")" = "$tree" ] && [ -x "$b/bin/radiance" ] && [ -d "$b/build-logs" ]; then
                echo "reusing build ${b%/} (same radiance source tree); logs: ${b%/}/build-logs/" >&2
                echo "${b%/}"
                return 0
            fi
        done
    fi
    run_id="rad-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}"
    log_dir="$BC_WORK_ROOT/build-logs/$run_id"
    # The build command owns persistent output; stdout stays the build dir for B=$(rad_build).
    if ! PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance \
        --experiment moe-remote-handoff --run-id "$run_id" >/dev/null 2>&1; then
        echo "build failed; logs: $log_dir" >&2
        [ ! -f "$log_dir/build.log" ] || grep -E ' error:|FAILED' "$log_dir/build.log" | sort -u | head >&2
        return 1
    fi
    echo "build logs: $log_dir" >&2
    b=$(python3 -c 'import json,sys; print(next(iter(json.load(open(sys.argv[1]))["lanes"].values()))["build_dir"])' "$log_dir/summary.json")
    [ -n "$tree" ] && echo "$tree" > "$b/.lab-source-tree"
    echo "$b"
}
