# Sourced by radiance lab scripts. rad_build <log>: prints the build directory of the moe-remote-handoff experiment for
# this checkout. A build is reused when its recorded source trees (git tree ids of engines/radiance -- components,
# patches, recipes -- and config) equal this checkout's, so a lab-script-only change does not rebuild (~5 min); any source change
# builds. The tree id is written to <build>/.lab-source-tree after a build.
rad_tree() {
    git rev-parse HEAD:engines/radiance HEAD:config 2>/dev/null
}
rad_build() {   # $1 build log path
    local tree b plan
    tree=$(rad_tree | tr '\n' ' ')
    [ -n "$(git status --porcelain --untracked-files=no -- engines/radiance config 2>/dev/null)" ] && tree=""   # dirty: always build
    if [ -n "$tree" ]; then
        for b in $(ls -dt "$BC_WORK_ROOT"/builds/*/*/ 2>/dev/null); do
            if [ -f "$b/.lab-source-tree" ] && [ "$(cat "$b/.lab-source-tree")" = "$tree" ] && [ -x "$b/bin/radiance" ]; then
                echo "reusing build ${b%/} (same radiance source tree)" > "$1"
                echo "${b%/}"
                return 0
            fi
        done
    fi
    PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance \
        --experiment moe-remote-handoff > "$1" 2>&1
    plan=$(grep -oE 'ok build_plan_id=[0-9a-f]+' "$1" | tail -1 | cut -d= -f2)
    [ -n "$plan" ] || { grep -E ' error:|FAILED' "$1" | sort -u | head >&2; return 1; }
    b=$(ls -d "$BC_WORK_ROOT"/builds/*/"$plan")
    [ -n "$tree" ] && echo "$tree" > "$b/.lab-source-tree"
    echo "$b"
}
