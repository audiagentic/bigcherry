#!/usr/bin/env bash
# shellcheck disable=SC2045
# Sourced by radiance lab scripts. rad_build <log>: prints a build directory with the production remote-MoE hand-off for
# this checkout. A build is reused when its recorded source trees (git tree ids of engines/radiance -- components,
# patches, recipes -- and config) and its ROCm SDK (root and version) equal this checkout's, so a lab-script-only change
# does not rebuild (~5 min); any source or SDK change builds. The key is written to <build>/.lab-source-tree after a build.
rad_tree() {
    local sdk
    git rev-parse HEAD:engines/radiance HEAD:config 2>/dev/null
    sdk=$(readlink -f "${BC_ROCM_ROOT:?no ROCm SDK selected (rocm-root in the environment file)}") \
        && echo "$sdk $(cat "$sdk/.info/version")"
}
rad_build() {   # $1 build log path
    local tree b plan
    # Same path and compiler hashing policy as CI. Lab host/toolchain parity
    # is enforced by ci_ccache.py before any main cache entry is imported.
    export CCACHE_DIR="$BC_WORK_ROOT/.ccache"
    CCACHE_BASEDIR=$(git rev-parse --show-toplevel)
    export CCACHE_BASEDIR
    export CCACHE_NOHASHDIR=true CCACHE_COMPILERCHECK=content
    tree=$(rad_tree | tr '\n' ' ')
    [ -n "$(git status --porcelain --untracked-files=no -- engines/radiance config 2>/dev/null)" ] && tree=""   # dirty: always build
    if [ -n "$tree" ]; then
        for b in $(ls -dt "$BC_WORK_ROOT"/builds/*/*/ 2>/dev/null); do
            if [ -f "$b/.lab-source-tree" ] && [ "$(cat "$b/.lab-source-tree")" = "$tree" ] && [ -x "$b/bin/radiance" ] && [ -x "$b/radiance_home/helpers/r3_moe_worker" ]; then
                echo "reusing build ${b%/} (same radiance source tree)" > "$1"
                echo "${b%/}"
                return 0
            fi
        done
    fi
    ccache -z
    if ! PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance > "$1" 2>&1; then
        ccache -s >> "$1"
        return 1
    fi
    ccache -s >> "$1"
    plan=$(grep -oE 'ok build_plan_id=[0-9a-f]+' "$1" | tail -1 | cut -d= -f2)
    [ -n "$plan" ] || { grep -E ' error:|FAILED' "$1" | sort -u | head >&2; return 1; }
    b=$(ls -d "$BC_WORK_ROOT"/builds/*/"$plan")
    [ -n "$tree" ] && echo "$tree" > "$b/.lab-source-tree"
    echo "$b"
}
