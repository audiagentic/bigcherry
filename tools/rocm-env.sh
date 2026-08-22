#!/usr/bin/env bash
# Selects a vendored ROCm/HIP toolchain version for the current shell.
#
# bigcherry keeps one or more full ROCm installs under vendor/rocm/<version>/
# (gitignored -- a straight copy of a ROCm install tree, either an AMD ROCm
# for Windows install (e.g. "C:\Program Files\AMD\ROCm\<version>") or a
# Linux ROCm install (e.g. "/opt/rocm-<version>")). This script points
# HIP_PATH / ROCM_PATH at one of them and prepends its bin/ to PATH, matching
# the HIP_PATH/ROCM_PATH convention already read by tools/bigcherry/toolchain.py
# and used throughout docs/reference/BUILD.md. Works identically on Windows
# (git-bash, hipcc.exe) and Linux (Brutus, hipcc) -- same script, same
# vendor/rocm/<version> layout, on both sides of the SMB share.
#
# Must be sourced, not executed, or the env vars only apply to a subshell:
#
#   source tools/rocm-env.sh 7.1
#   source tools/rocm-env.sh --list

_rocm_env_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

_rocm_env_hipcc_name() {
    # Windows install trees carry hipcc.exe; Linux trees carry hipcc.
    local d="$1"
    if [[ -f "${d}bin/hipcc.exe" ]]; then
        echo "hipcc.exe"
    elif [[ -f "${d}bin/hipcc" ]]; then
        echo "hipcc"
    fi
}

_rocm_env_main() {
    local repo_root vendor_root version target hipcc_name hipcc
    repo_root="$(_rocm_env_repo_root)"
    vendor_root="$repo_root/vendor/rocm"
    version="$1"

    if [[ -z "$version" || "$version" == "--list" || "$version" == "-l" ]]; then
        if [[ ! -d "$vendor_root" ]]; then
            echo "No vendored ROCm installs found under $vendor_root"
            echo "Populate one with: rsync -a /opt/rocm-<ver>/ '$vendor_root/<ver>/'  (Linux)"
            echo "                or: robocopy 'C:\\Program Files\\AMD\\ROCm\\<ver>' '$vendor_root\\<ver>' /E  (Windows)"
            return 0
        fi
        echo "Vendored ROCm versions under $vendor_root :"
        local found=0
        for d in "$vendor_root"/*/; do
            [[ -n "$(_rocm_env_hipcc_name "$d")" ]] || continue
            found=1
            local name mark
            name="$(basename "$d")"
            mark=""
            [[ "$ROCM_PATH" == "$vendor_root/$name" ]] && mark=" (active)"
            echo "  $name$mark"
        done
        [[ $found -eq 0 ]] && echo "  (none populated yet)"
        return 0
    fi

    target="$vendor_root/$version"
    hipcc_name="$(_rocm_env_hipcc_name "$target/")"
    if [[ -z "$hipcc_name" ]]; then
        echo "No vendored ROCm '$version' at $target (expected bin/hipcc or bin/hipcc.exe)." >&2
        echo "Run 'source tools/rocm-env.sh --list' to see what's available." >&2
        return 1
    fi

    export ROCM_PATH="$target"
    export HIP_PATH="$target"
    # Strip any previously-selected vendor/rocm bin dir, then prepend the new one.
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "^${vendor_root}/" | tr '\n' ':')"
    export PATH="$target/bin:$PATH"

    echo "ROCm $version selected: $target"
    "$target/bin/$hipcc_name" --version 2>&1 | head -1
}

_rocm_env_main "$@"
unset -f _rocm_env_repo_root _rocm_env_main
