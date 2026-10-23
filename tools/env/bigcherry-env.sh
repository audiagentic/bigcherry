#!/bin/bash
# Export the build host's environment from config/environment.local.toml.
#
#   source tools/bigcherry-env.sh              # default host
#   source tools/bigcherry-env.sh dev # a named host
#
# Exists so commands in docs and scripts can say $BC_MODEL_ROOT rather than
# hardcoding one person's paths. Everything machine-specific lives in
# config/environment.local.toml; this only reads it.
#
# Sets: BC_HOST BC_REPO_ROOT BC_CACHE_ROOT BC_MODEL_ROOT
#       BC_BENCH_HARNESS BC_ROCM_ROOT BC_TEST_PORT BC_PRODUCTION_PORT
# and, for convenience on the host itself, ROCM_PATH/HIP_PATH/PATH via the
# shim -- which campaigns require, because /opt/rocm exposes amdclang rather
# than clang and campaigns build compiler paths as <hip-path>/bin/clang.

_bc_env_root() {
    local d="${BASH_SOURCE[0]%/*}"
    ( cd "$d/../.." 2>/dev/null && pwd )
}

_bc_env_load() {
    local root cfg host
    root="$(_bc_env_root)"
    cfg="${BIGCHERRY_ENVIRONMENT:-$root/config/environment.local.toml}"
    host="${1:-}"
    if [ ! -f "$cfg" ]; then
        echo "bigcherry-env: no $cfg (copy config/environment.example.toml to config/environment.local.toml, or set BIGCHERRY_ENVIRONMENT)" >&2
        return 1
    fi
    # tomllib rather than grep/sed: the file is TOML and parsing it as text
    # would silently misread a multi-line string or a nested table.
    eval "$(python3 - "$cfg" "$host" <<'PY'
import sys, tomllib, shlex
cfg, want = sys.argv[1], (sys.argv[2] or "")
with open(cfg, "rb") as fh:
    doc = tomllib.load(fh)
host = want or doc.get("default-host")
h = (doc.get("host") or {}).get(host)
if not h:
    print(f'echo "bigcherry-env: unknown host {host!r}" >&2; false', end="")
    raise SystemExit(0)
pairs = {
    "BC_HOST": h.get("hostname", host),
    "BC_REPO_ROOT": h.get("repo-root", ""),
    "BC_CACHE_ROOT": h.get("cache-root", ""),
    "BC_MODEL_ROOT": h.get("model-root", ""),
    "BC_WORK_ROOT": h.get("work-root", ""),
    "BC_TMP_ROOT": h.get("tmp-root", ""),
    "BC_DEV_ROOT": h.get("dev-root", ""),
    "BC_BENCH_HARNESS": h.get("bench-harness", ""),
    "BC_ROCM_ROOT": h.get("rocm-root", ""),
    "BC_TEST_PORT": str(h.get("test-port", "")),
    "BC_PRODUCTION_PORT": str(h.get("production-port", "")),
}
for k, v in pairs.items():
    print(f"export {k}={shlex.quote(str(v))}")
# A compact device table, so scripts need not re-parse the TOML.
devs = h.get("devices") or []
print("export BC_DEVICES=" + shlex.quote(
    " ".join(f"{d.get('index')}:{d.get('arch')}" for d in devs)))
for d in devs:
    arch = str(d.get("arch", "")).upper()
    idx = d.get("index")
    print(f"export BC_DEV_{arch}_{idx}={idx}")
PY
)" || return 1

}

_bc_env_load "${1:-}"
