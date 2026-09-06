#!/bin/bash
# Export the build host's environment from config/environment.toml.
#
#   source tools/bigcherry-env.sh              # default host
#   source tools/bigcherry-env.sh build-server # a named host
#
# Exists so commands in docs and scripts can say $BC_MODEL_ROOT rather than
# hardcoding one person's paths. Everything machine-specific lives in
# config/environment.toml; this only reads it.
#
# Sets: BC_HOST BC_ADDRESS BC_HOME BC_REPO BC_CACHE BC_SHARE BC_MODEL_ROOT
#       BC_BENCH_HARNESS BC_ROCM BC_ROCM_SHIM BC_BENCH_PORT BC_PRODUCTION_PORT
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
    cfg="$root/config/environment.toml"
    host="${1:-}"
    if [ ! -f "$cfg" ]; then
        echo "bigcherry-env: no $cfg" >&2
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
    "BC_ADDRESS": h.get("address", ""),
    "BC_HOME": h.get("home", ""),
    "BC_REPO": h.get("repo", ""),
    "BC_CACHE": h.get("cache-root", ""),
    "BC_SHARE": h.get("share", ""),
    "BC_MODEL_ROOT": h.get("model-root", ""),
    "BC_BENCH_HARNESS": h.get("bench-harness", ""),
    "BC_ROCM": h.get("rocm", ""),
    "BC_ROCM_SHIM": h.get("rocm-shim", ""),
    "BC_BENCH_PORT": str(h.get("bench-port", "")),
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

    # Only meaningful when running ON the host; harmless elsewhere.
    if [ -n "${BC_ROCM_SHIM:-}" ] && [ -d "$BC_ROCM_SHIM" ]; then
        export ROCM_PATH="$BC_ROCM_SHIM" HIP_PATH="$BC_ROCM_SHIM"
        case ":$PATH:" in
            *":$BC_ROCM_SHIM/bin:"*) ;;
            *) export PATH="$BC_ROCM_SHIM/bin:$PATH" ;;
        esac
    fi
}

_bc_env_load "${1:-}"
