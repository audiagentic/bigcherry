#!/bin/bash
# Print the campaign work root: $BIGCHERRY_WORK_ROOT, else the [env] entry of
# the untracked config/environment.local.toml, else <repo>/work.
# Usage: work-root.sh <repo-root>
root=$1
if [ -n "${BIGCHERRY_WORK_ROOT:-}" ]; then echo "$BIGCHERRY_WORK_ROOT"; exit 0; fi
python3 - "$root" <<'PY'
import sys, tomllib
from pathlib import Path
root = Path(sys.argv[1])
local = root / "config" / "environment.local.toml"
env = tomllib.loads(local.read_text())["env"] if local.is_file() else {}
print(env.get("BIGCHERRY_WORK_ROOT") or root / "work")
PY
