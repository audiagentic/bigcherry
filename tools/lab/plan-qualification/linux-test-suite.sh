#!/bin/bash
# Run the full offline test suite on the Linux campaign host from a clean checkout of the pushed branch
# (covers the Linux-only tests that skip on the Windows controller). Queued so it never overlaps a
# timed measurement. Usage: linux-test-suite.sh <clone-dir> <branch>
set -u -o pipefail
clone=$1 branch=$2
src=$(cd "$(dirname "$0")/../../.." && pwd)
[ -d "$clone/.git" ] || git clone -q --no-checkout "$src" "$clone"
git -C "$clone" fetch -q "$src" "$branch" && git -C "$clone" checkout -q --detach FETCH_HEAD
echo "testing $(git -C "$clone" log -1 --oneline)"
# vendor/llama.cpp is gitignored, so a clean clone has none; tests that read the pinned source need it.
# Link the workspace's vendor checkout (tests read pristine bytes via `git show HEAD:` where it matters).
[ -e "$clone/vendor/llama.cpp" ] || { mkdir -p "$clone/vendor" && ln -s "$src/vendor/llama.cpp" "$clone/vendor/llama.cpp"; }
# Host settings are never committed: use this host's config/environment.local.toml.
[ -e "$clone/config/environment.local.toml" ] || ln -s "$src/config/environment.local.toml" "$clone/config/environment.local.toml"
# pytest in a venv next to the clone (the host python has none); same runner as the Windows suite.
venv=$clone.venv
[ -x "$venv/bin/pytest" ] || { python3 -m venv "$venv" && "$venv/bin/pip" install -q pytest; }
report=${BC_RUN_DIR:-$clone}/pytest-report.txt
cd "$clone" && PYTHONPATH="tools:." "$venv/bin/pytest" -q -p no:cacheprovider -rfE tools/tests > "$report" 2>&1
rc=$?
grep -E "^(FAILED|ERROR) " "$report" | sed "s/ - .*//" | head -100
tail -3 "$report"
echo "full report: $report"
exit $rc
