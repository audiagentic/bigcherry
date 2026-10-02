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
# pytest in a venv next to the clone (the host python has none); same runner as the Windows suite.
venv=$clone.venv
[ -x "$venv/bin/pytest" ] || { python3 -m venv "$venv" && "$venv/bin/pip" install -q pytest; }
cd "$clone" && PYTHONPATH="tools:." "$venv/bin/pytest" -q -p no:cacheprovider tools/tests 2>&1 | tail -40
