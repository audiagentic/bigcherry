"""MEN10 step g: production code must not hard-code engine names.

Engine identity flows from ``args.engine`` / ``ProjectContext`` /
``descriptor.engine``. Literal engine names and ``engine ==`` comparisons are
allowed only in ``core/engines.py`` and in modules that are llama.cpp-only
tools by design (listed in ALLOWLIST with a reason).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

PRODUCT_ROOT = Path(__file__).resolve().parents[2] / "bigcherry"

# Paths are POSIX-style, relative to tools/bigcherry. A trailing "/" is a prefix.
ALLOWLIST: dict[str, str] = {
    "core/engines.py": "the engine registry itself",
    "patch/campaign/": "patch campaign tooling is llama.cpp-only by design",
    "patch/validation_campaign.py": "llama.cpp-only patch validation campaign",
    "campaign/qualification_matrix.py": "llama.cpp-only qualification matrix",
    "release/patch_promote.py": "llama.cpp-only patch promotion tooling",
    "tuning/": "autotune tuning is llama.cpp-only by design",
    "cli/jobs.py": "jobs plane has no CLI engine selector yet",
    "jobs/worker.py": "jobs worker runs llama.cpp jobs only",
    "experiment/engine_quality.py": "per-engine quality adapters name their engine",
    "engine_adapters/": "per-engine adapters name their engine (MEN11)",
    "core/paths.py": "module-level LLAMACPP binding for this checkout's default engine layout",
    "cli/engine_scope.py": "maps not-yet-migrated commands to the llama.cpp-only scope",
    "cli/hardware.py": "hardware inventory CLI has its own argv and no engine selector (needs only work_root)",
    "check.py": "repo checks (source-audit, patch-catalog) audit the llama.cpp checkout; engine threading pending",
    "doctor.py": "doctor reports the llama.cpp project context; engine threading pending",
    "patch_admission.py": "admission gate evaluates the llama.cpp catalog; engine threading through gates pending",
    "release_validate.py": "release probe builds and runs llama.cpp by design",
    "re14_real_run.py": "campaign-build harness (RE14) is llama.cpp-only",
}

PATTERNS = (
    re.compile(r"""["'](?:llamacpp|radiance)["']"""),
    re.compile(r"engines\.LLAMACPP"),  # any use of the constant, compared or not
    re.compile(r"\bengine\s*[!=]=\s*[\"']"),
)


def _allowed(rel: str) -> bool:
    return any(rel == k or (k.endswith("/") and rel.startswith(k)) for k in ALLOWLIST)


class EngineLiteralLintTests(unittest.TestCase):
    def test_no_engine_literals_outside_allowlist(self) -> None:
        hits = []
        for path in sorted(PRODUCT_ROOT.rglob("*.py")):
            rel = path.relative_to(PRODUCT_ROOT).as_posix()
            if _allowed(rel):
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if any(p.search(line) for p in PATTERNS):
                    hits.append(f"{rel}:{n}: {line.strip()}")
        self.assertEqual(hits, [], "engine-name literals outside ALLOWLIST:\n" + "\n".join(hits))

    def test_allowlist_entries_exist(self) -> None:
        for key in ALLOWLIST:
            target = PRODUCT_ROOT / key.rstrip("/")
            self.assertTrue(target.exists(), f"stale ALLOWLIST entry: {key}")


if __name__ == "__main__":
    unittest.main()
