import json
import os
import subprocess
from pathlib import Path

import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


def candidate(number=10, **kw):
    return dict(number=number, author={"login": "owner"},
                headRefName="release-please--branches--main--components--bc-radiance",
                isCrossRepository=False, isDraft=False, autoMergeRequest=None) | kw


CASES = [
    ([candidate(20), candidate(10)], "CLEAN", "10"),
    ([candidate()], "BLOCKED", "10"),
    ([candidate()], "BEHIND", "10"),
    ([candidate()], "DIRTY", ""),
    ([candidate(autoMergeRequest={}), candidate(20)], "CLEAN", ""),
    ([candidate(author={"login": "other"}), candidate(isDraft=True),
      candidate(isCrossRepository=True), candidate(headRefName="arbitrary")], "CLEAN", ""),
    ([], "CLEAN", ""),
]

def check_queue(tmp_path, prs, state, expected):
    body = (ROOT / ".github/workflows/release-please.yml").read_text().split("      - name: queue eligible release PR\n", 1)[1]
    script = "\n".join(x[10:] for x in body.split("        run: |\n", 1)[1].splitlines())
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/bin/bash\n"
        "case \"$1 $2\" in\n"
        "  'api user') echo owner;;\n"
        "  'pr list') echo \"$PRS\";;\n"
        "  'pr view') echo \"$STATE\";;\n"
        "  'pr merge') echo \"$3\" > \"$MERGED\";;\n"
        "  *) exit 1;;\n"
        "esac\n"
    )
    gh.chmod(0o755)
    merged = tmp_path / "merged"
    env = dict(os.environ, PATH=str(tmp_path) + ":" + os.environ["PATH"],
               PRS=json.dumps(prs), STATE=json.dumps({"mergeStateStatus": state, "headRefOid": "abc"}),
               MERGED=str(merged))
    env.pop("BASH_ENV", None)
    subprocess.run(["bash", "-c", "gh() { bash "+str(gh)+" \"$@\"; }\n"+script], env=env, check=True)
    assert (merged.read_text().strip() if merged.exists() else "") == expected


class ReleaseAutomergeTests(unittest.TestCase):
    def test_queue(self):
        for prs, state, expected in CASES:
            with self.subTest(prs=prs, state=state), tempfile.TemporaryDirectory() as tmp:
                check_queue(Path(tmp), prs, state, expected)
