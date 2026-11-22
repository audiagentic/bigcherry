"""Every third-party action in a workflow is pinned to a full commit SHA with its version in a comment, and every workflow
names its token permissions."""
import re
import unittest

from bigcherry.core import paths

WORKFLOWS = sorted((paths.REPO_ROOT / ".github/workflows").glob("*.yml"))
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)(.*)$")


class WorkflowPinTests(unittest.TestCase):
    def test_every_external_action_is_a_full_sha_with_a_version_comment(self):
        seen = 0
        for workflow in WORKFLOWS:
            for number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), 1):
                match = USES.match(line)
                if not match or match.group(1).startswith("./"):
                    continue
                seen += 1
                with self.subTest(workflow=workflow.name, line=number):
                    self.assertRegex(match.group(1), r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$", line.strip())
                    self.assertRegex(match.group(2), r"#\s*v\d+\.\d+\.\d+", "the pinned version belongs in a comment")
        self.assertGreater(seen, 20)

    def test_every_workflow_declares_permissions(self):
        for workflow in WORKFLOWS:
            with self.subTest(workflow=workflow.name):
                text = workflow.read_text(encoding="utf-8")
                self.assertRegex(text, r"(?m)^permissions:\n  [a-z-]+: (read|write|none)", "no top-level permissions block")
                self.assertNotIn("write-all", text)


if __name__ == "__main__":
    unittest.main()
