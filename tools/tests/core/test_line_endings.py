from __future__ import annotations

import subprocess
import unittest


class RepositoryLineEndingTests(unittest.TestCase):
    def test_index_contains_no_crlf_text_files(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "--eol"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        offenders = [
            line for line in result.stdout.splitlines()
            if line.startswith("i/crlf") and "attr/-text" not in line
        ]
        self.assertEqual(
            offenders,
            [],
            "tracked text files stored with CRLF:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
