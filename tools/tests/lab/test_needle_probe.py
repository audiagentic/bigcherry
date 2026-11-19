"""needle_probe builds prompts at the requested depth and token count, and scores the exact code (no server needed)."""

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("needle_probe", ROOT / "tools/lab/radiance/needle_probe.py")
needle_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(needle_probe)

TEXT = ("A paragraph of ordinary documentation text that talks about builds, patches and plans.\n\n" * 4000)


class FakeServer:
    """Tokens are characters / ratio; the answer is whatever `answer(code)` says."""

    def __init__(self, ratio: float, answer=lambda code: code):
        self.ratio, self.answer, self.prompts = ratio, answer, []

    def __call__(self, url, model, prompt, max_tokens, timeout):
        self.prompts.append(prompt)
        match = re.search(r"vault is ([^.]*)\.", prompt)
        code = match.group(1) if match else ""
        return ({"usage": {"prompt_tokens": int(len(prompt) / self.ratio)},
                 "choices": [{"message": {"content": self.answer(code)}}]}, 0.0)


class NeedleProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._chat = needle_probe.chat

    def tearDown(self) -> None:
        needle_probe.chat = self._chat

    def test_the_needle_sits_at_the_requested_depth(self) -> None:
        needle_probe.chat = FakeServer(4.0)
        for depth in (0.1, 0.5, 0.9):
            needle_probe.chat.prompts.clear()
            row = needle_probe.case("u", "m", TEXT, 4.0, 20000, depth, 0, 1, 10)
            prompt = needle_probe.chat.prompts[-1]
            position = prompt.find("The access passphrase") / len(prompt)
            self.assertAlmostEqual(depth, position, delta=0.03)
            self.assertTrue(row["hit"])

    def test_a_prompt_off_target_is_rebuilt_with_the_servers_own_count(self) -> None:
        needle_probe.chat = FakeServer(3.7)        # the first estimate (2.06 chars a token) is far too long
        row = needle_probe.case("u", "m", TEXT, 2.06, 30000, 0.5, 0, 1, 10)
        self.assertLessEqual(abs(row["prompt_tokens"] - 30000) / 30000, 0.08)
        self.assertGreaterEqual(row["attempts"], 2)

    def test_only_the_exact_code_is_a_hit(self) -> None:
        needle_probe.chat = FakeServer(4.0, answer=lambda code: "I cannot provide the access passphrase.")
        self.assertFalse(needle_probe.case("u", "m", TEXT, 4.0, 10000, 0.5, 0, 1, 10)["hit"])
        needle_probe.chat = FakeServer(4.0, answer=lambda code: f"Sure: {code}")
        self.assertTrue(needle_probe.case("u", "m", TEXT, 4.0, 10000, 0.5, 0, 1, 10)["hit"])
        needle_probe.chat = FakeServer(4.0, answer=lambda code: code.replace("-", "_"))
        self.assertFalse(needle_probe.case("u", "m", TEXT, 4.0, 10000, 0.5, 0, 1, 10)["hit"])

    def test_cases_differ_by_offset_and_code(self) -> None:
        needle_probe.chat = FakeServer(4.0)
        rows = [needle_probe.case("u", "m", TEXT + str(i) * 100000, 4.0, 10000, 0.5, i, 1, 10) for i in range(3)]
        self.assertEqual(3, len({r["code"] for r in rows}))


if __name__ == "__main__":
    unittest.main()
