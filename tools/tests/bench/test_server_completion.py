"""tools.bigcherry.bench.server_completion -- repeated-completion MTP
benchmark harness. Metrics text fixtures match the REAL Prometheus
exposition format in vendor/llama.cpp/tools/server/server-task.cpp
(verified against source, not guessed) -- e.g.
llamacpp:spec_decode_num_draft_tokens_total, not the
spec_decode_num_drafted name an earlier draft design guessed."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.bench import server_completion as sc  # noqa: E402


def _metrics_text(draft: float, accepted: float, drafts: float, per_position: dict[int, float]) -> str:
    lines = [
        "# HELP llamacpp:spec_decode_num_draft_tokens_total Speculative: Total draft tokens generated",
        "# TYPE llamacpp:spec_decode_num_draft_tokens_total counter",
        f"llamacpp:spec_decode_num_draft_tokens_total {draft}",
        "# HELP llamacpp:spec_decode_num_accepted_tokens_total Speculative: Total draft tokens accepted",
        "# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter",
        f"llamacpp:spec_decode_num_accepted_tokens_total {accepted}",
        "# HELP llamacpp:spec_decode_num_drafts_total Speculative: Total speculative decoding verification steps",
        "# TYPE llamacpp:spec_decode_num_drafts_total counter",
        f"llamacpp:spec_decode_num_drafts_total {drafts}",
    ]
    if per_position:
        lines.append("# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position")
        lines.append("# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter")
        for position, count in per_position.items():
            lines.append(f'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position="{position}"}} {count}')
    return "\n".join(lines) + "\n"


class ParsePrometheusMetricsTests(unittest.TestCase):
    def test_parses_real_shaped_output(self):
        text = _metrics_text(426, 168, 86, {0: 86, 1: 55, 2: 21, 3: 6, 4: 0})
        metrics = sc.parse_prometheus_metrics(text)
        self.assertEqual(metrics.draft_tokens_total, 426)
        self.assertEqual(metrics.accepted_tokens_total, 168)
        self.assertEqual(metrics.drafts_total, 86)
        self.assertEqual(metrics.accepted_per_position, {0: 86, 1: 55, 2: 21, 3: 6, 4: 0})

    def test_missing_required_counter_raises(self):
        text = "\n".join(_metrics_text(1, 1, 1, {}).splitlines()[:3])  # only draft_tokens_total
        with self.assertRaises(sc.BenchmarkError):
            sc.parse_prometheus_metrics(text)

    def test_ignores_unrelated_lines(self):
        text = _metrics_text(10, 5, 3, {}) + "llamacpp:n_decode_total 999\n" + "# a comment\n"
        metrics = sc.parse_prometheus_metrics(text)
        self.assertEqual(metrics.draft_tokens_total, 10)

    def test_no_per_position_block_is_fine(self):
        metrics = sc.parse_prometheus_metrics(_metrics_text(10, 5, 3, {}))
        self.assertEqual(metrics.accepted_per_position, {})


class MetricsDeltaTests(unittest.TestCase):
    def test_normal_delta(self):
        before = sc.Metrics(100, 40, 20, {0: 20, 1: 10})
        after = sc.Metrics(150, 60, 30, {0: 30, 1: 15})
        delta = sc.metrics_delta(before, after)
        self.assertEqual(delta["draft_generated"], 50)
        self.assertEqual(delta["draft_accepted"], 20)
        self.assertEqual(delta["verification_cycles"], 10)
        self.assertEqual(delta["accepted_count_by_position"], [10, 5])

    def test_negative_scalar_delta_raises(self):
        before = sc.Metrics(100, 40, 20, {})
        after = sc.Metrics(90, 40, 20, {})  # server restarted, counters reset
        with self.assertRaises(sc.BenchmarkError):
            sc.metrics_delta(before, after)

    def test_negative_per_position_delta_raises(self):
        before = sc.Metrics(100, 40, 20, {0: 30})
        after = sc.Metrics(150, 60, 30, {0: 20})  # position 0 went backwards
        with self.assertRaises(sc.BenchmarkError):
            sc.metrics_delta(before, after)

    def test_new_position_appearing_is_fine(self):
        before = sc.Metrics(100, 40, 20, {0: 30})
        after = sc.Metrics(150, 60, 30, {0: 40, 1: 5})
        delta = sc.metrics_delta(before, after)
        self.assertEqual(delta["accepted_count_by_position"], [10, 5])


class LoadCorpusTests(unittest.TestCase):
    def test_loads_real_shipped_corpus(self):
        corpus_path = (
            Path(__file__).resolve().parents[2] / "bigcherry" / "bench" / "corpora" / "mtp-27b-v1.jsonl"
        )
        prompts, corpus_sha256 = sc.load_corpus(corpus_path)
        self.assertEqual(len(prompts), 24)
        self.assertEqual(len(corpus_sha256), 64)
        categories = {p.category for p in prompts}
        self.assertTrue({"prose", "code", "structured", "technical", "dialogue", "reasoning"} <= categories)

    def test_duplicate_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text(
                json.dumps({"id": "p1", "seed": 1, "category": "x", "prompt": "a"}) + "\n"
                + json.dumps({"id": "p1", "seed": 2, "category": "x", "prompt": "b"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(sc.BenchmarkError):
                sc.load_corpus(path)

    def test_missing_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text(json.dumps({"id": "p1", "seed": 1, "prompt": "a"}) + "\n", encoding="utf-8")
            with self.assertRaises(sc.BenchmarkError):
                sc.load_corpus(path)

    def test_empty_corpus_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(sc.BenchmarkError):
                sc.load_corpus(path)


class FakeTransport:
    """In-memory Transport implementing the same Protocol server_completion
    uses -- lets the harness's control-flow/fail-closed logic be tested
    without a real GPU/server."""

    def __init__(self, *, slots: int = 1, metrics_available: bool = True) -> None:
        self.slots = slots
        self.metrics_available = metrics_available
        self._draft = 0.0
        self._accepted = 0.0
        self._drafts = 0.0
        self._per_position: dict[int, float] = {}
        self.next_response: dict[str, Any] | None = None
        self.completion_calls: list[dict[str, Any]] = []

    def get_text(self, path: str) -> str:
        if path == "/metrics":
            if not self.metrics_available:
                raise sc.BenchmarkError("metrics disabled")
            return _metrics_text(self._draft, self._accepted, self._drafts, self._per_position)
        raise AssertionError(f"unexpected GET {path}")

    def get_json(self, path: str) -> Any:
        if path == "/slots":
            return [{} for _ in range(self.slots)]
        raise AssertionError(f"unexpected GET {path}")

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert path == "/completion"
        self.completion_calls.append(payload)
        response = self.next_response
        assert response is not None, "test must set next_response before each completion call"
        # Advance the underlying metrics counters to match the queued
        # response's own draft_n/draft_n_accepted -- matching what a real
        # server would actually do.
        timings = response.get("timings", {})
        # Advance from dedicated "_actual_*" fields, NOT draft_n/draft_n_accepted
        # -- a test tampering with the response's draft_n (to simulate a
        # real disagreement between /completion and /metrics) must not
        # also silently tamper with what /metrics itself reports.
        self._draft += timings.get("_actual_draft_n", timings.get("draft_n", 0))
        self._accepted += timings.get("_actual_draft_n_accepted", timings.get("draft_n_accepted", 0))
        self._drafts += timings.get("_verification_cycles", 1)
        for position, count in timings.get("_per_position_delta", {}).items():
            self._per_position[position] = self._per_position.get(position, 0.0) + count
        return response


def _completion_response(*, tokens_predicted: int, draft_n: int, draft_n_accepted: int,
                          verification_cycles: int = 1, per_position_delta: dict[int, float] | None = None,
                          predicted_ms: float = 1000.0) -> dict[str, Any]:
    return {
        "tokens_predicted": tokens_predicted,
        "tokens_evaluated": 10,
        "timings": {
            "draft_n": draft_n,
            "draft_n_accepted": draft_n_accepted,
            "predicted_ms": predicted_ms,
            "predicted_per_second": 1000.0 * tokens_predicted / predicted_ms,
            "_verification_cycles": verification_cycles,
            "_per_position_delta": per_position_delta or {},
        },
    }


def _config(**overrides) -> sc.SessionConfig:
    defaults = dict(
        session_id="s1", corpus_id="mtp-27b-v1", corpus_sha256="abc", bigcherry_revision="rev",
        llama_pin="b10687", llama_revision="c841aeeb8", model_id="qwen3.8-27b-q8_0",
        server_argv=(), spec_type="draft-mtp", spec_n_max=5, spec_draft_k="f16", spec_draft_v="f16",
        sampling=sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20),
        n_predict=256, order_seed=1,
    )
    defaults.update(overrides)
    return sc.SessionConfig(**defaults)


class ValidateServerTests(unittest.TestCase):
    def test_passes_with_one_slot_and_metrics(self):
        sc.validate_server(FakeTransport(slots=1))  # must not raise

    def test_fails_with_wrong_slot_count(self):
        with self.assertRaises(sc.BenchmarkError):
            sc.validate_server(FakeTransport(slots=4))

    def test_fails_without_metrics(self):
        with self.assertRaises(sc.BenchmarkError):
            sc.validate_server(FakeTransport(metrics_available=False))


class RunRequestTests(unittest.TestCase):
    def test_happy_path_computes_real_derived_fields(self):
        transport = FakeTransport()
        prompt = sc.CorpusPrompt(id="p1", seed=1, category="prose", prompt="hi")
        transport.next_response = _completion_response(
            tokens_predicted=256, draft_n=426, draft_n_accepted=168,
            verification_cycles=86, per_position_delta={0: 86, 1: 55, 2: 21, 3: 6},
        )
        record = sc.run_request(transport, prompt, _config(n_predict=256), pass_number=1, order_index=0)
        self.assertEqual(record["draft_generated"], 426)
        self.assertEqual(record["draft_accepted"], 168)
        self.assertEqual(record["verification_cycles"], 86)
        self.assertAlmostEqual(record["draft_acceptance"], 168 / 426)
        self.assertAlmostEqual(record["mean_accepted_length"], 1.0 + 168 / 86)
        self.assertEqual(record["accepted_count_by_position"], [86, 55, 21, 6])
        self.assertAlmostEqual(sum(record["acceptance_rate_by_position"]), 1.0)

    def test_short_response_fails_closed(self):
        transport = FakeTransport()
        prompt = sc.CorpusPrompt(id="p1", seed=1, category="prose", prompt="hi")
        transport.next_response = _completion_response(tokens_predicted=200, draft_n=10, draft_n_accepted=5)
        with self.assertRaises(sc.BenchmarkError):
            sc.run_request(transport, prompt, _config(n_predict=256), pass_number=1, order_index=0)

    def test_zero_tokens_predicted_fails_closed(self):
        transport = FakeTransport()
        prompt = sc.CorpusPrompt(id="p1", seed=1, category="prose", prompt="hi")
        transport.next_response = _completion_response(tokens_predicted=0, draft_n=0, draft_n_accepted=0)
        with self.assertRaises(sc.BenchmarkError):
            sc.run_request(transport, prompt, _config(n_predict=256), pass_number=1, order_index=0)

    def test_completion_metrics_disagreement_fails_closed(self):
        """gpt-dev-agent-recommended fail-closed check: /completion's own
        draft_n must agree with the /metrics delta, or something is wrong
        (concurrent traffic, restart, schema drift)."""
        transport = FakeTransport()
        prompt = sc.CorpusPrompt(id="p1", seed=1, category="prose", prompt="hi")
        response = _completion_response(tokens_predicted=256, draft_n=426, draft_n_accepted=168)
        response["timings"]["_actual_draft_n"] = 426  # what /metrics really advanced by
        response["timings"]["draft_n"] = 999  # but /completion CLAIMS a different value
        transport.next_response = response
        with self.assertRaises(sc.BenchmarkError):
            sc.run_request(transport, prompt, _config(n_predict=256), pass_number=1, order_index=0)


class RunSessionTests(unittest.TestCase):
    def test_produces_one_header_plus_two_passes_of_all_prompts(self):
        transport = FakeTransport()
        prompts = [sc.CorpusPrompt(id=f"p{i}", seed=i, category="x", prompt=f"prompt {i}") for i in range(4)]
        config = _config(n_predict=8)

        def make_response(*_args, **_kwargs):
            return _completion_response(tokens_predicted=8, draft_n=4, draft_n_accepted=2)

        # FakeTransport.next_response is static; set it once since every
        # request in this test uses the same shape.
        transport.next_response = _completion_response(tokens_predicted=8, draft_n=4, draft_n_accepted=2)

        records = sc.run_session(transport, prompts, config)
        header = records[0]
        self.assertEqual(header["kind"], "session")
        self.assertEqual(header["schema"], sc.SCHEMA)
        requests = records[1:]
        self.assertEqual(len(requests), 8)  # 2 passes x 4 prompts
        self.assertEqual([r["pass"] for r in requests], [1, 1, 1, 1, 2, 2, 2, 2])
        # 1 warmup call + 8 measured calls = 9 total completion calls
        self.assertEqual(len(transport.completion_calls), 9)

    def test_pass_orders_are_deterministic_and_differ(self):
        transport = FakeTransport()
        transport.next_response = _completion_response(tokens_predicted=8, draft_n=4, draft_n_accepted=2)
        prompts = [sc.CorpusPrompt(id=f"p{i}", seed=i, category="x", prompt=f"prompt {i}") for i in range(6)]
        records = sc.run_session(transport, prompts, _config(n_predict=8, order_seed=42))
        requests = records[1:]
        pass1_order = [r["prompt_id"] for r in requests if r["pass"] == 1]
        pass2_order = [r["prompt_id"] for r in requests if r["pass"] == 2]
        self.assertEqual(sorted(pass1_order), sorted(p.id for p in prompts))
        self.assertEqual(sorted(pass2_order), sorted(p.id for p in prompts))
        self.assertNotEqual(pass1_order, pass2_order)  # different deterministic shuffle

        # Re-running with the SAME order_seed must reproduce the SAME order.
        transport2 = FakeTransport()
        transport2.next_response = _completion_response(tokens_predicted=8, draft_n=4, draft_n_accepted=2)
        records2 = sc.run_session(transport2, prompts, _config(n_predict=8, order_seed=42))
        pass1_order2 = [r["prompt_id"] for r in records2[1:] if r["pass"] == 1]
        self.assertEqual(pass1_order, pass1_order2)


class WriteJsonlTests(unittest.TestCase):
    def test_writes_one_record_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            sc.write_jsonl([{"a": 1}, {"b": 2}], path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), {"a": 1})


if __name__ == "__main__":
    unittest.main()
