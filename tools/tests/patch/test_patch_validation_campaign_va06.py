"""VA06 slice: RD73 graph-cache resource-telemetry parsing.

PA36 RD73 legacy compatibility retirement: the telemetry parser moved
from shared validation_campaign.py (parse_rd73_resource_telemetry) into
RD73's producer (patches/1233_rd73_stable_graph_cache_key/
validation/producer.py, _parse_resource_telemetry). This test now
exercises the producer-side parser directly.

The peak-reduction (peak_rd73_resource_result) and activation-evidence
(evaluate_rd73_activation_evidence) shared functions were eliminated/
inlined into the producer's run() during the same retirement; their
unit tests were retired with them. The producer's run() end-to-end
coverage lives in test_patch_validation_campaign_rd73_contract_cli.py.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PRODUCER_DIR = Path(
    "patches/1233_rd73_stable_graph_cache_key/validation"
)
PRODUCER_MODULE = "patches_1233_rd73_stable_graph_cache_key_validation_producer_va06"


def _load_producer() -> Any:
    """Load the producer module with the required sys.modules registration."""
    spec = importlib.util.spec_from_file_location(
        PRODUCER_MODULE,
        PRODUCER_DIR / "producer.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRODUCER_MODULE] = module
    spec.loader.exec_module(module)
    return module


class ParseRd73ResourceTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.producer = _load_producer()

    def test_parses_all_readings_in_emission_order(self) -> None:
        text = (
            "some other log line\n"
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=386\n"
            "more noise\n"
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
        )
        self.assertEqual(self.producer._parse_resource_telemetry(text), (386, 651))

    def test_no_readings_returns_empty_tuple(self) -> None:
        self.assertEqual(self.producer._parse_resource_telemetry("nothing here\n"), ())

    def test_ignores_unrelated_bigcherry_lines(self) -> None:
        text = "BIGCHERRY_PATCH_HIT patch=1233_rd73 path=stable_graph_cache_key\n"
        self.assertEqual(self.producer._parse_resource_telemetry(text), ())

    def test_malformed_resource_line_raises(self) -> None:
        text = "BIGCHERRY_RD73_RESOURCE graph_cache_entries=notanumber\n"
        with self.assertRaises(self.producer.ValidationProducerError):
            self.producer._parse_resource_telemetry(text)

    def test_mixed_valid_and_malformed_raises(self) -> None:
        text = (
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=386\n"
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=\n"
            "BIGCHERRY_RD73_RESOURCE graph_cache_entries=651\n"
        )
        with self.assertRaises(self.producer.ValidationProducerError):
            self.producer._parse_resource_telemetry(text)


if __name__ == "__main__":
    unittest.main()
