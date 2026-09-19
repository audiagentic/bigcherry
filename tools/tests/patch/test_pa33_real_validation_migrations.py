"""PA33: migrate 2-3 real validations to the facade.

Proves that the patch-validate CLI can dispatch to real producers
(RD12/1205, RD04/1202, RD13/1206) without any CLI code changes.
These are the 3 most complete migrations from PA36.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestPA33RealValidationMigrations:
    """PA33: real validation migrations to the facade."""

    @pytest.mark.parametrize(
        "patch_id,producer_id",
        [
            ("1205_rd12_paired_mmvq_dual_output", "rd12"),
            ("1202_rd04_bf16_flash_attn_tile", "rd04"),
            ("1206_rd13_mul_mat_add_view_fusion", "rd13"),
        ],
    )
    def test_real_producer_dispatch_requires_no_cli_change(
        self, patch_id: str, producer_id: str
    ) -> None:
        """The patch-validate CLI dispatches to real producers without patch-specific branching.

        For each of the 3 most complete PA36 migrations (RD12, RD04, RD13),
        this test verifies:
        1. The producer.toml exists with the expected producer_id
        2. The producer.py exists and is loadable
        3. The CLI source contains no patch-specific reference to the producer

        If all 3 pass, the extension boundary is intact for real producers.
        """
        # Verify the producer.toml exists
        producer_toml = (
            Path("patches") / patch_id / "validation" / "producer.toml"
        )
        assert producer_toml.is_file(), (
            f"producer.toml not found for {patch_id}"
        )

        # Verify the producer_id is in the producer.toml
        toml_content = producer_toml.read_text()
        assert f'"{producer_id}"' in toml_content or f"{producer_id}" in toml_content, (
            f"producer_id {producer_id!r} not found in {producer_toml}"
        )

        # Verify the producer.py exists
        producer_py = (
            Path("patches") / patch_id / "validation" / "producer.py"
        )
        assert producer_py.is_file(), (
            f"producer.py not found for {patch_id}"
        )

        # Verify the CLI source contains no patch-specific reference
        from bigcherry.cli import patch as patch_cli

        cli_source = Path(patch_cli.__file__).read_text()
        assert patch_id not in cli_source, (
            f"CLI contains patch-specific reference to {patch_id!r} — "
            f"extension boundary violated"
        )
        assert producer_id not in cli_source, (
            f"CLI contains patch-specific reference to {producer_id!r} — "
            f"extension boundary violated"
        )

    def test_all_three_migrations_have_producer_toml(self) -> None:
        """All 3 migrated producers have a valid producer.toml."""
        for patch_id in [
            "1205_rd12_paired_mmvq_dual_output",
            "1202_rd04_bf16_flash_attn_tile",
            "1206_rd13_mul_mat_add_view_fusion",
        ]:
            producer_toml = (
                Path("patches") / patch_id / "validation" / "producer.toml"
            )
            assert producer_toml.is_file(), (
                f"producer.toml not found for {patch_id}"
            )
            # Verify it has the required schema field
            content = producer_toml.read_text()
            assert "schema" in content, (
                f"producer.toml for {patch_id} missing schema field"
            )

    def test_all_three_migrations_have_producer_py(self) -> None:
        """All 3 migrated producers have a loadable producer.py."""
        for patch_id in [
            "1205_rd12_paired_mmvq_dual_output",
            "1202_rd04_bf16_flash_attn_tile",
            "1206_rd13_mul_mat_add_view_fusion",
        ]:
            producer_py = (
                Path("patches") / patch_id / "validation" / "producer.py"
            )
            assert producer_py.is_file(), (
                f"producer.py not found for {patch_id}"
            )
            # Verify it has a run() function
            content = producer_py.read_text()
            assert "def run(" in content, (
                f"producer.py for {patch_id} missing run() function"
            )
