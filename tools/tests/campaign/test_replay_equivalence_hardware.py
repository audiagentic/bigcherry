"""PA26 stage 2: two-arm HIP hardware-run harness -- offline-testable via
injected lane_executor/runtime_runner. Real execute_campaign_lane and
real_hardware_runtime_runner are NEVER invoked here (no GPU, no compile).
"""

from __future__ import annotations

import dataclasses
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config as campaign_config  # noqa: E402
from bigcherry.core import paths  # noqa: E402
from bigcherry.core.artifacts import ArtifactRef  # noqa: E402
from bigcherry.build.builds import BuildPlan  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402
from bigcherry.campaign import replay_equivalence as offline  # noqa: E402
from bigcherry.campaign import replay_equivalence_hardware as hw  # noqa: E402
from bigcherry.campaign.lane import CampaignLaneResult  # noqa: E402
from bigcherry.tuning import replay as replay_module  # noqa: E402
from bigcherry.tuning import execution_audit  # noqa: E402

_RECIPES_PATH = paths.RECIPES


def _make_cache_blob(*, entry_count: int = 1) -> bytes:
    """Minimal structurally valid v5 replay cache blob (mirrors
    test_replay_equivalence.py's helper of the same name/layout -- kept
    local rather than cross-imported since test discovery does not
    guarantee sibling test modules are importable by bare name)."""
    name = b"winner-a\0"
    entry = (
        b"\x11" * 16
        + b"\x22" * 16
        + struct.pack("<I", 0)
        + struct.pack("<H", 1)
        + struct.pack("<iii", 0, 0, 0)
        + struct.pack("<BBBB", 0, 0, 0, 0)
        + b"\x33" * 16
        + b"\x44" * 16
        + struct.pack("<I", 0)
        + struct.pack("<H", 0)
        + struct.pack("<B", 0)
    )
    assert len(entry) == replay_module.ENT_SIZE, len(entry)
    payload = entry * entry_count + name
    header = bytearray()
    header += struct.pack(
        "<III", replay_module.MAGIC, replay_module.REPLAY_VERSION,
        replay_module.ARTIFACT_VERSION,
    )
    header += struct.pack(
        "<HH", replay_module.SIGNATURE_SCHEMA_VERSION,
        replay_module.HARDWARE_SCHEMA_VERSION,
    )
    header += struct.pack("<II", entry_count, len(name))
    header += b"\x55" * 16
    header += replay_module.blake2b_digest(payload)
    assert len(header) == replay_module.REPLAY_HEADER_SIZE
    return bytes(header) + payload


def _ref(kind: str, digest: str) -> ArtifactRef:
    return ArtifactRef(
        kind=kind, path=Path(f"/tmp/{kind}"), content_hash=digest, provenance={}
    )


#: One tempdir for the whole test process -- every _fake_result() call
#: writes its own uniquely-named real runtime-bundle manifest JSON file
#: into it (CampaignLaneResult.effective_configure/generated_compile_
#: inputs_hash read the manifest's own published JSON at runtime_bundle_ref
#: .path, not provenance -- see lane.py's _runtime_bundle_manifest()).
_MANIFEST_DIR = tempfile.mkdtemp(prefix="pa26-hw-test-manifests-")
_manifest_counter = [0]


def _write_bundle_manifest(
    *,
    bundle_digest: str,
    effective_build_id: str | None,
    effective_configure: dict[str, str] | None,
    generated_compile_inputs_hash: str | None,
) -> Path:
    _manifest_counter[0] += 1
    path = Path(_MANIFEST_DIR) / f"bundle-{_manifest_counter[0]}.json"
    manifest: dict[str, object] = {
        "entrypoint": "llama-server",
        "members": {},
        "runtime_bundle_hash": bundle_digest,
        "effective_build_id": effective_build_id,
        "effective_configure": effective_configure,
        "generated_compile_inputs_hash": generated_compile_inputs_hash,
        "generated_inputs_verification": "compiled-copy-v1",
        "toolchain": {},
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_json(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


#: The corpus fixture's single entry always names winner "winner-a"
#: (_make_cache_blob's hardcoded name). require_corpus_candidate_semantics
#: needs both the corpus producer's own manifest and each target arm's own
#: manifest to declare a semantically-matching "winner-a" candidate before
#: GGML_HIP_DISPATCH_REPLAY_REVISION_MATCH=0 is considered safe to rely on.
def _candidate_manifest(*, architectures: tuple[str, ...] = ("gfx1100",)) -> dict[str, object]:
    return {
        "manifest_hash": "irrelevant-for-this-test",
        "candidates": [
            {
                "stable_name": "winner-a",
                "family": "mmvq",
                "source_class": "native",
                "implementation_version": 1,
                "architectures": list(architectures),
                "config": {},
                "graph_safe": True,
                "deterministic": True,
                "implementation_digest": "irrelevant",
                "implementation_source_files": ["irrelevant.cu"],
            }
        ],
    }


def _next_manifest_path(label: str) -> Path:
    _manifest_counter[0] += 1
    return Path(_MANIFEST_DIR) / f"{label}-{_manifest_counter[0]}.json"


def _fake_result(
    *,
    source_name: str,
    resolved_revision: str = "a" * 40,
    inventory_digest: str = "inv-1",
    winners_digest: str = "win-1",
    binary_digest: str = "bin-1",
    bundle_digest: str = "bundle-1",
    effective_build_id: str | None = "effective-1",
    effective_configure: dict[str, str] | None = None,
    generated_compile_inputs_hash: str | None = "gci-1",
    #: (files: dict[str, str], compile_inputs: tuple[str, ...]) -- the
    #: generated-tree document's "files"/"compile_inputs" -- or None to
    #: leave manifest_ref/generated_tree_ref pointing at nonexistent paths
    #: (the default, unwritten -tmp- refs every other fixture already uses).
    generated_tree: tuple[dict[str, str], tuple[str, ...]] | None = None,
    manifest: dict[str, object] | None = None,
    targets: tuple[str, ...] = ("gfx1100",),
) -> CampaignLaneResult:
    build_plan = BuildPlan(
        source_slice_id=f"slice-{source_name}",
        phase="replay",
        platform="linux-multi",
        targets=targets,
        input_hashes=(("inventory", inventory_digest), ("promoted-winners", winners_digest)),
    )
    bundle_provenance = (
        {"build": {"effective_build_id": effective_build_id}}
        if effective_build_id is not None
        else {}
    )
    if effective_configure is None:
        effective_configure = {"GGML_HIP_DISPATCH_REPLAY": "ON"}
    manifest_path = _write_bundle_manifest(
        bundle_digest=bundle_digest,
        effective_build_id=effective_build_id,
        effective_configure=effective_configure,
        generated_compile_inputs_hash=generated_compile_inputs_hash,
    )
    return CampaignLaneResult(
        run_id=f"run-{source_name}",
        resolved_revision=resolved_revision,
        source_slice_id=f"slice-{source_name}",
        source_root=Path(f"/tmp/{source_name}"),
        build_plan=build_plan,
        workload_id="workload-1",
        source_metadata_ref=_ref("source-metadata", "meta-1"),
        input_refs=(
            ("inventory", _ref("inventory", inventory_digest)),
            ("promoted-winners", _ref("promoted-winners", winners_digest)),
        ),
        manifest_ref=(
            ArtifactRef(
                kind="manifest",
                path=_write_json(_next_manifest_path("manifest"), manifest),
                content_hash="manifest-1",
                provenance={},
            )
            if manifest is not None
            else _ref("manifest", "manifest-1")
        ),
        generated_tree_ref=(
            ArtifactRef(
                kind="generated-tree",
                path=_write_json(
                    _next_manifest_path("tree"),
                    {
                        "schema_version": 1,
                        "files": generated_tree[0],
                        "compile_inputs": list(generated_tree[1]),
                    },
                ),
                content_hash="tree-1",
                provenance={},
            )
            if generated_tree is not None
            else _ref("generated-tree", "tree-1")
        ),
        binary_ref=_ref("binary", binary_digest),
        runtime_bundle_ref=ArtifactRef(
            kind="runtime-bundle",
            path=manifest_path,
            content_hash=bundle_digest,
            provenance=bundle_provenance,
        ),
        smoke_ref=None,
    )


def _fake_lane_executor(results_by_source: dict[str, CampaignLaneResult]):
    def executor(spec, *, cfg, context, store, run_id):
        return results_by_source[spec.source_name]

    return executor


def _hit(**overrides) -> execution_audit.HitRecord:
    base = dict(
        dispatch="d1",
        signature="s1",
        candidate="w1",
        from_cache=True,
        recorded_calls=1,
    )
    base.update(overrides)
    return execution_audit.HitRecord(**base)


def _expectation(**overrides) -> hw.ReplayExpectation:
    base = dict(
        dispatch="d1",
        signature="s1",
        winner="w1",
        transform_id=0,
        match_kind=0,
        manifest_hash="mh-1",
    )
    base.update(overrides)
    return hw.ReplayExpectation(**base)


def _arm(**overrides) -> hw.ArmRuntimeResult:
    base = dict(
        process_success=True,
        clean_shutdown=True,
        output_digest="out-1",
        correctness_status="pass",
        model_hash="model-1",
        gpu_identity="gpu-1",
        runtime_args_digest="args-1",
        entries=(_hit(),),
    )
    base.update(overrides)
    return hw.ArmRuntimeResult(**base)


class ArmBuildIdentityTests(unittest.TestCase):
    def test_from_result_extracts_real_fields(self):
        result = _fake_result(source_name="bigcherry-native")
        identity = hw.ArmBuildIdentity.from_result(result)
        self.assertEqual(identity.resolved_revision, "a" * 40)
        self.assertEqual(identity.binary_digest, "bin-1")
        self.assertEqual(identity.runtime_bundle_digest, "bundle-1")
        self.assertEqual(identity.effective_build_id, "effective-1")
        self.assertEqual(
            dict(identity.effective_configure), {"GGML_HIP_DISPATCH_REPLAY": "ON"}
        )
        self.assertEqual(identity.generated_compile_inputs_hash, "gci-1")
        self.assertEqual(
            identity.input_hashes, (("inventory", "inv-1"), ("promoted-winners", "win-1"))
        )
        self.assertNotIn(
            "source_slice_id", dict(identity.build_plan_projection)
        )

    def test_missing_effective_build_id_reads_as_none(self):
        result = _fake_result(source_name="bigcherry-native", effective_build_id=None)
        identity = hw.ArmBuildIdentity.from_result(result)
        self.assertIsNone(identity.effective_build_id)


class RequireSharedBuildInputsTests(unittest.TestCase):
    def test_matching_inputs_pass(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(_fake_result(source_name="control")),
            candidate=hw.ArmBuildIdentity.from_result(_fake_result(source_name="candidate")),
        )
        hw.require_shared_build_inputs(delta)  # must not raise

    def test_negative_differing_revision_is_detected(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="control", resolved_revision="a" * 40)
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", resolved_revision="b" * 40)
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)

    def test_negative_differing_winners_input_is_detected(self):
        """If the two arms were fed different winner corpora, any later
        runtime divergence would be meaningless -- this must fail before
        the runner ever runs."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="control", winners_digest="win-1")
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", winners_digest="win-2")
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)

    def test_negative_differing_inventory_input_is_detected(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="control", inventory_digest="inv-1")
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", inventory_digest="inv-2")
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)

    def test_negative_differing_build_plan_projection_is_detected(self):
        """A BuildPlan field other than source_slice_id/composition
        diverging (e.g. compiled targets) must fail -- an unfair comparison."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="control", targets=("gfx1100",))
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", targets=("gfx1201",))
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)

    def test_differing_raw_effective_build_id_alone_is_tolerated(self):
        """GPT design correction (req_9cd5b140ca544ef8): raw effective_build_id
        is recorded for provenance but no longer required to match -- only
        the (normalized) effective_configure record is."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="control", effective_build_id="e1")
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", effective_build_id="e2")
            ),
        )
        hw.require_shared_build_inputs(delta)  # must not raise

    def test_differing_raw_generated_compile_inputs_hash_alone_is_tolerated(self):
        """GPT design correction (req_579777e899454ed9): raw
        generated_compile_inputs_hash is recorded for provenance but no
        longer required to match exactly -- it folds in per-candidate
        source-byte digests that legitimately differ between arms for an
        expected removed-module reason. The semantic generated-manifest/
        strict-file checks below are what require_shared_build_inputs
        actually enforces instead."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="control", generated_compile_inputs_hash="g1")
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", generated_compile_inputs_hash="g2")
            ),
        )
        hw.require_shared_build_inputs(delta)  # must not raise

    def test_negative_differing_compile_input_filename_set_is_detected(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    generated_tree=({"a.inc": "h1"}, ("a.inc",)),
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    generated_tree=({"a.inc": "h1", "b.inc": "h2"}, ("a.inc", "b.inc")),
                )
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)

    def test_negative_differing_strict_generated_file_content_is_detected(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    generated_tree=(
                        {"hip-autotune-registry.inc": "h1"},
                        ("hip-autotune-registry.inc",),
                    ),
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    generated_tree=(
                        {"hip-autotune-registry.inc": "h2"},
                        ("hip-autotune-registry.inc",),
                    ),
                )
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)

    def test_identity_only_generated_file_difference_is_tolerated(self):
        """hip-autotune-build-hash.h (EXPECTED_IDENTITY_ONLY_FILE) is not in
        STRICT_GENERATED_FILES -- its content may differ between arms."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    generated_tree=(
                        {"hip-autotune-build-hash.h": "h1"},
                        ("hip-autotune-build-hash.h",),
                    ),
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    generated_tree=(
                        {"hip-autotune-build-hash.h": "h2"},
                        ("hip-autotune-build-hash.h",),
                    ),
                )
            ),
        )
        hw.require_shared_build_inputs(delta)  # must not raise

    def test_negative_differing_manifest_candidate_config_is_detected(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    manifest={"candidates": [{"stable_name": "c1", "family": "mmq"}]},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    manifest={"candidates": [{"stable_name": "c2", "family": "mmq"}]},
                )
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)

    def test_manifest_identity_only_field_difference_is_tolerated(self):
        """generated_at/manifest_hash/build_descriptor (real per-run
        identity, not candidate-registry semantics) and per-candidate
        implementation_digest/implementation_source_files (raw source-byte
        digests that legitimately differ for an expected removed-module
        reason) may all differ between arms."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    manifest={
                        "generated_at": "2026-01-01T00:00:00Z",
                        "manifest_hash": "mh-1",
                        "build_descriptor": {"x": 1},
                        "candidates": [
                            {
                                "stable_name": "c1",
                                "family": "mmq",
                                "implementation_digest": "d1",
                                "implementation_source_files": ["a.cu"],
                            }
                        ],
                    },
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    manifest={
                        "generated_at": "2026-01-01T00:00:01Z",
                        "manifest_hash": "mh-2",
                        "build_descriptor": {"x": 2},
                        "candidates": [
                            {
                                "stable_name": "c1",
                                "family": "mmq",
                                "implementation_digest": "d2",
                                "implementation_source_files": ["a.cu", "b.cu"],
                            }
                        ],
                    },
                )
            ),
        )
        hw.require_shared_build_inputs(delta)  # must not raise

    def test_negative_unnormalized_effective_configure_difference_is_detected(self):
        """With no allowed_removed_modules, an OFF-vs-absent difference is
        NOT tolerated -- normalization is opt-in per module ownership."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    effective_configure={"GGML_HIP_AUTOTUNE": "OFF"},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", effective_configure={})
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)

    def test_whitelisted_off_to_absent_passes_with_owner_allowed(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    effective_configure={"GGML_HIP_AUTOTUNE": "OFF"},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", effective_configure={})
            ),
        )
        hw.require_shared_build_inputs(
            delta,
            allowed_removed_modules=frozenset({"0110_campaign_tune_record_build"}),
        )  # must not raise

    def test_differing_source_slice_and_build_plan_id_are_tolerated(self):
        """Composition legitimately differs between arms -- source_slice_id
        and build_plan_id must NOT be required to match."""
        control_result = _fake_result(source_name="control")
        candidate_result = _fake_result(source_name="candidate")
        self.assertNotEqual(
            control_result.source_slice_id, candidate_result.source_slice_id
        )
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(control_result),
            candidate=hw.ArmBuildIdentity.from_result(candidate_result),
        )
        hw.require_shared_build_inputs(delta)  # must not raise


class Pa26DerivedPathConfigureKeyTests(unittest.TestCase):
    """GPT design (req_13b71975cbe94c8d): GGML_HIP_AUTOTUNE_GENERATED_DIR is
    a build-local path derived from (source_slice_id, build_plan_id), which
    already legitimately differ between arms -- it must be validated then
    canonicalized (never blindly dropped), not compared raw."""

    def _identity_pair(self, *, control_dir: str, candidate_dir: str):
        control_result = _fake_result(
            source_name="control",
            effective_configure={"GGML_HIP_AUTOTUNE_GENERATED_DIR": control_dir},
        )
        candidate_result = _fake_result(
            source_name="candidate",
            effective_configure={"GGML_HIP_AUTOTUNE_GENERATED_DIR": candidate_dir},
        )
        control = hw.ArmBuildIdentity.from_result(control_result)
        candidate = hw.ArmBuildIdentity.from_result(candidate_result)
        return control, candidate

    def test_expected_shape_normalizes_and_passes(self):
        # Build once with placeholder paths just to learn the real
        # (source_slice_id, build_plan_id) ArmBuildIdentity.from_result
        # derives, then rebuild with the actual expected-shape paths.
        control, candidate = self._identity_pair(
            control_dir="placeholder", candidate_dir="placeholder"
        )
        control_dir = (
            f"/home/x/.cache/bigcherry/builds/{control.source_slice_id}/"
            f"{control.build_plan_id}/generated-inputs"
        )
        candidate_dir = (
            f"/home/x/.cache/bigcherry/builds/{candidate.source_slice_id}/"
            f"{candidate.build_plan_id}/generated-inputs"
        )
        control, candidate = self._identity_pair(
            control_dir=control_dir, candidate_dir=candidate_dir
        )
        delta = hw.BuildStageDelta(control=control, candidate=candidate)
        hw.require_shared_build_inputs(delta)  # must not raise

    def test_negative_unexpected_shape_fails_closed(self):
        control, candidate = self._identity_pair(
            control_dir="placeholder", candidate_dir="placeholder"
        )
        control_dir = "/not/the/expected/shape"
        candidate_dir = (
            f"/home/x/.cache/bigcherry/builds/{candidate.source_slice_id}/"
            f"{candidate.build_plan_id}/generated-inputs"
        )
        control, candidate = self._identity_pair(
            control_dir=control_dir, candidate_dir=candidate_dir
        )
        delta = hw.BuildStageDelta(control=control, candidate=candidate)
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(delta)


class Pa26EffectiveConfigureProjectionTests(unittest.TestCase):
    """GPT design's required test list (req_9cd5b140ca544ef8): each
    whitelisted OFF->absent normalizes and passes; every other asymmetry
    remains a hard failure."""

    _OWNER_0110 = frozenset({"0110_campaign_tune_record_build"})
    _OWNER_0810 = frozenset({"0810_replay_hit_diagnostics"})
    _OWNER_BOTH = frozenset(
        {"0110_campaign_tune_record_build", "0810_replay_hit_diagnostics"}
    )

    def test_whitelisted_off_to_absent_normalizes(self):
        control, candidate = hw.pa26_effective_configure_projection(
            {"GGML_HIP_AUTOTUNE": "OFF", "AMDGPU_TARGETS": "gfx1100"},
            {"AMDGPU_TARGETS": "gfx1100"},
            allowed_removed_modules=self._OWNER_0110,
        )
        self.assertEqual(control, candidate)

    def test_on_to_absent_is_not_normalized(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    effective_configure={"GGML_HIP_AUTOTUNE": "ON"},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", effective_configure={})
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(
                delta, allowed_removed_modules=self._OWNER_0110
            )

    def test_absent_to_off_wrong_direction_is_not_normalized(self):
        """The normalization is directional: control=OFF/candidate=absent is
        tolerated, but control=absent/candidate=OFF (the reverse) is not --
        that shape never legitimately arises from PA26's removed-module
        composition and must not be silently accepted."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="control", effective_configure={})
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    effective_configure={"GGML_HIP_AUTOTUNE": "OFF"},
                )
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(
                delta, allowed_removed_modules=self._OWNER_0110
            )

    def test_unknown_option_difference_still_fails(self):
        """An option difference with no PA26 owner mapping at all is never
        tolerated, regardless of allowed_removed_modules."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    effective_configure={"GGML_VULKAN": "OFF"},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", effective_configure={})
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(
                delta, allowed_removed_modules=self._OWNER_BOTH
            )

    def test_compiler_or_build_type_difference_still_fails(self):
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    effective_configure={"CMAKE_BUILD_TYPE": "Release"},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    effective_configure={"CMAKE_BUILD_TYPE": "Debug"},
                )
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(
                delta, allowed_removed_modules=self._OWNER_BOTH
            )

    def test_diagnostic_replay_diagnostics_difference_fails_when_0810_not_allowed(self):
        """The diagnostic pair's allowed_removed_modules excludes 0810 (it is
        re-added to the candidate) -- REPLAY_DIAGNOSTICS must match
        identically between diagnostic arms, never OFF/absent-normalized."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    effective_configure={"GGML_HIP_REPLAY_DIAGNOSTICS": "OFF"},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(source_name="candidate", effective_configure={})
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(
                delta, allowed_removed_modules=self._OWNER_0110
            )

    def test_diagnostic_dispatch_diagnostics_must_match_identically(self):
        """PA26's own ephemeral diagnostic build profile forces
        GGML_HIP_DISPATCH_DIAGNOSTICS=OFF on BOTH diagnostic arms (never
        ON/present on either) -- neither arm depends on 0110 for it, and it
        carries no owner mapping at all, so an identical OFF/OFF pair passes
        with no normalization needed."""
        delta = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    effective_configure={"GGML_HIP_DISPATCH_DIAGNOSTICS": "OFF"},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    effective_configure={"GGML_HIP_DISPATCH_DIAGNOSTICS": "OFF"},
                )
            ),
        )
        hw.require_shared_build_inputs(
            delta, allowed_removed_modules=self._OWNER_0110
        )  # must not raise

        # And a mismatch (one arm somehow carrying it ON) must still fail --
        # DISPATCH_DIAGNOSTICS has no PA26 owner mapping, so it is never
        # normalized away regardless of allowed_removed_modules.
        mismatched = hw.BuildStageDelta(
            control=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="control",
                    effective_configure={"GGML_HIP_DISPATCH_DIAGNOSTICS": "OFF"},
                )
            ),
            candidate=hw.ArmBuildIdentity.from_result(
                _fake_result(
                    source_name="candidate",
                    effective_configure={"GGML_HIP_DISPATCH_DIAGNOSTICS": "ON"},
                )
            ),
        )
        with self.assertRaises(hw.ReplayEquivalenceHardwareError):
            hw.require_shared_build_inputs(
                mismatched, allowed_removed_modules=self._OWNER_0110
            )


class Pa26DiagnosticBuildConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = campaign_config.load(_RECIPES_PATH)

    def test_ephemeral_build_forces_dispatch_diagnostics_off(self):
        ephemeral_cfg = hw.build_pa26_diagnostic_build_config(self.cfg)
        build = ephemeral_cfg.builds[hw.PA26_DIAGNOSTIC_BUILD_NAME]
        options = dict(build.options)
        self.assertEqual(options["GGML_HIP_DISPATCH_REPLAY"], "ON")
        self.assertEqual(options["GGML_HIP_REPLAY_DIAGNOSTICS"], "ON")
        self.assertEqual(options["GGML_HIP_DISPATCH_DIAGNOSTICS"], "OFF")
        # never persisted: the base cfg's own recipes-backed build is
        # untouched, and [build.replay-diagnostic] itself still carries its
        # original (unfair for PA26) options.
        self.assertNotIn(hw.PA26_DIAGNOSTIC_BUILD_NAME, self.cfg.builds)
        self.assertEqual(
            dict(self.cfg.builds["replay-diagnostic"].options).get(
                "GGML_HIP_DISPATCH_DIAGNOSTICS"
            ),
            "ON",
        )

    def test_diagnostic_specs_default_to_the_ephemeral_build_name(self):
        control_spec, candidate_spec = hw.build_diagnostic_hardware_specs(
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inventory_ref=_ref("inventory", "inv-1"),
            winners_ref=_ref("promoted-winners", "win-1"),
        )
        self.assertEqual(control_spec.build_name, hw.PA26_DIAGNOSTIC_BUILD_NAME)
        self.assertEqual(candidate_spec.build_name, hw.PA26_DIAGNOSTIC_BUILD_NAME)


class DiagnosticCompanionCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = campaign_config.load(_RECIPES_PATH)

    def test_diagnostic_candidate_is_production_candidate_plus_0810(self):
        diagnostic_cfg = offline.build_serving_core_diagnostic_config(self.cfg)
        diagnostic_ids = diagnostic_cfg.patch_sets[
            offline.SERVING_CORE_DIAGNOSTIC_PATCH_SET_NAME
        ].patches
        production_cfg = offline.build_serving_core_config(self.cfg)
        production_ids = production_cfg.patch_sets[
            offline.SERVING_CORE_PATCH_SET_NAME
        ].patches
        offline.require_diagnostic_matches_production(production_ids, diagnostic_ids)
        self.assertIn(offline.DIAGNOSTIC_ADDBACK_MODULE, diagnostic_ids)
        self.assertNotIn(offline.DIAGNOSTIC_ADDBACK_MODULE, production_ids)

    def test_negative_diagnostic_missing_addback_module_is_detected(self):
        with self.assertRaises(offline.ReplayEquivalenceError):
            offline.require_diagnostic_matches_production(
                ("0100_cmake_options", "0200_dispatch_hook"),
                ("0100_cmake_options", "0200_dispatch_hook"),
            )

    def test_negative_diagnostic_composition_drift_is_detected(self):
        with self.assertRaises(offline.ReplayEquivalenceError):
            offline.require_diagnostic_matches_production(
                ("0100_cmake_options", "0200_dispatch_hook"),
                ("0100_cmake_options", offline.DIAGNOSTIC_ADDBACK_MODULE),
            )


class ExecuteTwoArmBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = campaign_config.load(_RECIPES_PATH)
        cls.catalog = patchset.catalog()

    def test_real_composition_plus_fake_build_wires_together(self):
        control_spec, candidate_spec = hw.build_hardware_specs(
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inventory_ref=_ref("inventory", "inv-1"),
            winners_ref=_ref("promoted-winners", "win-1"),
        )
        self.assertEqual(control_spec.source_name, "bigcherry-native")
        self.assertEqual(candidate_spec.source_name, offline.SERVING_CORE_SOURCE_NAME)

        results_by_source = {
            "bigcherry-native": _fake_result(source_name="bigcherry-native"),
            offline.SERVING_CORE_SOURCE_NAME: _fake_result(
                source_name=offline.SERVING_CORE_SOURCE_NAME
            ),
        }
        build_delta, control_result, candidate_result = hw.execute_two_arm_build(
            self.cfg,
            self.catalog,
            context=SimpleNamespace(),
            store=SimpleNamespace(),
            control_spec=control_spec,
            candidate_spec=candidate_spec,
            run_id_prefix="pa26-test",
            lane_executor=_fake_lane_executor(results_by_source),
        )
        self.assertEqual(control_result.source_slice_id, "slice-bigcherry-native")
        self.assertEqual(
            candidate_result.source_slice_id,
            f"slice-{offline.SERVING_CORE_SOURCE_NAME}",
        )
        self.assertEqual(
            build_delta.control.input_hashes, build_delta.candidate.input_hashes
        )

    def test_stale_composition_fails_before_any_build_call(self):
        """If the offline composition check would fail, the fake executor
        must never even be called -- confirms build never proceeds on a bad
        composition."""
        pruned_framework = dataclasses.replace(
            self.cfg.patch_sets["framework"],
            patches=tuple(
                pid
                for pid in self.cfg.patch_sets["framework"].patches
                if pid != offline.EXPECTED_REMOVED_MODULES[0]
            ),
        )
        stale_cfg = dataclasses.replace(
            self.cfg,
            patch_sets={**self.cfg.patch_sets, "framework": pruned_framework},
        )
        control_spec, candidate_spec = hw.build_hardware_specs(
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inventory_ref=_ref("inventory", "inv-1"),
            winners_ref=_ref("promoted-winners", "win-1"),
        )

        def executor(spec, *, cfg, context, store, run_id):
            raise AssertionError("lane_executor must not be called")

        with self.assertRaises(offline.ReplayEquivalenceError):
            hw.execute_two_arm_build(
                stale_cfg,
                self.catalog,
                context=SimpleNamespace(),
                store=SimpleNamespace(),
                control_spec=control_spec,
                candidate_spec=candidate_spec,
                run_id_prefix="pa26-test",
                lane_executor=executor,
            )

    def test_diagnostic_pair_wires_through_the_diagnostic_builder(self):
        control_spec, candidate_spec = hw.build_diagnostic_hardware_specs(
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inventory_ref=_ref("inventory", "inv-1"),
            winners_ref=_ref("promoted-winners", "win-1"),
        )
        self.assertEqual(
            candidate_spec.source_name, offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME
        )
        results_by_source = {
            "bigcherry-native": _fake_result(source_name="bigcherry-native"),
            offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME: _fake_result(
                source_name=offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME
            ),
        }
        build_delta, control_result, candidate_result = hw.execute_two_arm_build(
            self.cfg,
            self.catalog,
            context=SimpleNamespace(),
            store=SimpleNamespace(),
            control_spec=control_spec,
            candidate_spec=candidate_spec,
            run_id_prefix="pa26-diag-test",
            candidate_cfg_builder=offline.build_serving_core_diagnostic_config,
            lane_executor=_fake_lane_executor(results_by_source),
        )
        self.assertEqual(
            candidate_result.source_slice_id,
            f"slice-{offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME}",
        )


class CompareRuntimeResultsTests(unittest.TestCase):
    def _compare(self, control, candidate, expected=None):
        if expected is None:
            expected = {"d1": _expectation()}
        return hw.compare_runtime_results(control, candidate, expected=expected)

    def test_identical_arms_are_equivalent(self):
        comparison = self._compare(_arm(), _arm())
        self.assertTrue(comparison.equivalent)
        self.assertEqual(comparison.differences, ())

    def test_negative_two_empty_result_sets_are_not_equivalent(self):
        """The coverage gap GPT flagged: empty vs empty must never read as
        equivalent when dispatches were actually expected."""
        empty = _arm(entries=())
        comparison = self._compare(empty, empty, expected={"d1": _expectation()})
        self.assertFalse(comparison.equivalent)
        reasons = {d.get("reason") for d in comparison.differences}
        self.assertIn("missing", reasons)

    def test_negative_output_digest_mismatch_is_detected(self):
        comparison = self._compare(
            _arm(output_digest="o1"), _arm(output_digest="o2")
        )
        self.assertFalse(comparison.equivalent)

    def test_negative_process_failure_is_detected(self):
        comparison = self._compare(
            _arm(process_success=True), _arm(process_success=False)
        )
        self.assertFalse(comparison.equivalent)

    def test_negative_clean_shutdown_mismatch_is_detected(self):
        comparison = self._compare(
            _arm(clean_shutdown=True), _arm(clean_shutdown=False)
        )
        self.assertFalse(comparison.equivalent)

    def test_negative_model_or_gpu_identity_mismatch_is_detected(self):
        comparison = self._compare(
            _arm(model_hash="m1"), _arm(model_hash="m2")
        )
        self.assertFalse(comparison.equivalent)

    def test_negative_winner_or_signature_mismatch_is_detected(self):
        comparison = self._compare(
            _arm(entries=(_hit(candidate="w1"),)),
            _arm(entries=(_hit(candidate="w2"),)),
        )
        self.assertFalse(comparison.equivalent)

    def test_call_count_mismatch_alone_is_not_detected(self):
        """recorded_calls is NOT authoritative (L1/L2 warm-cache bypass) --
        GPT design correction: it must be excluded from equality."""
        comparison = self._compare(
            _arm(entries=(_hit(recorded_calls=1),)),
            _arm(entries=(_hit(recorded_calls=99),)),
        )
        self.assertTrue(comparison.equivalent)

    def test_negative_from_cache_mismatch_is_detected(self):
        comparison = self._compare(
            _arm(entries=(_hit(from_cache=True),)),
            _arm(entries=(_hit(from_cache=False),)),
        )
        self.assertFalse(comparison.equivalent)

    def test_negative_missing_entry_in_one_arm_is_detected(self):
        control = _arm(entries=(_hit(dispatch="d1"), _hit(dispatch="d2")))
        candidate = _arm(entries=(_hit(dispatch="d1"),))
        expected = {"d1": _expectation(dispatch="d1"), "d2": _expectation(dispatch="d2")}
        comparison = self._compare(control, candidate, expected=expected)
        self.assertFalse(comparison.equivalent)
        reasons = {d.get("reason") for d in comparison.differences}
        self.assertIn("missing", reasons)

    def test_extra_entry_beyond_expected_is_not_detected(self):
        """GPT design correction: extra/fallback dispatches in the hit log
        beyond the expected corpus are NOT a PA26 failure."""
        control = _arm(entries=(_hit(dispatch="d1"),))
        candidate = _arm(
            entries=(_hit(dispatch="d1"), _hit(dispatch="d2"))
        )
        comparison = self._compare(control, candidate, expected={"d1": _expectation()})
        self.assertTrue(comparison.equivalent)


class RealHardwareRuntimeRunnerTests(unittest.TestCase):
    def test_default_runner_raises_runtime_not_evaluated(self):
        with self.assertRaises(hw.RuntimeNotEvaluated):
            hw.real_hardware_runtime_runner(
                _fake_result(source_name="x"),
                offline.WinnersCorpus(sha256="0" * 64, header={}, entries=()),
            )

    def test_runtime_not_evaluated_is_not_a_bare_not_implemented_error(self):
        """A runner that raises a plain NotImplementedError (a real bug, or
        an unrelated missing-feature error) must NOT be silently downgraded
        to NOT_EVALUATED by build_hardware_receipt -- only the specific
        RuntimeNotEvaluated exception may do that."""
        self.assertFalse(issubclass(hw.RuntimeNotEvaluated, NotImplementedError))


class BuildHardwareReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = campaign_config.load(_RECIPES_PATH)
        cls.catalog = patchset.catalog()

    def _cache_path(self, directory: str, entry_count: int = 1) -> Path:
        cache_path = Path(directory) / "winners.cache"
        cache_path.write_bytes(_make_cache_blob(entry_count=entry_count))
        return cache_path

    def _specs_and_results(self):
        control_spec, candidate_spec = hw.build_hardware_specs(
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inventory_ref=_ref("inventory", "inv-1"),
            winners_ref=_ref("promoted-winners", "win-1"),
        )
        results_by_source = {
            "bigcherry-native": _fake_result(source_name="bigcherry-native"),
            offline.SERVING_CORE_SOURCE_NAME: _fake_result(
                source_name=offline.SERVING_CORE_SOURCE_NAME
            ),
            offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME: _fake_result(
                source_name=offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME
            ),
        }
        return control_spec, candidate_spec, results_by_source

    def _specs_and_results_with_matching_manifests(self):
        """Same as ``_specs_and_results`` but the DIAGNOSTIC-pair-relevant
        results (``bigcherry-native``, used for both the production AND
        diagnostic control specs, and ``SERVING_CORE_DIAGNOSTIC_SOURCE_NAME``)
        carry a real, semantically-matching manifest -- required by
        ``require_corpus_candidate_semantics`` before a non-default
        runtime_runner is exercised. The production candidate
        (``SERVING_CORE_SOURCE_NAME``) gets the SAME manifest content so
        ``require_shared_build_inputs``'s generated_manifest_projection
        equality check on the production pair still holds."""
        control_spec, candidate_spec = hw.build_hardware_specs(
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inventory_ref=_ref("inventory", "inv-1"),
            winners_ref=_ref("promoted-winners", "win-1"),
        )
        manifest = _candidate_manifest()
        results_by_source = {
            "bigcherry-native": _fake_result(
                source_name="bigcherry-native", manifest=manifest
            ),
            offline.SERVING_CORE_SOURCE_NAME: _fake_result(
                source_name=offline.SERVING_CORE_SOURCE_NAME, manifest=manifest
            ),
            offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME: _fake_result(
                source_name=offline.SERVING_CORE_DIAGNOSTIC_SOURCE_NAME,
                manifest=manifest,
            ),
        }
        return control_spec, candidate_spec, results_by_source

    def _diagnostic_specs(self):
        return hw.build_diagnostic_hardware_specs(
            platform_name="linux-multi",
            architectures=("gfx1100",),
            inventory_ref=_ref("inventory", "inv-1"),
            winners_ref=_ref("promoted-winners", "win-1"),
        )

    def test_default_runner_yields_not_evaluated_receipt_with_real_build_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = self._cache_path(directory)
            control_spec, candidate_spec, results_by_source = self._specs_and_results()
            receipt = hw.build_hardware_receipt(
                self.cfg,
                self.catalog,
                cache_path,
                bigcherry_revision="0" * 40,
                context=SimpleNamespace(),
                store=SimpleNamespace(),
                control_spec=control_spec,
                candidate_spec=candidate_spec,
                run_id_prefix="pa26-test",
                lane_executor=_fake_lane_executor(results_by_source),
                # runtime_runner omitted: exercises the real default stub.
            )
            self.assertEqual(receipt["runtime"]["status"], "NOT_EVALUATED")
            self.assertEqual(
                receipt["decision_equivalence"]["status"], "NOT_EVALUATED"
            )
            self.assertIn("control", receipt["build_identity"])
            self.assertIn("candidate", receipt["build_identity"])
            self.assertEqual(
                receipt["build_identity"]["control"]["binary_digest"], "bin-1"
            )
            self.assertEqual(
                receipt["build_identity"]["control"]["effective_build_id"],
                "effective-1",
            )

    def test_unexpected_error_from_runner_propagates_not_downgraded(self):
        """A runner defect (not RuntimeNotEvaluated) must propagate as a
        real failure, never read as NOT_EVALUATED."""
        with tempfile.TemporaryDirectory() as directory:
            cache_path = self._cache_path(directory)
            control_spec, candidate_spec, results_by_source = self._specs_and_results()

            diagnostic_control_spec, diagnostic_candidate_spec = self._diagnostic_specs()

            def broken_runner(result, corpus):
                raise ValueError("parser bug")

            with self.assertRaises(ValueError):
                hw.build_hardware_receipt(
                    self.cfg,
                    self.catalog,
                    cache_path,
                    bigcherry_revision="0" * 40,
                    context=SimpleNamespace(),
                    store=SimpleNamespace(),
                    control_spec=control_spec,
                    candidate_spec=candidate_spec,
                    diagnostic_control_spec=diagnostic_control_spec,
                    diagnostic_candidate_spec=diagnostic_candidate_spec,
                    run_id_prefix="pa26-test",
                    lane_executor=_fake_lane_executor(results_by_source),
                    runtime_runner=broken_runner,
                )

    def test_non_default_runner_without_diagnostic_specs_fails_closed(self):
        """A non-default runtime_runner supplied without the diagnostic
        pair specs must fail closed, never silently run against the
        (diagnostics-free) production pair."""
        with tempfile.TemporaryDirectory() as directory:
            cache_path = self._cache_path(directory)
            control_spec, candidate_spec, results_by_source = self._specs_and_results()

            def matching_runner(result, corpus):
                return _arm(entries=())

            with self.assertRaises(hw.ReplayEquivalenceHardwareError):
                hw.build_hardware_receipt(
                    self.cfg,
                    self.catalog,
                    cache_path,
                    bigcherry_revision="0" * 40,
                    context=SimpleNamespace(),
                    store=SimpleNamespace(),
                    control_spec=control_spec,
                    candidate_spec=candidate_spec,
                    run_id_prefix="pa26-test",
                    lane_executor=_fake_lane_executor(results_by_source),
                    runtime_runner=matching_runner,
                )

    def test_injected_runner_producing_matching_results_yields_evaluated_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = self._cache_path(directory)
            control_spec, candidate_spec, results_by_source = (
                self._specs_and_results_with_matching_manifests()
            )
            diagnostic_control_spec, diagnostic_candidate_spec = self._diagnostic_specs()
            producer_manifest_path = _write_json(
                _next_manifest_path("producer-manifest"), _candidate_manifest()
            )

            def matching_runner(result, corpus):
                return _arm(
                    entries=tuple(
                        _hit(
                            dispatch=entry["dispatch"],
                            signature=entry["signature"],
                            candidate=entry["winner"],
                        )
                        for entry in corpus.entries
                    )
                )

            receipt = hw.build_hardware_receipt(
                self.cfg,
                self.catalog,
                cache_path,
                bigcherry_revision="0" * 40,
                context=SimpleNamespace(),
                store=SimpleNamespace(),
                control_spec=control_spec,
                candidate_spec=candidate_spec,
                diagnostic_control_spec=diagnostic_control_spec,
                diagnostic_candidate_spec=diagnostic_candidate_spec,
                corpus_producer_manifest_path=producer_manifest_path,
                run_id_prefix="pa26-test",
                lane_executor=_fake_lane_executor(results_by_source),
                runtime_runner=matching_runner,
            )
            self.assertEqual(receipt["runtime"]["status"], "EVALUATED")
            self.assertTrue(receipt["runtime"]["equivalent"])
            self.assertEqual(receipt["decision_equivalence"]["differences"], [])

    def test_injected_runner_producing_divergent_output_fails_receipt(self):
        """The most important hardware-stage negative fixture: a real
        divergence in resolved output must surface as a non-equivalent,
        EVALUATED receipt -- never silently pass."""
        with tempfile.TemporaryDirectory() as directory:
            cache_path = self._cache_path(directory)
            control_spec, candidate_spec, results_by_source = (
                self._specs_and_results_with_matching_manifests()
            )
            diagnostic_control_spec, diagnostic_candidate_spec = self._diagnostic_specs()
            producer_manifest_path = _write_json(
                _next_manifest_path("producer-manifest"), _candidate_manifest()
            )
            call_state = {"n": 0}

            def divergent_runner(result, corpus):
                call_state["n"] += 1
                digest = "control-output" if call_state["n"] == 1 else "candidate-output"
                return _arm(
                    output_digest=digest,
                    entries=tuple(
                        _hit(
                            dispatch=entry["dispatch"],
                            signature=entry["signature"],
                            candidate=entry["winner"],
                        )
                        for entry in corpus.entries
                    ),
                )

            receipt = hw.build_hardware_receipt(
                self.cfg,
                self.catalog,
                cache_path,
                bigcherry_revision="0" * 40,
                context=SimpleNamespace(),
                store=SimpleNamespace(),
                control_spec=control_spec,
                candidate_spec=candidate_spec,
                diagnostic_control_spec=diagnostic_control_spec,
                diagnostic_candidate_spec=diagnostic_candidate_spec,
                corpus_producer_manifest_path=producer_manifest_path,
                run_id_prefix="pa26-test",
                lane_executor=_fake_lane_executor(results_by_source),
                runtime_runner=divergent_runner,
            )
            self.assertEqual(receipt["runtime"]["status"], "EVALUATED")
            self.assertFalse(receipt["runtime"]["equivalent"])
            self.assertTrue(receipt["decision_equivalence"]["differences"])


if __name__ == "__main__":
    unittest.main()
