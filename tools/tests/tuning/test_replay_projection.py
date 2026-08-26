"""HI121 M4: end-to-end tests for replay_projection.project_measurements() --
selective, capability-gated reuse feeding into the UNCHANGED replay.build()."""

from __future__ import annotations

import json
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import catalog  # noqa: E402
from bigcherry.tuning import dispatch_abi  # noqa: E402
from bigcherry.tuning import replay as replay_module  # noqa: E402
from bigcherry.tuning import replay_projection as rp  # noqa: E402
from bigcherry.source.audit import git_revision  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_SQL = REPO_ROOT / "sql" / "dispatch-db.sql"

EPOCH = dispatch_abi.SIGNATURE_IDENTITY_EPOCH

CORE_ONLY_HEX = "0" * 31 + "1"  # CORE_SIGNATURE_V1 only
ALL_FIVE_HEX = "0000000000000000000000000000001f"  # CORE + all 4 HI118 presence caps


def _git_init_and_commit(root: Path) -> str:
    """Real git repo, real commit -- _load_target_capabilities() verifies
    vendor_root's ACTUAL git revision against what a manifest claims, so a
    fixture claiming a made-up 40-hex-char revision can no longer pass."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=root, check=True)
    revision, _dirty = git_revision(root, check_dirty=False)
    return revision


def _write_fixture_vendor(tmp_path: Path, *, producer_capabilities_hex: str) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n    GGML_TYPE_F32  = 0,\n    GGML_TYPE_Q8_0 = 8,\n};\n"
        "enum ggml_op {\n    GGML_OP_NONE,\n    GGML_OP_ADD,\n    GGML_OP_MUL_MAT,\n"
        "    GGML_OP_MUL_MAT_ID,\n    GGML_OP_GLU,\n    GGML_OP_COUNT,\n};\n",
        encoding="utf-8",
    )
    (vendor / "ggml" / "src" / "ggml.c").write_text(
        "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n"
        '    [GGML_TYPE_F32] = {\n        .type_name = "f32",\n    },\n'
        '    [GGML_TYPE_Q8_0] = {\n        .type_name = "q8_0",\n    },\n'
        "};\n"
        'static const char * GGML_OP_NAME[GGML_OP_COUNT] = {\n'
        '    "NONE",\n    "ADD",\n    "MUL_MAT",\n    "MUL_MAT_ID",\n    "GLU",\n'
        "};\n",
        encoding="utf-8",
    )
    lo = int(producer_capabilities_hex[16:], 16)
    hi = int(producer_capabilities_hex[:16], 16)
    (vendor / "ggml" / "src" / "ggml-cuda").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src" / "ggml-cuda" / "hip-autotune-types.h").write_text(
        f"#define GGML_HIP_PRODUCER_CAPABILITIES_LO UINT64_C({hex(lo)})\n"
        f"#define GGML_HIP_PRODUCER_CAPABILITIES_HI UINT64_C({hex(hi)})\n",
        encoding="utf-8",
    )
    cuda = vendor / "ggml" / "src" / "ggml-cuda"
    for name in (
        "hip-autotune-dispatch.cu", "mmvq.cu", "mmq.cu", "mmvf.cu", "mmf.cu", "ggml-cuda.cu",
        # HI124 (adversarial-review follow-up): the real per-family direct
        # #include headers, now part of the hashed implementation slice --
        # the fixture must provide them or candidate_implementation_digest()
        # fails closed with a missing-file CatalogError.
        "common.cuh", "mmq.cuh", "quantize.cuh", "mmid.cuh",
        "mmvq.cuh", "mmvq-autotune.cuh", "unary.cuh", "vecdotq.cuh",
        "mmvf.cuh", "convert.cuh", "mmf.cuh",
    ):
        (cuda / name).write_text(f"// fixture implementation: {name}\n", encoding="utf-8")
    (cuda / "mmq-config-rdna3.cuh").write_text("// fixture MMQ table\n", encoding="utf-8")
    return vendor


def _make_manifest(*, producer_capabilities_hex: str, source_revision: str,
                   vendor_root: Path | None = None) -> dict:
    families = ("mmvq", "mmq", "mmvf", "mmf", "blas")
    manifest = {
        "artifact_version": 1,
        "variant_set": "inventory",
        "source_revision": source_revision,
        "architectures": ["gfx1100"],
        "signature_schema_version": EPOCH,
        "hardware_schema_version": 1,
        "producer_capabilities": producer_capabilities_hex,
        "candidates": [
            {
                "stable_name": f"{f}:native:v1", "family": f, "source_class": "native_wrapper",
                "implementation_version": 1, "architectures": ["gfx1100"], "architecture_mask": 1,
                "graph_safe": True, "deterministic": True, "config": {},
            }
            for f in families
        ],
        "summary": {
            "total": len(families),
            "by_family": dict.fromkeys(families, 1),
            "by_source_class": {"native_wrapper": len(families)},
        },
    }
    if vendor_root is not None:
        for candidate in manifest["candidates"]:
            candidate["implementation_source_files"] = list(
                catalog.candidate_implementation_source_files(candidate)
            )
            candidate["implementation_digest"] = catalog.candidate_implementation_digest(
                candidate, vendor_root
            )
    manifest["manifest_hash"] = catalog.manifest_hash(manifest)
    manifest["build_descriptor"] = catalog.build_descriptor(manifest)
    return manifest


class ProjectMeasurementsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

        self.vendor = _write_fixture_vendor(self.tmp_path, producer_capabilities_hex=ALL_FIVE_HEX)
        self.source_revision = _git_init_and_commit(self.vendor)

        self.dispatch_db = self.tmp_path / "dispatch.sqlite"
        self.conn = sqlite3.connect(str(self.dispatch_db))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))

        self.manifest = _make_manifest(
            producer_capabilities_hex=ALL_FIVE_HEX,
            source_revision=self.source_revision,
            vendor_root=self.vendor,
        )
        self.manifest_path = self.tmp_path / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set, build_descriptor_hash) VALUES (?, ?, ?, 1, 'inventory', ?)",
            (
                self.source_revision, self.manifest["manifest_hash"], EPOCH,
                self.manifest["build_descriptor"]["descriptor_hash"],
            ),
        )
        self.build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # S1: plain MUL_MAT, requires CORE only.
        self.s1_hex = "11" * 16
        s1_canonical = {"schema_version": EPOCH, "op": 2, "flags": 0, "fusion": 0, "glu_op": 0}
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', ?, 'MUL_MAT', 'f32', 'f32', 'f32', 1, 1, 1, ?)",
            (bytes.fromhex(self.s1_hex), EPOCH, json.dumps(s1_canonical)),
        )
        # S2: GLU with all HI118 content flags zero -- requires all 4 presence caps.
        self.s2_hex = "22" * 16
        s2_canonical = {"schema_version": EPOCH, "op": 4, "flags": 1 << 3, "fusion": 2, "glu_op": 2}
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', ?, 'MUL_MAT_ID', 'q8_0', 'f32', 'f32', 1, 1, 1, ?)",
            (bytes.fromhex(self.s2_hex), EPOCH, json.dumps(s2_canonical)),
        )

        hardware_hex = "bb" * 16
        self.conn.execute(
            "INSERT INTO hardware (hardware_digest, architecture, architecture_code, "
            "wave_size, compute_units, feature_flags, canonical_json) VALUES "
            "(?, 'gfx1100', 1100, 32, 1, 0, '{}')",
            (bytes.fromhex(hardware_hex),),
        )
        hardware_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES "
            "(?, 'mmvq:native:v1', 'mmvq', 'native_wrapper', 1, '[\"gfx1100\"]', 1, 1, 1, '{}')",
            (self.build_id,),
        )
        candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        s1_signature_id = self.conn.execute(
            "SELECT signature_id FROM signature WHERE signature_digest = ?",
            (bytes.fromhex(self.s1_hex),),
        ).fetchone()[0]
        s2_signature_id = self.conn.execute(
            "SELECT signature_id FROM signature WHERE signature_digest = ?",
            (bytes.fromhex(self.s2_hex),),
        ).fetchone()[0]
        for signature_id, signature_hex in ((s1_signature_id, self.s1_hex), (s2_signature_id, self.s2_hex)):
            self.conn.execute(
                "INSERT INTO measurement (build_id, hardware_id, signature_id, dispatch_digest, "
                "candidate_id, objective, stage, accepted) VALUES (?, ?, ?, ?, ?, 'latency', 'final', 1)",
                (
                    self.build_id, hardware_id, signature_id,
                    bytes.fromhex(replay_module.portable_tuning_key(hardware_hex, signature_hex)),
                    candidate_id,
                ),
            )
            self.conn.execute(
                "INSERT INTO winner (build_id, hardware_id, signature_id, objective, dispatch_digest, "
                "candidate_id, stable_name, native_stable_name, is_native, improvement_pct, "
                "median_us, p95_us) VALUES (?, ?, ?, 'latency', ?, ?, ?, ?, 1, 0.0, 1.0, 1.0)",
                (
                    self.build_id, hardware_id, signature_id,
                    bytes.fromhex(replay_module.portable_tuning_key(hardware_hex, signature_hex)),
                    candidate_id, "mmvq:native:v1", "mmvq:native:v1",
                ),
            )
        self.conn.commit()

        self.header = {
            "kind": "header",
            "artifact_version": 1,
            "source_revision": self.source_revision,
            "manifest_hash": self.manifest["manifest_hash"],
            "build_descriptor_hash": self.manifest["build_descriptor"]["descriptor_hash"],
            "producer_capabilities": ALL_FIVE_HEX,
            "variant_set": "inventory",
        }
        hardware_hex = "bb" * 16
        self.s1_result = {
            "kind": "result",
            "dispatch": replay_module.portable_tuning_key(hardware_hex, self.s1_hex),
            "signature": self.s1_hex, "hardware": hardware_hex,
            "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
            "source_revision": self.source_revision, "manifest_hash": self.manifest["manifest_hash"],
        }
        self.s2_result = {
            "kind": "result",
            "dispatch": replay_module.portable_tuning_key(hardware_hex, self.s2_hex),
            "signature": self.s2_hex, "hardware": hardware_hex,
            "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
            "source_revision": self.source_revision, "manifest_hash": self.manifest["manifest_hash"],
        }
        self.measurements_path = self.tmp_path / "measurements.jsonl"
        with self.measurements_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.header) + "\n")
            handle.write(json.dumps(self.s1_result) + "\n")
            handle.write(json.dumps(self.s2_result) + "\n")

    def _rewrite_result(self, signature_hex: str, **updates: str) -> None:
        lines = self.measurements_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            record = json.loads(line)
            if record.get("signature") == signature_hex:
                record.update(updates)
                lines[index] = json.dumps(record)
                break
        else:
            self.fail(f"no result row for signature {signature_hex}")
        self.measurements_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def tearDown(self):
        self.conn.close()

    def _set_source_capabilities(self, mask_hex: str) -> None:
        self.conn.execute(
            "INSERT INTO build_capability (build_id, backend, producer_capabilities) "
            "VALUES (?, 'hip', ?)",
            (self.build_id, bytes.fromhex(mask_hex)),
        )
        self.conn.commit()

    def test_missing_source_capability_row_raises(self):
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_full_capabilities_both_sides_retains_both_rows(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        output = self.tmp_path / "out.jsonl"
        summary = rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
            target_manifest_path=self.manifest_path, vendor_root=self.vendor,
        )
        self.assertEqual(summary.examined, 2)
        self.assertEqual(summary.retained, 2)
        self.assertEqual(summary.omitted_missing_producer_capability, 0)
        self.assertEqual(summary.omitted_missing_target_capability, 0)
        self.assertEqual(summary.omitted_unsupported_domain, 0)

        lines = output.read_text(encoding="utf-8").splitlines()
        kinds = [json.loads(line)["kind"] for line in lines]
        self.assertEqual(kinds, ["header", "result", "result"])

    def test_core_only_source_capability_omits_glu_row_but_keeps_mul_mat(self):
        # S1 (plain MUL_MAT) needs only CORE; S2 (GLU) needs all 4 HI118
        # presence caps too -- a CORE-only source producer cannot certify S2.
        self._set_source_capabilities(CORE_ONLY_HEX)
        source_manifest = json.loads(json.dumps(self.manifest))
        source_manifest["producer_capabilities"] = CORE_ONLY_HEX
        source_manifest["manifest_hash"] = catalog.manifest_hash(source_manifest)
        source_manifest["build_descriptor"] = catalog.build_descriptor(source_manifest)
        source_manifest_path = self.tmp_path / "core-only-source-manifest.json"
        source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
        self.conn.execute(
            "UPDATE build SET manifest_hash = ?, build_descriptor_hash = ? WHERE build_id = ?",
            (
                source_manifest["manifest_hash"],
                source_manifest["build_descriptor"]["descriptor_hash"],
                self.build_id,
            ),
        )
        self.conn.commit()
        header = dict(self.header, producer_capabilities=CORE_ONLY_HEX)
        header["manifest_hash"] = source_manifest["manifest_hash"]
        header["build_descriptor_hash"] = source_manifest["build_descriptor"]["descriptor_hash"]
        self.measurements_path.write_text(
            json.dumps(header) + "\n" + json.dumps(self.s1_result) + "\n" + json.dumps(self.s2_result) + "\n",
            encoding="utf-8",
        )
        output = self.tmp_path / "out.jsonl"
        summary = rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=source_manifest_path,
            target_manifest_path=self.manifest_path, vendor_root=self.vendor,
        )
        self.assertEqual(summary.retained, 1)
        self.assertEqual(summary.omitted_missing_producer_capability, 1)
        results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                   if json.loads(line).get("kind") == "result"]
        self.assertEqual(results[0]["signature"], self.s1_hex)

    def test_core_only_target_omits_glu_row_even_with_full_source_capability(self):
        # HI121's own added strengthening: the TARGET must also hold the
        # required capabilities, not just the source.
        self._set_source_capabilities(ALL_FIVE_HEX)
        core_only_vendor = _write_fixture_vendor(
            self.tmp_path / "core-only-target", producer_capabilities_hex=CORE_ONLY_HEX
        )
        core_only_revision = _git_init_and_commit(core_only_vendor)
        core_only_manifest = _make_manifest(
            producer_capabilities_hex=CORE_ONLY_HEX, source_revision=core_only_revision,
            vendor_root=core_only_vendor,
        )
        core_only_manifest_path = self.tmp_path / "core-only-manifest.json"
        core_only_manifest_path.write_text(json.dumps(core_only_manifest), encoding="utf-8")

        output = self.tmp_path / "out.jsonl"
        summary = rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
            target_manifest_path=core_only_manifest_path, vendor_root=core_only_vendor,
        )
        self.assertEqual(summary.retained, 1)
        self.assertEqual(summary.omitted_missing_target_capability, 1)

    def test_projected_file_feeds_into_unchanged_replay_build(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        output = self.tmp_path / "out.jsonl"
        rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
            target_manifest_path=self.manifest_path, vendor_root=self.vendor,
        )
        ggml_h = self.tmp_path / "ggml.h"
        ggml_h.write_text("enum ggml_type { GGML_TYPE_F32 = 0 };\n", encoding="utf-8")
        blob = replay_module.build(output, self.manifest_path, ggml_h)
        header, entries = replay_module.read_cache(blob)
        self.assertEqual(header["version"], replay_module.REPLAY_VERSION)
        self.assertEqual(len(entries), 2)

    def test_genuine_cross_revision_projection_feeds_replay_build(self):
        # The central multi-generation use case: source and target are
        # DIFFERENT real revisions/builds. Without the output header being
        # target-bound, replay.build()'s own real requirement (producer
        # header source_revision == target manifest source_revision) would
        # reject this unconditionally, even though every row is legitimately
        # capability-compatible.
        self._set_source_capabilities(ALL_FIVE_HEX)
        target_vendor = _write_fixture_vendor(self.tmp_path / "target-root", producer_capabilities_hex=ALL_FIVE_HEX)
        # Distinct tree content so this real commit hashes differently from
        # self.vendor's -- otherwise two git commits with byte-identical
        # trees made within the same second can hash identically.
        (target_vendor / "MARKER.txt").write_text("target-root\n", encoding="utf-8")
        target_revision = _git_init_and_commit(target_vendor)
        self.assertNotEqual(target_revision, self.source_revision)
        target_manifest = _make_manifest(
            producer_capabilities_hex=ALL_FIVE_HEX,
            source_revision=target_revision,
            vendor_root=target_vendor,
        )
        target_manifest_path = self.tmp_path / "target-manifest.json"
        target_manifest_path.write_text(json.dumps(target_manifest), encoding="utf-8")

        output = self.tmp_path / "out.jsonl"
        summary = rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
            target_manifest_path=target_manifest_path, vendor_root=target_vendor,
        )
        self.assertEqual(summary.retained, 2)

        output_header = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(output_header["source_revision"], target_revision)
        self.assertEqual(output_header["manifest_hash"], target_manifest["manifest_hash"])
        self.assertEqual(output_header["hi121_source_provenance"]["source_revision"], self.source_revision)
        self.assertEqual(output_header["hi121_source_provenance"]["source_build_id"], self.build_id)

        ggml_h = self.tmp_path / "ggml.h"
        ggml_h.write_text("enum ggml_type { GGML_TYPE_F32 = 0 };\n", encoding="utf-8")
        blob = replay_module.build(output, target_manifest_path, ggml_h)
        header, entries = replay_module.read_cache(blob)
        self.assertEqual(header["version"], replay_module.REPLAY_VERSION)
        self.assertEqual(len(entries), 2)

    def test_identical_candidate_descriptors_with_changed_kernel_are_not_reused(self):
        """The old descriptor-only check accepted this exact false positive."""
        self._set_source_capabilities(ALL_FIVE_HEX)
        target_vendor = _write_fixture_vendor(
            self.tmp_path / "changed-kernel-root",
            producer_capabilities_hex=ALL_FIVE_HEX,
        )
        (target_vendor / "ggml" / "src" / "ggml-cuda" / "mmvq.cu").write_text(
            "// different kernel implementation\n", encoding="utf-8"
        )
        target_revision = _git_init_and_commit(target_vendor)
        target_manifest = _make_manifest(
            producer_capabilities_hex=ALL_FIVE_HEX,
            source_revision=target_revision,
            vendor_root=target_vendor,
        )
        target_manifest_path = self.tmp_path / "changed-kernel-manifest.json"
        target_manifest_path.write_text(json.dumps(target_manifest), encoding="utf-8")

        source_candidate = self.manifest["candidates"][0]
        target_candidate = target_manifest["candidates"][0]
        self.assertEqual(
            {k: v for k, v in source_candidate.items() if k not in (
                "implementation_digest", "implementation_source_files"
            )},
            {k: v for k, v in target_candidate.items() if k not in (
                "implementation_digest", "implementation_source_files"
            )},
        )
        self.assertNotEqual(
            source_candidate["implementation_digest"],
            target_candidate["implementation_digest"],
        )

        output = self.tmp_path / "changed-kernel-out.jsonl"
        summary = rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id,
            source_manifest_path=self.manifest_path,
            target_manifest_path=target_manifest_path, vendor_root=target_vendor,
        )
        self.assertEqual(summary.retained, 0)
        self.assertEqual(summary.omitted_candidate_mismatch, 2)

    def test_target_manifest_with_forged_implementation_digest_rejected(self):
        """Adversarial-review follow-up: _load_target_capabilities() proved
        vendor_root IS the exact, clean revision the target manifest claims,
        but never re-verified that a candidate's embedded implementation_digest
        was actually computed from THAT root's real files -- a target
        manifest could copy an old (source-side) digest into its own
        candidate entries, self-consistently recompute manifest_hash/
        descriptor from the tampered dict, pass the git-revision check (a
        real, correctly-identified, clean checkout), and still get treated
        as equivalent to the source. This must now be rejected."""
        self._set_source_capabilities(ALL_FIVE_HEX)
        target_vendor = _write_fixture_vendor(
            self.tmp_path / "forged-digest-root", producer_capabilities_hex=ALL_FIVE_HEX,
        )
        (target_vendor / "ggml" / "src" / "ggml-cuda" / "mmvq.cu").write_text(
            "// a genuinely different kernel implementation\n", encoding="utf-8"
        )
        target_revision = _git_init_and_commit(target_vendor)
        target_manifest = _make_manifest(
            producer_capabilities_hex=ALL_FIVE_HEX,
            source_revision=target_revision,
            vendor_root=target_vendor,
        )
        # Forge: copy the SOURCE's own (stale, real-for-a-different-root)
        # mmvq implementation_digest over the target's freshly-recomputed
        # one, then recompute manifest_hash/descriptor from the tampered
        # dict so the manifest is fully self-consistent.
        source_mmvq = next(c for c in self.manifest["candidates"] if c["family"] == "mmvq")
        target_mmvq = next(c for c in target_manifest["candidates"] if c["family"] == "mmvq")
        self.assertNotEqual(source_mmvq["implementation_digest"], target_mmvq["implementation_digest"])
        target_mmvq["implementation_digest"] = source_mmvq["implementation_digest"]
        target_mmvq["implementation_source_files"] = list(source_mmvq["implementation_source_files"])
        target_manifest["manifest_hash"] = catalog.manifest_hash(target_manifest)
        target_manifest["build_descriptor"] = catalog.build_descriptor(target_manifest)

        target_manifest_path = self.tmp_path / "forged-digest-manifest.json"
        target_manifest_path.write_text(json.dumps(target_manifest), encoding="utf-8")

        with self.assertRaisesRegex(rp.ProjectionError, "does not match the digest independently recomputed"):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "forged-digest-out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id,
                source_manifest_path=self.manifest_path,
                target_manifest_path=target_manifest_path, vendor_root=target_vendor,
            )

    def test_retained_rows_are_preserved_byte_for_byte(self):
        # round 9's explicit requirement: retained result rows must be the
        # ORIGINAL bytes, not a json.dumps() re-serialization (which can
        # silently change whitespace/escaping/key order).
        self._set_source_capabilities(ALL_FIVE_HEX)
        # Rewrite the measurements file with deliberately unusual (but valid)
        # formatting for the S1 result line -- extra whitespace, different
        # key order -- that json.dumps(..., separators=(",", ":")) would
        # normalize away if the row were re-serialized.
        original_lines = self.measurements_path.read_text(encoding="utf-8").splitlines()
        s1_line_index = next(
            i for i, line in enumerate(original_lines)
            if json.loads(line).get("signature") == self.s1_hex
        )
        reordered = {"kind": "result", "winner": self.s1_result["winner"]}
        reordered.update({k: v for k, v in self.s1_result.items() if k not in reordered})
        odd_line = json.dumps(reordered, indent=None, separators=(", ", ": "))
        original_lines[s1_line_index] = odd_line
        self.measurements_path.write_text("\n".join(original_lines) + "\n", encoding="utf-8")

        output = self.tmp_path / "out.jsonl"
        rp.project_measurements(
            self.measurements_path, output,
            dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
            target_manifest_path=self.manifest_path, vendor_root=self.vendor,
        )
        output_lines = output.read_text(encoding="utf-8").splitlines()
        matching = [line for line in output_lines if self.s1_hex in line]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0], odd_line, "retained row must be the original raw bytes, not re-serialized")

    def test_header_capability_forgery_is_rejected(self):
        # A measurements artifact whose header CLAIMS a different producer_
        # capabilities than what's actually DB-attested for source_build_id
        # must be rejected -- trusting the DB attestation alone (without
        # checking THIS artifact's own claim against it) would let a forged/
        # different artifact silently inherit someone else's attestation.
        self._set_source_capabilities(ALL_FIVE_HEX)
        forged_header = dict(self.header)
        forged_header["producer_capabilities"] = CORE_ONLY_HEX
        lines = [json.dumps(forged_header)] + [
            line for line in self.measurements_path.read_text(encoding="utf-8").splitlines()[1:]
        ]
        self.measurements_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_source_manifest_hash_mismatch_with_db_is_rejected(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        forged_source = json.loads(json.dumps(self.manifest))
        forged_source["candidates"][0]["config"] = {"forged": True}
        forged_source["manifest_hash"] = catalog.manifest_hash(forged_source)
        forged_source["build_descriptor"] = catalog.build_descriptor(forged_source)
        forged_source_path = self.tmp_path / "forged-source-manifest.json"
        forged_source_path.write_text(json.dumps(forged_source), encoding="utf-8")
        with self.assertRaisesRegex(rp.ProjectionError, "DB manifest_hash"):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id,
                source_manifest_path=forged_source_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_null_db_descriptor_is_not_projectable(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        self.conn.execute(
            "UPDATE build SET build_descriptor_hash = NULL WHERE build_id = ?",
            (self.build_id,),
        )
        self.conn.commit()
        with self.assertRaisesRegex(rp.ProjectionError, "descriptor-less"):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id,
                source_manifest_path=self.manifest_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_target_manifest_without_descriptor_is_not_projectable(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        target_manifest = json.loads(json.dumps(self.manifest))
        del target_manifest["build_descriptor"]
        target_manifest_path = self.tmp_path / "target-without-descriptor.json"
        target_manifest_path.write_text(json.dumps(target_manifest), encoding="utf-8")
        with self.assertRaisesRegex(rp.ProjectionError, "target manifest has no build_descriptor"):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id,
                source_manifest_path=self.manifest_path,
                target_manifest_path=target_manifest_path, vendor_root=self.vendor,
            )

    def test_forged_winner_claim_is_rejected_when_candidate_exists_in_both_manifests(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        # This is a real candidate in both manifests, so manifest candidate
        # equivalence alone would accept the forged claim.
        forged_name = "mmq:native:v1"
        self._rewrite_result(self.s1_hex, winner=forged_name)
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_forged_native_claim_is_rejected_when_candidate_exists_in_both_manifests(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        forged_name = "mmq:native:v1"
        self._rewrite_result(self.s1_hex, native=forged_name)
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_forged_hardware_claim_is_rejected(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        self._rewrite_result(self.s1_hex, hardware="cc" * 16)
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_forged_winner_native_and_hardware_claims_are_rejected_together(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        forged_name = "mmq:native:v1"
        self._rewrite_result(self.s1_hex, winner=forged_name, native=forged_name, hardware="cc" * 16)
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_wrong_source_build_id_header_mismatch_raises(self):
        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES (?, 'different', ?, 1, 'inventory')",
            ("c" * 40, EPOCH),
        )
        other_build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO build_capability (build_id, backend, producer_capabilities) "
            "VALUES (?, 'hip', ?)",
            (other_build_id, bytes.fromhex(ALL_FIVE_HEX)),
        )
        self.conn.commit()
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=other_build_id, source_manifest_path=self.manifest_path,
                target_manifest_path=self.manifest_path, vendor_root=self.vendor,
            )

    def test_cli_project_replay_command(self):
        from bigcherry.cli import main as cli_main

        self._set_source_capabilities(ALL_FIVE_HEX)
        output = self.tmp_path / "out.jsonl"
        exit_code = cli_main.main([
            "project-replay", str(self.measurements_path),
            "--dispatch-db", str(self.dispatch_db),
            "--source-build-id", str(self.build_id),
            "--source-manifest", str(self.manifest_path),
            "--target-manifest", str(self.manifest_path),
            "--vendor-root", str(self.vendor),
            "--output", str(output),
            "--json",
        ])
        self.assertEqual(exit_code, 0)
        self.assertTrue(output.is_file())
        results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                   if json.loads(line).get("kind") == "result"]
        self.assertEqual(len(results), 2)

    def test_target_manifest_from_wrong_source_root_raises(self):
        self._set_source_capabilities(ALL_FIVE_HEX)
        wrong_vendor = _write_fixture_vendor(self.tmp_path / "wrong-root", producer_capabilities_hex=CORE_ONLY_HEX)
        _git_init_and_commit(wrong_vendor)  # a real, but DIFFERENT, revision than self.manifest claims
        with self.assertRaises(rp.ProjectionError):
            rp.project_measurements(
                self.measurements_path, self.tmp_path / "out.jsonl",
                dispatch_db=self.dispatch_db, source_build_id=self.build_id, source_manifest_path=self.manifest_path,
                # manifest claims self.source_revision but this vendor_root is at a different
                # real revision (and also only declares CORE_ONLY_HEX, not ALL_FIVE_HEX)
                target_manifest_path=self.manifest_path, vendor_root=wrong_vendor,
            )


class SourceBuildIdProvenanceBindingTests(unittest.TestCase):
    """Adversarial-review follow-up (HI126 composition gap): replay.build()'s
    _validate_correctness_gate() previously trusted a projected artifact's
    hi121_source_provenance.source_build_id as-is (only range/type checked),
    feeding it directly into resolve_promotion_identity()'s build_id
    constraint with no proof that build_id actually corresponds to the REST
    of the same provenance tuple recorded right beside it in the same
    header. An attacker could retarget source_build_id to a DIFFERENT real
    build sharing the same dispatch/signature/hardware and inherit ITS
    correctness evidence."""

    def _make_db(self, root: Path) -> tuple[Path, int, int]:
        db_path = root / "dispatch.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set, build_descriptor_hash) VALUES "
            "(?, 'aaaa', 1, 1, 'inventory', 'desc-a')",
            ("a" * 40,),
        )
        build_a = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set, build_descriptor_hash) VALUES "
            "(?, 'bbbb', 1, 1, 'inventory', 'desc-b')",
            ("b" * 40,),
        )
        build_b = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()
        return db_path, build_a, build_b

    def test_source_build_id_pointing_at_wrong_build_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dispatch_db, build_a, build_b = self._make_db(root)
            entries = {
                "d" * 32: {
                    "kind": "result", "dispatch": "d" * 32,
                    "winner": "candidate", "native": "native",
                    "signature": "c" * 32, "hardware": "e" * 32,
                },
            }
            # Header claims build_a's own real source_revision/manifest_hash,
            # but the source_build_id field points at build_b instead.
            forged_provenance = {
                "source_build_id": build_b,
                "source_revision": "a" * 40,
                "manifest_hash": "aaaa",
                "build_descriptor_hash": "desc-a",
            }
            with self.assertRaisesRegex(SystemExit, "not bound to the rest"):
                replay_module._validate_correctness_gate(
                    entries, dispatch_db, build_b, forged_provenance,
                )

    def test_source_build_id_matching_its_own_provenance_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dispatch_db, build_a, _build_b = self._make_db(root)
            entries = {
                "d" * 32: {
                    "kind": "result", "dispatch": "d" * 32,
                    "winner": "mmvq:native:v1", "native": "mmvq:native:v1",
                },
            }
            real_provenance = {
                "source_build_id": build_a,
                "source_revision": "a" * 40,
                "manifest_hash": "aaaa",
                "build_descriptor_hash": "desc-a",
            }
            # winner == native here, so the correctness gate never even
            # needs to resolve a binding -- this just proves a CORRECT
            # provenance tuple doesn't spuriously raise.
            replay_module._validate_correctness_gate(
                entries, dispatch_db, build_a, real_provenance,
            )

    def test_source_build_id_without_provenance_tuple_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dispatch_db, build_a, _build_b = self._make_db(root)
            entries = {
                "d" * 32: {
                    "kind": "result", "dispatch": "d" * 32,
                    "winner": "candidate", "native": "native",
                    "signature": "c" * 32, "hardware": "e" * 32,
                },
            }
            with self.assertRaisesRegex(SystemExit, "without its"):
                replay_module._validate_correctness_gate(
                    entries, dispatch_db, build_a, None,
                )


if __name__ == "__main__":
    unittest.main()
