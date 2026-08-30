"""HTR03: offline tests for the versioned behavioral corpus manifest --
schema validation, class-tag/requirement fail-closed behavior, edition
immutability (content digest changes iff vectors/params change), and
applicability resolution (including the mandatory "zero matches is loud,
not silent" contract)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import behavioral_corpus as bc  # noqa: E402


def _write_prompt(fixtures_dir: Path, name: str, content: str = "hello world") -> str:
    (fixtures_dir / name).write_text(content, encoding="utf-8")
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _manifest_toml(*, edition="ed-v1", classes=("mtp-speculative",), vectors_toml="") -> str:
    classes_str = ", ".join(f'"{c}"' for c in classes)
    return f"""
schema_version = 1
edition = "{edition}"
classes = [{classes_str}]

{vectors_toml}
"""


def _vector_toml(*, id="v1", applies_to=("mtp-speculative",), requirements=("mtp-telemetry",),
                  prompt_file="p.txt", prompt_sha256="deadbeef", n_predict=128, seed=42,
                  scenario="s", provenance="HI141") -> str:
    applies_str = ", ".join(f'"{c}"' for c in applies_to)
    reqs_str = ", ".join(f'"{r}"' for r in requirements)
    return f"""
[[vectors]]
id = "{id}"
prompt_file = "{prompt_file}"
prompt_sha256 = "{prompt_sha256}"
n_predict = {n_predict}
seed = {seed}
applies_to = [{applies_str}]
requirements = [{reqs_str}]
scenario = "{scenario}"
provenance = "{provenance}"
"""


class LoadCorpusEditionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixtures = Path(self._tmp.name)

    def _write_manifest(self, text: str) -> Path:
        path = self.fixtures / "corpus.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_a_valid_manifest(self):
        sha = _write_prompt(self.fixtures, "p.txt")
        manifest = self._write_manifest(
            _manifest_toml(vectors_toml=_vector_toml(prompt_sha256=sha))
        )
        edition = bc.load_corpus_edition(manifest, self.fixtures)
        self.assertEqual(edition.edition, "ed-v1")
        self.assertEqual(len(edition.vectors), 1)
        self.assertEqual(edition.vectors[0].id, "v1")

    def test_rejects_unsupported_schema_version(self):
        manifest = self._write_manifest("schema_version = 2\nedition = \"e\"\nclasses = []\n")
        with self.assertRaises(bc.BehavioralCorpusError):
            bc.load_corpus_edition(manifest, self.fixtures)

    def test_rejects_unknown_applies_to_class(self):
        sha = _write_prompt(self.fixtures, "p.txt")
        manifest = self._write_manifest(_manifest_toml(
            vectors_toml=_vector_toml(prompt_sha256=sha, applies_to=("not-a-declared-class",))
        ))
        with self.assertRaises(bc.BehavioralCorpusError):
            bc.load_corpus_edition(manifest, self.fixtures)

    def test_rejects_unknown_requirement(self):
        sha = _write_prompt(self.fixtures, "p.txt")
        manifest = self._write_manifest(_manifest_toml(
            vectors_toml=_vector_toml(prompt_sha256=sha, requirements=("not-a-real-requirement",))
        ))
        with self.assertRaises(bc.BehavioralCorpusError):
            bc.load_corpus_edition(manifest, self.fixtures)

    def test_rejects_content_drift_from_declared_sha256(self):
        _write_prompt(self.fixtures, "p.txt", content="original")
        manifest = self._write_manifest(_manifest_toml(
            vectors_toml=_vector_toml(prompt_sha256="0" * 64)  # wrong digest on purpose
        ))
        with self.assertRaises(bc.BehavioralCorpusError):
            bc.load_corpus_edition(manifest, self.fixtures)

    def test_rejects_duplicate_vector_ids(self):
        sha = _write_prompt(self.fixtures, "p.txt")
        manifest = self._write_manifest(_manifest_toml(
            vectors_toml=_vector_toml(id="dup", prompt_sha256=sha) + _vector_toml(id="dup", prompt_sha256=sha)
        ))
        with self.assertRaises(bc.BehavioralCorpusError):
            bc.load_corpus_edition(manifest, self.fixtures)

    def test_rejects_empty_applies_to(self):
        sha = _write_prompt(self.fixtures, "p.txt")
        manifest = self._write_manifest(_manifest_toml(
            vectors_toml=_vector_toml(prompt_sha256=sha, applies_to=())
        ))
        with self.assertRaises(bc.BehavioralCorpusError):
            bc.load_corpus_edition(manifest, self.fixtures)

    def test_content_digest_changes_iff_content_changes(self):
        sha = _write_prompt(self.fixtures, "p.txt")
        manifest_a = self._write_manifest(_manifest_toml(
            vectors_toml=_vector_toml(prompt_sha256=sha, n_predict=128)
        ))
        edition_a = bc.load_corpus_edition(manifest_a, self.fixtures)

        # Identical content, re-parsed -- digest must be stable.
        edition_a2 = bc.load_corpus_edition(manifest_a, self.fixtures)
        self.assertEqual(edition_a.content_digest, edition_a2.content_digest)

        # A real parameter change must change the digest.
        manifest_b = self._write_manifest(_manifest_toml(
            vectors_toml=_vector_toml(prompt_sha256=sha, n_predict=256)
        ))
        edition_b = bc.load_corpus_edition(manifest_b, self.fixtures)
        self.assertNotEqual(edition_a.content_digest, edition_b.content_digest)


class ResolveApplicableVectorsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixtures = Path(self._tmp.name)

    def _edition(self, classes, vectors_toml):
        sha = _write_prompt(self.fixtures, "p.txt")
        manifest = self.fixtures / "corpus.toml"
        manifest.write_text(
            _manifest_toml(classes=classes, vectors_toml=vectors_toml.replace("__SHA__", sha)),
            encoding="utf-8",
        )
        return bc.load_corpus_edition(manifest, self.fixtures)

    def test_intersecting_class_selects_the_vector(self):
        edition = self._edition(
            ("mtp-speculative", "other-class"),
            _vector_toml(prompt_sha256="__SHA__", applies_to=("mtp-speculative",)),
        )
        applicable = bc.resolve_applicable_vectors(edition, ("mtp-speculative",))
        self.assertEqual(len(applicable), 1)

    def test_non_intersecting_class_selects_nothing_silently_when_no_classes_requested(self):
        edition = self._edition(
            ("mtp-speculative",),
            _vector_toml(prompt_sha256="__SHA__", applies_to=("mtp-speculative",)),
        )
        applicable = bc.resolve_applicable_vectors(edition, ())
        self.assertEqual(applicable, ())

    def test_requested_class_matching_zero_vectors_raises_loud(self):
        edition = self._edition(
            ("mtp-speculative", "other-class"),
            _vector_toml(prompt_sha256="__SHA__", applies_to=("mtp-speculative",)),
        )
        with self.assertRaises(bc.BehavioralCorpusError):
            bc.resolve_applicable_vectors(edition, ("other-class",))

    def test_unknown_requested_class_raises(self):
        edition = self._edition(
            ("mtp-speculative",),
            _vector_toml(prompt_sha256="__SHA__", applies_to=("mtp-speculative",)),
        )
        with self.assertRaises(bc.BehavioralCorpusError):
            bc.resolve_applicable_vectors(edition, ("nonexistent-class",))


if __name__ == "__main__":
    unittest.main()
