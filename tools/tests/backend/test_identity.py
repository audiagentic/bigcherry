"""RRVP02 Stage 1 resolved-stack wire identity contracts."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.backend.identity import (
    BackendIdentityError,
    IdentityAttribute,
    ResolvedStackIdentity,
    SoftwareComponentIdentity,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _present_component(
    name: str,
    *attributes: IdentityAttribute,
) -> SoftwareComponentIdentity:
    return SoftwareComponentIdentity(name=name, state="present", attributes=attributes)


class ResolvedStackIdentityTests(unittest.TestCase):
    def test_wire_and_fingerprint_are_versioned_and_stable(self):
        identity = ResolvedStackIdentity(
            backend="hip",
            components=(SoftwareComponentIdentity("provider.ck", "missing"),),
        )
        self.assertEqual(
            identity.canonical_json_bytes(),
            b'{"backend":"hip","components":{"provider.ck":{"attributes":{},"state":"missing"}},'
            b'"schema":"bigcherry/resolved-stack-identity/v1"}',
        )
        self.assertEqual(identity.fingerprint, "36ec1d666cf99053550677ed11543ff3")

    def test_component_and_attribute_order_is_canonical(self):
        a = ResolvedStackIdentity(
            backend="hip",
            components=(
                _present_component(
                    "runtime.hip",
                    IdentityAttribute("version", "present", "7.1"),
                    IdentityAttribute("name", "present", "HIP Runtime"),
                ),
                _present_component(
                    "provider.rocblas",
                    IdentityAttribute("artifact_sha256", "present", _SHA_A),
                ),
            ),
        )
        b = ResolvedStackIdentity(
            backend="hip",
            components=(
                _present_component(
                    "provider.rocblas",
                    IdentityAttribute("artifact_sha256", "present", _SHA_A),
                ),
                _present_component(
                    "runtime.hip",
                    IdentityAttribute("name", "present", "HIP Runtime"),
                    IdentityAttribute("version", "present", "7.1"),
                ),
            ),
        )
        self.assertEqual(a, b)
        self.assertEqual(a.canonical_json_bytes(), b.canonical_json_bytes())
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_round_trip_is_byte_identical(self):
        identity = ResolvedStackIdentity(
            backend="hip",
            components=(
                _present_component(
                    "compiler.cxx",
                    IdentityAttribute("vendor", "present", "AMD"),
                    IdentityAttribute("version", "unknown"),
                ),
                SoftwareComponentIdentity(name="provider.hipblaslt", state="missing"),
            ),
        )
        parsed = ResolvedStackIdentity.from_document(identity.document())
        self.assertEqual(parsed, identity)
        self.assertEqual(parsed.canonical_json_bytes(), identity.canonical_json_bytes())
        self.assertEqual(parsed.fingerprint, identity.fingerprint)

    def test_missing_and_unknown_are_distinct_identity(self):
        missing = ResolvedStackIdentity(
            backend="hip",
            components=(SoftwareComponentIdentity("provider.ck", "missing"),),
        )
        unknown = ResolvedStackIdentity(
            backend="hip",
            components=(SoftwareComponentIdentity("provider.ck", "unknown"),),
        )
        self.assertNotEqual(missing.document(), unknown.document())
        self.assertNotEqual(missing.fingerprint, unknown.fingerprint)

    def test_present_attribute_requires_value(self):
        with self.assertRaisesRegex(BackendIdentityError, "requires a non-blank"):
            IdentityAttribute("version", "present")
        with self.assertRaisesRegex(BackendIdentityError, "requires a non-blank"):
            IdentityAttribute("version", "present", "  ")

    def test_missing_or_unknown_attribute_forbids_value(self):
        for state in ("missing", "unknown"):
            with self.subTest(state=state):
                with self.assertRaisesRegex(BackendIdentityError, "must have value=None"):
                    IdentityAttribute("version", state, "7.1")  # type: ignore[arg-type]

    def test_missing_or_unknown_component_forbids_attributes(self):
        attribute = IdentityAttribute("version", "present", "7.1")
        for state in ("missing", "unknown"):
            with self.subTest(state=state):
                with self.assertRaisesRegex(BackendIdentityError, "cannot carry attributes"):
                    SoftwareComponentIdentity(
                        "provider.ck", state, (attribute,)  # type: ignore[arg-type]
                    )

    def test_duplicate_names_are_rejected(self):
        with self.assertRaisesRegex(BackendIdentityError, "duplicate attributes"):
            _present_component(
                "runtime.hip",
                IdentityAttribute("version", "present", "7.1"),
                IdentityAttribute("version", "present", "7.2"),
            )
        component = _present_component("runtime.hip")
        with self.assertRaisesRegex(BackendIdentityError, "duplicate components"):
            ResolvedStackIdentity(backend="hip", components=(component, component))

    def test_identity_changes_for_version_artifact_or_backend(self):
        base = ResolvedStackIdentity(
            backend="hip",
            components=(
                _present_component(
                    "provider.rocblas",
                    IdentityAttribute("version", "present", "4.3.0"),
                    IdentityAttribute("artifact_sha256", "present", _SHA_A),
                ),
            ),
        )
        changed_version = ResolvedStackIdentity(
            backend="hip",
            components=(
                _present_component(
                    "provider.rocblas",
                    IdentityAttribute("version", "present", "4.4.0"),
                    IdentityAttribute("artifact_sha256", "present", _SHA_A),
                ),
            ),
        )
        changed_artifact = ResolvedStackIdentity(
            backend="hip",
            components=(
                _present_component(
                    "provider.rocblas",
                    IdentityAttribute("version", "present", "4.3.0"),
                    IdentityAttribute("artifact_sha256", "present", _SHA_B),
                ),
            ),
        )
        changed_backend = ResolvedStackIdentity(
            backend="cuda",
            components=base.components,
        )
        self.assertNotEqual(base.fingerprint, changed_version.fingerprint)
        self.assertNotEqual(base.fingerprint, changed_artifact.fingerprint)
        self.assertNotEqual(base.fingerprint, changed_backend.fingerprint)

    def test_wire_parser_is_strict(self):
        valid = ResolvedStackIdentity(
            backend="hip",
            components=(_present_component("runtime.hip"),),
        ).document()
        for mutated in (
            {**valid, "extra": 1},
            {**valid, "schema": "bigcherry/resolved-stack-identity/v2"},
            {**valid, "components": []},
        ):
            with self.subTest(mutated=mutated):
                with self.assertRaises(BackendIdentityError):
                    ResolvedStackIdentity.from_document(mutated)

        with self.assertRaises(BackendIdentityError):
            ResolvedStackIdentity.from_document(
                {
                    "schema": "bigcherry/resolved-stack-identity/v1",
                    "backend": "hip",
                    "components": {
                        "runtime.hip": {
                            "state": "present",
                            "attributes": {
                                "version": {"state": "present", "value": "7.1", "extra": 1}
                            },
                        }
                    },
                }
            )

    def test_closed_attribute_vocabulary_excludes_volatile_identity(self):
        for name in ("path", "hostname", "timestamp", "bdf", "ordinal", "architecture"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(BackendIdentityError, "unknown identity attribute"):
                    IdentityAttribute(name, "present", "x")  # type: ignore[arg-type]

    def test_artifact_sha256_is_strict_lowercase_hex(self):
        for value in ("a" * 63, "A" * 64, "g" * 64):
            with self.subTest(value=value):
                with self.assertRaisesRegex(BackendIdentityError, "64 lowercase hexadecimal"):
                    IdentityAttribute("artifact_sha256", "present", value)

    def test_instances_are_frozen(self):
        attribute = IdentityAttribute("version", "present", "7.1")
        component = _present_component("runtime.hip", attribute)
        identity = ResolvedStackIdentity(backend="hip", components=(component,))
        with self.assertRaises(FrozenInstanceError):
            identity.backend = "cuda"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            component.state = "missing"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            attribute.value = "7.2"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
