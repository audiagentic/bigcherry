"""Single canonical source for BigCherry's HIP dispatch signature/hardware
schema version numbers on the Python side (HI119 review follow-up).

Before this module existed, "the current schema is 1" was independently
hand-maintained in six places: the real C++ #define
(src/ggml/src/ggml-cuda/hip-autotune-types.h's GGML_HIP_SIGNATURE_SCHEMA_
VERSION), tuning/replay.py's own SIGNATURE_SCHEMA_VERSION/HARDWARE_SCHEMA_
VERSION module constants, two `header.get("signature_schema", 1)` fallback
sites (catalog.py, inventory.py) that conflated "current schema" with
"legacy header omitted the field", one hardcoded literal in catalog.py's
export-manifest builder, and the SQL schema_meta seed row in sql/dispatch-
db.sql -- nothing checked that any of them agreed, so a future bump could
silently update some sites and not others.

This module is a leaf: it imports nothing from this package, so every other
tuning module (replay, catalog, inventory, signature_mapping, ...) can import
it without a dependency cycle (replay.py already imports catalog.py).

tools/tests/test_dispatch_abi_agreement.py enforces this module's value
against the real C++ #define (parsed directly out of hip-autotune-types.h,
the same technique signature_mapping.py already uses for the real ggml enum
tables) and against sql/dispatch-db.sql's schema_meta seed row -- so a future
bump that misses one of the three fails a fast offline test instead of
silently drifting.
"""

from __future__ import annotations

# HI121 (round 9): the identity epoch is now FROZEN at 2. Additive/scoped
# semantic content (a new independent flag bit, a new optional field scoped
# to the op classes that actually use it) and new producer capabilities
# (tools/bigcherry/tuning/hip_capabilities.py) NEVER bump this -- the whole
# point of HI121's two-axis model (semantic content vs. producer knowledge)
# is that a new semantic distinction is expressed as a new capability bit
# plus (if needed) a new content field, not as a global epoch bump that
# invalidates every existing signature's digest regardless of relevance.
#
# Bump this ONLY when the canonical signature REPRESENTATION/encoding itself
# changes incompatibly (e.g. the canonicalization algorithm changes, or an
# EXISTING hashed field's on-wire meaning is reinterpreted such that old and
# new semantics cannot be distinguished through content + capability
# applicability alone) -- and bump GGML_HIP_SIGNATURE_SCHEMA_VERSION in
# hip-autotune-types.h and sql/dispatch-db.sql's schema_meta seed row in the
# same commit. This is exactly the HI118/HI119 case that forced the 1->2
# bump: schema 1 had no way to represent "producer evaluated bias/scale
# presence at all", so the reinterpretation was of the canonical
# representation's own completeness, not an ordinary additive change.
SIGNATURE_IDENTITY_EPOCH = 2

# Compatibility name: several existing call sites and serialized artifact
# fields (JSONL headers, manifests) already say "signature_schema" -- this
# alias keeps that vocabulary working without a second independently
# hand-maintained constant. Always equal to SIGNATURE_IDENTITY_EPOCH; never
# assign it separately.
SIGNATURE_SCHEMA_VERSION = SIGNATURE_IDENTITY_EPOCH
HARDWARE_SCHEMA_VERSION = 1

# The schema a historical artifact meant when it PREDATES this field
# existing in its own header/JSONL row at all -- a distinct concept from
# "the current schema", and must never silently track SIGNATURE_SCHEMA_
# VERSION as it's bumped forward, or every old artifact missing the field
# would be silently reinterpreted as the new current schema instead of the
# schema it actually was produced under.
LEGACY_MISSING_SIGNATURE_SCHEMA_VERSION = 1
LEGACY_MISSING_HARDWARE_SCHEMA_VERSION = 1
