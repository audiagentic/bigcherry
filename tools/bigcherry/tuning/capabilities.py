"""Generic, backend-neutral 128-bit capability mask (HI121 M1a).

A capability bit means "this producer's code knew how to correctly evaluate
semantic question X" -- a distinct axis from a signature's own CONTENT flags
("this dispatch actually HAS property X"). Content flags are hashed into a
signature's digest; capability masks are never hashed anywhere -- they are
compatibility-gate metadata, checked offline (see hip_capabilities.py and
signature_capabilities.py), never folded into signature identity.

This module knows nothing about what any bit MEANS -- HIP and Vulkan each
define their own independent capability registries (hip_capabilities.py,
a future vk_capabilities.py) using this as their shared 128-bit primitive,
with fully independent bit namespaces per backend (see docs/planning/active/
hip-autotune/HI121.md).

Canonical representations, all equivalent:
  * logical value: an unsigned 128-bit integer
  * text (JSON/manifest/DB-adjacent): exactly 32 lowercase hex digits,
    most-significant byte first (e.g. "0000000000000000000000000000001f")
  * bytes (SQLite BLOB, wire): exactly 16 bytes, big-endian
  * C++ words: two uint64_t, LO and HI halves
"""

from __future__ import annotations

from dataclasses import dataclass

_BIT_WIDTH = 128
_BYTE_LENGTH = 16
_HEX_LENGTH = 32
_MAX_VALUE = (1 << _BIT_WIDTH) - 1
_WORD_MASK = (1 << 64) - 1


class CapabilityMaskError(ValueError):
    """The capability mask is malformed or outside the 128-bit domain."""


@dataclass(frozen=True)
class CapabilityMask128:
    """An immutable unsigned 128-bit capability bitmask."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise CapabilityMaskError(f"capability mask value must be an int, got {type(self.value).__name__}")
        if self.value < 0 or self.value > _MAX_VALUE:
            raise CapabilityMaskError(
                f"capability mask value {self.value!r} is outside the unsigned 128-bit domain "
                f"[0, {_MAX_VALUE}]"
            )

    @classmethod
    def from_words(cls, lo: int, hi: int) -> "CapabilityMask128":
        """Build a mask from low/high unsigned 64-bit words."""
        for name, word in (("lo", lo), ("hi", hi)):
            if isinstance(word, bool) or not isinstance(word, int):
                raise CapabilityMaskError(f"{name} word must be an int, got {type(word).__name__}")
            if word < 0 or word > _WORD_MASK:
                raise CapabilityMaskError(f"{name} word {word!r} does not fit in an unsigned 64-bit word")
        return cls((hi << 64) | lo)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CapabilityMask128":
        """Decode the canonical 16-byte big-endian storage representation."""
        if not isinstance(data, (bytes, bytearray)):
            raise CapabilityMaskError(f"capability mask bytes must be bytes, got {type(data).__name__}")
        if len(data) != _BYTE_LENGTH:
            raise CapabilityMaskError(
                f"capability mask must be exactly {_BYTE_LENGTH} bytes, got {len(data)}"
            )
        return cls(int.from_bytes(bytes(data), byteorder="big", signed=False))

    @classmethod
    def from_hex(cls, value: str) -> "CapabilityMask128":
        """Decode the canonical 32-lowercase-hex artifact representation."""
        if not isinstance(value, str):
            raise CapabilityMaskError(f"capability mask hex must be a str, got {type(value).__name__}")
        if len(value) != _HEX_LENGTH or value != value.lower():
            raise CapabilityMaskError(
                f"capability mask hex must be exactly {_HEX_LENGTH} lowercase hex characters, got {value!r}"
            )
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise CapabilityMaskError(f"capability mask hex is not valid hexadecimal: {value!r}") from exc
        return cls.from_bytes(raw)

    def to_words(self) -> tuple[int, int]:
        """Return the mask as (lo, hi) unsigned 64-bit words."""
        return self.value & _WORD_MASK, (self.value >> 64) & _WORD_MASK

    def to_bytes(self) -> bytes:
        """Encode the canonical 16-byte big-endian database representation."""
        return self.value.to_bytes(_BYTE_LENGTH, byteorder="big", signed=False)

    def to_hex(self) -> str:
        """Encode the canonical 32-lowercase-hex artifact representation."""
        return self.to_bytes().hex()

    def contains(self, required: "CapabilityMask128") -> bool:
        """Return whether this mask contains every bit set in `required`."""
        if not isinstance(required, CapabilityMask128):
            raise CapabilityMaskError(f"required must be a CapabilityMask128, got {type(required).__name__}")
        return (self.value & required.value) == required.value

    def __or__(self, other: "CapabilityMask128") -> "CapabilityMask128":
        if not isinstance(other, CapabilityMask128):
            return NotImplemented
        return CapabilityMask128(self.value | other.value)
