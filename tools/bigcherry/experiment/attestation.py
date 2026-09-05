"""VA25: execution attestation -- proof that a measurement ran where it claimed.

THE INVARIANT. No measurement is valid unless the execution environment
ACTUALLY OBSERVED BY THAT MEASURED PROCESS positively matches the identity the
lane resolved for it. Missing or unrecognised attestation is INVALID, never
neutral: absence of an error is not evidence of correct execution.

WHY THIS EXISTS. A ROCm/HIP init failure does not stop llama.cpp; it falls
back to CPU and still prints a well-formed results table labelled backend
"ROCm". Measured on real hardware 2026-09-05, same binary, same model, same
command, differing only in device-selection environment:

    failed to initialize ROCm: no ROCm-capable device is detected
    | qwen35 9B Q6_K | ROCm | pp512 |   43.70 +/- 0.06 |
    | qwen35 9B Q6_K | ROCm | tg128 |   10.46 +/- 0.00 |

    found 1 ROCm devices ... gfx1201, VRAM 32624 MiB
    | qwen35 9B Q6_K | ROCm | pp512 | 3160.50 +/- 31.80 |
    | qwen35 9B Q6_K | ROCm | tg128 |   69.57 +/- 0.07  |

72x and 6.7x wrong, and the CPU run reported the TIGHTER variance. The only
signal was one stderr line above the table.

That is uniquely dangerous for this project. Promotion runs a paired A/B with
a block bootstrap and a pre-registered confidence rule, and the outlier policy
is deliberately frozen as "keep every valid pair, never trim, never use a
statistical test to discard". A silently-CPU arm therefore produces an
enormous, entirely plausible effect with tight intervals, and the frozen
policy obliges us to KEEP it -- unless it is caught here, as a VALIDITY
FAILURE, upstream of statistics. The frozen doctrine already names this
category ("required telemetry is missing or corrupt"), defined independently
of any expected benefit.

It has already cost this project twice: RD08's false subject_hit=false on
2026-09-01 (gfx1030's HIP runtime failed to detect the device and a trigger
check quietly ran CPU-only), and VA22's gfx1201 double-env-selector bug, which
recurred on 2026-09-05 despite being documented -- which is the argument for
enforcing this in code rather than in a runbook.

IDENTITY vs TELEMETRY. Only identity fields are compared; telemetry is
recorded for diagnosis and never gates. Device LOCATOR is identity, not just
architecture: this host has two gfx1100 cards, so architecture alone cannot
detect a run silently moving between them.

Attestation must be derived from the measured process's OWN backend
initialisation output. rocminfo, environment variables, build targets and the
results-table backend label prove intent or capability -- never execution.

Design reviewed adversarially (dev-gpt-agent, req_7bd183755a0643e7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

#: Stable reason codes. Callers branch on these rather than on message text.
ATTESTATION_MISSING = "ATTESTATION_MISSING"
ATTESTATION_CORRUPT = "ATTESTATION_CORRUPT"
BACKEND_MISMATCH = "BACKEND_MISMATCH"
DEVICE_COUNT_MISMATCH = "DEVICE_COUNT_MISMATCH"
ARCH_MISMATCH = "ARCH_MISMATCH"
DEVICE_ID_MISMATCH = "DEVICE_ID_MISMATCH"

REASON_CODES: tuple[str, ...] = (
    ATTESTATION_MISSING, ATTESTATION_CORRUPT, BACKEND_MISMATCH,
    DEVICE_COUNT_MISMATCH, ARCH_MISMATCH, DEVICE_ID_MISMATCH,
)


class AttestationError(RuntimeError):
    """Raised when a measurement cannot be accepted as real execution."""

    def __init__(self, message: str, *, reasons: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.reasons: tuple[str, ...] = tuple(reasons)


@dataclass(frozen=True)
class ObservedDevice:
    """One accelerator the measured process actually initialised."""

    architecture: str
    #: Stable locator -- UUID where available, else PCI bus:device.function.
    #: None means the backend did not report one; that is recorded but cannot
    #: be compared, so a lane that declares expected locators and receives
    #: None fails DEVICE_ID_MISMATCH rather than passing by omission.
    locator: str | None = None


@dataclass(frozen=True)
class ExecutionIdentity:
    """What a lane REQUIRES of the environment, resolved before measuring.

    ``architectures`` is ordered by device index, so a two-device lane whose
    cards are enumerated in the wrong order is a mismatch rather than a
    set-equality pass -- device order is load-bearing for tensor split.
    """

    backend: str
    architectures: tuple[str, ...]
    locators: tuple[str, ...] | None = None

    @property
    def device_count(self) -> int:
        return len(self.architectures)

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("ExecutionIdentity.backend must be non-empty")
        if not self.architectures:
            raise ValueError(
                "ExecutionIdentity.architectures must be non-empty -- a lane that "
                "declares no expected device cannot attest anything"
            )
        if self.locators is not None and len(self.locators) != len(self.architectures):
            raise ValueError(
                "ExecutionIdentity.locators must be one per architecture when given"
            )


@dataclass(frozen=True)
class ExecutionAttestation:
    """What the measured process was OBSERVED to initialise."""

    backend: str | None
    devices: tuple[ObservedDevice, ...] = ()
    telemetry: Mapping[str, object] = field(default_factory=dict)
    #: Set when the output positively indicated an initialisation FAILURE
    #: (as opposed to merely lacking a success line).
    failure_signature: str | None = None

    @property
    def device_count(self) -> int:
        return len(self.devices)

    def document(self) -> dict[str, object]:
        """Serialisable form for the evidence record."""
        return {
            "schema_version": 1,
            "backend": self.backend,
            "device_count": self.device_count,
            "devices": [
                {"architecture": d.architecture, "locator": d.locator}
                for d in self.devices
            ],
            "failure_signature": self.failure_signature,
            "telemetry": dict(self.telemetry),
        }


def compare_execution_identity(
    expected: ExecutionIdentity, observed: ExecutionAttestation | None,
) -> tuple[str, ...]:
    """THE canonical comparator. Returns () on PASS, else reason codes.

    Deliberately the only place this comparison is implemented. The recurring
    defect in this codebase is a check existing in one place and not reaching
    the path that needs it, so every enforcement layer -- runner, evidence
    writer, promotion gate -- must call THIS, not re-derive the logic.
    """
    if observed is None:
        return (ATTESTATION_MISSING,)
    if observed.failure_signature is not None:
        return (ATTESTATION_CORRUPT,)
    if observed.backend is None or not observed.devices:
        # Output parsed but carried no positive evidence of initialisation.
        return (ATTESTATION_MISSING,)

    reasons: list[str] = []
    if observed.backend.lower() != expected.backend.lower():
        reasons.append(BACKEND_MISMATCH)
    if observed.device_count != expected.device_count:
        reasons.append(DEVICE_COUNT_MISMATCH)
        # Per-device comparisons below would be meaningless at a different
        # cardinality; report the cardinality fault alone.
        return tuple(reasons)

    if tuple(d.architecture for d in observed.devices) != expected.architectures:
        reasons.append(ARCH_MISMATCH)

    if expected.locators is not None:
        observed_locators = tuple(d.locator for d in observed.devices)
        if any(loc is None for loc in observed_locators):
            # A lane that pinned locators must not pass because the backend
            # declined to report them.
            reasons.append(DEVICE_ID_MISMATCH)
        elif observed_locators != expected.locators:
            reasons.append(DEVICE_ID_MISMATCH)

    return tuple(reasons)


def require_execution_identity(
    expected: ExecutionIdentity, observed: ExecutionAttestation | None, *, context: str,
) -> None:
    """Fail closed. Raises AttestationError carrying the reason codes."""
    reasons = compare_execution_identity(expected, observed)
    if not reasons:
        return
    detail = ", ".join(reasons)
    raise AttestationError(
        f"{context}: execution attestation failed [{detail}] -- expected "
        f"{expected.backend} {expected.device_count}x{list(expected.architectures)}, "
        f"observed {observed.document() if observed is not None else 'nothing'}. "
        f"A measurement whose execution environment cannot be positively "
        f"confirmed is not evidence.",
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# ROCm attestor
# ---------------------------------------------------------------------------

#: Positive initialisation. llama.cpp prints e.g.
#:   ggml_cuda_init: found 2 ROCm devices:
_ROCM_FOUND = re.compile(r"ggml_cuda_init:\s*found\s+(\d+)\s+ROCm device")

#: Per-device line, e.g.
#:   Device 0: AMD Radeon Graphics, gfx1201 (0x1201), VMM: no, Wave Size: 32
_ROCM_DEVICE = re.compile(
    r"Device\s+(?P<index>\d+):\s*(?P<name>[^,]+),\s*(?P<arch>gfx[0-9a-fA-F]+)"
)

#: Known initialisation-failure signatures. Matching any of these is a
#: positive statement that the backend did NOT come up -- distinct from
#: merely lacking a success line, and reported as ATTESTATION_CORRUPT.
_ROCM_FAILURES: tuple[re.Pattern[str], ...] = (
    re.compile(r"failed to initialize ROCm"),
    re.compile(r"no ROCm-capable device is detected"),
    re.compile(r"hipErrorNoDevice"),
)

_VRAM = re.compile(r"VRAM:\s*(\d+)\s*MiB")


# ---------------------------------------------------------------------------
# llama-server attestor
# ---------------------------------------------------------------------------
#
# llama-server does NOT route ggml's init logs to its own stream. Measured on
# real hardware 2026-09-05: running with -v (verbosity threshold INT_MAX)
# produced 2684 log lines and ZERO "ggml_cuda_init" lines. So the llama-bench
# attestor above cannot be reused here, and no verbosity flag fixes it.
#
# What llama-server DOES emit is better for our purposes, because it carries
# the PCI locator that distinguishes this host's two gfx1100 cards:
#
#   I llama_prepare_model_devices: using device ROCm0 (AMD Radeon RX 7900 XTX)
#         (0000:03:00.0) - 24520 MiB free
#   D load_tensors: layer 0 assigned to device ROCm0, is_swa = 0
#
# The second is stronger evidence than device DETECTION: it proves tensors were
# actually assigned to the device, which is what a CPU fallback would not do.
#
# CAVEAT, found the hard way and the reason this is not yet wired into RD73's
# lanes: the "using device" line comes from the device-memory-FITTING path, and
# RD73's servers must pass "--fit off" (required alongside -sm tensor, else
# llama.cpp aborts with "llama_params_fit is not implemented for
# SPLIT_MODE_TENSOR"). With --fit off that path is skipped, so the line never
# appears -- RD73's archived logs contain I and W lines but no ROCm device
# mention whatsoever. The "layer N assigned" line is DEBUG level and so is
# filtered at the verbosity those lanes ran at.
#
# Consequence: a validation server lane must run at a verbosity that emits the
# layer-assignment lines, and must not rely on the fitting path. Until that is
# done, server lanes cannot attest, which is recorded in VA25 rather than
# worked around.

_SERVER_USING_DEVICE = re.compile(
    r"using device (?P<backend>[A-Za-z]+)(?P<index>\d+)\s*"
    r"\((?P<name>[^)]*)\)\s*"
    r"\((?P<locator>[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d)\)"
)

_SERVER_LAYER_ASSIGNED = re.compile(
    r"assigned to device (?P<backend>[A-Za-z]+)(?P<index>\d+)"
)


def parse_llama_server_attestation(
    output: str, *, architecture_by_locator: Mapping[str, str] | None = None,
) -> ExecutionAttestation | None:
    """Attest a llama-server process from its own log stream.

    Returns None when the output carries no device evidence at all -- which
    the comparator treats as ATTESTATION_MISSING, never as a pass.

    ``architecture_by_locator`` maps PCI BDF -> gfx arch, because the server's
    device line reports a marketing name and locator but not the ISA. The
    mapping is host configuration, so it is supplied rather than guessed; a
    locator absent from it yields an unknown architecture, which then fails
    ARCH_MISMATCH rather than silently matching.
    """
    if not output:
        return None

    for pattern in _ROCM_FAILURES:
        match = pattern.search(output)
        if match is not None:
            return ExecutionAttestation(
                backend=None, devices=(), failure_signature=match.group(0),
            )

    lookup = dict(architecture_by_locator or {})
    by_index: dict[int, ObservedDevice] = {}
    backend: str | None = None

    for match in _SERVER_USING_DEVICE.finditer(output):
        backend = match.group("backend")
        locator = match.group("locator").lower()
        by_index[int(match.group("index"))] = ObservedDevice(
            architecture=lookup.get(locator, "<unknown>"), locator=locator,
        )

    # Layer assignment proves tensors really went to the device. It cannot
    # supply a locator, so it only contributes devices the "using device"
    # lines did not already report.
    assigned_indices: set[int] = set()
    for match in _SERVER_LAYER_ASSIGNED.finditer(output):
        backend = backend or match.group("backend")
        assigned_indices.add(int(match.group("index")))
    for index in assigned_indices:
        by_index.setdefault(index, ObservedDevice(architecture="<unknown>", locator=None))

    if not by_index:
        return None

    devices = tuple(by_index[i] for i in sorted(by_index))
    telemetry: dict[str, object] = {
        "layers_assigned_to_devices": sorted(assigned_indices),
    }
    return ExecutionAttestation(
        backend=backend, devices=devices, telemetry=telemetry,
    )


def parse_rocm_attestation(output: str) -> ExecutionAttestation | None:
    """Parse ROCm backend-init evidence out of a measured process's output.

    Returns None when the output carries NO attestation at all -- which the
    comparator treats as ATTESTATION_MISSING, never as a pass. Text parsing is
    acceptable only because it fails closed in every ambiguous case; the
    durable answer is structured backend-init output, and this function is the
    single place that would be replaced by it.
    """
    if not output:
        return None

    for pattern in _ROCM_FAILURES:
        match = pattern.search(output)
        if match is not None:
            return ExecutionAttestation(
                backend=None, devices=(), failure_signature=match.group(0),
            )

    found = _ROCM_FOUND.search(output)
    if found is None:
        return None

    declared = int(found.group(1))
    devices: list[ObservedDevice] = []
    for match in _ROCM_DEVICE.finditer(output):
        devices.append(ObservedDevice(architecture=match.group("arch"), locator=None))
        if len(devices) == declared:
            break

    if len(devices) != declared:
        # The count line and the per-device lines disagree: the output is not
        # trustworthy, so refuse it rather than guessing which is right.
        return ExecutionAttestation(
            backend=None, devices=(),
            failure_signature=(
                f"ggml_cuda_init reported {declared} device(s) but "
                f"{len(devices)} device line(s) were parsed"
            ),
        )

    telemetry: dict[str, object] = {}
    vram = _VRAM.search(output)
    if vram is not None:
        telemetry["first_device_vram_mib"] = int(vram.group(1))

    return ExecutionAttestation(
        backend="ROCm", devices=tuple(devices), telemetry=telemetry,
    )
