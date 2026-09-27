"""Pure measurement-window state machine used by the privileged RCD11 helper.

The eventual root helper performs side effects; this module validates requests
and legal transitions so crash/timeout/boot recovery can be unit-tested without
systemd, Slurm or llama-swap.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WindowError(RuntimeError):
    pass


class WindowState(str, Enum):
    CLOSED = "closed"
    REQUESTED = "requested"
    DRAINING = "draining"
    ACTIVE = "active"
    CLOSING = "closing"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class WindowRequest:
    window_id: str
    execution_id: str
    target_device_ids: tuple[str, ...]
    inventory_hash: str
    production_config_hash: str
    deadline_unix_ns: int

    def __post_init__(self) -> None:
        if not self.window_id.strip() or not self.execution_id.strip():
            raise ValueError("window_id and execution_id are required")
        if not self.target_device_ids:
            raise ValueError("measurement window needs target_device_ids")
        if len(set(self.target_device_ids)) != len(self.target_device_ids):
            raise ValueError("target_device_ids contains duplicates")
        if not self.inventory_hash or not self.production_config_hash:
            raise ValueError("inventory and production config hashes are required")
        if self.deadline_unix_ns <= 0:
            raise ValueError("deadline must be positive")


@dataclass(frozen=True)
class WindowRecord:
    state: WindowState
    request: WindowRequest | None = None
    reason: str | None = None


def transition(
    record: WindowRecord,
    action: str,
    *,
    request: WindowRequest | None = None,
    reason: str | None = None,
) -> WindowRecord:
    """Apply one idempotent/legal transition.

    `recover` is legal from every non-closed state. `close` is idempotent from
    CLOSED and funnels ACTIVE/REQUESTED/DRAINING through CLOSING. Side-effect
    drivers must not restart production until CLOSING/RECOVERY reaches CLOSED.
    """
    state = record.state
    if action == "request":
        if request is None:
            raise WindowError("request action requires WindowRequest")
        if state == WindowState.CLOSED:
            return WindowRecord(WindowState.REQUESTED, request)
        if state == WindowState.REQUESTED and record.request == request:
            return record
        raise WindowError(f"cannot request window from {state.value}")
    if action == "begin-drain":
        if state == WindowState.DRAINING:
            return record
        if state != WindowState.REQUESTED:
            raise WindowError(f"cannot begin drain from {state.value}")
        return WindowRecord(WindowState.DRAINING, record.request)
    if action == "activate":
        if state == WindowState.ACTIVE:
            return record
        if state != WindowState.DRAINING:
            raise WindowError(f"cannot activate from {state.value}")
        return WindowRecord(WindowState.ACTIVE, record.request)
    if action == "close":
        if state == WindowState.CLOSED:
            return record
        if state == WindowState.CLOSING:
            return record
        if state in {WindowState.REQUESTED, WindowState.DRAINING, WindowState.ACTIVE}:
            return WindowRecord(WindowState.CLOSING, record.request, reason)
        if state == WindowState.RECOVERY:
            return record
        raise WindowError(f"cannot close from {state.value}")
    if action == "closed":
        if state == WindowState.CLOSED:
            return record
        if state not in {WindowState.CLOSING, WindowState.RECOVERY}:
            raise WindowError(f"cannot mark closed from {state.value}")
        return WindowRecord(WindowState.CLOSED, None, reason)
    if action == "recover":
        if state == WindowState.CLOSED:
            return record
        if state == WindowState.RECOVERY:
            return record
        return WindowRecord(WindowState.RECOVERY, record.request, reason or "recovery")
    raise WindowError(f"unknown measurement-window action: {action}")


def deadline_expired(record: WindowRecord, *, now_unix_ns: int) -> bool:
    return bool(
        record.request is not None
        and record.state != WindowState.CLOSED
        and now_unix_ns >= record.request.deadline_unix_ns
    )
