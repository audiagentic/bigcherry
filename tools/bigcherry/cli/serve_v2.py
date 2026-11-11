"""Inactive S13 CLI adapter bindings for v2 start and cross-process stop.

S13 switches the command router atomically after full contract acceptance.
Do not register these on the shipped v1 command path beforehand.
"""
from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

from ..hardware import gfxoff, lease
from ..serving.runtime_adapters import (
    LifecycleLimits, request_stop, start_with_adapters,
)
from . import layout_cmds, serve


def start(args: Namespace, *, source, writer, validator, limits: LifecycleLimits,
          lease_ok: Callable = lease.held_by_this_process,
          **kwargs) -> dict:
    """Bind the existing CLI inventory to one S11/S12 v2 process owner."""
    return start_with_adapters(
        name=args.name, run_id=args.run_id,
        resolve=lambda name: layout_cmds._resolve(args),
        source=source, writer=writer, validator=validator,
        record_dir=serve._record_dir(),
        journal_dir=serve._journal_dir(),
        lease_ok=lease_ok, limits=limits,
        gfxoff_capture=kwargs.pop("gfxoff_capture", gfxoff.sample_cards),
        **kwargs,
    )


def stop(args: Namespace, *, limits: LifecycleLimits, **kwargs) -> dict:
    """Request stop from the still-running start owner; never kill from a PID."""
    return request_stop(args.run_id, serve._record_dir(), limits=limits, **kwargs)
