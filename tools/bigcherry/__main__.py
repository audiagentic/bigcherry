"""The ``bigcherry`` command line.

One command per stage of taking a new llama.cpp release into production:

    pull -> audit -> apply -> generate -> build

The durable job-control plane is available as ``bigcherry jobs ...`` and
accepted/discovered hardware operations as ``bigcherry hardware ...``.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, cast


def main(argv: list[str] | None = None) -> int:
    """Delegate package entrypoint; route durable control-plane domains."""
    effective = list(sys.argv[1:] if argv is None else argv)
    if effective[:1] == ["jobs"]:
        jobs_main = cast(Any, importlib.import_module("bigcherry.cli.jobs").main)
        return int(jobs_main(effective[1:]))
    if effective[:1] == ["hardware"]:
        hardware_main = cast(Any, importlib.import_module("bigcherry.cli.hardware").main)
        return int(hardware_main(effective[1:]))
    cli_main = cast(Any, importlib.import_module("bigcherry.cli.main").main)
    return int(cli_main(effective))


if __name__ == "__main__":
    raise SystemExit(main())
