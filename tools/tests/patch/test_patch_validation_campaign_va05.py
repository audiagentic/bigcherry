"""VA05 hardware-free tests: run_rd58_state_restore_evidence() -- an
RD58-scoped validation-domain state-restore correctness/activation/
controls producer, analogous to RD04/RD08's producers but for a
correctness-only contract with no performance claim. Hardware-free via
an injected fake subprocess.run().
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class _Result:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


