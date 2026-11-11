"""Audited S8 build-scope rebind producer (no GPU required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.serving.calibration_rebind import main

if __name__ == "__main__":
    raise SystemExit(main())
