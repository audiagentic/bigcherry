"""GPU-lease-gated S8 hardware sweep and P2 measurement-file producer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.serving.s8_lab import main

if __name__ == "__main__":
    raise SystemExit(main())
