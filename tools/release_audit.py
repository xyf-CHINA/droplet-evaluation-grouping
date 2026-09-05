#!/usr/bin/env python3
"""Compatibility entry point for the canonical public-release audit."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.public_release_audit import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
