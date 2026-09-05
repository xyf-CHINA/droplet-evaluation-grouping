"""Public test configuration."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
