#!/usr/bin/env python3
"""Integrated entry point for the ZIP mixed-effects forecasting model."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mixed_effect_zip.fit_zip_mixed_effects import main


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc))
