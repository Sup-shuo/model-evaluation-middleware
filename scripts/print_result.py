#!/usr/bin/env python3
"""Render a saved run from either a source checkout or an installed package."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model_evaluation.result_report import main


if __name__ == "__main__":
    raise SystemExit(main())
