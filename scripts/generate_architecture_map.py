#!/usr/bin/env python3
"""Run the canonical architecture-map generator shipped with the econhdfe skill."""
from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "econhdfe" / "scripts" / "architecture_map.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
