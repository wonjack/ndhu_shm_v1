"""v5 pipeline entry point for D003 (NDHU 圖書館, 6F SRC, sensor at 1F).

Reviewer_v4 fixes (see src/pipeline_common_v5.py for details):
symmetric-bandpass f_min check, explicit segmentation source, PGA-matched
and deseasonalized baseline comparisons.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline_common_v5 import run_pipeline
from src.station_config import STATIONS


if __name__ == "__main__":
    run_pipeline(STATIONS["D003"])
