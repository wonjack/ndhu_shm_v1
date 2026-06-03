"""v5 pipeline entry point for D002 (NDHU 教育學院, 4F RC, sensor at 1F).

Reviewer_v4 fixes (see main_v5_D001.py and src/pipeline_common_v5.py for
details): symmetric-bandpass f_min check, explicit segmentation source,
PGA-matched and deseasonalized baseline comparisons.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline_common_v5 import run_pipeline
from src.station_config import STATIONS


if __name__ == "__main__":
    run_pipeline(STATIONS["D002"])
