"""v5 pipeline entry point for D001 (NDHU 人社院, 4F RC, sensor at 2F).

v5 wraps src.pipeline_common_v5.run_pipeline, which extends v4 with the
four reviewer_v4 code-level fixes:
  (3) symmetric-bandpass f_min sensitivity check (per-event CSV column
      `f_min_sym` + summary_fmin_symmetric_vs_asymmetric.{png,json}),
  (5) explicit `segment_source` column,
  (6) baseline_comparison_pga_matched.{png,json},
  (10) baseline_comparison_deseasonalized.{png,json}.
Outputs go to D:/PHD/project2026/output_v5_D001/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline_common_v5 import run_pipeline
from src.station_config import STATIONS


if __name__ == "__main__":
    run_pipeline(STATIONS["D001"])
