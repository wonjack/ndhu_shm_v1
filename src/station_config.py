"""Per-station deployment metadata for the v4 pipeline.

Centralised here so main_v3_DXXX.py can be a thin wrapper around
src.pipeline_common.run_pipeline(STATIONS["DXXX"]).

Reviewer items addressed (command_v1.md):
- §2.4 NCREE input physics: ``ncree_distance_m`` records straight-line
  distance from each structural station to the NCREE free-field reference.
  NCREE itself is the *surface* companion of a borehole instrument, so the
  ground-floor-equivalent free-field assumption holds, but the path
  difference (230–1077 m) is non-trivial and must be reported.
- §4.1 SSI: ``footprint_m2`` and ``vs30_ms`` feed the simplified
  Veletsos-Meek / NIST GCR 12-917-21 §2 period-lengthening estimate
  (see src.ssi_bounds, populated in Phase 2).

Vs30 source: HWA018 志學國小 borehole (2008-09-24, 31.5 m profile),
NCREE 強震測站場址工程地質資料庫. HWA018 sits in 志學村, immediately
adjacent to the NDHU campus; Vs30 = 462.57 m/s corresponds to NEHRP
Site Class C.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Single Vs30 value covers the whole campus — HWA018 is the closest
# instrumented borehole and all five D-stations are within ~2 km on the
# same alluvial-fan terrace.
CAMPUS_VS30_MS: float = 462.57         # HWA018, NCREE EGDT 2008
CAMPUS_SITE_CLASS: str = "C"           # NEHRP (360 < Vs30 ≤ 760 m/s)
CAMPUS_SOIL_DENSITY_KGM3: float = 1900.0
CAMPUS_SOIL_POISSON: float = 0.35


@dataclass(frozen=True)
class StationConfig:
    """Static deployment parameters for one structural station."""

    # Identity
    station_id: str                    # "D001" .. "D006"
    locname: str                       # 中文建物名稱
    sensor_floor_label: str            # "1F" | "2F" | ...

    # Structural / code-empirical parameters
    height_m: float
    structure: str                     # "RC" | "SRC" | "STEEL" | "OTHER"

    # Site / geometry (per station_site.md)
    lat: float
    lon: float
    elev_m: float
    footprint_m2: float
    ncree_distance_m: float            # straight-line to NCREE free-field

    # Pipeline plumbing — names of the matcher module/attributes
    matcher_module: str                # "src.matcher_ncree_d001"
    palert_attr: str                   # "palert_d001" — attribute on Event

    # Optional second sensor (D006 only)
    has_upper_sensor: bool = False
    upper_sensor_label: Optional[str] = None    # e.g. "4F"
    upper_palert_attr: Optional[str] = None     # e.g. "palert_4f"


# Taiwan seismic design code empirical period: T = k · h^(3/4).
_STRUCTURE_K = {"STEEL": 0.085, "RC": 0.070, "SRC": 0.070, "OTHER": 0.050}


def empirical_period(cfg: StationConfig) -> float:
    return _STRUCTURE_K[cfg.structure] * (cfg.height_m ** 0.75)


def empirical_frequency(cfg: StationConfig) -> float:
    return 1.0 / empirical_period(cfg)


def code_band(cfg: StationConfig) -> tuple[float, float, float]:
    """Return (f_emp, f_emp_min, f_emp_max) for the ±40% code band."""
    f_emp = empirical_frequency(cfg)
    return f_emp, f_emp / 1.4, f_emp * 1.4


STATIONS: dict[str, StationConfig] = {
    "D001": StationConfig(
        station_id="D001",
        locname="東華人社院",
        sensor_floor_label="2F",
        height_m=12.0,
        structure="RC",
        lat=23.894972, lon=121.542278, elev_m=41.0,
        footprint_m2=10470.0,
        ncree_distance_m=868.0,
        matcher_module="src.matcher_ncree_d001",
        palert_attr="palert_d001",
    ),
    "D002": StationConfig(
        station_id="D002",
        locname="東華教育學院",
        sensor_floor_label="1F",
        height_m=12.0,
        structure="RC",
        lat=23.896278, lon=121.539861, elev_m=40.0,
        footprint_m2=8442.0,
        ncree_distance_m=1077.0,
        matcher_module="src.matcher_ncree_d002",
        palert_attr="palert_d002",
    ),
    "D003": StationConfig(
        station_id="D003",
        locname="東華圖書館",
        sensor_floor_label="1F",
        height_m=18.0,
        structure="SRC",
        lat=23.896389, lon=121.542306, elev_m=46.0,
        footprint_m2=5918.0,
        ncree_distance_m=832.0,
        matcher_module="src.matcher_ncree_d003",
        palert_attr="palert_d003",
    ),
    "D005": StationConfig(
        station_id="D005",
        locname="東華學生活動中心",
        sensor_floor_label="1F",
        height_m=6.0,
        structure="RC",
        lat=23.899222, lon=121.541222, elev_m=45.0,
        footprint_m2=4152.0,
        ncree_distance_m=989.0,
        matcher_module="src.matcher_ncree_d005",
        palert_attr="palert_d005",
    ),
    "D006": StationConfig(
        station_id="D006",
        locname="東華環境院",
        sensor_floor_label="1F",
        height_m=12.0,
        structure="RC",
        lat=23.895944, lon=121.548355, elev_m=58.0,
        footprint_m2=3608.0,
        ncree_distance_m=230.0,
        matcher_module="src.matcher_ncree",          # dual-station matcher
        palert_attr="palert_1f",
        has_upper_sensor=True,
        upper_sensor_label="4F",
        upper_palert_attr="palert_4f",
    ),
}
