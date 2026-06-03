"""NCREE-driven event matcher.

Builds the canonical event list from NCREE Freefield SAC files, joins against
the DH catalog (CWB local time, +8h from NCREE UTC), and locates matching
P-alert D006 (1F) / W10F (4F) SAC files within a UTC tolerance window.

Filename conventions
--------------------
- NCREE:   YYYYMMDDHHMMSSG.00.{E,N,Z}.SAC          (UTC)
- P-alert: {STA}.HL{E,N,Z}.TW.YYYYMMDDHHMMSS.SAC   (UTC, = trigger time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Optional

import pandas as pd


NCREE_DIR = Path(r"D:\PHD\Poster2026\data\NCREE_Freefield")
PALERT_1F_DIR = Path(r"D:\PHD\Poster2026\data\D006")
PALERT_4F_DIR = Path(r"D:\PHD\Poster2026\data\W10F")
CATALOG_PATH = Path(r"D:\PHD\DH_catlog.xlsx")

PALERT_TOLERANCE_SEC = 60
CATALOG_TOLERANCE_SEC = 60

NCREE_NAME_RE = re.compile(r"^(\d{14})G?\d*\.00\.([ENZ])\.SAC$", re.IGNORECASE)
PALERT_NAME_RE = re.compile(
    r"^(?P<sta>[A-Z0-9]+)\.HL(?P<comp>[ENZ])\.TW\.(?P<ts>\d{14})\.SAC$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NcreeEvent:
    event_id: str            # YYYYMMDDHHMMSS (UTC = NCREE filename)
    utc_time: datetime       # NCREE/P-alert reference (UTC)
    local_time: datetime     # CWB local = utc_time + 8h
    ncree_paths: dict        # {'E','N','Z': Path}
    palert_1f: dict          # {'E','N','Z': Path}
    palert_4f: dict          # {'E','N','Z': Path}
    catalog_meta: Optional[dict] = field(default=None)


@dataclass
class SkipReport:
    event_id: str
    reason: str


# --------------------------------------------------------------------------- #
# NCREE scanning
# --------------------------------------------------------------------------- #
def scan_ncree_events(ncree_dir: Path = NCREE_DIR) -> dict[str, dict[str, Path]]:
    """Group NCREE SAC files by 14-digit UTC timestamp -> {component: path}."""
    if not ncree_dir.exists():
        return {}
    events: dict[str, dict[str, Path]] = {}
    for path in ncree_dir.glob("*.SAC"):
        m = NCREE_NAME_RE.match(path.name)
        if not m:
            continue
        ts, comp = m.group(1), m.group(2).upper()
        events.setdefault(ts, {})[comp] = path
    return events


def _parse_palert_name(name: str) -> Optional[tuple[str, str, datetime]]:
    m = PALERT_NAME_RE.match(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group("ts"), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return m.group("sta").upper(), m.group("comp").upper(), dt


def index_palert_dir(palert_dir: Path) -> list[tuple[datetime, str, Path]]:
    """Return [(trigger_dt_utc, component, path), ...]."""
    if not palert_dir.exists():
        return []
    out = []
    for path in palert_dir.glob("*.SAC"):
        parsed = _parse_palert_name(path.name)
        if parsed is None:
            continue
        _, comp, dt = parsed
        out.append((dt, comp, path))
    return out


def find_palert_group(
    index: list[tuple[datetime, str, Path]],
    target_utc: datetime,
    tolerance_sec: int = PALERT_TOLERANCE_SEC,
) -> Optional[dict[str, Path]]:
    """Return {'E','N','Z': Path} if all three components found within tolerance.

    Strategy: cluster candidates by their exact filename timestamp; pick the
    cluster whose timestamp is closest to target_utc and contains all 3 comps.
    """
    clusters: dict[datetime, dict[str, Path]] = {}
    for dt, comp, path in index:
        if abs((dt - target_utc).total_seconds()) > tolerance_sec:
            continue
        clusters.setdefault(dt, {})[comp] = path
    best: Optional[dict[str, Path]] = None
    best_diff = None
    for dt, comps in clusters.items():
        if not {"E", "N", "Z"}.issubset(comps):
            continue
        diff = abs((dt - target_utc).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = {k: comps[k] for k in ("E", "N", "Z")}
    return best


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    # Normalize column names by position (sheet has mojibake headers).
    expected_cols = [
        "internal_id", "date", "time_cwb", "lon", "lat", "magnitude",
        "depth_km", "epicenter", "sensitivity", "supplement",
        "pga_0m", "pga_20m", "pga_30m", "pga_58m", "pga_78m",
        "_unused1", "_unused2",
    ]
    if df.shape[1] >= len(expected_cols):
        df = df.iloc[:, : len(expected_cols)].copy()
        df.columns = expected_cols
    else:
        raise ValueError(f"Catalog shape unexpected: {df.shape}")

    # Build local datetime: date is int-like YYYYMMDD; time_cwb may be "HH:MM:SS.s" or datetime.time
    def _combine(row) -> Optional[pd.Timestamp]:
        d = row["date"]
        t = row["time_cwb"]
        if pd.isna(d) or pd.isna(t):
            return None
        try:
            d_str = str(int(d))
            if len(d_str) != 8:
                return None
            if hasattr(t, "strftime"):
                t_str = t.strftime("%H:%M:%S.%f")
            else:
                t_str = str(t)
            return pd.to_datetime(f"{d_str} {t_str}", errors="coerce")
        except Exception:
            return None

    df["t_local"] = df.apply(_combine, axis=1)
    df = df.dropna(subset=["t_local"]).sort_values("t_local").reset_index(drop=True)
    return df


def lookup_catalog(
    df: pd.DataFrame, t_local: datetime, tolerance_sec: int = CATALOG_TOLERANCE_SEC
) -> Optional[dict]:
    if df.empty:
        return None
    # Use absolute time difference (vectorized)
    diffs = (df["t_local"] - pd.Timestamp(t_local)).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx].total_seconds() > tolerance_sec:
        return None
    row = df.loc[idx]
    return {
        "magnitude": row["magnitude"],
        "depth_km": row["depth_km"],
        "epicenter": row["epicenter"],
        "lon": row["lon"],
        "lat": row["lat"],
        "sensitivity": row["sensitivity"],
        "pga_0m": row["pga_0m"],
        "pga_20m": row["pga_20m"],
        "pga_30m": row["pga_30m"],
        "pga_58m": row["pga_58m"],
        "pga_78m": row["pga_78m"],
        "t_local": row["t_local"].to_pydatetime(),
    }


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def build_event_list(
    *,
    ncree_dir: Path = NCREE_DIR,
    palert_1f_dir: Path = PALERT_1F_DIR,
    palert_4f_dir: Path = PALERT_4F_DIR,
    catalog_path: Path = CATALOG_PATH,
) -> tuple[list[NcreeEvent], list[SkipReport]]:
    catalog = load_catalog(catalog_path)
    pal_1f_idx = index_palert_dir(palert_1f_dir)
    pal_4f_idx = index_palert_dir(palert_4f_dir)
    ncree = scan_ncree_events(ncree_dir)

    events: list[NcreeEvent] = []
    skipped: list[SkipReport] = []
    for ts, comps in sorted(ncree.items()):
        if not {"E", "N", "Z"}.issubset(comps):
            skipped.append(SkipReport(ts, "ncree_missing_component"))
            continue
        try:
            # NCREE filename is UTC (verified empirically against catalog & P-alert)
            t_utc = datetime.strptime(ts, "%Y%m%d%H%M%S")
        except ValueError:
            skipped.append(SkipReport(ts, "ncree_bad_timestamp"))
            continue
        t_local = t_utc + timedelta(hours=8)  # CWB local for catalog lookup

        meta = lookup_catalog(catalog, t_local)
        # catalog miss is not fatal, just annotated below.

        g_1f = find_palert_group(pal_1f_idx, t_utc)
        if g_1f is None:
            skipped.append(SkipReport(ts, "palert_d006_missing"))
            continue
        g_4f = find_palert_group(pal_4f_idx, t_utc)
        if g_4f is None:
            skipped.append(SkipReport(ts, "palert_w10f_missing"))
            continue

        events.append(
            NcreeEvent(
                event_id=ts,
                utc_time=t_utc,
                local_time=t_local,
                ncree_paths={k: comps[k] for k in ("E", "N", "Z")},
                palert_1f=g_1f,
                palert_4f=g_4f,
                catalog_meta=meta,
            )
        )
    return events, skipped


if __name__ == "__main__":
    events, skipped = build_event_list()
    print(f"Matched events: {len(events)}")
    print(f"Skipped:        {len(skipped)}")
    from collections import Counter
    print("Skip reasons:", Counter(s.reason for s in skipped))
    print()
    print("Sample matched events:")
    for ev in events[:5]:
        meta = ev.catalog_meta
        meta_str = (
            f"M{meta['magnitude']} {meta['epicenter']}"
            if meta else "<catalog_missing>"
        )
        print(f"  {ev.event_id}  UTC={ev.utc_time}  {meta_str}")
    cat_hit = sum(1 for e in events if e.catalog_meta is not None)
    print(f"\nCatalog hit rate: {cat_hit}/{len(events)}")
