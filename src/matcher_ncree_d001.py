"""NCREE-driven event matcher for D001 (NDHU Humanities & Social Sciences building).

Builds the canonical event list from NCREE Freefield SAC files, joins against
the DH catalog (CWB local time, +8h from NCREE UTC), and locates matching
P-alert D001 (2F) SAC files within a UTC tolerance window.

Filename conventions
--------------------
- NCREE:   YYYYMMDDHHMMSSG.00.{E,N,Z}.SAC          (UTC)
- P-alert: {STA}.HL{E,N,Z}.TW.YYYYMMDDHHMMSS.SAC   (UTC, = trigger time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.matcher_ncree import (
    NCREE_NAME_RE,
    PALERT_NAME_RE,
    PALERT_TOLERANCE_SEC,
    CATALOG_TOLERANCE_SEC,
    scan_ncree_events,
    index_palert_dir,
    find_palert_group,
    load_catalog,
    lookup_catalog,
    SkipReport,
)


NCREE_DIR = Path(r"D:\PHD\Poster2026\data\NCREE_Freefield")
PALERT_D001_DIR = Path(r"D:\PHD\Poster2026\data\D001")
CATALOG_PATH = Path(r"D:\PHD\DH_catlog.xlsx")


@dataclass(frozen=True)
class NcreeD001Event:
    event_id: str            # YYYYMMDDHHMMSS (UTC = NCREE filename)
    utc_time: datetime       # NCREE/P-alert reference (UTC)
    local_time: datetime     # CWB local = utc_time + 8h
    ncree_paths: dict        # {'E','N','Z': Path}
    palert_d001: dict        # {'E','N','Z': Path}  (D001 sensor at 2F)
    catalog_meta: Optional[dict] = field(default=None)


def build_event_list(
    *,
    ncree_dir: Path = NCREE_DIR,
    palert_d001_dir: Path = PALERT_D001_DIR,
    catalog_path: Path = CATALOG_PATH,
) -> tuple[list[NcreeD001Event], list[SkipReport]]:
    catalog = load_catalog(catalog_path)
    pal_d001_idx = index_palert_dir(palert_d001_dir)
    ncree = scan_ncree_events(ncree_dir)

    events: list[NcreeD001Event] = []
    skipped: list[SkipReport] = []
    for ts, comps in sorted(ncree.items()):
        if not {"E", "N", "Z"}.issubset(comps):
            skipped.append(SkipReport(ts, "ncree_missing_component"))
            continue
        try:
            t_utc = datetime.strptime(ts, "%Y%m%d%H%M%S")
        except ValueError:
            skipped.append(SkipReport(ts, "ncree_bad_timestamp"))
            continue
        t_local = t_utc + timedelta(hours=8)

        meta = lookup_catalog(catalog, t_local)

        g_d001 = find_palert_group(pal_d001_idx, t_utc)
        if g_d001 is None:
            skipped.append(SkipReport(ts, "palert_d001_missing"))
            continue

        events.append(
            NcreeD001Event(
                event_id=ts,
                utc_time=t_utc,
                local_time=t_local,
                ncree_paths={k: comps[k] for k in ("E", "N", "Z")},
                palert_d001=g_d001,
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
    cat_hit = sum(1 for e in events if e.catalog_meta is not None)
    print(f"\nCatalog hit rate: {cat_hit}/{len(events)}")
