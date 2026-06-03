"""NCREE-driven event matcher for D003 (NDHU Library building).

Sensor at 1F of a 6-storey SRC (steel-reinforced concrete) building.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.matcher_ncree import (
    scan_ncree_events,
    index_palert_dir,
    find_palert_group,
    load_catalog,
    lookup_catalog,
    SkipReport,
)


NCREE_DIR = Path(r"D:\PHD\Poster2026\data\NCREE_Freefield")
PALERT_D003_DIR = Path(r"D:\PHD\Poster2026\data\D003")
CATALOG_PATH = Path(r"D:\PHD\DH_catlog.xlsx")


@dataclass(frozen=True)
class NcreeD003Event:
    event_id: str
    utc_time: datetime
    local_time: datetime
    ncree_paths: dict
    palert_d003: dict        # D003 sensor at 1F
    catalog_meta: Optional[dict] = field(default=None)


def build_event_list(
    *,
    ncree_dir: Path = NCREE_DIR,
    palert_d003_dir: Path = PALERT_D003_DIR,
    catalog_path: Path = CATALOG_PATH,
) -> tuple[list[NcreeD003Event], list[SkipReport]]:
    catalog = load_catalog(catalog_path)
    pal_idx = index_palert_dir(palert_d003_dir)
    ncree = scan_ncree_events(ncree_dir)

    events: list[NcreeD003Event] = []
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

        g = find_palert_group(pal_idx, t_utc)
        if g is None:
            skipped.append(SkipReport(ts, "palert_d003_missing"))
            continue

        events.append(
            NcreeD003Event(
                event_id=ts,
                utc_time=t_utc,
                local_time=t_local,
                ncree_paths={k: comps[k] for k in ("E", "N", "Z")},
                palert_d003=g,
                catalog_meta=meta,
            )
        )
    return events, skipped


if __name__ == "__main__":
    events, skipped = build_event_list()
    print(f"Matched events: {len(events)}")
    print(f"Skipped:        {len(skipped)}")
