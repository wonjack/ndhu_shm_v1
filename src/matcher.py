from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, Optional


BASE_DIR = Path(r"D:\PHD\project2026")
CWA_Z_DIR = BASE_DIR / "CWA" / "z"
CWA_E_DIR = BASE_DIR / "CWA" / "e"
CWA_N_DIR = BASE_DIR / "CWA" / "n"
PALERT_1F_DIR = BASE_DIR / "palert" / "D006"
PALERT_4F_DIR = BASE_DIR / "palert" / "W10F"
PALERT_TOLERANCE_SEC = 60


@dataclass(frozen=True)
class Event:
    cwa_z: Path
    cwa_e: Path
    cwa_n: Path
    palert_1f: list[Path]
    palert_4f: list[Path]


def get_cwa_events(cwa_z_dir: Path = CWA_Z_DIR) -> list[Path]:
    if not cwa_z_dir.exists():
        return []
    return sorted(cwa_z_dir.glob("*.mseed"), key=lambda p: p.name)


def find_matching_files(
    cwa_event: Path | str,
    *,
    cwa_e_dir: Path = CWA_E_DIR,
    cwa_n_dir: Path = CWA_N_DIR,
    palert_1f_dir: Path = PALERT_1F_DIR,
    palert_4f_dir: Path = PALERT_4F_DIR,
    tolerance_sec: int = PALERT_TOLERANCE_SEC,
) -> Optional[Event]:
    cwa_event_path = Path(cwa_event)
    if not cwa_event_path.exists():
        return None

    cwa_e = cwa_e_dir / cwa_event_path.name
    cwa_n = cwa_n_dir / cwa_event_path.name
    if not cwa_e.exists() or not cwa_n.exists():
        return None

    cwa_dt = _parse_cwa_time(cwa_event_path.name)
    if cwa_dt is None:
        return None

    palert_1f = _find_palert_group(palert_1f_dir, cwa_dt, tolerance_sec)
    if palert_1f is None:
        return None

    palert_4f = _find_palert_group(palert_4f_dir, cwa_dt, tolerance_sec)
    if palert_4f is None:
        return None

    return Event(
        cwa_z=cwa_event_path,
        cwa_e=cwa_e,
        cwa_n=cwa_n,
        palert_1f=palert_1f,
        palert_4f=palert_4f,
    )


def _parse_cwa_time(filename: str) -> Optional[datetime]:
    stem = Path(filename).stem
    try:
        return datetime.strptime(stem, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _parse_palert_time(filename: str) -> Optional[datetime]:
    name = Path(filename).name
    match = re.search(r"(\d{14})", name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _find_palert_group(
    palert_dir: Path, cwa_dt: datetime, tolerance_sec: int
) -> Optional[list[Path]]:
    if not palert_dir.exists():
        return None

    candidates: dict[str, dict[str, object]] = {}
    for path in palert_dir.glob("*.SAC"):
        palert_dt = _parse_palert_time(path.name)
        if palert_dt is None:
            continue
        diff = abs((palert_dt - cwa_dt).total_seconds())
        if diff > tolerance_sec:
            continue
        key = palert_dt.strftime("%Y%m%d%H%M%S")
        entry = candidates.setdefault(key, {"dt": palert_dt, "paths": []})
        entry["paths"].append(path)

    best_paths = _select_best_group(candidates.values(), cwa_dt)
    if best_paths is None:
        return None
    return best_paths


def _select_best_group(
    groups: Iterable[dict[str, object]], cwa_dt: datetime
) -> Optional[list[Path]]:
    best_diff = None
    best_paths: Optional[list[Path]] = None
    for group in groups:
        paths = list(group["paths"])
        if len(paths) < 3:
            continue
        group_dt = group["dt"]
        diff = abs((group_dt - cwa_dt).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_paths = sorted(paths, key=lambda p: p.name)[:3]
    return best_paths
