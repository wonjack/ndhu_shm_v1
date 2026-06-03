"""Backfill catalog/PGA fields into already-written v4 CSVs.

The original DH_catlog.xlsx ran out at 2022-11; our pipeline's YEAR_MIN=2023
filter dropped every catalog match, leaving catalog_matched=False and empty
PGA fields. The newer NCREE_datasheel.xlsx covers 2017–2025 with PGA at the
0/20/30/70 m depths used at the NDHU borehole.

This script:
  1. Loads NCREE_datasheel.xlsx (NDHU_DH sheet) and builds a clean catalog.
  2. For each structural_history_log_v4_DXXX.csv, re-matches every event by
     UTC→CWB-local time (+8h), within ±60 s.
  3. Backfills magnitude, depth_km, epicenter, pga_0m, pga_20m, pga_30m,
     pga_70m, sensitivity, catalog_matched. Other columns untouched.
  4. Re-renders plot_summary_scatter so summary_softening_vs_pga.png now
     gets data.

Safe to re-run — idempotent w.r.t. the input CSV.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline_common import BASE_DIR, plot_summary_scatter
from src.station_config import STATIONS


NEW_CATALOG = BASE_DIR / "NCREE_datasheel.xlsx"
TOLERANCE_SEC = 60
LOCAL_OFFSET = timedelta(hours=8)


def load_new_catalog(path: Path = NEW_CATALOG) -> pd.DataFrame:
    """Return a DataFrame sorted by t_local with these columns:
    t_local, magnitude, depth_km, epicenter, pga_0m, pga_20m, pga_30m, pga_70m.

    The new sheet has 16 columns; we read by *position* because the header
    is Chinese mojibake.
    """
    raw = pd.read_excel(path, sheet_name="NDHU_DH", engine="openpyxl")
    # Positional column map (verified by inspection):
    # 0 internal_id, 1 date (YYYYMMDD int), 2 time_cwb,
    # 3 lon, 4 lat, 5 magnitude, 6 depth_km, 7 epicenter,
    # 8 sensitivity_class (A/B/...), 9 supplement,
    # 10 pga_0m, 11 pga_20m, 12 pga_30m, 13 pga_70m
    cols = ["internal_id", "date", "time_cwb",
            "lon", "lat", "magnitude", "depth_km", "epicenter",
            "sensitivity", "supplement",
            "pga_0m", "pga_20m", "pga_30m", "pga_70m"]
    df = raw.iloc[:, :len(cols)].copy()
    df.columns = cols

    def _combine(row):
        d = row["date"]; t = row["time_cwb"]
        if pd.isna(d) or pd.isna(t):
            return None
        try:
            d_str = str(int(d))
            if len(d_str) != 8:
                return None
            t_str = t.strftime("%H:%M:%S.%f") if hasattr(t, "strftime") else str(t)
            return pd.to_datetime(f"{d_str} {t_str}", errors="coerce")
        except Exception:
            return None

    df["t_local"] = df.apply(_combine, axis=1)
    df = df.dropna(subset=["t_local"]).sort_values("t_local").reset_index(drop=True)
    return df


def _lookup(cat: pd.DataFrame, t_local: pd.Timestamp,
            tolerance_sec: int = TOLERANCE_SEC):
    diffs = (cat["t_local"] - t_local).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx].total_seconds() > tolerance_sec:
        return None
    return cat.loc[idx]


def backfill_csv(csv_path: Path, cat: pd.DataFrame) -> dict:
    df = pd.read_csv(csv_path)
    if "pga_70m" not in df.columns:
        # Insert pga_70m right after pga_30m for tidiness; otherwise append.
        if "pga_30m" in df.columns:
            ix = list(df.columns).index("pga_30m") + 1
            df.insert(ix, "pga_70m", "")
        else:
            df["pga_70m"] = ""
    pga_cols = ["pga_0m", "pga_20m", "pga_30m", "pga_70m"]
    text_cols = ["magnitude", "depth_km", "epicenter", "sensitivity"]

    # Build a unique event_id -> UTC datetime map.
    df["_utc"] = pd.to_datetime(df["utc_time"])
    n_hit = 0
    n_total = 0
    pga0_filled = 0
    for ev_id, sub in df.groupby("event_id"):
        utc = sub["_utc"].iloc[0]
        loc = utc + LOCAL_OFFSET
        n_total += 1
        hit = _lookup(cat, loc)
        mask = df["event_id"] == ev_id
        if hit is None:
            df.loc[mask, "catalog_matched"] = "False"
            continue
        n_hit += 1
        df.loc[mask, "catalog_matched"] = "True"
        for c in text_cols + pga_cols:
            val = hit.get(c)
            if pd.isna(val):
                continue
            if c in pga_cols and c == "pga_0m":
                pga0_filled += 1
            df.loc[mask, c] = val
    df = df.drop(columns=["_utc"])
    df.to_csv(csv_path, index=False)
    return {"events": n_total, "hits": n_hit, "pga0_filled": pga0_filled}


def main():
    cat = load_new_catalog()
    print(f"new catalog: {len(cat)} rows, "
          f"{cat['t_local'].min()} .. {cat['t_local'].max()}")
    for sid in ("D001", "D002", "D003", "D005", "D006"):
        cfg = STATIONS[sid]
        csv = BASE_DIR / f"structural_history_log_v4_{sid}.csv"
        out_root = BASE_DIR / f"output_v4_{sid}"
        if not csv.exists():
            print(f"[{sid}] CSV missing, skipping")
            continue
        stats = backfill_csv(csv, cat)
        print(f"[{sid}] events={stats['events']}  "
              f"catalog_hits={stats['hits']}  pga0_filled={stats['pga0_filled']}")
        # Re-render summary plots with backfilled PGA.
        try:
            csv_for_plot = csv
            if sid == "D006":
                # D006 uses legacy v3.1 dual-sensor schema. Project the 4F
                # columns onto the v4 single-station schema in a temp CSV so
                # plot_summary_scatter works.
                csv_for_plot = _adapt_d006_csv(csv)
            plot_summary_scatter(csv_for_plot, out_root, cfg)
            print(f"[{sid}] summary plots refreshed")
        except Exception as exc:
            print(f"[{sid}] plot_summary_scatter FAILED: {exc}")


def _adapt_d006_csv(src: Path) -> Path:
    """Rename D006 dual-sensor columns to the v4 single-station schema,
    using 4F (W10F) as the primary structural floor."""
    df = pd.read_csv(src)
    df = df.rename(columns={
        "f_iapp_4f":       "f_iapp",
        "f_min_4f":        "f_min",
        "f_99app_4f":      "f_99app",
        "tf_peak_freq_4f": "tf_peak_freq",
        "tf_peak_amp_4f":  "tf_peak_amp",
        "tf_f1_4f":        "tf_f1",
        "damping_ratio_4f":"damping_ratio",
        "fft_peak_d006":   "fft_peak_d006",  # already correct
    })
    tmp = src.with_name(src.stem + "_4f_view.csv")
    df.to_csv(tmp, index=False)
    return tmp


if __name__ == "__main__":
    main()
