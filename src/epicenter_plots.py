"""Two summary maps describing the event catalogue used in v4 analysis.

1. epicenter_map.png — azimuthal-equidistant scatter of all matched events
   within 200 km of NDHU, plus the M7.2 mainshock as a labelled star and
   M ≥ 5 events highlighted.

2. back_azimuth_rose.png — polar histogram of the back-azimuth (NDHU →
   epicenter) showing angular coverage of the dataset.

Reads from NCREE_datasheel.xlsx + any one of the structural_history_log_v4
CSVs (to take the event_id list actually used by the pipeline). Output goes
to ``output_v4_summary/`` so the same maps cover all 5 stations.

Run standalone:
    python -m src.epicenter_plots
"""
from __future__ import annotations

import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backfill_pga import load_new_catalog
from src.pipeline_common import BASE_DIR, MAINSHOCK_EVENT_ID
from src.station_config import STATIONS


# Campus reference: take the centroid of the five D-stations.
CAMPUS_LAT = float(np.mean([cfg.lat for cfg in STATIONS.values()]))
CAMPUS_LON = float(np.mean([cfg.lon for cfg in STATIONS.values()]))

RADIUS_KM = 200.0
LOCAL_OFFSET = timedelta(hours=8)
MATCH_TOL_SEC = 60


# --------------------------------------------------------------------------- #
# Geodesy
# --------------------------------------------------------------------------- #
def _haversine_azeqd(lon: np.ndarray, lat: np.ndarray,
                     lon0: float, lat0: float):
    """Azimuthal equidistant projection (m).

    Returns (x, y) where +x = East, +y = North, distance from (lon0, lat0)
    preserved. Pure numpy, good to <1% for distances within ~1000 km.
    """
    R = 6371008.8                                       # mean Earth radius (m)
    phi0 = math.radians(lat0); lam0 = math.radians(lon0)
    phi  = np.radians(lat);    lam  = np.radians(lon)
    dlam = lam - lam0
    cos_c = (np.sin(phi0) * np.sin(phi)
             + np.cos(phi0) * np.cos(phi) * np.cos(dlam))
    cos_c = np.clip(cos_c, -1.0, 1.0)
    c = np.arccos(cos_c)
    # k = c / sin(c), with limit 1 as c→0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(c > 1e-12, c / np.sin(c), 1.0)
    x = R * k * np.cos(phi) * np.sin(dlam)
    y = R * k * (np.cos(phi0) * np.sin(phi)
                 - np.sin(phi0) * np.cos(phi) * np.cos(dlam))
    return x, y, R * c                                  # x_m, y_m, distance_m


def _back_azimuth(lon: np.ndarray, lat: np.ndarray,
                  lon0: float, lat0: float) -> np.ndarray:
    """Forward azimuth NDHU → epicenter, in degrees clockwise from North."""
    phi0 = math.radians(lat0); lam0 = math.radians(lon0)
    phi  = np.radians(lat);    lam  = np.radians(lon)
    dlam = lam - lam0
    y_az = np.sin(dlam) * np.cos(phi)
    x_az = (np.cos(phi0) * np.sin(phi)
            - np.sin(phi0) * np.cos(phi) * np.cos(dlam))
    az = (np.degrees(np.arctan2(y_az, x_az)) + 360.0) % 360.0
    return az


# --------------------------------------------------------------------------- #
# Data loader
# --------------------------------------------------------------------------- #
def load_events_used() -> pd.DataFrame:
    """Cross-join: events the pipeline actually processed × catalog metadata.

    Returns DataFrame with columns event_id, utc_time, lon, lat, magnitude,
    depth_km, distance_km, back_azimuth_deg.
    """
    # Use the D006 csv as the canonical event list (it had the same matcher
    # universe as D001-D005). We only need event_id and utc_time.
    src_csv = BASE_DIR / "structural_history_log_v4_D006.csv"
    df = pd.read_csv(src_csv, usecols=["event_id", "utc_time", "comp"],
                     dtype={"event_id": str})
    df = df[df["comp"] == "E"].drop(columns=["comp"]).reset_index(drop=True)
    df["t_utc"] = pd.to_datetime(df["utc_time"])
    df["t_local"] = df["t_utc"] + LOCAL_OFFSET

    cat = load_new_catalog()
    # Time-window join. Catalog is sorted; use searchsorted for speed.
    cat_t = cat["t_local"].values.astype("datetime64[ns]")

    out = {"lon": [], "lat": [], "magnitude": [], "depth_km": []}
    for t in df["t_local"]:
        t64 = np.datetime64(t)
        diffs = np.abs(cat_t - t64)
        idx = int(diffs.argmin())
        if diffs[idx].astype("timedelta64[s]").astype(int) > MATCH_TOL_SEC:
            out["lon"].append(np.nan); out["lat"].append(np.nan)
            out["magnitude"].append(np.nan); out["depth_km"].append(np.nan)
            continue
        row = cat.iloc[idx]
        out["lon"].append(row["lon"])
        out["lat"].append(row["lat"])
        out["magnitude"].append(row["magnitude"])
        out["depth_km"].append(row["depth_km"])
    for k, v in out.items():
        df[k] = v
    df = df.dropna(subset=["lon", "lat", "magnitude"]).reset_index(drop=True)

    # Geodesy
    x_m, y_m, d_m = _haversine_azeqd(df["lon"].values, df["lat"].values,
                                     CAMPUS_LON, CAMPUS_LAT)
    df["x_km"] = x_m / 1000.0
    df["y_km"] = y_m / 1000.0
    df["distance_km"] = d_m / 1000.0
    df["back_azimuth_deg"] = _back_azimuth(df["lon"].values, df["lat"].values,
                                           CAMPUS_LON, CAMPUS_LAT)
    # Apply the 200-km cut the user asked for.
    df = df[df["distance_km"] <= RADIUS_KM].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Plot 1: epicenter map with coastline (cartopy AzimuthalEquidistant)
# --------------------------------------------------------------------------- #
def plot_epicenter_map(df: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.AzimuthalEquidistant(central_longitude=CAMPUS_LON,
                                     central_latitude=CAMPUS_LAT)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection=proj)

    # Bound the map to ~210 km in projection metres
    half = RADIUS_KM * 1.18 * 1000.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)

    # Land + coastline + admin borders from Natural Earth (10 m = highest res)
    ax.add_feature(cfeature.LAND.with_scale("10m"),
                   facecolor="#f5f0e6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                   facecolor="#e7f0f6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                   edgecolor="#555555", linewidth=0.7, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   edgecolor="#888888", linewidth=0.5, linestyle=":", zorder=2)

    # lat/lon graticule
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.4,
                      color="#aaaaaa", alpha=0.6, linestyle="--", zorder=1)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "#444"}
    gl.ylabel_style = {"size": 8, "color": "#444"}

    # 50/100/150/200 km distance rings — circles in projection space
    for r in (50, 100, 150, 200):
        circle = mpatches.Circle((0, 0), r * 1000.0,
                                 transform=proj,
                                 facecolor="none",
                                 edgecolor="#7a7a7a",
                                 linewidth=0.8,
                                 linestyle="--" if r < 200 else "-",
                                 alpha=0.7, zorder=3)
        ax.add_patch(circle)
        ax.text(0, r * 1000.0, f" {r} km",
                color="#555555", fontsize=7, va="bottom", ha="left",
                transform=proj, zorder=4)
    # Cardinal direction labels
    for lbl, dx, dy in [("N", 0, 215), ("E", 215, 0),
                        ("S", 0, -215), ("W", -215, 0)]:
        ax.text(dx * 1000.0, dy * 1000.0, lbl, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#333333",
                transform=proj, zorder=4)

    # Background events: M < 5
    small = df[df["magnitude"] < 5.0]
    ax.scatter(small["lon"], small["lat"], transform=data_crs,
               s=(small["magnitude"].clip(2.0, 7.0) ** 2) * 1.5,
               c="#707070", alpha=0.40, edgecolors="none", zorder=5,
               label=f"M < 5  (n={len(small)})")

    # M ≥ 5 events (excluding mainshock so the legend doesn't double-count)
    big = df[(df["magnitude"] >= 5.0) & (df["event_id"] != MAINSHOCK_EVENT_ID)]
    ax.scatter(big["lon"], big["lat"], transform=data_crs,
               s=(big["magnitude"].clip(5.0, 7.5) ** 2) * 5,
               marker="^", c="tab:orange", alpha=0.88,
               edgecolors="black", linewidths=0.6, zorder=6,
               label=f"M ≥ 5  (n={len(big)})")

    # M7.2 mainshock
    main = df[df["event_id"] == MAINSHOCK_EVENT_ID]
    if not main.empty:
        ax.scatter(main["lon"], main["lat"], transform=data_crs,
                   s=420, marker="*", c="red",
                   edgecolors="black", linewidths=1.4, zorder=10,
                   label="M7.2 mainshock (2024-04-03)")
        for _, r in main.iterrows():
            ax.annotate(f"M{r['magnitude']:.1f}  {r['event_id']}",
                        xy=(r["lon"], r["lat"]),
                        xycoords=data_crs._as_mpl_transform(ax),
                        xytext=(20, 18), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="red",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="red", alpha=0.92),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.1),
                        zorder=11)

    # NDHU campus
    ax.scatter(CAMPUS_LON, CAMPUS_LAT, transform=data_crs,
               s=230, marker="*", c="tab:blue",
               edgecolors="black", linewidths=1.2, zorder=10,
               label="NDHU campus (5-station centroid)")

    ax.set_title(
        f"Epicenter distribution — {len(df)} matched events within "
        f"{RADIUS_KM:.0f} km of NDHU\n"
        f"Catalogue: NCREE_datasheel.xlsx  2017–2025  "
        f"(coastline: Natural Earth 10 m)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Plot 2: back-azimuth rose
# --------------------------------------------------------------------------- #
def plot_back_azimuth_rose(df: pd.DataFrame, out_path: Path, *,
                           n_bins: int = 36):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)                          # clockwise

    edges = np.linspace(0, 360, n_bins + 1)
    width = np.deg2rad(360.0 / n_bins)
    centres = np.deg2rad((edges[:-1] + edges[1:]) / 2)

    # Stack: small events as outer band, M ≥ 5 as inner band so both visible.
    az = df["back_azimuth_deg"].values
    az_big = df.loc[df["magnitude"] >= 5.0, "back_azimuth_deg"].values

    counts_all, _ = np.histogram(az,     bins=edges)
    counts_big, _ = np.histogram(az_big, bins=edges)
    counts_small  = counts_all - counts_big

    # Stacked: M ≥ 5 at the base (orange), M < 5 stacked on top (grey).
    # Both visible; total height = counts_all.
    ax.bar(centres, counts_big, width=width, bottom=0.0,
           color="tab:orange", alpha=0.95, edgecolor="black", linewidth=0.4,
           label=f"M ≥ 5  (n={len(az_big)})", zorder=4)
    ax.bar(centres, counts_small, width=width, bottom=counts_big,
           color="#9a9a9a", alpha=0.75, edgecolor="white", linewidth=0.4,
           label=f"M < 5  (n={len(az) - len(az_big)})", zorder=3)

    # Mainshock back-azimuth as a dashed radial line
    main = df[df["event_id"] == MAINSHOCK_EVENT_ID]
    if not main.empty:
        az_main = float(main["back_azimuth_deg"].iloc[0])
        rmax = counts_all.max() * 1.15
        ax.plot([np.deg2rad(az_main)] * 2, [0, rmax],
                color="red", linewidth=2.0, linestyle="--",
                label=f"M7.2 mainshock  (az={az_main:.0f}°)")
        ax.scatter([np.deg2rad(az_main)], [rmax], marker="*",
                   c="red", s=200, edgecolors="black",
                   linewidths=1.0, zorder=10)

    ax.set_thetagrids(np.arange(0, 360, 30),
                      labels=["N", "30°", "60°", "E", "120°", "150°",
                              "S", "210°", "240°", "W", "300°", "330°"])
    ax.set_rlabel_position(135)
    ax.set_title(
        f"Back-azimuth coverage — NDHU → {len(df)} epicentres\n"
        f"(bin = {360/n_bins:.0f}°, radius = event count)",
        fontsize=12, pad=15,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10),
              fontsize=9, framealpha=0.93)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = BASE_DIR / "output_v4_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_events_used()
    print(f"loaded {len(df)} matched events within {RADIUS_KM:.0f} km")
    print(f"  M < 5 : {(df['magnitude'] < 5).sum()}")
    print(f"  M >=5 : {(df['magnitude'] >= 5).sum()}")
    print(f"  M >=6 : {(df['magnitude'] >= 6).sum()}")
    plot_epicenter_map(df, out_dir / "epicenter_map.png")
    plot_back_azimuth_rose(df, out_dir / "back_azimuth_rose.png")
    df.to_csv(out_dir / "events_used_catalog.csv", index=False)
    print(f"wrote {out_dir / 'epicenter_map.png'}")
    print(f"wrote {out_dir / 'back_azimuth_rose.png'}")


if __name__ == "__main__":
    main()
