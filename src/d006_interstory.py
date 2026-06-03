"""§4.3 (reviewer) — Snieder-Şafak inter-story deconvolution for D006.

D006 has two collocated sensors (1F + 4F = W10F), which is the rare
configuration that *does* allow direct separation of fixed-base structural
frequency from soil-structure interaction (Snieder & Şafak 2006 BSSA;
Nakata et al. 2013; Todorovska 2009). For each event we:

1. Deconvolve the 4F response by the 1F response with the same water-level
   regularisation used in the NCREE→struct pipeline. The resulting causal
   impulse response is the upgoing-wave Green's function from 1F to 4F.
2. Pick the first major causal peak inside the physically plausible window
   [0.02, 0.4] s — this is the one-way shear-wave travel time τ.
3. Estimate the fundamental fixed-base frequency as f_fixed = 1 / (4·τ).
   The factor 4 comes from the quarter-wavelength resonance of a uniform
   shear beam with fixed base and free top (Şafak 1999, JEE).

This estimate is independent of soil compliance because the input *and*
output are inside the structure — any rigid-body translation/rocking of
the foundation cancels in the ratio. Compared with the NCREE→4F transfer
function peak, the gap is a direct measure of SSI period-lengthening at
this site, on a per-event basis.

Outputs (under output_v4_D006/interstory/):
  - interstory_summary.csv          per-event f_fixed and τ for E/N
  - interstory_timeseries.png       f_fixed vs time, alongside f_iapp for context
  - interstory_irf_<event_id>.png   per-event IRF panel(s) for QC

We compute this for every catalogue event ≥ 2023 just like the main
pipeline, but the loop is much cheaper since we skip CWT/FFT/plots.
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
from obspy import read
from scipy.signal.windows import tukey

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline_common import (
    BASE_DIR,
    MAINSHOCK_EVENT_ID,
    PRE_EVENT_SEC,
    POST_EVENT_SEC,
    TARGET_FS,
    TUKEY_ALPHA,
    WATERLEVEL_EPS,
    YEAR_MIN,
    _resample_to,
    _slice_window,
)
from src.station_config import STATIONS, code_band, empirical_frequency


# Physical τ range for D006 (4-storey RC, sensors at 1F + 4F, ~9 m of structure
# between them). Shear-beam quarter-wave resonance gives f_fixed = 1/(4τ); the
# code-empirical band for 4F RC is ~1.6–3.1 Hz which maps to τ ∈ [80, 160] ms.
# We allow a wider [30, 300] ms window so coseismic softening down to ~0.8 Hz
# is still captured, but anything outside this window is likely a reverberation
# or noise pick rather than the first upgoing wavetrain.
TAU_MIN_SEC = 0.03
TAU_MAX_SEC = 0.30


def _interstory_irf(x_1f, y_4f, fs, epsilon=WATERLEVEL_EPS):
    """Causal IRF of 4F response deconvolved by 1F input.

    Both signals must be the same length and pre-aligned. Returns time-domain
    IRF, time axis, frequency axis, and |H(f)| so callers can also use the
    spectrum (v5: needed for half-power damping ratio).
    """
    n = min(len(x_1f), len(y_4f))
    x = (x_1f[:n] - np.nanmean(x_1f[:n])) * tukey(n, alpha=TUKEY_ALPHA)
    y = (y_4f[:n] - np.nanmean(y_4f[:n])) * tukey(n, alpha=TUKEY_ALPHA)
    X = np.fft.rfft(x); Y = np.fft.rfft(y)
    power = np.abs(X) ** 2
    water = epsilon * power.max()
    H = Y * np.conjugate(X) / np.maximum(power, water)
    irf = np.fft.irfft(H, n=n)
    t = np.arange(n) / fs
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return irf, t, freqs, np.abs(H)


def _zeta_halfpower(freqs, mag, f0, halfwidth_hz=1.5):
    """Half-power bandwidth damping ratio around f0 (v5 reviewer §4)."""
    if not np.isfinite(f0) or f0 <= 0:
        return float("nan")
    lo, hi = max(0.05, f0 - halfwidth_hz), f0 + halfwidth_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 8:
        return float("nan")
    sub_f, sub_a = freqs[mask], mag[mask]
    i0 = int(np.argmin(np.abs(sub_f - f0)))
    peak = sub_a[i0]
    if peak <= 0:
        return float("nan")
    target = peak / np.sqrt(2.0)
    j = i0
    while j > 0 and sub_a[j] > target:
        j -= 1
    f1 = sub_f[j] if j != i0 else float("nan")
    k = i0
    while k < len(sub_a) - 1 and sub_a[k] > target:
        k += 1
    f2 = sub_f[k] if k != i0 else float("nan")
    if not (np.isfinite(f1) and np.isfinite(f2)):
        return float("nan")
    zeta = (f2 - f1) / (2.0 * f0)
    if zeta <= 0 or zeta > 0.5:
        return float("nan")
    return float(zeta)


def _estimate_tau(irf, t, *, t_min=TAU_MIN_SEC, t_max=TAU_MAX_SEC):
    """First major causal peak of |IRF| in [t_min, t_max] s.

    Returns τ in seconds (or NaN if no defensible peak). We use the envelope
    (|IRF|) so the polarity convention of the deconvolution doesn't matter.
    """
    mask = (t >= t_min) & (t <= t_max)
    if not mask.any():
        return float("nan"), float("nan")
    env = np.abs(irf[mask])
    if env.max() <= 0:
        return float("nan"), float("nan")
    idx_local = int(np.argmax(env))
    tau = float(t[mask][idx_local])
    return tau, float(env[idx_local])


def _load_event_traces(event, comp):
    """Return (1F trace, 4F trace) for one component."""
    p1 = event.palert_1f[comp]
    p4 = event.palert_4f[comp]
    return read(str(p1))[0], read(str(p4))[0]


def process_one_event(event, *, save_plot_dir: Path | None = None):
    """Returns dict with per-component f_fixed_hz, tau_sec, etc."""
    result = {"event_id": event.event_id,
              "utc_time": event.utc_time.isoformat()}
    for comp in ("E", "N", "Z"):
        try:
            tr1, tr4 = _load_event_traces(event, comp)
        except Exception as exc:
            result[f"err_{comp}"] = f"load: {exc}"
            continue
        tr1 = _resample_to(tr1, TARGET_FS)
        tr4 = _resample_to(tr4, TARGET_FS)
        t_ref = tr1.stats.starttime
        x = _slice_window(tr1, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, TARGET_FS)
        y = _slice_window(tr4, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, TARGET_FS)
        i0 = int(PRE_EVENT_SEC * TARGET_FS)
        post_n = int(POST_EVENT_SEC * TARGET_FS)
        x_post = np.nan_to_num(x[i0:i0 + post_n])
        y_post = np.nan_to_num(y[i0:i0 + post_n])
        if (np.abs(x_post) > 0).sum() < int(10 * TARGET_FS):
            result[f"err_{comp}"] = "too_short"
            continue
        irf, t, freqs, Hmag = _interstory_irf(x_post, y_post, TARGET_FS)
        tau, peak_amp = _estimate_tau(irf, t)
        result[f"tau_sec_{comp}"] = tau
        f_fixed = (1.0 / (4.0 * tau)) if np.isfinite(tau) and tau > 0 else float("nan")
        result[f"f_fixed_hz_{comp}"] = f_fixed
        result[f"irf_peak_{comp}"] = peak_amp
        # v5 reviewer §4: damping ratio of the interstory transfer function
        # at the fixed-base mode (half-power bandwidth on |H_{4F/1F}(f)|).
        result[f"zeta_inter_{comp}"] = _zeta_halfpower(freqs, Hmag, f_fixed)

        if save_plot_dir is not None and comp in ("E", "N"):
            _plot_irf(irf, t, tau, comp, event.event_id,
                      save_plot_dir / f"interstory_irf_{event.event_id}_{comp}.png")
    return result


def _plot_irf(irf, t, tau, comp, event_id, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t, irf, color="black", linewidth=0.7)
    ax.axvspan(TAU_MIN_SEC, TAU_MAX_SEC, color="cyan", alpha=0.10,
               label=f"τ search [{TAU_MIN_SEC*1000:.0f}, {TAU_MAX_SEC*1000:.0f}] ms")
    if np.isfinite(tau):
        f_fixed = 1.0 / (4.0 * tau)
        ax.axvline(tau, color="tab:red", linestyle="--", linewidth=1.2,
                   label=f"τ = {tau*1000:.1f} ms  →  f_fixed = 1/(4τ) = {f_fixed:.2f} Hz")
    ax.set_xlim(-0.05, 0.6)
    ax.set_xlabel("Lag (s) — 4F response deconvolved by 1F")
    ax.set_ylabel("IRF amplitude")
    ax.set_title(f"D006 inter-story IRF (Snieder-Şafak 2006) — {event_id} {comp}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_interstory(out_root: Path | None = None,
                   mainshock_only: bool = False) -> Path:
    """Process all D006 events ≥ YEAR_MIN. Set mainshock_only=True for a
    quick test that only does the M7.2 event."""
    cfg = STATIONS["D006"]
    out_root = out_root or (BASE_DIR / "output_v4_D006" / "interstory")
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "interstory_summary.csv"

    matcher = importlib.import_module(cfg.matcher_module)
    events, _ = matcher.build_event_list()
    events = [e for e in events if e.event_id[:4] >= YEAR_MIN]
    if mainshock_only:
        events = [e for e in events if e.event_id == MAINSHOCK_EVENT_ID]

    # Per-event IRF plots only for the M7.2 mainshock (cheap, illustrative).
    plot_dir = out_root if mainshock_only else None
    rows = []
    print(f"D006 inter-story: {len(events)} events")
    for i, event in enumerate(events):
        is_mainshock = event.event_id == MAINSHOCK_EVENT_ID
        r = process_one_event(event,
                              save_plot_dir=out_root if is_mainshock else None)
        rows.append(r)
        if i % 50 == 0:
            print(f"  {i}/{len(events)}  {event.event_id}")
    cols = ["event_id", "utc_time"]
    for comp in ("E", "N", "Z"):
        cols += [f"tau_sec_{comp}", f"f_fixed_hz_{comp}", f"irf_peak_{comp}",
                 f"zeta_inter_{comp}", f"err_{comp}"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    print(f"Wrote {csv_path}")

    _plot_timeseries(csv_path, out_root, cfg)
    return out_root


def _plot_timeseries(csv_path: Path, out_dir: Path, cfg):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    df["t"] = pd.to_datetime(df["utc_time"])
    f_emp, f_emp_min, f_emp_max = code_band(cfg)

    # Try to overlay f_iapp from the main D006 CSV for reference.
    f_iapp_csv = BASE_DIR / "structural_history_log_v4_D006.csv"
    if not f_iapp_csv.exists():
        f_iapp_csv = BASE_DIR / "structural_history_log_v3.1.csv"
    iapp_df = None
    if f_iapp_csv.exists():
        try:
            iapp_df = pd.read_csv(f_iapp_csv)
            iapp_df["t"] = pd.to_datetime(iapp_df["utc_time"])
            # Pick the 4F column if present in v3.1 schema, else f_iapp.
            if "f_iapp_4f" in iapp_df.columns:
                iapp_df["f_ref"] = pd.to_numeric(iapp_df["f_iapp_4f"], errors="coerce")
            elif "f_iapp" in iapp_df.columns:
                iapp_df["f_ref"] = pd.to_numeric(iapp_df["f_iapp"], errors="coerce")
            else:
                iapp_df = None
        except Exception:
            iapp_df = None

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, comp in zip(axes, ("E", "N")):
        ax.axhspan(f_emp_min, f_emp_max, color="cyan", alpha=0.12,
                   label=f"code band [{f_emp_min:.2f},{f_emp_max:.2f}]")
        ax.axhline(f_emp, color="cyan", linestyle="--", linewidth=1.0, alpha=0.7)
        y = pd.to_numeric(df[f"f_fixed_hz_{comp}"], errors="coerce")
        ax.scatter(df["t"], y, s=20, c="tab:red", alpha=0.6, edgecolors="none",
                   label=f"f_fixed_base = 1/(4τ)  [Snieder-Şafak, SSI-free]")
        if iapp_df is not None:
            sub_ref = iapp_df[iapp_df.get("comp", comp) == comp] if "comp" in iapp_df.columns else iapp_df
            ax.scatter(sub_ref["t"], sub_ref["f_ref"], s=12, c="tab:blue",
                       alpha=0.35, edgecolors="none",
                       label="f_iapp (NCREE→4F path, includes SSI)")
        ax.set_ylabel(f"{comp}  Freq (Hz)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    axes[-1].set_xlabel("Event Time (UTC)")
    fig.suptitle(
        f"D006 ({cfg.sensor_floor_label}+{cfg.upper_sensor_label}) — "
        f"inter-story deconvolution (Snieder & Şafak 2006)\n"
        f"Difference (f_iapp − f_fixed) bounds the SSI period-lengthening on a per-event basis.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "interstory_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mainshock-only", action="store_true",
                        help="only process the M7.2 event (fast smoke test)")
    args = parser.parse_args()
    run_interstory(mainshock_only=args.mainshock_only)
