"""Shared v5 pipeline (reviewer_v4 revisions).

Forked from pipeline_common.py with surgical changes for the v5 round of
reviewer responses (see ``reviewer_v4.md``). The four code-level fixes:

(3) Symmetric-bandpass f_min sensitivity check. The original f_min uses
    an asymmetric window [f_c/1.7, f_c*1.15] that biases the estimator
    toward "softening". v5 also computes a symmetric, equal-log-width
    band [f_c/r, f_c*r] where r=sqrt(1.7*1.15)~=1.398, and reports both
    values plus their difference in the per-event CSV. A new summary
    plot (``summary_fmin_symmetric_vs_asymmetric.png``) compares the
    Δf/f distributions so reviewer (3) can be checked directly.

(5) STA/LTA + Arias 5/95/99% segmentation source is now explicit in the
    README and CSV (column ``segment_source = "structure"``) so the
    reviewer's question about whether segmentation is done on the
    free-field or structural record is answered in the artefact itself.

(6) ``_baseline_comparison`` adds a PGA-matched variant: for each post-
    mainshock event we find the nearest pre-mainshock event by log-PGA
    and compute the median difference on that matched subsample, on
    top of the original ±90 d bootstrap. Output:
    ``baseline_comparison_pga_matched.{png,json}``.

(10) ``_baseline_comparison`` additionally computes an STL-deseasonalized
     f_iapp series (monthly median + STL with period=12) and re-runs the
     pre/post bootstrap on the residual + trend (environment-effect
     removed). Output: ``baseline_comparison_deseasonalized.{png,json}``.

Outputs go to ``BASE_DIR / f"output_v5_{station_id}"`` and
``structural_history_log_v5_{sid}.csv`` so v3/v4 results stay intact for
side-by-side comparison.
"""
from __future__ import annotations

import csv
import gc
import importlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
from obspy import read
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.signal.windows import tukey
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from src.station_config import (
    CAMPUS_SITE_CLASS,
    CAMPUS_VS30_MS,
    StationConfig,
    code_band,
    empirical_frequency,
)


# --- Standard limitation text appended to summary figure captions ---
SSI_LIMITATION_NOTE = (
    "Limitations: (i) NCREE is the surface companion of a borehole instrument and acts as the\n"
    "free-field input; the 230–1077 m offset to each D-station introduces path effects (see\n"
    "station distance in CSV metadata). (ii) Observed coseismic f_min drops conflate structural\n"
    "and soil nonlinearity; soil shear-modulus reduction during strong shaking contributes a\n"
    "comparable order of magnitude (Todorovska 2009 BSSA; Stewart et al. 1999). SSI is NOT\n"
    "separated here; see ssi_estimates.csv for Veletsos-Meek upper-bound period-lengthening per\n"
    "NIST GCR 12-917-21 §2. Only D006 (1F+4F sensors) permits direct SSI removal via inter-story\n"
    "deconvolution (Snieder & Şafak 2006)."
)


# ---------------------------------------------------------------------------
# Module-level constants (shared by all stations)
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"D:\PHD\project2026")

TARGET_FS = 100.0
PRE_EVENT_SEC = 30.0
POST_EVENT_SEC = 120.0
YEAR_MIN = "2023"

CWT_MIN_FREQ = 0.4

TUKEY_ALPHA = 0.05
WATERLEVEL_EPS = 0.05
KO_BANDWIDTH = 30
KO_LOG_BINS = 300
STA_SEC = 1.0
LTA_SEC = 10.0
STA_LTA_THRESHOLD = 2.0
ARIAS_G = 9.80665

MAINSHOCK_UTC = "2024-04-02 23:58:11"
MAINSHOCK_EVENT_ID = "20240402235754"


# ---------------------------------------------------------------------------
# Per-run paths
# ---------------------------------------------------------------------------
def output_paths(cfg: StationConfig) -> dict[str, Path]:
    sid = cfg.station_id
    root = BASE_DIR / f"output_v5_{sid}"
    return {
        "root": root,
        "log": BASE_DIR / f"processing_errors_v5_{sid}.log",
        "skip_log": BASE_DIR / f"skipped_events_v5_{sid}.log",
        "csv": BASE_DIR / f"structural_history_log_v5_{sid}.csv",
    }


# v5 (reviewer §3): symmetric-bandpass control band. The asymmetric
# multipliers (1.7 down, 1.15 up) allow ~41% softening but only ~15%
# headroom upward -- a reviewer-flagged source of one-sided bias. The
# symmetric band uses the geometric-mean half-width so [lo, hi] is
# centred on f_c in log space:
#     r_sym = sqrt(1.7 * 1.15) ~= 1.398
# i.e. f_c * r_sym  vs  f_c / r_sym  -> equal +/- log-width.
F_MIN_BAND_DOWN_ASYM = 1.7
F_MIN_BAND_UP_ASYM = 1.15
F_MIN_BAND_R_SYM = float(np.sqrt(F_MIN_BAND_DOWN_ASYM * F_MIN_BAND_UP_ASYM))

# v5 (reviewer §5): segmentation (STA/LTA + Arias 5/95/99) is run on the
# *structural* station record. This is the same as v4 but now recorded as
# an explicit CSV column so reviewers can see it without reading the code.
SEGMENT_SOURCE = "structure"


def csv_columns(cfg: StationConfig) -> list[str]:
    sid_lower = cfg.station_id.lower()
    return [
        "event_id", "utc_time", "local_time",
        "magnitude", "depth_km", "epicenter",
        "pga_0m", "pga_20m", "pga_30m", "pga_58m", "pga_78m",
        "sensitivity", "catalog_matched",
        "comp",
        # --- transfer function ---
        "tf_peak_freq", "tf_peak_amp",
        "tf_f1",                         # constrained to code band (reference only)
        "tf_peak_in_codeband",           # bool: is the unconstrained TF peak inside the code band?
        # --- code-empirical reference ---
        "f_empirical", "f_emp_min", "f_emp_max",
        # --- Astorga 2018 operational modal frequencies ---
        "f_iapp", "f_min", "f_99app",
        # v5 (reviewer §3): symmetric-bandpass f_min for bias control
        "f_min_sym", "f_min_diff_pct",
        "t_first_arrival_sec", "t5_sec", "t95_sec", "t99_sec",
        "arias_total",
        "damping_ratio",
        "fft_peak_ncree", f"fft_peak_{sid_lower}",
        # --- per-station deployment metadata (constant within station) ---
        "ncree_distance_m", "vs30_ms", "site_class",
        # v5 (reviewer §5): which record drives STA/LTA + Arias t5/95/99
        "segment_source",
        "status",
    ]


# ---------------------------------------------------------------------------
# Signal-processing primitives (verbatim from v3 main files)
# ---------------------------------------------------------------------------
def _resample_to(trace, fs):
    if abs(trace.stats.sampling_rate - fs) < 1e-6:
        return trace
    tr = trace.copy()
    tr.resample(fs)
    return tr


def _slice_window(trace, t_ref, pre_sec, post_sec, fs):
    t0 = trace.stats.starttime
    n = trace.stats.npts
    expected_len = int((pre_sec + post_sec) * fs)
    out = np.full(expected_len, np.nan)
    sample_at_ref = int(round((t_ref - t0) * fs))
    src_start = sample_at_ref - int(pre_sec * fs)
    src_end = src_start + expected_len
    a = max(0, src_start)
    b = min(n, src_end)
    if b <= a:
        return out
    out[a - src_start : b - src_start] = trace.data[a:b]
    return out


def _peak_freq(freqs, amp, fmin=0.4, fmax=20.0):
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return float("nan"), float("nan")
    sub_f = freqs[mask]
    sub_a = amp[mask]
    i = int(np.argmax(sub_a))
    return float(sub_f[i]), float(sub_a[i])


def _tapered(x, alpha=TUKEY_ALPHA):
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return x.copy()
    return (x - x.mean()) * tukey(len(x), alpha=alpha)


def _ko_smooth(freqs, amp, fmin=0.1, fmax=25.0,
               n_log_bins=KO_LOG_BINS, bandwidth=KO_BANDWIDTH):
    log_f = np.logspace(np.log10(max(fmin, freqs[freqs > 0].min())),
                        np.log10(min(fmax, freqs[-1])), n_log_bins)
    amp_log = np.interp(log_f, freqs, amp)
    smoothed = konno_ohmachi_smoothing(
        amp_log.astype(np.float32), log_f.astype(np.float32),
        bandwidth=bandwidth, normalize=False,
    )
    return log_f, np.asarray(smoothed, dtype=float)


def _smoothed_fft(signal, fs):
    x = _tapered(signal)
    n = len(x)
    if n < 8:
        return np.array([np.nan]), np.array([np.nan])
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.abs(np.fft.rfft(x)) / n
    return _ko_smooth(freqs, amp)


def _damping_from_raw_H(input_sig, output_sig, fs, f0,
                        epsilon=WATERLEVEL_EPS, halfwidth_hz=1.5):
    if np.isnan(f0) or f0 <= 0:
        return float("nan")
    n = min(len(input_sig), len(output_sig))
    if n < 64:
        return float("nan")
    x = _tapered(input_sig[:n])
    y = _tapered(output_sig[:n])
    X = np.fft.rfft(x); Y = np.fft.rfft(y)
    power = np.abs(X) ** 2
    water = epsilon * power.max()
    H = np.abs(Y * np.conjugate(X) / np.maximum(power, water))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    lo, hi = max(0.05, f0 - halfwidth_hz), f0 + halfwidth_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 8:
        return float("nan")
    return _half_power_damping(freqs[mask], H[mask], f0)


def _half_power_damping(freqs, amp, f0):
    if np.isnan(f0):
        return float("nan")
    i0 = int(np.argmin(np.abs(freqs - f0)))
    peak = amp[i0]
    if peak <= 0:
        return float("nan")
    target = peak / np.sqrt(2)
    j = i0
    while j > 0 and amp[j] > target:
        j -= 1
    f1 = freqs[j] if j != i0 else np.nan
    k = i0
    while k < len(amp) - 1 and amp[k] > target:
        k += 1
    f2 = freqs[k] if k != i0 else np.nan
    if np.isnan(f1) or np.isnan(f2) or f0 <= 0:
        return float("nan")
    zeta = (f2 - f1) / (2.0 * f0)
    if zeta <= 0 or zeta > 0.5:
        return float("nan")
    return float(zeta)


def _event_segments(signal_with_pre, fs, pre_event_sec=PRE_EVENT_SEC):
    n = len(signal_with_pre)
    nsta = int(STA_SEC * fs)
    nlta = int(LTA_SEC * fs)
    if n < nlta + nsta + 50:
        return None
    sig = np.nan_to_num(signal_with_pre - np.nanmean(signal_with_pre))
    if not np.any(np.abs(sig) > 0):
        return None
    cft = classic_sta_lta(sig, nsta, nlta)
    pre_n = int(pre_event_sec * fs)
    search_lo = max(nlta, pre_n - int(5 * fs))
    search_hi = min(n, pre_n + int(30 * fs))
    region = cft[search_lo:search_hi]
    above = np.where(region > STA_LTA_THRESHOLD)[0]
    if len(above) > 0:
        t_first = search_lo + int(above[0])
    else:
        t_first = pre_n
    arias = np.cumsum(sig ** 2)
    arias_max = float(arias[-1])
    if arias_max <= 0:
        return None
    arias_norm = arias / arias_max
    t5 = int(np.searchsorted(arias_norm, 0.05))
    t95 = int(np.searchsorted(arias_norm, 0.95))
    t99 = int(np.searchsorted(arias_norm, 0.99))
    arias_total = arias_max * (np.pi / (2.0 * ARIAS_G)) / fs
    return {
        "t_first": t_first, "t5": t5, "t95": t95, "t99": t99,
        "t_first_sec": t_first / fs - pre_event_sec,
        "t5_sec":  t5 / fs - pre_event_sec,
        "t95_sec": t95 / fs - pre_event_sec,
        "t99_sec": t99 / fs - pre_event_sec,
        "arias_total": arias_total,
    }


def _modal_in_segment(signal_full, fs, segments, key_a, key_b, fmin, fmax):
    a, b = segments[key_a], segments[key_b]
    if b - a < int(2 * fs):
        return float("nan"), float("nan")
    seg = signal_full[a:b]
    seg = seg[~np.isnan(seg)]
    if len(seg) < int(2 * fs):
        return float("nan"), float("nan")
    f, amp = _smoothed_fft(seg, fs)
    return _peak_freq(f, amp, fmin=fmin, fmax=fmax)


def _f_min_strong_motion(signal_full, fs, segments, fmin, fmax,
                         f_center=None, mode="asymmetric"):
    """v5: ``mode`` selects between the v4 asymmetric band (default, kept for
    backward comparability) and a symmetric equal-log-width band whose
    half-width matches the geometric mean of the asymmetric multipliers
    (reviewer §3 bias test).
    """
    a, b = segments["t5"], segments["t95"]
    if b - a < int(2 * fs):
        return float("nan"), float("nan")
    dsm_rms = float(np.sqrt(np.nanmean(np.square(signal_full[a:b]))))
    if not np.isfinite(dsm_rms) or dsm_rms < 1.0:
        return float("nan"), float("nan")
    if f_center is not None and not np.isnan(f_center) and f_center > 0.5:
        if mode == "symmetric":
            lo = max(0.3, f_center / F_MIN_BAND_R_SYM)
            hi = min(fs / 2 - 0.1, f_center * F_MIN_BAND_R_SYM)
        else:
            lo = max(0.3, f_center / F_MIN_BAND_DOWN_ASYM)
            hi = min(fs / 2 - 0.1, f_center * F_MIN_BAND_UP_ASYM)
    else:
        lo, hi = fmin, fmax
    if hi - lo < 0.3:
        return float("nan"), float("nan")
    sos = butter(6, [lo, hi], btype="band", fs=fs, output="sos")
    sig = np.nan_to_num(signal_full)
    sig = sig - sig.mean()
    bp = sosfiltfilt(sos, sig)
    analytic = hilbert(bp)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs
    inst_freq = np.append(inst_freq, inst_freq[-1])
    inst_freq = uniform_filter1d(inst_freq, size=int(fs))
    envelope = np.abs(analytic)
    if_seg = inst_freq[a:b]
    env_seg = envelope[a:b]
    if env_seg.size == 0 or env_seg.max() <= 0:
        return float("nan"), float("nan")
    if_lower = max(lo * 1.02, fmin)
    if_upper = min(hi * 0.98, fmax * 1.05)
    high_env = env_seg > 0.5 * env_seg.max()
    in_band = (if_seg >= if_lower) & (if_seg <= if_upper)
    valid = high_env & in_band
    if valid.sum() < int(0.5 * fs):
        return float("nan"), float("nan")
    f_med = float(np.median(if_seg[valid]))
    valid_idx = np.flatnonzero(valid)
    rel = valid_idx[int(np.argmin(np.abs(if_seg[valid_idx] - f_med)))]
    abs_idx = a + int(rel)
    return f_med, abs_idx / fs - PRE_EVENT_SEC


def _transfer_function(input_sig, output_sig, fs, epsilon=WATERLEVEL_EPS):
    n = min(len(input_sig), len(output_sig))
    x = _tapered(input_sig[:n])
    y = _tapered(output_sig[:n])
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    power = np.abs(X) ** 2
    water = epsilon * power.max()
    denom = np.maximum(power, water)
    H = Y * np.conjugate(X) / denom
    freqs_lin = np.fft.rfftfreq(n, 1.0 / fs)
    return _ko_smooth(freqs_lin, np.abs(H))


def _sliding_window_tf(input_sig, output_sig, fs, f_emp_min, f_emp_max,
                       window_sec=10.0, overlap_sec=5.0,
                       fmin_plot=0.4, fmax_plot=15.0,
                       epsilon=WATERLEVEL_EPS, t_offset_sec=0.0):
    """§2.1 (reviewer): time-varying transfer function H(f, t).

    The event-averaged |H(f)| mixes pre-event linear-elastic, coseismic
    nonlinear, and post-event recovery into a single spectrum and violates
    the LTI assumption invoked in §2.3. Sliding-window deconvolution
    (Nakata & Snieder 2014 BSSA; Mikael et al. 2013) re-applies the same
    water-level deconvolution on overlapping windows so we can read off
    the coseismic frequency drop and recovery directly.

    Returns dict with keys: time (centre of each window, s relative to
    NCREE trigger), freqs (Hz), H (n_windows × n_freqs |H| magnitude).
    """
    n = min(len(input_sig), len(output_sig))
    nw = int(window_sec * fs)
    step = max(1, int((window_sec - overlap_sec) * fs))
    if n < nw + step:
        return None
    centres = []
    rows = []
    freqs_lin = np.fft.rfftfreq(nw, 1.0 / fs)
    mask_plot = (freqs_lin >= fmin_plot) & (freqs_lin <= fmax_plot)
    f_plot = freqs_lin[mask_plot]
    for start in range(0, n - nw + 1, step):
        x = _tapered(input_sig[start:start + nw])
        y = _tapered(output_sig[start:start + nw])
        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)
        power = np.abs(X) ** 2
        water = epsilon * (power.max() if power.size else 1.0)
        H = np.abs(Y * np.conjugate(X) / np.maximum(power, water))
        rows.append(H[mask_plot])
        centres.append((start + nw / 2) / fs + t_offset_sec)
    if not rows:
        return None
    H_grid = np.vstack(rows)
    # Ridge: code-band-constrained peak per window (purely descriptive overlay)
    band = (f_plot >= f_emp_min) & (f_plot <= f_emp_max)
    if band.any():
        masked = H_grid.copy()
        masked[:, ~band] = -np.inf
        ridge_idx = np.argmax(masked, axis=1)
        ridge = f_plot[ridge_idx]
    else:
        ridge = np.full(len(centres), np.nan)
    return {
        "time": np.asarray(centres),
        "freqs": f_plot,
        "H": H_grid,
        "ridge": ridge,
        "window_sec": window_sec,
        "overlap_sec": overlap_sec,
    }


def _plot_sliding_tf(out_path, comp, tv, cfg, segs=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.5))
    # log magnitude for dynamic range
    pcm = ax.pcolormesh(tv["time"], tv["freqs"], np.log10(tv["H"].T + 1e-12),
                        shading="auto", cmap="viridis")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("log10 |H(f, t)|")
    _draw_empirical_band(ax, cfg, axis="y", in_legend=True)
    ax.plot(tv["time"], tv["ridge"], color="white", linestyle="--",
            linewidth=1.0, alpha=0.85, label="code-band ridge")
    if segs is not None:
        for tval, c, lbl in [
            (segs.get("t_first_sec"), "tab:red",    "STA/LTA"),
            (segs.get("t5_sec"),  "tab:purple",     "Arias 5%"),
            (segs.get("t95_sec"), "tab:purple",     "Arias 95%"),
            (segs.get("t99_sec"), "tab:gray",       "Arias 99%"),
        ]:
            if tval is not None and not np.isnan(tval):
                ax.axvline(tval, color=c, linestyle=":", linewidth=1.1,
                           alpha=0.8, zorder=3)
    ax.set_xlabel("Time (s) — t=0 at NCREE trigger")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(tv["freqs"].min(), tv["freqs"].max())
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.set_title(
        f"Time-varying |H(f, t)| — {comp}   "
        f"window={tv['window_sec']:.0f}s, overlap={tv['overlap_sec']:.0f}s   "
        f"(Reviewer §2.1: relaxes the LTI assumption)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cwt_response(signal_full, fs, f_emp_min, f_emp_max,
                  pre_event_sec=PRE_EVENT_SEC,
                  fmin=0.4, fmax=15.0, n_bins=120):
    from src import processor
    sig = np.nan_to_num(signal_full - np.nanmean(signal_full))
    cwt_freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_bins)
    cwt_matrix, freqs = processor.compute_cwt(sig, fs, freqs=cwt_freqs)
    n = len(sig)
    time = np.arange(n) / fs - pre_event_sec
    band_mask = (freqs >= f_emp_min) & (freqs <= f_emp_max)
    if band_mask.any():
        power = np.abs(cwt_matrix)
        masked = power.copy()
        masked[~band_mask, :] = -np.inf
        ridge_idx = np.argmax(masked, axis=0)
        ridge = freqs[ridge_idx]
    else:
        ridge = np.full(n, np.nan)
    return {"cwt": cwt_matrix, "freqs": freqs, "time": time, "ridge": ridge}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _draw_empirical_band(ax, cfg: StationConfig, axis="y", *,
                         in_legend: bool = False):
    """Draw the Taiwan-code empirical band on a freq axis.

    The band+centre-line are visually distinctive (cyan span + dashed line)
    so they don't usually need their own legend entries. Set in_legend=True
    on single-panel figures (e.g. per-event TF) where we want them labelled.
    """
    f_emp, f_emp_min, f_emp_max = code_band(cfg)
    band_kw = dict(color="cyan", alpha=0.12, zorder=1)
    line_kw = dict(color="cyan", linestyle="--", linewidth=1.2, alpha=0.9, zorder=2)
    if in_legend:
        band_kw["label"] = f"code band [{f_emp_min:.2f}, {f_emp_max:.2f}] Hz"
        line_kw["label"] = f"f_emp = {f_emp:.2f} Hz"
    if axis == "y":
        ax.axhspan(f_emp_min, f_emp_max, **band_kw)
        ax.axhline(f_emp, **line_kw)
    else:
        ax.axvspan(f_emp_min, f_emp_max, **band_kw)
        ax.axvline(f_emp, **line_kw)


def _plot_raw(out_path, fs, comp, traces_with_times, cfg,
              t_ref_label="NCREE trigger", segments=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    seg_lines = []
    if segments is not None:
        seg_lines = [
            (segments.get("t_first_sec"), "First arrival (STA/LTA)", "tab:red"),
            (segments.get("t5_sec"),  "Arias 5% (DSM start)",  "tab:purple"),
            (segments.get("t95_sec"), "Arias 95% (DSM end)",   "tab:purple"),
            (segments.get("t99_sec"), "Arias 99%",             "tab:gray"),
        ]
    for ax, (label, t, data) in zip(axes, traces_with_times):
        valid = ~np.isnan(data)
        ax.plot(t[valid], data[valid], linewidth=0.6)
        ax.set_ylabel(f"{label}\n(gal)")
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
        for tval, _lbl, color in seg_lines:
            if tval is None or np.isnan(tval):
                continue
            ax.axvline(tval, color=color, linewidth=1.0, alpha=0.7, linestyle="--")
        if valid.any():
            peak = np.nanmax(np.abs(data))
            ax.text(0.99, 0.95, f"|peak|={peak:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.grid(alpha=0.3)
    if seg_lines:
        handles = [plt.Line2D([0], [0], color=c, linestyle="--", linewidth=1.2,
                              label=lbl) for _, lbl, c in seg_lines]
        axes[0].legend(handles=handles, loc="upper right", fontsize=7,
                       framealpha=0.9)
    axes[-1].set_xlabel(f"Time (s) — t=0 at {t_ref_label}")
    fig.suptitle(f"Raw waveforms ({comp})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fft(out_path, fs, comp, traces, cfg):
    import matplotlib.pyplot as plt
    _, f_emp_min, f_emp_max = code_band(cfg)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    peaks = {}
    struct_key = f"{cfg.station_id}({cfg.sensor_floor_label})"
    colors = {"NCREE": "tab:blue", struct_key: "tab:green"}
    for ax, (label, data) in zip(axes, traces):
        d = data[~np.isnan(data)]
        if len(d) == 0:
            continue
        freqs, amp = _smoothed_fft(d, fs)
        c = colors.get(label, "black")
        _draw_empirical_band(ax, cfg, axis="x", in_legend=True)
        ax.semilogy(freqs, amp, linewidth=0.9, color=c)
        pf, pa = _peak_freq(freqs, amp)
        peaks[label] = pf
        ax.axvline(pf, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.annotate(f"{label}  peak = {pf:.2f} Hz",
                    xy=(pf, pa), xytext=(8, -4), textcoords="offset points",
                    fontsize=9, color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow",
                              ec=c, alpha=0.9))
        pf_b, pa_b = _peak_freq(freqs, amp, fmin=f_emp_min, fmax=f_emp_max)
        peaks[f"{label}_band"] = pf_b
        if not np.isnan(pf_b):
            ax.scatter([pf_b], [pa_b], marker="*", c="magenta", s=120,
                       edgecolor="white", linewidth=1, zorder=5)
            ax.annotate(f"{label}  f₁ (code band) = {pf_b:.2f} Hz",
                        xy=(pf_b, pa_b), xytext=(8, -32),
                        textcoords="offset points",
                        fontsize=9, color="magenta", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="magenta", alpha=0.9))
        ax.set_ylabel(f"{label}\nAmplitude")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"FFT spectra ({comp})")
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(0, 25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return peaks


def _plot_transfer_function(out_path, comp, tf, cfg):
    import matplotlib.pyplot as plt
    _, f_emp_min, f_emp_max = code_band(cfg)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    freqs, mag = tf
    _draw_empirical_band(ax, cfg, axis="x", in_legend=True)
    ax.semilogy(freqs, mag, linewidth=0.8)
    pf, pa = _peak_freq(freqs, mag, fmin=0.4, fmax=15)
    ax.axvline(pf, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.annotate(f"peak = {pf:.2f} Hz\n|H| = {pa:.2g}",
                xy=(pf, pa), xytext=(10, 0), textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    f1, a1 = _peak_freq(freqs, mag, fmin=f_emp_min, fmax=f_emp_max)
    if not np.isnan(f1):
        ax.scatter([f1], [a1], marker="*", c="magenta", s=140,
                   edgecolor="white", linewidth=1, zorder=5)
        ax.annotate(f"f₁ (code band) = {f1:.2f} Hz",
                    xy=(f1, a1), xytext=(10, -25), textcoords="offset points",
                    fontsize=9, color="magenta", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="magenta", alpha=0.9))
    ax.set_ylabel(f"NCREE(free-field) → {cfg.station_id} ({cfg.sensor_floor_label})\n|H(f)|  composite soil+structure")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 20)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlabel("Frequency (Hz)")
    fig.suptitle(
        f"Transfer Function (Water-level Deconvolution) — {comp}\n"
        f"Path: NCREE (surface companion of borehole, d={cfg.ncree_distance_m:.0f} m) → "
        f"{cfg.station_id} ({cfg.sensor_floor_label})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"peak": (pf, pa), "_f1": f1}


def _plot_cwt_annotated(feat, out_path, title, segs, ops, label_pre, cfg,
                        y_max=15.0):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.pcolormesh(feat["time"], feat["freqs"], np.abs(feat["cwt"]),
                  shading="auto", cmap="viridis", zorder=1)
    ax.plot(feat["time"], feat["ridge"], color="white", linestyle="--",
            linewidth=1.0, alpha=0.85, label="1st-mode ridge (code-band)")
    _draw_empirical_band(ax, cfg, axis="y", in_legend=True)
    if segs is not None:
        for tval, c in [(segs.get("t_first_sec"), "tab:red"),
                        (segs.get("t5_sec"),  "tab:purple"),
                        (segs.get("t95_sec"), "tab:purple"),
                        (segs.get("t99_sec"), "tab:gray")]:
            if tval is not None and not np.isnan(tval):
                ax.axvline(tval, color=c, linestyle=":", linewidth=1.0,
                           alpha=0.7, zorder=3)
    if ops is not None and segs is not None:
        pts = [
            ("f_iapp",  segs["t_first_sec"] / 2, ops.get("f_iapp"),  "tab:blue"),
            ("f_min",   ops.get("t_min", (segs["t5_sec"] + segs["t95_sec"]) / 2),
                        ops.get("f_min"), "tab:orange"),
            ("f_99app", (segs["t95_sec"] + segs["t99_sec"]) / 2,
                        ops.get("f_99app"), "tab:green"),
        ]
        for nm, tx, fy, color in pts:
            if fy is None or np.isnan(fy):
                continue
            ax.scatter([tx], [fy], c=color, s=140, marker="o",
                       edgecolor="black", linewidth=1.4, zorder=10)
            ax.annotate(f"{nm}\n{fy:.2f} Hz",
                        xy=(tx, fy), xytext=(8, 10),
                        textcoords="offset points",
                        color=color, fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=color, alpha=0.92),
                        zorder=11)
    ax.set_xlabel("Time (s) — t=0 at NCREE trigger")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, y_max)
    ax.set_title(f"{title}  —  CWT({label_pre} response signal)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-event processor (runs in worker process)
# ---------------------------------------------------------------------------
def _load_traces(event, cfg: StationConfig):
    out = {"NCREE": {}, cfg.station_id: {}}
    for c, p in event.ncree_paths.items():
        out["NCREE"][c] = read(str(p))[0]
    palert_paths = getattr(event, cfg.palert_attr)
    for c, p in palert_paths.items():
        out[cfg.station_id][c] = read(str(p))[0]
    return out


def _row(event, meta, cfg, comp, feat, tf_info, fft_peaks, operational, status):
    f_emp, f_emp_min, f_emp_max = code_band(cfg)
    sid = cfg.station_id
    sid_lower = sid.lower()

    def _g(k):
        v = meta.get(k) if meta else None
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else v

    def _fmt(v, spec=".4f"):
        if v is None:
            return ""
        try:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return ""
        except Exception:
            pass
        return f"{v:{spec}}"

    op = operational or {}
    peak = tf_info.get("peak", (None, None)) if tf_info else (None, None)
    struct_key = f"{sid}({cfg.sensor_floor_label})"
    # §2.7: flag whether the *unconstrained* TF peak falls inside the code band.
    # This replaces tf_f1 as the primary indicator and breaks the circular logic
    # ("we always find peaks inside the code band because we restricted search to it").
    if peak[0] is not None and not (isinstance(peak[0], float) and np.isnan(peak[0])):
        tf_peak_in_codeband = bool(f_emp_min <= peak[0] <= f_emp_max)
    else:
        tf_peak_in_codeband = None
    return {
        "event_id": event.event_id,
        "utc_time": event.utc_time.isoformat(),
        "local_time": event.local_time.isoformat(),
        "magnitude": _g("magnitude"), "depth_km": _g("depth_km"), "epicenter": _g("epicenter"),
        "pga_0m": _g("pga_0m"), "pga_20m": _g("pga_20m"),
        "pga_30m": _g("pga_30m"), "pga_58m": _g("pga_58m"), "pga_78m": _g("pga_78m"),
        "sensitivity": _g("sensitivity"),
        "catalog_matched": bool(meta),
        "comp": comp,
        "tf_peak_freq": f"{peak[0]:.3f}" if peak[0] is not None and not np.isnan(peak[0]) else "",
        "tf_peak_amp":  f"{peak[1]:.4g}" if peak[1] is not None and not np.isnan(peak[1]) else "",
        "tf_f1": f"{tf_info['_f1']:.3f}" if tf_info and not np.isnan(tf_info.get('_f1', np.nan)) else "",
        "tf_peak_in_codeband": "" if tf_peak_in_codeband is None else str(tf_peak_in_codeband),
        "f_empirical": f"{f_emp:.3f}",
        "f_emp_min":   f"{f_emp_min:.3f}",
        "f_emp_max":   f"{f_emp_max:.3f}",
        "f_iapp":   _fmt(op.get("f_iapp"), ".3f"),
        "f_min":    _fmt(op.get("f_min"),  ".3f"),
        "f_99app":  _fmt(op.get("f_99app"), ".3f"),
        "f_min_sym":       _fmt(op.get("f_min_sym"), ".3f"),
        "f_min_diff_pct":  _fmt(op.get("f_min_diff_pct"), ".2f"),
        "t_first_arrival_sec": _fmt(op.get("t_first_arrival_sec"), ".2f"),
        "t5_sec":  _fmt(op.get("t5_sec"),  ".2f"),
        "t95_sec": _fmt(op.get("t95_sec"), ".2f"),
        "t99_sec": _fmt(op.get("t99_sec"), ".2f"),
        "arias_total": _fmt(op.get("arias_total"), ".4g"),
        "damping_ratio": _fmt(tf_info.get("_zeta") if tf_info else None, ".4f"),
        "fft_peak_ncree": f"{fft_peaks['NCREE']:.3f}" if fft_peaks else "",
        f"fft_peak_{sid_lower}": f"{fft_peaks[struct_key]:.3f}" if fft_peaks else "",
        "ncree_distance_m": f"{cfg.ncree_distance_m:.0f}",
        "vs30_ms": f"{CAMPUS_VS30_MS:.1f}",
        "site_class": CAMPUS_SITE_CLASS,
        "segment_source": SEGMENT_SOURCE,
        "status": status or "OK",
    }


def process_event(event, cfg: StationConfig):
    paths = output_paths(cfg)
    event_dir = paths["root"] / event.event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    _, f_emp_min, f_emp_max = code_band(cfg)
    f_emp = empirical_frequency(cfg)
    f_min_lo = f_emp * 0.5
    f_min_hi = f_emp_max
    sid = cfg.station_id

    try:
        streams = _load_traces(event, cfg)
        meta = event.catalog_meta or {}
        t_ref = streams["NCREE"]["Z"].stats.starttime

        for comp in ("E", "N", "Z"):
            ncree_tr = _resample_to(streams["NCREE"][comp], TARGET_FS)
            struct_tr = _resample_to(streams[sid][comp], TARGET_FS)

            t_axis = np.arange(int((PRE_EVENT_SEC + POST_EVENT_SEC) * TARGET_FS)) / TARGET_FS - PRE_EVENT_SEC
            ncree_full = _slice_window(ncree_tr, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, TARGET_FS)
            struct_full = _slice_window(struct_tr, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, TARGET_FS)

            for arr in (ncree_full, struct_full):
                m = ~np.isnan(arr)
                if m.any():
                    arr[m] -= arr[m].mean()

            post_n = int(POST_EVENT_SEC * TARGET_FS)
            i0 = int(PRE_EVENT_SEC * TARGET_FS)
            ncree_post = np.nan_to_num(ncree_full[i0:i0 + post_n])
            struct_post = np.nan_to_num(struct_full[i0:i0 + post_n])

            if (np.abs(ncree_post) > 0).sum() < int(10 * TARGET_FS):
                rows.append(_row(event, meta, cfg, comp, None, None, None, None, "too_short"))
                continue

            # ---- Transfer function (event-averaged) ----
            tf = _transfer_function(ncree_post, struct_post, TARGET_FS)
            tf_info = _plot_transfer_function(
                event_dir / f"transfer_fn_{comp}.png", comp, tf, cfg,
            )
            f1, _ = _peak_freq(tf[0], tf[1], fmin=f_emp_min, fmax=f_emp_max)
            tf_info["_f1"] = f1
            zeta = _damping_from_raw_H(ncree_post, struct_post, TARGET_FS, f1)
            tf_info["_zeta"] = zeta

            # ---- Segment-based operational features ----
            segs = _event_segments(struct_full, TARGET_FS, pre_event_sec=PRE_EVENT_SEC)
            if segs is None:
                segs = {
                    "t_first": int(PRE_EVENT_SEC * TARGET_FS),
                    "t5":  int(PRE_EVENT_SEC * TARGET_FS),
                    "t95": len(struct_full) - 1,
                    "t99": len(struct_full) - 1,
                    "t_first_sec": 0.0, "t5_sec": 0.0,
                    "t95_sec": POST_EVENT_SEC, "t99_sec": POST_EVENT_SEC,
                    "arias_total": float("nan"),
                }
            min_pre_n = int(10 * TARGET_FS)
            ipre_end = segs["t_first"]
            if ipre_end < min_pre_n:
                ipre_end = max(min_pre_n, int(PRE_EVENT_SEC * TARGET_FS) - int(2 * TARGET_FS))
            f_iapp, _ = _modal_in_segment(struct_full, TARGET_FS,
                                          {"a": 0, "b": ipre_end},
                                          "a", "b", f_emp_min, f_emp_max)
            f_min, t_min = _f_min_strong_motion(
                struct_full, TARGET_FS, segs,
                f_min_lo, f_min_hi, f_center=f_iapp, mode="asymmetric")
            # v5 reviewer §3: symmetric-bandpass bias check on the same record.
            f_min_sym, _ = _f_min_strong_motion(
                struct_full, TARGET_FS, segs,
                f_min_lo, f_min_hi, f_center=f_iapp, mode="symmetric")
            f_99, _ = _modal_in_segment(struct_full, TARGET_FS,
                                        {"a": segs["t95"], "b": segs["t99"]},
                                        "a", "b", f_emp_min, f_emp_max)
            if (f_iapp is not None and not np.isnan(f_iapp) and f_iapp > 0
                    and f_min_sym is not None and not np.isnan(f_min_sym)):
                f_min_diff_pct = (f_min_sym - f_min) / f_iapp * 100.0
            else:
                f_min_diff_pct = float("nan")
            operational = {
                "f_iapp": f_iapp, "f_min": f_min, "f_99app": f_99,
                "f_min_sym": f_min_sym, "f_min_diff_pct": f_min_diff_pct,
                "t_first_arrival_sec": segs["t_first_sec"],
                "t5_sec": segs["t5_sec"], "t95_sec": segs["t95_sec"],
                "t99_sec": segs["t99_sec"], "arias_total": segs["arias_total"],
            }

            # ---- Time-varying TF (sliding-window deconvolution, §2.1) ----
            # The event-averaged TF mixes pre-event linear-elastic, coseismic
            # nonlinear, and post-event recovery regimes, contradicting the LTI
            # assumption. NCREE has no pre-event samples, so we slide on the
            # post-event [0, 120] s window only.
            try:
                tv = _sliding_window_tf(
                    ncree_post, struct_post, TARGET_FS,
                    f_emp_min, f_emp_max,
                    window_sec=10.0, overlap_sec=5.0,
                    t_offset_sec=0.0,           # ncree_post starts at NCREE trigger
                )
                if tv is not None:
                    _plot_sliding_tf(
                        event_dir / f"transfer_fn_tv_{comp}.png", comp, tv, cfg,
                        segs=segs,
                    )
            except Exception as exc:
                logging.error("sliding-window TF %s %s: %s",
                              event.event_id, comp, exc)

            try:
                feat = _cwt_response(struct_full, TARGET_FS, f_emp_min, f_emp_max,
                                     pre_event_sec=PRE_EVENT_SEC)
            except ValueError as exc:
                rows.append(_row(event, meta, cfg, comp, None, tf_info, None,
                                 operational, f"feat_fail:{exc}"))
                continue

            ops = {"f_iapp": f_iapp, "f_min": f_min, "f_99app": f_99, "t_min": t_min}
            _plot_cwt_annotated(
                feat, event_dir / f"cwt_{comp}.png",
                title=f"{event.event_id} {comp}  {sid} ({cfg.sensor_floor_label})",
                segs=segs, ops=ops,
                label_pre=f"{sid} {cfg.sensor_floor_label}", cfg=cfg, y_max=15.0,
            )

            struct_key = f"{sid}({cfg.sensor_floor_label})"
            fft_peaks = _plot_fft(
                event_dir / f"fft_{comp}.png", TARGET_FS, comp,
                [("NCREE", ncree_post), (struct_key, struct_post)], cfg,
            )

            _plot_raw(
                event_dir / f"raw_{comp}.png", TARGET_FS, comp,
                [
                    ("NCREE", t_axis, ncree_full),
                    (struct_key, t_axis, struct_full),
                ], cfg, segments=segs,
            )

            rows.append(_row(event, meta, cfg, comp, feat, tf_info, fft_peaks,
                             operational, "OK"))
        return event.event_id, rows, None
    except Exception as exc:
        return event.event_id, rows, f"{type(exc).__name__}: {exc}"
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Summary plots (v3 verbatim, with cfg-driven labels)
# ---------------------------------------------------------------------------
def _rolling_trend(times, values, window=30):
    import pandas as pd
    t_arr = np.asarray(times.values if hasattr(times, "values") else times)
    v_arr = np.asarray(values.values if hasattr(values, "values") else values,
                       dtype=float)
    s = pd.Series(v_arr, index=t_arr).dropna().sort_index()
    if len(s) < max(5, window // 2):
        return s.index.values, s.values
    rolled = s.rolling(window=window, min_periods=max(3, window // 4),
                       center=True).median()
    return rolled.index.values, rolled.values


def _linear_fit(times_pd, y_pd, *, hac_lags: int | None = None):
    """OLS slope-per-year fit with optional Newey-West HAC standard errors.

    Reviewer §3.1: aftershock clustering means residuals are time-correlated,
    so plain OLS standard errors are biased. Use Newey-West HAC with lag
    L ≈ floor(4·(T/100)^(2/9)) when ``hac_lags`` is None.
    """
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats as scs

    t_arr = pd.to_datetime(times_pd).values
    y_arr = pd.to_numeric(y_pd, errors="coerce").values
    m = ~np.isnan(y_arr)
    if m.sum() < 10:
        return None
    t_used = t_arr[m]
    y_used = y_arr[m]
    t0 = t_used.min()
    x = (t_used - t0).astype("timedelta64[s]").astype(float) / (365.25 * 24 * 3600)
    X = sm.add_constant(x)

    ols = sm.OLS(y_used, X).fit()
    if hac_lags is None:
        # Newey-West rule of thumb (Newey & West 1994 ECMA, Stock & Watson)
        hac_lags = max(1, int(np.floor(4 * (len(x) / 100.0) ** (2.0 / 9.0))))
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)

    intercept = float(ols.params[0])
    slope = float(ols.params[1])
    p_ols = float(ols.pvalues[1])
    p_hac = float(hac.pvalues[1])
    se_hac = float(hac.bse[1])
    r2 = float(ols.rsquared)
    n = int(len(x))

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_fit = intercept + slope * x_grid
    # CI uses HAC standard error: SE(intercept) + x·SE(slope) under the
    # diagonal approximation (cov between intercept and slope is small for
    # demeaned regressors and good enough for visualisation).
    se_inter = float(hac.bse[0])
    se_y = np.sqrt(se_inter ** 2 + (x_grid * se_hac) ** 2)
    tcrit = scs.t.ppf(0.975, df=max(1, n - 2))
    ci_lo = y_fit - tcrit * se_y
    ci_hi = y_fit + tcrit * se_y
    tt = t0 + (x_grid * 365.25 * 24 * 3600 * 1e9).astype("timedelta64[ns]")
    return {
        "tt": tt, "yy": y_fit, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "slope_per_yr": slope,
        "p_value": p_hac,        # primary p-value is HAC; OLS kept as p_ols
        "p_ols":   p_ols,
        "intercept": intercept,
        "r2": r2,
        "n": n,
        "hac_lags": hac_lags,
    }


def _bh_fdr(pvals: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted q-values.

    Returns q[i] aligned with the input p[i] order. NaNs in input are kept
    as NaN in output.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out.tolist()
    pf = p[finite]
    n = len(pf)
    order = np.argsort(pf)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q_sorted = pf[order] * n / np.arange(1, n + 1)
    # Enforce monotonicity (BH step-up)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(n)
    q[order] = q_sorted
    out[finite] = np.clip(q, 0.0, 1.0)
    return out.tolist()


def _plot_summary_one(csv_path, out_dir, kind, cfg: StationConfig):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["t"] = pd.to_datetime(df["utc_time"])

    sid = cfg.station_id
    flr = cfg.sensor_floor_label

    if kind == "cwt":
        # Short labels for legend; long-form definitions live in README.md.
        series = [
            ("f_iapp",  "tab:blue",   "f_iapp"),
            ("f_min",   "tab:orange", "f_min"),
            ("f_99app", "tab:green",  "f_99app"),
        ]
        title_root = f"{sid} ({flr}) — Operational Modal Frequencies"
    else:
        # §2.7: tf_peak_freq is the PRIMARY metric (0.4–15 Hz unconstrained);
        # tf_f1 (code-band-restricted) is reference only. See README.md.
        series = [
            ("tf_peak_freq", "black",      "TF peak"),
            ("tf_f1",        "tab:purple", "f₁ (code-band)"),
        ]
        title_root = f"{sid} ({flr}) — Transfer-Function Peaks"

    trend_colors = {
        "tab:blue":   "#08306b", "tab:orange": "#7f2704",
        "tab:green":  "#00441b", "black":      "#d62728",
        "tab:purple": "#4a148c",
    }

    def _marker(col):
        if col == "tf_peak_freq": return "x"
        if col == "tf_f1":        return "*"
        return "o"

    def _scatter_layer(ax, sub):
        for col, color, label in series:
            y = pd.to_numeric(sub[col], errors="coerce")
            ax.scatter(sub["t"].values, y.values, c=color, s=22, alpha=0.45,
                       marker=_marker(col), label=label, zorder=3,
                       edgecolors="none")

    # Long definitions and SSI limitation moved to README.md / ssi_estimates.md;
    # figure footnote is a single pointer.
    footnote = "Definitions & limitations: see README.md and ssi_estimates.md"

    def _finalize(fig, axes, subtitle, out_path, *, legend_ncol=3):
        for ax in axes:
            ax.grid(alpha=0.3)
            leg = ax.legend(loc="upper right", fontsize=8,
                            framealpha=0.90, ncol=legend_ncol,
                            handlelength=1.6, columnspacing=1.0,
                            borderpad=0.3, labelspacing=0.3)
            leg.set_zorder(20)
        axes[-1].set_xlabel("Event Time (UTC)")
        fig.suptitle(f"{title_root} — {subtitle}", fontsize=13)
        fig.text(0.5, 0.01, footnote, fontsize=8, color="#666666", ha="center")
        # Leave only a sliver of bottom margin for the footnote; plot fills the rest.
        fig.tight_layout(rect=(0, 0.03, 1.0, 0.96))
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].sort_values("t")
        _draw_empirical_band(ax, cfg, axis="y")
        _scatter_layer(ax, sub)
        ax.set_ylabel(f"{comp}  Freq (Hz)")
    _finalize(fig, axes, "Event Scatter", out_dir / f"summary_{kind}_scatter.png")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].sort_values("t")
        _draw_empirical_band(ax, cfg, axis="y")
        _scatter_layer(ax, sub)
        for col, color, label in series:
            y = pd.to_numeric(sub[col], errors="coerce")
            tt, yy = _rolling_trend(sub["t"], y, window=30)
            # No label here — keep legend to the bare series markers; the
            # rolling line is implicit from the colour match.
            ax.plot(tt, yy, color=trend_colors[color], linewidth=2.6,
                    alpha=0.95, zorder=10, solid_capstyle="round")
        ax.set_ylabel(f"{comp}  Freq (Hz)")
    _finalize(fig, axes, "Scatter + Rolling-Median Trend (window n=30)",
              out_dir / f"summary_{kind}_rolling.png")

    # --- Full-period OLS with HAC + BH FDR -----------------------------
    # Collect HAC p-values across (component × series) cells, then apply BH FDR.
    cells_full: list[tuple[str, str, dict]] = []   # (comp, label, lf-dict)
    for comp in ("E", "N", "Z"):
        sub = df[df["comp"] == comp].sort_values("t")
        for col, color, label in series:
            y = pd.to_numeric(sub[col], errors="coerce")
            lf = _linear_fit(sub["t"], y)
            cells_full.append((comp, col, lf, color, label))
    pvals_full = [c[2]["p_value"] if c[2] is not None else float("nan") for c in cells_full]
    qvals_full = _bh_fdr(pvals_full)
    cell_q_full = {(c[0], c[1]): qvals_full[i] for i, c in enumerate(cells_full)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].sort_values("t")
        _draw_empirical_band(ax, cfg, axis="y")
        _scatter_layer(ax, sub)
        stat_lines = []
        for col, color, label in series:
            lf = next((c[2] for c in cells_full
                       if c[0] == comp and c[1] == col), None)
            if lf is None:
                continue
            q = cell_q_full.get((comp, col), float("nan"))
            tc = trend_colors[color]
            ax.fill_between(lf["tt"], lf["ci_lo"], lf["ci_hi"],
                            color=tc, alpha=0.18, zorder=5, linewidth=0)
            ax.plot(lf["tt"], lf["yy"], color=tc, linewidth=2.4,
                    linestyle="-", alpha=0.95, zorder=10)
            sig = "✓" if (np.isfinite(q) and q < 0.05) else "✗"
            stat_lines.append(
                f"{label:<8s} {lf['slope_per_yr']*1000:+5.1f} mHz/yr  "
                f"q={q:.2g} {sig}  n={lf['n']}"
            )
        if stat_lines:
            ax.text(0.995, 0.03, "\n".join(stat_lines),
                    transform=ax.transAxes, fontsize=7, family="monospace",
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#888888", alpha=0.92), zorder=15)
        ax.set_ylabel(f"{comp}  Freq (Hz)")
    _finalize(fig, axes,
              "OLS + Newey-West HAC (BH-FDR q-values, 95% CI)",
              out_dir / f"summary_{kind}_ols.png")

    import pandas as pd
    cutoff = pd.Timestamp(MAINSHOCK_UTC)
    # --- Segmented HAC fits + BH FDR across all 12 cells (3 comp × 2 series × pre/post)
    cells_seg: list[tuple[str, str, str, dict, str, str]] = []
    for comp in ("E", "N", "Z"):
        sub = df[df["comp"] == comp].sort_values("t")
        pre_mask = sub["t"] < cutoff
        post_mask = sub["t"] >= cutoff
        for col, color, label in series:
            y_all = pd.to_numeric(sub[col], errors="coerce")
            cells_seg.append((comp, col, "PRE",
                              _linear_fit(sub.loc[pre_mask, "t"], y_all[pre_mask]),
                              color, label))
            cells_seg.append((comp, col, "POST",
                              _linear_fit(sub.loc[post_mask, "t"], y_all[post_mask]),
                              color, label))
    pvals_seg = [c[3]["p_value"] if c[3] is not None else float("nan") for c in cells_seg]
    qvals_seg = _bh_fdr(pvals_seg)
    cell_q_seg = {(c[0], c[1], c[2]): qvals_seg[i] for i, c in enumerate(cells_seg)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].sort_values("t")
        _draw_empirical_band(ax, cfg, axis="y")
        _scatter_layer(ax, sub)
        ax.axvline(cutoff, color="red", linestyle="-.", linewidth=1.4,
                   alpha=0.7, zorder=6, label="M7.2 mainshock")
        stat_lines = []
        for col, color, label in series:
            tc = trend_colors[color]
            lf_pre = next((c[3] for c in cells_seg
                           if c[0] == comp and c[1] == col and c[2] == "PRE"), None)
            lf_post = next((c[3] for c in cells_seg
                            if c[0] == comp and c[1] == col and c[2] == "POST"), None)
            q_pre  = cell_q_seg.get((comp, col, "PRE"),  float("nan"))
            q_post = cell_q_seg.get((comp, col, "POST"), float("nan"))
            if lf_pre is not None:
                ax.fill_between(lf_pre["tt"], lf_pre["ci_lo"], lf_pre["ci_hi"],
                                color=tc, alpha=0.15, zorder=5, linewidth=0)
                ax.plot(lf_pre["tt"], lf_pre["yy"], color=tc, linewidth=2.2,
                        linestyle="-", alpha=0.95, zorder=10)
                sig = "✓" if (np.isfinite(q_pre) and q_pre < 0.05) else "✗"
                stat_lines.append(
                    f"{label:<8s} PRE  {lf_pre['slope_per_yr']*1000:+5.1f} mHz/yr  "
                    f"q={q_pre:.2g} {sig}  n={lf_pre['n']}"
                )
            if lf_post is not None:
                ax.fill_between(lf_post["tt"], lf_post["ci_lo"], lf_post["ci_hi"],
                                color=tc, alpha=0.15, zorder=5, linewidth=0)
                ax.plot(lf_post["tt"], lf_post["yy"], color=tc, linewidth=2.2,
                        linestyle="--", alpha=0.95, zorder=10)
                sig = "✓" if (np.isfinite(q_post) and q_post < 0.05) else "✗"
                stat_lines.append(
                    f"{label:<8s} POST {lf_post['slope_per_yr']*1000:+5.1f} mHz/yr  "
                    f"q={q_post:.2g} {sig}  n={lf_post['n']}"
                )
        if stat_lines:
            ax.text(0.995, 0.03, "\n".join(stat_lines),
                    transform=ax.transAxes, fontsize=7, family="monospace",
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#888888", alpha=0.92), zorder=15)
        ax.set_ylabel(f"{comp}  Freq (Hz)")
    _finalize(fig, axes,
              "Segmented OLS+HAC — Pre vs Post M7.2 (solid=PRE, dashed=POST)",
              out_dir / f"summary_{kind}_ols_segmented.png")


def plot_concept_diagram(event, out_path, cfg: StationConfig):
    import matplotlib.pyplot as plt
    _, f_emp_min, f_emp_max = code_band(cfg)
    f_emp = empirical_frequency(cfg)
    f_min_lo = f_emp * 0.5
    f_min_hi = f_emp_max
    sid = cfg.station_id
    flr = cfg.sensor_floor_label

    streams = _load_traces(event, cfg)
    t_ref = streams["NCREE"]["Z"].stats.starttime
    fs = TARGET_FS
    comp = "N"
    struct_tr = _resample_to(streams[sid][comp], fs)
    struct_full = _slice_window(struct_tr, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, fs)
    finite = ~np.isnan(struct_full)
    if finite.any():
        struct_full[finite] -= struct_full[finite].mean()
    t_axis = np.arange(len(struct_full)) / fs - PRE_EVENT_SEC

    segs = _event_segments(struct_full, fs, pre_event_sec=PRE_EVENT_SEC)
    if segs is None:
        return
    f_iapp, _ = _modal_in_segment(struct_full, fs,
                                  {"a": 0, "b": segs["t_first"]},
                                  "a", "b", f_emp_min, f_emp_max)
    f_min, t_min = _f_min_strong_motion(struct_full, fs, segs,
                                        f_min_lo, f_min_hi, f_center=f_iapp)
    f_99, _ = _modal_in_segment(struct_full, fs,
                                {"a": segs["t95"], "b": segs["t99"]},
                                "a", "b", f_emp_min, f_emp_max)

    if not np.isnan(f_iapp) and f_iapp > 0:
        lo = max(0.3, f_iapp / F_MIN_BAND_DOWN_ASYM); hi = min(fs / 2 - 0.1, f_iapp * F_MIN_BAND_UP_ASYM)
        sos = butter(6, [lo, hi], btype="band", fs=fs, output="sos")
        bp = sosfiltfilt(sos, np.nan_to_num(struct_full))
        analytic = hilbert(bp)
        phase = np.unwrap(np.angle(analytic))
        if_curve = np.diff(phase) / (2 * np.pi) * fs
        if_curve = np.append(if_curve, if_curve[-1])
        if_curve = uniform_filter1d(if_curve, size=int(fs))
    else:
        if_curve = np.full_like(struct_full, np.nan)

    fig, (ax_a, ax_f) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    ax_a.plot(t_axis, struct_full, color="black", linewidth=0.6, zorder=3)
    ax_a.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax_a.axvspan(t_axis[0], segs["t_first_sec"], color="tab:blue", alpha=0.15,
                 label="f_iapp window (pre-event ambient)")
    ax_a.axvspan(segs["t5_sec"], segs["t95_sec"], color="tab:orange", alpha=0.15,
                 label="DSM (Arias 5–95%) → f_min")
    ax_a.axvspan(segs["t95_sec"], segs["t99_sec"], color="tab:green", alpha=0.20,
                 label="Post-DSM (95–99% Arias) → f_99app")
    for tval, lbl, c in [
        (segs["t_first_sec"], "STA/LTA arrival", "tab:red"),
        (segs["t5_sec"],  "Arias 5%",  "tab:purple"),
        (segs["t95_sec"], "Arias 95%", "tab:purple"),
        (segs["t99_sec"], "Arias 99%", "tab:gray"),
    ]:
        ax_a.axvline(tval, color=c, linestyle="--", linewidth=1.0, alpha=0.75)
    ax_a.set_ylabel(f"{sid} ({flr}) N-axis\nAcceleration (gal)")
    ax_a.grid(alpha=0.3)
    ax_a.set_title(f"Operational Modal-Frequency Definitions — Example: {event.event_id}",
                   fontsize=12)
    ax_a.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
                fontsize=8, framealpha=0.95)

    ax_f.plot(t_axis, if_curve, color="tab:orange", linewidth=0.8, alpha=0.6,
              label="Hilbert instantaneous freq (bandpass on f_iapp)")
    _draw_empirical_band(ax_f, cfg, axis="y", in_legend=True)
    pts = [
        ("f_iapp", segs["t_first_sec"] / 2, f_iapp, "tab:blue"),
        ("f_min", t_min if t_min is not None else (segs["t5_sec"] + segs["t95_sec"]) / 2,
         f_min, "tab:orange"),
        ("f_99app", (segs["t95_sec"] + segs["t99_sec"]) / 2, f_99, "tab:green"),
    ]
    for lbl, tx, fy, c in pts:
        if fy is None or np.isnan(fy):
            continue
        ax_f.scatter([tx], [fy], c=c, s=160, marker="o",
                     edgecolor="black", linewidth=1.2, zorder=10)
        ax_f.annotate(f"{lbl}\n= {fy:.2f} Hz",
                      xy=(tx, fy), xytext=(8, 12),
                      textcoords="offset points",
                      fontsize=10, color=c, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                ec=c, alpha=0.9))
    if not (np.isnan(f_iapp) or np.isnan(f_min)):
        ax_f.annotate("",
                      xy=(t_min if t_min else (segs["t5_sec"] + segs["t95_sec"]) / 2, f_min),
                      xytext=(segs["t_first_sec"] / 2, f_iapp),
                      arrowprops=dict(arrowstyle="->", color="red", lw=1.8,
                                      alpha=0.6))
        drop_pct = (f_iapp - f_min) / f_iapp * 100
        midx = (segs["t_first_sec"] / 2 + (segs["t5_sec"] + segs["t95_sec"]) / 2) / 2
        midy = (f_iapp + f_min) / 2
        ax_f.text(midx, midy + 0.05,
                  f"coseismic\nsoftening {drop_pct:.0f}%",
                  color="red", fontsize=9, fontweight="bold", ha="center")
    if not (np.isnan(f_99) or np.isnan(f_min) or np.isnan(f_iapp)):
        ax_f.text(0.99, 0.05,
                  f"Recovery (f_99 vs pre-event): "
                  f"{(f_99/f_iapp-1)*100:+.0f}%",
                  transform=ax_f.transAxes, ha="right",
                  fontsize=9, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    ax_f.set_xlabel("Time (s) — t=0 at NCREE trigger")
    ax_f.set_ylabel("1st-mode\nFrequency (Hz)")
    ax_f.set_ylim(max(0.5, f_emp_min * 0.6), f_emp_max * 1.3)
    ax_f.grid(alpha=0.3)
    ax_f.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
                fontsize=8, framealpha=0.95)

    fig.tight_layout(rect=(0, 0, 0.78, 1.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_softening_vs_pga(csv_path: Path, out_dir: Path, cfg: StationConfig):
    """§3.3 (reviewer): dose-response of coseismic softening vs PGA.

    Plots (f_iapp - f_min) / f_iapp vs pga_0m on log-x for each component,
    with OLS fit on log(PGA). Pooling all magnitudes into a single median
    obscures this trend; this figure exposes it.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["t"] = pd.to_datetime(df["utc_time"])
    df["f_iapp_n"] = pd.to_numeric(df["f_iapp"], errors="coerce")
    df["f_min_n"]  = pd.to_numeric(df["f_min"],  errors="coerce")
    # v5: prefer pga_0m if catalog metadata was populated, otherwise fall
    # back to arias_total (always data-derived). Earlier code died with
    # "no data" in every panel because the CWA catalog merge wasn't done
    # and pga_0m/pga_30m were all empty.
    pga_raw = pd.to_numeric(df.get("pga_0m"), errors="coerce")
    if pga_raw.notna().sum() < 10 or (pga_raw > 0).sum() < 10:
        df["pga_n"] = pd.to_numeric(df.get("arias_total"), errors="coerce")
        intensity_label = "arias_total  (m/s, log scale)  [PGA fallback]"
    else:
        df["pga_n"] = pga_raw
        intensity_label = "pga_0m  (gal, log scale)"
    df["softening"] = (df["f_iapp_n"] - df["f_min_n"]) / df["f_iapp_n"]

    cutoff = pd.Timestamp(MAINSHOCK_UTC)
    sid = cfg.station_id
    flr = cfg.sensor_floor_label

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp]
        sub = sub.dropna(subset=["softening", "pga_n"])
        sub = sub[(sub["pga_n"] > 0) & (sub["softening"] > -0.5) & (sub["softening"] < 0.95)]
        if sub.empty:
            ax.text(0.5, 0.5, f"{comp}: no data", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        pre  = sub[sub["t"] <  cutoff]
        post = sub[sub["t"] >= cutoff]
        ax.scatter(pre["pga_n"],  pre["softening"]  * 100, s=20, alpha=0.55,
                   c="tab:blue", label=f"pre  (n={len(pre)})", edgecolors="none")
        ax.scatter(post["pga_n"], post["softening"] * 100, s=20, alpha=0.55,
                   c="tab:red",  label=f"post (n={len(post)})", edgecolors="none")
        x = np.log10(sub["pga_n"].values)
        y = sub["softening"].values * 100
        stat_txt = ""
        if len(x) >= 10:
            res = stats.linregress(x, y)
            xg = np.linspace(x.min(), x.max(), 100)
            ax.plot(10 ** xg, res.intercept + res.slope * xg,
                    color="black", linewidth=2.0, alpha=0.85, label="pooled OLS")
            sig = "✓" if res.pvalue < 0.05 else "✗"
            stat_txt = (f"slope = {res.slope:+.2f} %/decade  "
                        f"p={res.pvalue:.2g} {sig}  R²={res.rvalue**2:.2f}  n={len(x)}")
        if stat_txt:
            # Bottom-left: low-PGA / low-softening corner is usually sparse.
            ax.text(0.005, 0.03, stat_txt, transform=ax.transAxes, fontsize=7,
                    family="monospace", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#888888", alpha=0.92), zorder=15)
        ax.axhline(0, color="gray", linewidth=0.7, alpha=0.5)
        ax.set_xscale("log")
        ax.set_ylabel(f"{comp}\nsoftening [%]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=3)
    axes[-1].set_xlabel(intensity_label)
    fig.suptitle(f"{sid} ({flr}) — Coseismic Softening vs PGA", fontsize=13)
    fig.text(0.5, 0.01,
             "softening = (f_iapp − f_min)/f_iapp.  See README.md for SSI & input notes.",
             fontsize=8, color="#666666", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1.0, 0.96))
    fig.savefig(out_dir / "summary_softening_vs_pga.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _baseline_comparison(csv_path: Path, out_dir: Path, cfg: StationConfig,
                         window_days: int = 90, n_boot: int = 2000):
    """§3.4 (reviewer): pre vs post mainshock long-term baselines.

    Compares median f_iapp in two ±window_days windows around the M7.2
    mainshock (2024-04-02 23:58 UTC). f_99app/f_iapp inside a single
    event measures *short-term recovery* (seconds), not *long-term aging*
    (months). True permanent stiffness change must compare baselines from
    different events on each side of the mainshock.

    Outputs:
      baseline_comparison.json  — numeric summary
      baseline_comparison.png   — f_iapp time series with both windows shaded
    """
    import json
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["t"] = pd.to_datetime(df["utc_time"])
    df["f_iapp_n"] = pd.to_numeric(df["f_iapp"], errors="coerce")
    cutoff = pd.Timestamp(MAINSHOCK_UTC)
    pre_window  = (cutoff - pd.Timedelta(days=window_days), cutoff)
    post_window = (cutoff, cutoff + pd.Timedelta(days=window_days))

    rng = np.random.default_rng(20240403)
    summary: dict = {
        "mainshock_utc": MAINSHOCK_UTC,
        "window_days": window_days,
        "bootstrap_iterations": n_boot,
        "comp": {},
    }
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].dropna(subset=["f_iapp_n"]).sort_values("t")
        pre = sub[(sub["t"] >= pre_window[0])  & (sub["t"] < pre_window[1])]["f_iapp_n"].values
        post = sub[(sub["t"] >= post_window[0]) & (sub["t"] < post_window[1])]["f_iapp_n"].values

        def _boot_median_ci(arr):
            if len(arr) < 5:
                return float("nan"), float("nan"), float("nan")
            boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
            meds = np.median(boot, axis=1)
            return float(np.median(arr)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

        pre_med, pre_lo, pre_hi = _boot_median_ci(pre)
        post_med, post_lo, post_hi = _boot_median_ci(post)
        if np.isnan(pre_med) or np.isnan(post_med):
            delta_hz = float("nan"); delta_pct = float("nan")
        else:
            delta_hz = post_med - pre_med
            delta_pct = delta_hz / pre_med * 100

        # 2-sample bootstrap of the median difference for significance
        sig_note = ""
        if not (np.isnan(pre_med) or np.isnan(post_med)):
            boot_diffs = (np.median(rng.choice(post, size=(n_boot, len(post)), replace=True), axis=1)
                          - np.median(rng.choice(pre,  size=(n_boot, len(pre)),  replace=True), axis=1))
            ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
            crosses_zero = ci_lo <= 0 <= ci_hi
            sig_note = f"  Δ CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] Hz  ({'NOT sig' if crosses_zero else 'sig at 95%'})"
        else:
            ci_lo = ci_hi = float("nan")

        summary["comp"][comp] = {
            "pre":  {"n": int(len(pre)),  "median_hz": pre_med,  "ci95": [pre_lo,  pre_hi]},
            "post": {"n": int(len(post)), "median_hz": post_med, "ci95": [post_lo, post_hi]},
            "delta_hz": delta_hz,
            "delta_pct": delta_pct,
            "delta_ci95": [float(ci_lo), float(ci_hi)],
        }

        ax.scatter(sub["t"], sub["f_iapp_n"], s=10, alpha=0.40,
                   c="#888888", edgecolors="none", label="f_iapp")
        ax.axvspan(*pre_window,  color="tab:blue", alpha=0.10)
        ax.axvspan(*post_window, color="tab:red",  alpha=0.10)
        if not np.isnan(pre_med):
            ax.hlines(pre_med, *pre_window,  colors="tab:blue", linewidth=2.8,
                      label=f"pre  (n={len(pre)})")
        if not np.isnan(post_med):
            ax.hlines(post_med, *post_window, colors="tab:red", linewidth=2.8,
                      label=f"post (n={len(post)})")
        ax.axvline(cutoff, color="red", linestyle="-.", linewidth=1.2, alpha=0.7)
        ax.set_ylabel(f"{comp}\nf_iapp (Hz)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=3)
        if not np.isnan(pre_med) and not np.isnan(post_med):
            stat_txt = (
                f"pre  median = {pre_med:.3f} [{pre_lo:.3f}, {pre_hi:.3f}] Hz\n"
                f"post median = {post_med:.3f} [{post_lo:.3f}, {post_hi:.3f}] Hz\n"
                f"Δ = {delta_hz:+.3f} Hz ({delta_pct:+.1f}%)  "
                f"CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] Hz  "
                f"{'✓ sig' if not (ci_lo <= 0 <= ci_hi) else '✗ n.s.'}"
            )
            # Upper-left: pre-mainshock data is sparse (n=15 over 90 d)
            # leaving plenty of empty space at the top-left.
            ax.text(0.005, 0.97, stat_txt, transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#888888", alpha=0.92), zorder=15)
    axes[-1].set_xlabel("Event Time (UTC)")
    fig.suptitle(
        f"{cfg.station_id} ({cfg.sensor_floor_label}) — Pre vs Post M7.2 "
        f"f_iapp Baseline (±{window_days} d, bootstrap 95% CI)",
        fontsize=13,
    )
    fig.text(0.5, 0.01, "See README.md for SSI & methodology notes.",
             fontsize=8, color="#666666", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1.0, 0.96))
    fig.savefig(out_dir / "baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "baseline_comparison.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def _plot_stl_detrend(csv_path: Path, out_dir: Path, cfg: StationConfig,
                      monthly_period: int = 12):
    """§4.2 (reviewer): STL seasonal decomposition of f_iapp.

    Reviewer noted that the −5 to −10 mHz/yr slope is comparable to seasonal
    temperature-driven f₁ swings (~3–5% range, Clinton 2006). We resample
    f_iapp to a monthly median and fit STL with period=12 months to expose
    the seasonal cycle separately from any long-term trend.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from statsmodels.tsa.seasonal import STL

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["t"] = pd.to_datetime(df["utc_time"])
    cutoff = pd.Timestamp(MAINSHOCK_UTC)

    # 3 cols (observed / trend / seasonal). Dropped the residual column —
    # high-frequency noise is already obvious in the observed scatter and adds
    # nothing for the seasonal-vs-trend question. Wider figure gives the
    # x-axis date labels room to breathe.
    fig, axes = plt.subplots(3, 3, figsize=(17, 10), sharex="col")
    decomp_summary: dict = {}
    for row, comp in enumerate(("E", "N", "Z")):
        sub = df[df["comp"] == comp].copy()
        sub["f"] = pd.to_numeric(sub["f_iapp"], errors="coerce")
        sub = sub.dropna(subset=["f"]).set_index("t").sort_index()
        # monthly median, drop empty months
        monthly = sub["f"].resample("MS").median().dropna()
        if len(monthly) < 2 * monthly_period + 4:
            for col in range(3):
                axes[row, col].text(0.5, 0.5,
                                    f"{comp}: only {len(monthly)} monthly bins "
                                    f"(need ≥ {2*monthly_period+4})",
                                    transform=axes[row, col].transAxes,
                                    ha="center", va="center", fontsize=9)
            decomp_summary[comp] = {"status": "insufficient_data",
                                    "n_months": int(len(monthly))}
            continue
        try:
            stl = STL(monthly, period=monthly_period, robust=True).fit()
        except Exception as exc:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, f"{comp}: STL failed: {exc}",
                                    transform=axes[row, col].transAxes,
                                    ha="center", va="center", fontsize=9)
            decomp_summary[comp] = {"status": "stl_failed", "error": str(exc)}
            continue

        season_amp = float(stl.seasonal.max() - stl.seasonal.min())
        trend_change = float(stl.trend.iloc[-1] - stl.trend.iloc[0])
        trend_yrs = (monthly.index[-1] - monthly.index[0]).days / 365.25
        decomp_summary[comp] = {
            "status": "ok",
            "n_months": int(len(monthly)),
            "seasonal_peak_to_peak_hz": season_amp,
            "trend_change_hz": trend_change,
            "trend_slope_per_yr": trend_change / trend_yrs if trend_yrs > 0 else float("nan"),
        }

        # Column 0: observed monthly medians
        axes[row, 0].scatter(monthly.index, monthly.values, s=22, c="tab:blue",
                             alpha=0.7, edgecolors="none")
        axes[row, 0].plot(monthly.index, monthly.values, color="tab:blue",
                          linewidth=0.6, alpha=0.5)
        axes[row, 0].axvline(cutoff, color="red", linestyle="-.", linewidth=1.0,
                             alpha=0.6)
        axes[row, 0].set_ylabel(f"{comp}\nobserved f_iapp (Hz)")
        axes[row, 0].grid(alpha=0.3)
        # Column 1: trend
        axes[row, 1].plot(stl.trend.index, stl.trend.values, color="tab:red", linewidth=2)
        axes[row, 1].axvline(cutoff, color="red", linestyle="-.", linewidth=1.0,
                             alpha=0.6)
        axes[row, 1].set_ylabel("STL trend")
        axes[row, 1].grid(alpha=0.3)
        axes[row, 1].text(0.02, 0.95,
                          f"Δtrend = {trend_change*1000:+.1f} mHz "
                          f"over {trend_yrs:.1f} yr",
                          transform=axes[row, 1].transAxes, fontsize=8,
                          va="top", bbox=dict(boxstyle="round,pad=0.25",
                                              fc="white", alpha=0.85))
        # Column 2: seasonal
        axes[row, 2].plot(stl.seasonal.index, stl.seasonal.values,
                          color="tab:green", linewidth=1.2)
        axes[row, 2].axhline(0, color="gray", linewidth=0.5)
        axes[row, 2].set_ylabel("STL seasonal")
        axes[row, 2].grid(alpha=0.3)
        axes[row, 2].text(0.02, 0.95,
                          f"peak-to-peak = {season_amp*1000:.1f} mHz",
                          transform=axes[row, 2].transAxes, fontsize=8,
                          va="top", bbox=dict(boxstyle="round,pad=0.25",
                                              fc="white", alpha=0.85))
    for col, ttl in enumerate(("Observed (monthly median)", "Trend",
                                "Seasonal (12-mo period)")):
        axes[0, col].set_title(ttl, fontsize=11)
    for col in range(3):
        axes[-1, col].set_xlabel("Time (UTC)")
        # Rotate dates so they don't overlap on the narrower 3-col layout.
        for lbl in axes[-1, col].get_xticklabels():
            lbl.set_rotation(35)
            lbl.set_ha("right")
    fig.suptitle(
        f"{cfg.station_id} ({cfg.sensor_floor_label}) — STL decomposition of f_iapp",
        fontsize=13,
    )
    fig.text(0.5, 0.01,
             "Seasonal cycle separated from secular trend. "
             "See README.md for interpretation notes.",
             fontsize=8, color="#666666", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1.0, 0.96))
    fig.savefig(out_dir / "summary_stl_detrend.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    import json
    with (out_dir / "stl_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(decomp_summary, fh, indent=2, ensure_ascii=False)


def _baseline_comparison_pga_matched(csv_path: Path, out_dir: Path,
                                     cfg: StationConfig, n_boot: int = 2000):
    """v5 reviewer §6: PGA-matched pre/post baseline.

    The unmatched bootstrap (`_baseline_comparison`) is contaminated by the
    fact that the pre-mainshock window is dominated by weak events while the
    post-mainshock window is dominated by strong aftershocks. We match each
    post-mainshock event to its nearest pre-mainshock counterpart on
    ``log10(pga_30m)`` and compare the median f_iapp within the matched
    pair set. A 2-sample bootstrap of the median difference gives the
    significance.

    The matching is one-to-many (each pre event may serve multiple post
    events); we keep only matched pairs where |Δlog10 PGA| <= 0.25 dex
    (i.e. PGA within a factor of ~1.8). Pairs outside that tolerance are
    discarded with a count reported in the JSON.
    """
    import json
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["t"] = pd.to_datetime(df["utc_time"])
    df["f_iapp_n"] = pd.to_numeric(df["f_iapp"], errors="coerce")
    # v5: prefer pga_30m if catalog metadata was populated, otherwise fall
    # back to arias_total (a data-derived shaking-intensity proxy that is
    # always available even when the CWB catalog merge produces no PGA).
    pga = pd.to_numeric(df.get("pga_30m"), errors="coerce")
    if pga.notna().sum() < 10 or (pga > 0).sum() < 10:
        proxy = "arias_total"
        intensity = pd.to_numeric(df.get("arias_total"), errors="coerce")
    else:
        proxy = "pga_30m"
        intensity = pga
    df["intensity"] = intensity
    df["log_intensity"] = np.log10(intensity.where(intensity > 0))
    cutoff = pd.Timestamp(MAINSHOCK_UTC)
    rng = np.random.default_rng(20260403)

    summary: dict = {
        "mainshock_utc": MAINSHOCK_UTC,
        "method": f"nearest-neighbor on log10({proxy}), tolerance 0.25 dex",
        "intensity_proxy": proxy,
        "bootstrap_iterations": n_boot,
        "comp": {},
    }
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].dropna(subset=["f_iapp_n", "log_intensity"])
        pre = sub[sub["t"] < cutoff]
        post = sub[sub["t"] >= cutoff]
        if len(pre) < 3 or len(post) < 3:
            ax.text(0.5, 0.5, f"{comp}: insufficient matched events",
                    transform=ax.transAxes, ha="center")
            summary["comp"][comp] = {"status": "insufficient"}
            continue
        pre_lp = pre["log_intensity"].values
        pre_fi = pre["f_iapp_n"].values
        matched_pre_idx = []
        matched_post_idx = []
        pre_arr_idx = np.arange(len(pre))
        for j, lp in enumerate(post["log_intensity"].values):
            k = int(np.argmin(np.abs(pre_lp - lp)))
            if abs(pre_lp[k] - lp) <= 0.25:
                matched_pre_idx.append(int(pre_arr_idx[k]))
                matched_post_idx.append(j)
        if len(matched_post_idx) < 5:
            ax.text(0.5, 0.5, f"{comp}: only {len(matched_post_idx)} matched pairs",
                    transform=ax.transAxes, ha="center")
            summary["comp"][comp] = {"status": "few_matches",
                                     "n_matched": int(len(matched_post_idx))}
            continue
        f_pre_m = pre_fi[matched_pre_idx]
        f_post_m = post["f_iapp_n"].values[matched_post_idx]
        delta_med = float(np.median(f_post_m) - np.median(f_pre_m))
        delta_pct = float(delta_med / np.median(f_pre_m) * 100)
        boot = (np.median(rng.choice(f_post_m, size=(n_boot, len(f_post_m)),
                                      replace=True), axis=1)
                - np.median(rng.choice(f_pre_m,  size=(n_boot, len(f_pre_m)),
                                       replace=True), axis=1))
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        summary["comp"][comp] = {
            "status": "ok",
            "n_matched_pairs": int(len(matched_post_idx)),
            "pre_median_hz":  float(np.median(f_pre_m)),
            "post_median_hz": float(np.median(f_post_m)),
            "delta_hz": delta_med,
            "delta_pct": delta_pct,
            "delta_ci95": [float(ci_lo), float(ci_hi)],
        }
        # Scatter (pre vs post on x = log PGA, y = f_iapp)
        ax.scatter(pre["log_intensity"], pre["f_iapp_n"], s=18, c="tab:blue",
                   alpha=0.55, label=f"pre (n={len(pre)})")
        ax.scatter(post["log_intensity"], post["f_iapp_n"], s=18, c="tab:red",
                   alpha=0.55, label=f"post (n={len(post)})")
        # Connect matched pairs
        for ip, ipo in zip(matched_pre_idx, matched_post_idx):
            ax.plot([pre_lp[ip], post["log_intensity"].values[ipo]],
                    [pre_fi[ip], post["f_iapp_n"].values[ipo]],
                    color="black", alpha=0.10, linewidth=0.5)
        ax.set_ylabel(f"{comp}\nf_iapp (Hz)")
        ax.set_xlabel(f"log10 {proxy}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
        stat_txt = (f"matched pairs = {len(matched_post_idx)}  "
                    f"Δ = {delta_med:+.3f} Hz ({delta_pct:+.1f}%)  "
                    f"CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] "
                    f"{'sig' if not (ci_lo <= 0 <= ci_hi) else 'n.s.'}")
        ax.text(0.005, 0.97, stat_txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#888888", alpha=0.92))
    fig.suptitle(
        f"{cfg.station_id} ({cfg.sensor_floor_label}) — PGA-matched pre vs "
        f"post M7.2 baseline (reviewer §3.5)", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.03, 1.0, 0.96))
    fig.savefig(out_dir / "baseline_comparison_pga_matched.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "baseline_comparison_pga_matched.json").open(
            "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def _baseline_comparison_deseasonalized(csv_path: Path, out_dir: Path,
                                        cfg: StationConfig,
                                        window_days: int = 90,
                                        n_boot: int = 2000):
    """v5 reviewer §10: pre/post M7.2 baseline AFTER removing the STL
    seasonal cycle (period = 12 months) from monthly-median f_iapp.

    The STL seasonal mean of each calendar month is subtracted from every
    event in that calendar month; the resulting "deseasonalized" series
    isolates trend + irregular and lets the pre/post comparison rule out
    the reviewer's environment-effect alternative explanation.
    """
    import json
    import matplotlib.pyplot as plt
    import pandas as pd
    from statsmodels.tsa.seasonal import STL

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["t"] = pd.to_datetime(df["utc_time"])
    df["f_iapp_n"] = pd.to_numeric(df["f_iapp"], errors="coerce")
    cutoff = pd.Timestamp(MAINSHOCK_UTC)
    rng = np.random.default_rng(20260404)
    summary: dict = {"mainshock_utc": MAINSHOCK_UTC,
                     "window_days": window_days,
                     "method": "STL period=12 monthly; calendar-month seasonal mean subtracted from each event",
                     "comp": {}}
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].dropna(subset=["f_iapp_n"]).copy()
        if sub.empty:
            summary["comp"][comp] = {"status": "empty"}
            continue
        monthly = (sub.set_index("t")["f_iapp_n"]
                       .resample("MS").median().dropna())
        if len(monthly) < 24:
            # Not enough monthly bins for STL period=12 -> just demean
            seasonal_by_month = pd.Series(0.0, index=range(1, 13))
            summary["comp"][comp] = {"status": "stl_skipped_short_series"}
        else:
            try:
                stl = STL(monthly, period=12, robust=True).fit()
                seasonal_by_month = (stl.seasonal.groupby(
                    stl.seasonal.index.month).mean())
            except Exception as exc:
                seasonal_by_month = pd.Series(0.0, index=range(1, 13))
                summary["comp"][comp] = {"status": f"stl_failed:{exc}"}
        # Subtract seasonal contribution from every event
        sub["seasonal"] = sub["t"].dt.month.map(seasonal_by_month).fillna(0.0)
        sub["f_deseason"] = sub["f_iapp_n"] - sub["seasonal"]
        pre_w  = (cutoff - pd.Timedelta(days=window_days), cutoff)
        post_w = (cutoff, cutoff + pd.Timedelta(days=window_days))
        pre  = sub[(sub["t"] >= pre_w[0])  & (sub["t"] < pre_w[1])]["f_deseason"].values
        post = sub[(sub["t"] >= post_w[0]) & (sub["t"] < post_w[1])]["f_deseason"].values
        if len(pre) < 3 or len(post) < 3:
            ax.text(0.5, 0.5, f"{comp}: insufficient events in ±{window_days} d windows",
                    transform=ax.transAxes, ha="center")
            summary["comp"].setdefault(comp, {})["status"] = "insufficient_window"
            continue
        pre_med = float(np.median(pre))
        post_med = float(np.median(post))
        delta = post_med - pre_med
        delta_pct = delta / pre_med * 100.0 if pre_med != 0 else float("nan")
        boot = (np.median(rng.choice(post, size=(n_boot, len(post)),
                                      replace=True), axis=1)
                - np.median(rng.choice(pre, size=(n_boot, len(pre)),
                                        replace=True), axis=1))
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        entry = summary["comp"].setdefault(comp, {})
        entry.update({
            "n_pre": int(len(pre)), "n_post": int(len(post)),
            "pre_median_deseason_hz": pre_med,
            "post_median_deseason_hz": post_med,
            "delta_hz": delta, "delta_pct": delta_pct,
            "delta_ci95": [float(ci_lo), float(ci_hi)],
        })
        ax.scatter(sub["t"], sub["f_deseason"], s=10, alpha=0.4,
                   c="#888888", edgecolors="none",
                   label="f_iapp (deseasonalized)")
        ax.axvspan(*pre_w,  color="tab:blue", alpha=0.10)
        ax.axvspan(*post_w, color="tab:red",  alpha=0.10)
        ax.hlines(pre_med, *pre_w,  colors="tab:blue", linewidth=2.6,
                  label=f"pre (n={len(pre)})")
        ax.hlines(post_med, *post_w, colors="tab:red", linewidth=2.6,
                  label=f"post (n={len(post)})")
        ax.axvline(cutoff, color="red", linestyle="-.", linewidth=1.2, alpha=0.7)
        ax.set_ylabel(f"{comp}\nf_iapp - seasonal (Hz)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=3)
        stat_txt = (
            f"pre  = {pre_med:.3f}   post = {post_med:.3f}   "
            f"Δ = {delta:+.3f} ({delta_pct:+.1f}%)   "
            f"CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] "
            f"{'sig' if not (ci_lo <= 0 <= ci_hi) else 'n.s.'}"
        )
        ax.text(0.005, 0.97, stat_txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#888888", alpha=0.92))
    axes[-1].set_xlabel("Event Time (UTC)")
    fig.suptitle(
        f"{cfg.station_id} ({cfg.sensor_floor_label}) — Pre vs Post M7.2, "
        f"STL-deseasonalized f_iapp (reviewer §3.6/§4.6)", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.03, 1.0, 0.96))
    fig.savefig(out_dir / "baseline_comparison_deseasonalized.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "baseline_comparison_deseasonalized.json").open(
            "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def _plot_fmin_symmetric_vs_asymmetric(csv_path: Path, out_dir: Path,
                                        cfg: StationConfig):
    """v5 reviewer §3: compare softening % computed with the asymmetric vs
    symmetric event-adaptive bandpass. A systematic gap = estimator-band bias.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import json
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["f_iapp_n"] = pd.to_numeric(df["f_iapp"], errors="coerce")
    df["f_min_n"] = pd.to_numeric(df["f_min"], errors="coerce")
    df["f_min_sym_n"] = pd.to_numeric(df["f_min_sym"], errors="coerce")
    df = df.dropna(subset=["f_iapp_n", "f_min_n", "f_min_sym_n"])
    if df.empty:
        return
    df["soft_asym_pct"] = (df["f_iapp_n"] - df["f_min_n"]) / df["f_iapp_n"] * 100
    df["soft_sym_pct"]  = (df["f_iapp_n"] - df["f_min_sym_n"]) / df["f_iapp_n"] * 100

    summary = {"comp": {}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp]
        if len(sub) < 5:
            ax.text(0.5, 0.5, f"{comp}: n={len(sub)}",
                    transform=ax.transAxes, ha="center")
            continue
        diffs = sub["soft_asym_pct"] - sub["soft_sym_pct"]
        med_diff = float(diffs.median())
        ax.scatter(sub["soft_sym_pct"], sub["soft_asym_pct"], s=14, alpha=0.5,
                   edgecolors="none")
        lim = max(sub["soft_asym_pct"].max(), sub["soft_sym_pct"].max(), 5)
        ax.plot([0, lim], [0, lim], color="red", linestyle="--", linewidth=1,
                label="1:1")
        ax.set_xlabel("softening % (symmetric band)")
        ax.set_ylabel("softening % (asymmetric band, paper)")
        ax.set_title(f"{comp}  n={len(sub)}  median(asym − sym) = {med_diff:+.1f} pp")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        summary["comp"][comp] = {
            "n": int(len(sub)),
            "median_softening_asym_pct": float(sub["soft_asym_pct"].median()),
            "median_softening_sym_pct":  float(sub["soft_sym_pct"].median()),
            "median_diff_asym_minus_sym_pp": med_diff,
        }
    fig.suptitle(
        f"{cfg.station_id} ({cfg.sensor_floor_label}) — f_min estimator bias check: "
        f"asymmetric vs symmetric bandpass (reviewer §3)", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1.0, 0.95))
    fig.savefig(out_dir / "summary_fmin_symmetric_vs_asymmetric.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "summary_fmin_symmetric_vs_asymmetric.json").open(
            "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def plot_summary_scatter(csv_path: Path, out_dir: Path, cfg: StationConfig):
    _plot_summary_one(csv_path, out_dir, "cwt", cfg)
    _plot_summary_one(csv_path, out_dir, "tf", cfg)
    _plot_softening_vs_pga(csv_path, out_dir, cfg)
    _baseline_comparison(csv_path, out_dir, cfg)
    try:
        _plot_stl_detrend(csv_path, out_dir, cfg)
    except Exception as exc:
        logging.error("STL decomposition failed: %s", exc)
    # ---- v5 reviewer additions ----
    try:
        _baseline_comparison_pga_matched(csv_path, out_dir, cfg)
    except Exception as exc:
        logging.error("PGA-matched baseline failed: %s", exc)
    try:
        _baseline_comparison_deseasonalized(csv_path, out_dir, cfg)
    except Exception as exc:
        logging.error("Deseasonalized baseline failed: %s", exc)
    try:
        _plot_fmin_symmetric_vs_asymmetric(csv_path, out_dir, cfg)
    except Exception as exc:
        logging.error("f_min sym/asym comparison failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        filename=str(log_path),
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )


def _process_event_for_worker(args):
    event, cfg = args
    return process_event(event, cfg)


def _write_readme(cfg: StationConfig, out_root: Path) -> None:
    sid = cfg.station_id
    f_emp, f_emp_min, f_emp_max = code_band(cfg)
    txt = f"""# {sid} ({cfg.locname}) — v4 pipeline outputs

## Station metadata
- Location: {cfg.locname}  ({cfg.lat:.6f}, {cfg.lon:.6f})  elev {cfg.elev_m:.0f} m
- Structure: {cfg.structure}, {cfg.height_m:.0f} m  (~{max(1, int(round(cfg.height_m/3.0)))} floors)
- Sensor floor: {cfg.sensor_floor_label}
- Footprint: {cfg.footprint_m2:.0f} m²
- Distance to NCREE free-field input: {cfg.ncree_distance_m:.0f} m
- Site (HWA018 borehole, ~adjacent campus): Vs30 = {CAMPUS_VS30_MS:.1f} m/s, NEHRP class {CAMPUS_SITE_CLASS}

## Code-empirical reference
- Taiwan Seismic Code §2.6: T = k·h^(3/4)
- f_empirical = {f_emp:.3f} Hz
- Code band [f_emp/1.4, f_emp×1.4] = [{f_emp_min:.3f}, {f_emp_max:.3f}] Hz

## Per-event outputs (in each YYYYMMDDHHMMSS/ folder)
- `raw_{{E,N,Z}}.png`        Raw waveforms with Arias/STA-LTA markers
- `fft_{{E,N,Z}}.png`        NCREE + structure FFT spectra (KO-smoothed)
- `transfer_fn_{{E,N,Z}}.png`     Event-averaged |H(f)| water-level deconv
- `transfer_fn_tv_{{E,N,Z}}.png`  **NEW** time-varying |H(f, t)|, 10 s window / 5 s overlap (§2.1)
- `cwt_{{E,N,Z}}.png`        CWT response with f_iapp/f_min/f_99app annotations

## Summary outputs (root of this folder)
- `summary_cwt_*.png`               f_iapp/f_min/f_99app time series (scatter/rolling/OLS/segmented)
- `summary_tf_*.png`                TF peak (PRIMARY) + f₁ (reference) time series — see §2.7 note
- `summary_softening_vs_pga.png`    **NEW** dose-response (f_iapp − f_min)/f_iapp vs PGA (§3.3)
- `baseline_comparison.png/.json`   **NEW** pre vs post M7.2 baselines (±90 d) with bootstrap CIs (§3.4)
- `ssi_estimates.csv/.md`           **NEW** Veletsos-Meek SSI period-lengthening bounds (§4.1)
- `concept_frequencies.png`         Astorga 2018 definitions illustrated with the M7.2 record
- `structural_history_log_v4_{sid}.csv`  Master per-event-component table

## Reviewer fixes applied in v5 (vs v4)
- §3 (estimator bias): per-event CSV now reports both `f_min` (asymmetric
  bandpass `[f_c/{F_MIN_BAND_DOWN_ASYM}, f_c*{F_MIN_BAND_UP_ASYM}]`, paper
  default) and `f_min_sym` (symmetric equal-log-width band
  `[f_c/{F_MIN_BAND_R_SYM:.3f}, f_c*{F_MIN_BAND_R_SYM:.3f}]`). A new summary
  plot `summary_fmin_symmetric_vs_asymmetric.{{png,json}}` shows the
  softening % distribution under both choices so the 50% drop can be
  inspected for one-sided estimator bias.
- §5 (segmentation source): STA/LTA + Arias 5/95/99 windows are computed on
  the *{SEGMENT_SOURCE}* record; this is now logged in the
  `segment_source` column of the per-event CSV.
- §6 (PGA-matched baseline): in addition to the ±90-day bootstrap, the v5
  pipeline writes `baseline_comparison_pga_matched.{{png,json}}` where each
  post-mainshock event is matched to its nearest pre-mainshock counterpart
  on log10(PGA_30m). This neutralises the weak-vs-strong sample-selection
  bias that contaminates the unmatched comparison.
- §10 (deseasonalization): `baseline_comparison_deseasonalized.{{png,json}}`
  repeats the ±90-day bootstrap after subtracting the calendar-month mean
  of the STL seasonal component from every event's f_iapp. Δ surviving
  this correction is interpretable as a non-seasonal change in f_iapp.

## Reviewer fixes applied in v4 (vs v3)
- §2.1: time-varying TF (`transfer_fn_tv_*.png`) relaxes LTI assumption.
- §2.4: TF panel titles + CSV columns `ncree_distance_m`, `vs30_ms`, `site_class`
  spell out the borehole-surface→structure composite-path interpretation.
- §2.7: `tf_peak_freq` (0.4–15 Hz unconstrained) is now the primary indicator;
  `tf_f1` (code-band-restricted) is reported as reference only, and a new
  boolean column `tf_peak_in_codeband` flags out-of-band events. This breaks
  the circular use of the code band as both the search range and the
  validation target.
- §3.3: `summary_softening_vs_pga.png` shows the dose-response with pre/post
  split and OLS on log(PGA).
- §3.4: `baseline_comparison.{{png,json}}` compares ±90-day windows of f_iapp
  around the M7.2 mainshock with bootstrap CIs — this measures long-term
  permanent change, not the seconds-scale recovery captured by f_99app/f_iapp.
- §4.1: `ssi_estimates.*` reports Veletsos-Meek / Pais-Kausel /
  NIST GCR 12-917-21 period-lengthening bounds. For NDHU on NEHRP Class C
  ground (Vs ≈ 460 m/s), σ = h/(Vs·T) ≈ 0.04 and Δf_SSI ≈ 1–2%, so
  observed 15–30% coseismic drops are dominated by structural and
  soil-strain nonlinearity rather than small-strain SSI. SSI is not
  *separated* here — only bounded.

## Known limitations (still open)
- Principal-axis transformation is not yet performed; E/N components are in
  instrument coordinates, so each axis mixes structural X- and Y-modes.
- Damping ratio uses only the half-power bandwidth method; no RDT/ARX
  cross-check (reviewer §4.4).
- Environmental (temperature/humidity/occupancy) detrending: STL-based
  detrending lives in Phase 3 outputs (see README in repo root).
"""
    (out_root / "README.md").write_text(txt, encoding="utf-8")


def run_pipeline(cfg: StationConfig):
    paths = output_paths(cfg)
    _setup_logging(paths["log"])
    paths["root"].mkdir(parents=True, exist_ok=True)
    # Drop README + SSI bounds before processing so they're available even if
    # the run is interrupted.
    _write_readme(cfg, paths["root"])
    from src.ssi_bounds import write_estimates
    write_estimates(paths["root"])

    matcher = importlib.import_module(cfg.matcher_module)
    events, skipped = matcher.build_event_list()
    events = [e for e in events if e.event_id[:4] >= YEAR_MIN]
    print(f"[{cfg.station_id}] Filtered to {YEAR_MIN}+: {len(events)} events")

    with paths["skip_log"].open("w", encoding="utf-8") as f:
        for s in skipped:
            f.write(f"{s.event_id}\t{s.reason}\n")

    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns(cfg))
        writer.writeheader()
        n_workers = max(1, (os.cpu_count() or 2) - 2)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_event_for_worker, (e, cfg)) for e in events]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{cfg.station_id} events"):
                event_id, rows, err = fut.result()
                if err:
                    logging.error("Event %s failed: %s", event_id, err)
                for r in rows:
                    writer.writerow(r)

    plot_summary_scatter(paths["csv"], paths["root"], cfg)

    m7_event = next((e for e in events if e.event_id == MAINSHOCK_EVENT_ID), None)
    if m7_event is not None:
        try:
            plot_concept_diagram(m7_event, paths["root"] / "concept_frequencies.png", cfg)
            print(f"[{cfg.station_id}] Concept figure: "
                  f"{paths['root'] / 'concept_frequencies.png'}")
        except Exception as exc:
            logging.error("concept figure failed: %s", exc)

    print(f"\n[{cfg.station_id}] Outputs: {paths['root']}\n"
          f"[{cfg.station_id}] CSV: {paths['csv']}")
