"""v5 pipeline for D006 dual-sensor deployment (1F + 4F, NCREE-referenced).

Forked from main_v3_D006.py with the reviewer_v4 fixes:

(3) Symmetric-bandpass f_min sensitivity check. Per-event CSV now records
    both f_min (asymmetric paper default) and f_min_sym (equal-log-width
    band, r = sqrt(1.7*1.15) ~= 1.398), for both the 1F and 4F sensors.
    A new summary plot `summary_fmin_symmetric_vs_asymmetric_D006.png`
    shows the bias distribution.
(4) D006 all-event damping comparison (single-station NCREE->4F vs
    Snieder-Safak inter-story 1F->4F) is written as
    `damping_all_events.{png,json}` with bootstrap CIs — addresses the
    reviewer's complaint that the 16% gap rested on a single event with
    no uncertainty band.
(5) Explicit `segment_source` label in per-event CSV (= "W10F-4F").
(6, 10) PGA-matched and STL-deseasonalized baselines run via the shared v5
    pipeline (see src/pipeline_common_v5.py).

Outputs go to D:/PHD/project2026/output_v5_D006/.
"""
from __future__ import annotations

import csv
import gc
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from obspy import read
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.signal.windows import tukey
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src import processor
from src.matcher_ncree import build_event_list, NcreeEvent


BASE_DIR = Path(r"D:\PHD\project2026")
OUTPUT_ROOT = BASE_DIR / "output_v5_D006"
LOG_PATH = BASE_DIR / "processing_errors_v5_D006.log"
SKIP_LOG_PATH = BASE_DIR / "skipped_events_v5_D006.log"
CSV_PATH = BASE_DIR / "structural_history_log_v5_D006.csv"

TARGET_FS = 100.0
PRE_EVENT_SEC = 30.0     # seconds before NCREE trigger (P-alert side only)
POST_EVENT_SEC = 120.0   # seconds after NCREE trigger (all three stations)
YEAR_MIN = "2023"        # focus window for analysis

# --- Building parameters (W10F deployment, per Taiwan seismic design code) ---
# 4-storey RC building -> T = 0.070 * h^(3/4), h in meters.
# Code clause: T_actual ≤ 1.4 * T_empirical  ->  f_actual ≥ f_emp / 1.4.
BUILDING_HEIGHT_M = 12.0          # 4 floors × 3 m/floor (default)
BUILDING_STRUCTURE = "RC"         # "RC" | "STEEL" | "OTHER"
_STRUCTURE_K = {"STEEL": 0.085, "RC": 0.070, "OTHER": 0.050}
T_EMPIRICAL = _STRUCTURE_K[BUILDING_STRUCTURE] * (BUILDING_HEIGHT_M ** 0.75)
F_EMPIRICAL = 1.0 / T_EMPIRICAL
F_EMP_MIN = F_EMPIRICAL / 1.4
F_EMP_MAX = F_EMPIRICAL * 1.4
# Wider band for coseismic minimum search — allows up to 50% softening
# during strong shaking (beyond the design code's ±40% linear-elastic band).
# Cf. Astorga 2018 observations of co-seismic f drops in Tohoku sequence.
F_MIN_SEARCH_LO = F_EMPIRICAL * 0.5
F_MIN_SEARCH_HI = F_EMP_MAX

# v5 reviewer §3: asymmetric vs symmetric bandpass for event-adaptive IF
# search around f_iapp. Asymmetric (paper default) = (1.7, 1.15); symmetric
# = (r, r) with r = sqrt(1.7*1.15) ~= 1.398, equal log-width control band.
F_MIN_BAND_DOWN_ASYM = 1.7
F_MIN_BAND_UP_ASYM = 1.15
F_MIN_BAND_R_SYM = float(np.sqrt(F_MIN_BAND_DOWN_ASYM * F_MIN_BAND_UP_ASYM))
SEGMENT_SOURCE = "W10F-4F"

# Lower CWT mask so 1st-mode (~2.2 Hz) is well inside the searchable band.
CWT_MIN_FREQ = 0.4

# --- Signal-processing constants (per literature) -----------------------------
TUKEY_ALPHA = 0.05               # 5% cosine taper on both ends (mitigate leakage)
WATERLEVEL_EPS = 0.05            # Snieder & Şafak 2006 used 0.10; we use 0.05
                                 # as compromise: more LF detail than Snieder while
                                 # avoiding excessive amplification of noise notches.
                                 # Cf. Clayton & Wiggins 1976 frequency-domain decon.
KO_BANDWIDTH = 30                # Konno-Ohmachi b; 20–40 is standard for HVSR/TF
KO_LOG_BINS = 300                # downsample FFT to 300 log-spaced bins for KO speed
STA_SEC = 1.0                    # STA/LTA classic parameters (Astorga 2018 §rsPWV)
LTA_SEC = 10.0
STA_LTA_THRESHOLD = 2.0
ARIAS_G = 9.80665                # m/s²; Arias intensity normalization constant

CSV_COLUMNS = [
    "event_id", "utc_time", "local_time",
    "magnitude", "depth_km", "epicenter",
    "pga_0m", "pga_20m", "pga_30m", "pga_58m", "pga_78m",
    "sensitivity", "catalog_matched",
    "comp",
    "f_init_4f", "f_lowest_4f", "f_final_4f", "t_lowest_4f",
    "f_init_1f", "f_lowest_1f", "f_final_1f", "t_lowest_1f",
    "tf_peak_freq_4f", "tf_peak_amp_4f",
    "tf_peak_freq_1f", "tf_peak_amp_1f",
    "tf_f1_4f", "tf_f1_1f",          # 1st-mode peak within empirical band
    "f_empirical", "f_emp_min", "f_emp_max",
    # New operational definitions (Astorga 2018 BSSA / Michel-Guéguen)
    "f_iapp_4f", "f_min_4f", "f_99app_4f",
    "f_iapp_1f", "f_min_1f", "f_99app_1f",
    # v5 reviewer §3: symmetric-bandpass control
    "f_min_sym_4f", "f_min_diff_pct_4f",
    "f_min_sym_1f", "f_min_diff_pct_1f",
    "t_first_arrival_sec", "t5_sec", "t95_sec", "t99_sec",
    "arias_total",
    "damping_ratio_4f",              # half-power bandwidth ζ on H(f)
    "damping_ratio_1f",
    "fft_peak_ncree", "fft_peak_d006", "fft_peak_w10f",
    "fft_peak_cb_ncree", "fft_peak_cb_d006", "fft_peak_cb_w10f",
    "segment_source",                # v5 reviewer §5
    "status",
]


def _setup_logging() -> None:
    logging.basicConfig(
        filename=str(LOG_PATH),
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w",
    )


def _resample_to(trace, fs):
    if abs(trace.stats.sampling_rate - fs) < 1e-6:
        return trace
    tr = trace.copy()
    tr.resample(fs)
    return tr


def _load_traces(event: NcreeEvent):
    out = {"NCREE": {}, "D006": {}, "W10F": {}}
    for c, p in event.ncree_paths.items():
        out["NCREE"][c] = read(str(p))[0]
    for c, p in event.palert_1f.items():
        out["D006"][c] = read(str(p))[0]
    for c, p in event.palert_4f.items():
        out["W10F"][c] = read(str(p))[0]
    return out


def _slice_window(trace, t_ref, pre_sec, post_sec, fs):
    """Return data[(pre_sec + post_sec)*fs] aligned so index pre_sec*fs == t_ref.

    Out-of-bounds samples filled with NaN. Caller decides how to handle NaN.
    """
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


# --- Signal-processing helpers (windowing, smoothing, segmentation) ----------
def _tapered(x, alpha=TUKEY_ALPHA):
    """Demean + Tukey-windowed copy of x (mitigates spectral leakage)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return x.copy()
    return (x - x.mean()) * tukey(len(x), alpha=alpha)


def _ko_smooth(freqs, amp, fmin=0.1, fmax=25.0,
                n_log_bins=KO_LOG_BINS, bandwidth=KO_BANDWIDTH):
    """Resample to log-spaced grid then apply Konno-Ohmachi smoothing.

    Returns (log_freqs, smoothed_amp). Faster than KO on dense linear grid
    (300² vs N² ops). Cf. Konno & Ohmachi 1998 BSSA; obspy implementation.
    """
    log_f = np.logspace(np.log10(max(fmin, freqs[freqs > 0].min())),
                         np.log10(min(fmax, freqs[-1])), n_log_bins)
    amp_log = np.interp(log_f, freqs, amp)
    smoothed = konno_ohmachi_smoothing(
        amp_log.astype(np.float32), log_f.astype(np.float32),
        bandwidth=bandwidth, normalize=False,
    )
    return log_f, np.asarray(smoothed, dtype=float)


def _smoothed_fft(signal, fs):
    """Tukey-tapered FFT + KO smoothing on log-spaced grid.

    Returns (log_freqs, smoothed_amp).
    """
    x = _tapered(signal)
    n = len(x)
    if n < 8:
        return np.array([np.nan]), np.array([np.nan])
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.abs(np.fft.rfft(x)) / n
    return _ko_smooth(freqs, amp)


def _damping_from_raw_H(input_sig, output_sig, fs, f0,
                          epsilon=WATERLEVEL_EPS, halfwidth_hz=1.5):
    """Half-power bandwidth damping ζ from UNSMOOTHED |H(f)| near f0.

    KO smoothing widens peaks by O(b/f) so applying half-power on smoothed
    H over-estimates ζ. We re-run the FFT/deconv here without smoothing and
    compute ζ in a narrow band [f0-hw, f0+hw].
    """
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
    """Half-power bandwidth damping ratio ζ at resonance f0.

    ζ ≈ (f₂ − f₁) / (2·f0), where f₁, f₂ are −3 dB points around f0.
    Returns NaN if peak cannot be bracketed (e.g. boundary peak).
    """
    if np.isnan(f0):
        return float("nan")
    i0 = int(np.argmin(np.abs(freqs - f0)))
    peak = amp[i0]
    if peak <= 0:
        return float("nan")
    target = peak / np.sqrt(2)  # -3 dB amplitude
    # walk left
    j = i0
    while j > 0 and amp[j] > target:
        j -= 1
    f1 = freqs[j] if j != i0 else np.nan
    # walk right
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
    """Detect first-arrival and Arias 5%/95%/99% segment points.

    signal_with_pre: array spanning [-pre_event_sec, +POST_EVENT_SEC] sec, in
    sample units; sample 0 = NCREE trigger − pre_event_sec.

    Returns dict with sample indices and corresponding seconds (relative to
    NCREE trigger, t=0).  None if signal is too short / silent.
    """
    n = len(signal_with_pre)
    nsta = int(STA_SEC * fs)
    nlta = int(LTA_SEC * fs)
    if n < nlta + nsta + 50:
        return None
    # NaN-safe: STA/LTA on demeaned, NaN-filled signal
    sig = np.nan_to_num(signal_with_pre - np.nanmean(signal_with_pre))
    if not np.any(np.abs(sig) > 0):
        return None
    cft = classic_sta_lta(sig, nsta, nlta)
    pre_n = int(pre_event_sec * fs)
    # Limit STA/LTA search to [-5s, +30s] around NCREE trigger to avoid
    # picking aftershock onsets late in the window.
    search_lo = max(nlta, pre_n - int(5 * fs))
    search_hi = min(n, pre_n + int(30 * fs))
    region = cft[search_lo:search_hi]
    above = np.where(region > STA_LTA_THRESHOLD)[0]
    if len(above) > 0:
        t_first = search_lo + int(above[0])
    else:
        t_first = pre_n  # fallback to NCREE trigger
    # Arias intensity (cumulative). Use full signal squared.
    arias = np.cumsum(sig ** 2)
    arias_max = float(arias[-1])
    if arias_max <= 0:
        return None
    arias_norm = arias / arias_max
    t5  = int(np.searchsorted(arias_norm, 0.05))
    t95 = int(np.searchsorted(arias_norm, 0.95))
    t99 = int(np.searchsorted(arias_norm, 0.99))
    arias_total = arias_max * (np.pi / (2.0 * ARIAS_G)) / fs  # physical units (m/s)
    return {
        "t_first": t_first, "t5": t5, "t95": t95, "t99": t99,
        "t_first_sec": t_first / fs - pre_event_sec,
        "t5_sec":  t5  / fs - pre_event_sec,
        "t95_sec": t95 / fs - pre_event_sec,
        "t99_sec": t99 / fs - pre_event_sec,
        "arias_total": arias_total,
    }


def _modal_in_segment(signal_full, fs, segments, key_a, key_b, fmin, fmax):
    """FFT peak in segment[key_a:key_b] within frequency band [fmin, fmax]."""
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
    """Coseismic 1st-mode frequency via Hilbert instantaneous frequency.

    Methodology (literature-grounded):
      1. Narrow event-adaptive bandpass [f_iapp/1.5, f_iapp×1.5] to keep the
         signal single-mode (Di Tommaso et al. 2012 BV-filter; Todorovska 2007).
      2. Analytic signal via Hilbert transform → unwrap phase → IF = dφ/dt.
         Standard instantaneous-frequency definition (Trifunac 2001 Part II
         §"Instantaneous apparent frequency"; Cohen 1989).
      3. Smooth IF with 1-s moving average (Savitzky-Golay-equivalent
         denoising of phase fluctuations, cf. Astorga 2018 §rsPWV).
      4. Energy-gated median: median of IF samples where local envelope
         > 50% of DSM max. This is the robust analog of Astorga 2018's
         "frequency at peak energy" picker on rsPWV ridges.
    Returns (f_min_Hz, t_at_f_min_sec).
    """
    a, b = segments["t5"], segments["t95"]
    if b - a < int(2 * fs):
        return float("nan"), float("nan")
    # SNR filter: skip f_min computation if structural response in DSM is too
    # weak to reliably engage the 1st mode (Astorga 2018 bins events by PTA;
    # we use a simple RMS threshold). Below 1 gal RMS the IF estimate is
    # dominated by measurement noise.
    dsm_rms = float(np.sqrt(np.nanmean(np.square(signal_full[a:b]))))
    if not np.isfinite(dsm_rms) or dsm_rms < 1.0:
        return float("nan"), float("nan")
    if f_center is not None and not np.isnan(f_center) and f_center > 0.5:
        # v5 reviewer §3: bias check between asymmetric paper band and a
        # symmetric (equal-log-width) control band. Asymmetric is wide on
        # the softening side / narrow on the hardening side; symmetric uses
        # r = sqrt(1.7 * 1.15) on both sides so the picker has equal room.
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
    # Instantaneous frequency via Hilbert transform (Di Tommaso 2012 BV-filter
    # equivalent: narrow bandpass → analytic signal → unwrapped phase → df/dt).
    # Gives continuous (non-quantized) frequency tracking of the 1st mode.
    analytic = hilbert(bp)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs
    inst_freq = np.append(inst_freq, inst_freq[-1])  # pad to len n
    # Smooth instantaneous freq (1 s moving average) to reduce phase noise
    inst_freq = uniform_filter1d(inst_freq, size=int(fs))
    # Envelope for energy gating
    envelope = np.abs(analytic)
    # Restrict to DSM window
    if_seg = inst_freq[a:b]
    env_seg = envelope[a:b]
    if env_seg.size == 0 or env_seg.max() <= 0:
        return float("nan"), float("nan")
    # Energy gating: only count samples in the high-amplitude phase (envelope
    # > 50% of its DSM max). Median of IF in this set = "frequency the structure
    # actually resonated at during peak shaking" — robust to IF variance of
    # forced SDOF systems (cf. Astorga 2018 picks at IA energy percentiles;
    # using top-envelope median is the equivalent robust estimator).
    if_lower = max(lo * 1.02, fmin)
    if_upper = min(hi * 0.98, fmax * 1.05)
    high_env = env_seg > 0.5 * env_seg.max()
    in_band = (if_seg >= if_lower) & (if_seg <= if_upper)
    valid = high_env & in_band
    if valid.sum() < int(0.5 * fs):     # need at least 0.5 s of valid samples
        return float("nan"), float("nan")
    f_med = float(np.median(if_seg[valid]))
    # Locate time of f_med (sample closest to median IF among valid)
    valid_idx = np.flatnonzero(valid)
    rel = valid_idx[int(np.argmin(np.abs(if_seg[valid_idx] - f_med)))]
    abs_idx = a + int(rel)
    return f_med, abs_idx / fs - PRE_EVENT_SEC


def _transfer_function(input_sig, output_sig, fs, epsilon=WATERLEVEL_EPS):
    """Frequency-domain water-level deconvolution with leakage & noise control.

    Pipeline:
      1. Tukey (5% cosine) taper on both x and y to suppress edge leakage.
      2. rFFT, water-level water = epsilon · max(|X|²).  ε=0.01 (Snieder & Şafak
         2006 used 0.1; we use 0.01 for better LF response).
      3. Konno-Ohmachi smoothing (b=30) on |H(f)| via log-spaced 300-bin grid.

    Returns (log_freqs, smoothed |H(f)|).
    """
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


def _cwt_w10f(signal_full, fs, pre_event_sec=PRE_EVENT_SEC,
               fmin=0.4, fmax=15.0, n_bins=120):
    """CWT of the raw structural-response signal (W10F or D006).

    Per Todorovska 2007 (Gabor / wavelet of building response) and
    Astorga 2018 (rsPWV time-frequency tracking on building floor records),
    the time-varying modal frequency should be extracted from the response
    signal itself, not from a deconvolved IRF (which carries deconvolution
    artifacts). Returns:
      cwt:        2-D complex CWT matrix (n_freq × n_time)
      freqs:      log-spaced frequencies (Hz)
      time:       full time axis with t=0 at NCREE trigger
      ridge:      ridge frequency (max-energy bin) restricted to the code band
                  to lock onto the 1st mode and avoid jumps to higher modes.
    """
    sig = np.nan_to_num(signal_full - np.nanmean(signal_full))
    cwt_freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_bins)
    cwt_matrix, freqs = processor.compute_cwt(sig, fs, freqs=cwt_freqs)
    n = len(sig)
    time = np.arange(n) / fs - pre_event_sec
    # Ridge confined to the empirical code band so we track the 1st mode.
    band_mask = (freqs >= F_EMP_MIN) & (freqs <= F_EMP_MAX)
    if band_mask.any():
        power = np.abs(cwt_matrix)
        masked = power.copy()
        masked[~band_mask, :] = -np.inf
        ridge_idx = np.argmax(masked, axis=0)
        ridge = freqs[ridge_idx]
    else:
        ridge = np.full(n, np.nan)
    return {
        "cwt": cwt_matrix, "freqs": freqs, "time": time, "ridge": ridge,
    }


# -------------------- plotting helpers (with annotations) --------------------
def _plot_raw(out_path, fs, comp, traces_with_times, t_ref_label="NCREE trigger",
               segments=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
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
            ax.axvline(tval, color=color, linewidth=1.0, alpha=0.7,
                       linestyle="--")
        if valid.any():
            peak = np.nanmax(np.abs(data))
            ax.text(0.99, 0.95, f"|peak|={peak:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.grid(alpha=0.3)
    if seg_lines:
        handles = [plt.Line2D([0], [0], color=c, linestyle="--", linewidth=1.2,
                                label=lbl)
                   for _, lbl, c in seg_lines]
        axes[0].legend(handles=handles, loc="upper right", fontsize=7,
                        framealpha=0.9)
    axes[-1].set_xlabel(f"Time (s) — t=0 at {t_ref_label}")
    fig.suptitle(f"Raw waveforms ({comp})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fft(out_path, fs, comp, traces):
    """3 stacked subplots so peak labels never overlap."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    peaks = {}
    colors = {"NCREE": "tab:blue", "D006(1F)": "tab:orange", "W10F(4F)": "tab:green"}
    for ax, (label, data) in zip(axes, traces):
        d = data[~np.isnan(data)]
        if len(d) == 0:
            continue
        freqs, amp = _smoothed_fft(d, fs)
        c = colors.get(label, "black")
        _draw_empirical_band(ax, axis="x")
        ax.semilogy(freqs, amp, linewidth=0.9, color=c)
        # Full-band peak (no frequency restriction)
        pf, pa = _peak_freq(freqs, amp)
        peaks[label] = pf
        ax.axvline(pf, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.annotate(f"{label}  peak = {pf:.2f} Hz",
                    xy=(pf, pa), xytext=(8, -4), textcoords="offset points",
                    fontsize=9, color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow",
                              ec=c, alpha=0.9))
        # Code-band peak (within empirical code band → 1st-mode candidate)
        pf_cb, pa_cb = _peak_freq(freqs, amp, fmin=F_EMP_MIN, fmax=F_EMP_MAX)
        if not np.isnan(pf_cb) and abs(pf_cb - pf) > 0.05:
            # Only show separately if it's not the same peak as full-band
            ax.scatter([pf_cb], [pa_cb], marker="*", c="magenta", s=140,
                        edgecolor="white", linewidth=1.0, zorder=8)
            # Position label below the marker to avoid overlap with full-band
            # annotation (which is placed above-right of the global peak)
            ax.annotate(f"f₁ (code band) = {pf_cb:.2f} Hz",
                         xy=(pf_cb, pa_cb), xytext=(8, -32),
                         textcoords="offset points",
                         fontsize=9, color="magenta", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                    ec="magenta", alpha=0.92),
                         zorder=9)
        elif not np.isnan(pf_cb):
            # Same peak — just mark with star but no duplicate label
            ax.scatter([pf_cb], [pa_cb], marker="*", c="magenta", s=140,
                        edgecolor="white", linewidth=1.0, zorder=8)
        peaks[f"{label}_cb"] = pf_cb
        ax.set_ylabel(f"{label}\nAmplitude")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"FFT spectra ({comp})")
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(0, 25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return peaks


def _plot_transfer_function(out_path, comp, tf_1f, tf_4f):
    """tf_1f / tf_4f = (freqs, |H(f)|) tuples."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    out_peaks = {}
    for ax, label, (freqs, mag) in zip(
        axes,
        ("NCREE → D006 (1F)", "NCREE → W10F (4F)"),
        (tf_1f, tf_4f),
    ):
        _draw_empirical_band(ax, axis="x")
        ax.semilogy(freqs, mag, linewidth=0.8)
        pf, pa = _peak_freq(freqs, mag, fmin=0.4, fmax=15)
        out_peaks[label] = (pf, pa)
        ax.axvline(pf, color="r", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.annotate(f"peak = {pf:.2f} Hz\n|H| = {pa:.2g}",
                    xy=(pf, pa), xytext=(10, 0), textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
        # Constrained 1st-mode peak within empirical band
        f1, a1 = _peak_freq(freqs, mag, fmin=F_EMP_MIN, fmax=F_EMP_MAX)
        if not np.isnan(f1):
            ax.scatter([f1], [a1], marker="*", c="magenta", s=140,
                       edgecolor="white", linewidth=1, zorder=5)
            ax.annotate(f"f₁ (code band) = {f1:.2f} Hz",
                        xy=(f1, a1), xytext=(10, -25), textcoords="offset points",
                        fontsize=9, color="magenta", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="magenta", alpha=0.9))
        ax.set_ylabel(f"{label}\n|H(f)|")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 20)
        ax.legend(loc="lower right", fontsize=7)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"Transfer Function (Water-level Deconvolution) — {comp}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_peaks


def _draw_empirical_band(ax, axis="y"):
    """Overlay the code-empirical 1st-mode estimate as a band + reference line.

    axis="y" -> horizontal lines (for CWT / summary scatter);
    axis="x" -> vertical lines (for FFT / TF spectra).
    """
    if axis == "y":
        ax.axhspan(F_EMP_MIN, F_EMP_MAX, color="cyan", alpha=0.12, zorder=1,
                   label=f"Code band [{F_EMP_MIN:.2f}, {F_EMP_MAX:.2f}] Hz")
        ax.axhline(F_EMPIRICAL, color="cyan", linestyle="--", linewidth=1.2,
                   alpha=0.9, zorder=2,
                   label=f"f_emp = {F_EMPIRICAL:.2f} Hz")
    else:
        ax.axvspan(F_EMP_MIN, F_EMP_MAX, color="cyan", alpha=0.15, zorder=1)
        ax.axvline(F_EMPIRICAL, color="cyan", linestyle="--", linewidth=1.2,
                   alpha=0.9, zorder=2,
                   label=f"f_emp = {F_EMPIRICAL:.2f} Hz")


def _plot_cwt_annotated(feat, out_path, title, segs, ops, label_pre, y_max=15.0):
    """CWT-on-response plot with operational SHM points overlaid.

    feat must be returned by _cwt_w10f().
    ops = dict of operational frequencies (f_iapp, f_min, f_99app) and times.
    Three colored circles mark f_iapp / f_min / f_99app at their respective
    measurement-window centers, matching the concept figure.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.pcolormesh(feat["time"], feat["freqs"], np.abs(feat["cwt"]),
                   shading="auto", cmap="viridis", zorder=1)
    # Ridge (restricted to code band → tracks 1st mode only)
    ax.plot(feat["time"], feat["ridge"], color="white", linestyle="--",
             linewidth=1.0, alpha=0.85, label="1st-mode ridge (code-band)")
    _draw_empirical_band(ax, axis="y")
    # Mark Arias segment boundaries
    if segs is not None:
        for tval, c in [(segs.get("t_first_sec"), "tab:red"),
                         (segs.get("t5_sec"),  "tab:purple"),
                         (segs.get("t95_sec"), "tab:purple"),
                         (segs.get("t99_sec"), "tab:gray")]:
            if tval is not None and not np.isnan(tval):
                ax.axvline(tval, color=c, linestyle=":", linewidth=1.0,
                            alpha=0.7, zorder=3)
    # Operational points (f_iapp, f_min, f_99app)
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


# -------------------- main processor --------------------
def process_event(event: NcreeEvent):
    event_dir = OUTPUT_ROOT / event.event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        streams = _load_traces(event)
        meta = event.catalog_meta or {}
        t_ref = streams["NCREE"]["Z"].stats.starttime  # NCREE filename = trigger

        for comp in ("E", "N", "Z"):
            ncree_tr = _resample_to(streams["NCREE"][comp], TARGET_FS)
            d006_tr = _resample_to(streams["D006"][comp], TARGET_FS)
            w10f_tr = _resample_to(streams["W10F"][comp], TARGET_FS)

            # Window: [-30, +120] sec around NCREE trigger.
            t_axis = np.arange(int((PRE_EVENT_SEC + POST_EVENT_SEC) * TARGET_FS)) / TARGET_FS - PRE_EVENT_SEC
            ncree_full = _slice_window(ncree_tr, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, TARGET_FS)
            d006_full = _slice_window(d006_tr, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, TARGET_FS)
            w10f_full = _slice_window(w10f_tr, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, TARGET_FS)

            # Demean using only finite (non-NaN) samples
            for arr in (ncree_full, d006_full, w10f_full):
                m = ~np.isnan(arr)
                if m.any():
                    arr[m] -= arr[m].mean()

            # Post-event-only window for deconv & CWT (NCREE has no pre data)
            post_n = int(POST_EVENT_SEC * TARGET_FS)
            i0 = int(PRE_EVENT_SEC * TARGET_FS)
            ncree_post = ncree_full[i0:i0 + post_n]
            d006_post = d006_full[i0:i0 + post_n]
            w10f_post = w10f_full[i0:i0 + post_n]

            # Replace any residual NaN with 0 (rare boundary cases)
            ncree_post = np.nan_to_num(ncree_post)
            d006_post = np.nan_to_num(d006_post)
            w10f_post = np.nan_to_num(w10f_post)

            if (np.abs(ncree_post) > 0).sum() < int(10 * TARGET_FS):
                rows.append(_row(event, meta, comp, None, None, None, None, None, "too_short"))  # noqa
                continue

            # ---- Frequency-domain transfer functions ----
            tf_1f = _transfer_function(ncree_post, d006_post, TARGET_FS)
            tf_4f = _transfer_function(ncree_post, w10f_post, TARGET_FS)
            tf_peaks = _plot_transfer_function(
                event_dir / f"transfer_fn_{comp}.png", comp, tf_1f, tf_4f
            )
            # 1st-mode constrained search within empirical band [F_EMP_MIN, F_EMP_MAX]
            f1_1f, h1_1f = _peak_freq(tf_1f[0], tf_1f[1], fmin=F_EMP_MIN, fmax=F_EMP_MAX)
            f1_4f, h1_4f = _peak_freq(tf_4f[0], tf_4f[1], fmin=F_EMP_MIN, fmax=F_EMP_MAX)
            tf_peaks["_f1_1f"] = f1_1f
            tf_peaks["_f1_4f"] = f1_4f
            # Damping ratio ζ via half-power bandwidth on UNSMOOTHED |H(f)|
            zeta_4f = _damping_from_raw_H(ncree_post, w10f_post, TARGET_FS, f1_4f)
            zeta_1f = _damping_from_raw_H(ncree_post, d006_post, TARGET_FS, f1_1f)
            tf_peaks["_zeta_4f"] = zeta_4f
            tf_peaks["_zeta_1f"] = zeta_1f

            # ---- Segment-based operational features (Astorga 2018) ----
            # Use W10F (4F) to determine first arrival + Arias intensity windows
            segs = _event_segments(w10f_full, TARGET_FS, pre_event_sec=PRE_EVENT_SEC)
            if segs is None:
                segs = {
                    "t_first": int(PRE_EVENT_SEC * TARGET_FS),
                    "t5":  int(PRE_EVENT_SEC * TARGET_FS),
                    "t95": len(w10f_full) - 1,
                    "t99": len(w10f_full) - 1,
                    "t_first_sec": 0.0, "t5_sec": 0.0,
                    "t95_sec": POST_EVENT_SEC, "t99_sec": POST_EVENT_SEC,
                    "arias_total": float("nan"),
                }
            # f_iapp = pre-event ambient. Need at least 10 s of pre-event signal
            # for reliable FFT (resolution ≤ 0.1 Hz at 1st mode). If STA/LTA
            # fired early (close to window start), fall back to a fixed
            # [0, t_first - 2s] minimum or the full pre-NCREE window.
            min_pre_n = int(10 * TARGET_FS)
            ipre_end = segs["t_first"]
            if ipre_end < min_pre_n:
                # Use fixed pre-NCREE window [0, PRE_EVENT_SEC * fs - 2s]
                ipre_end = max(min_pre_n, int(PRE_EVENT_SEC * TARGET_FS) - int(2 * TARGET_FS))
            f_iapp_4f, _ = _modal_in_segment(w10f_full, TARGET_FS,
                                              {"a": 0, "b": ipre_end},
                                              "a", "b", F_EMP_MIN, F_EMP_MAX)
            f_iapp_1f, _ = _modal_in_segment(d006_full, TARGET_FS,
                                              {"a": 0, "b": ipre_end},
                                              "a", "b", F_EMP_MIN, F_EMP_MAX)
            # f_min = minimum 1st-mode instantaneous frequency during DSM.
            # Event-adaptive bandpass centered on f_iapp to keep IF narrow-band.
            f_min_4f, t_min_4f = _f_min_strong_motion(
                w10f_full, TARGET_FS, segs,
                F_MIN_SEARCH_LO, F_MIN_SEARCH_HI,
                f_center=f_iapp_4f, mode="asymmetric")
            f_min_1f, t_min_1f = _f_min_strong_motion(
                d006_full, TARGET_FS, segs,
                F_MIN_SEARCH_LO, F_MIN_SEARCH_HI,
                f_center=f_iapp_1f, mode="asymmetric")
            # v5 reviewer §3: symmetric-band control measurement on both
            # sensors (timing not needed — only the value).
            f_min_sym_4f, _ = _f_min_strong_motion(
                w10f_full, TARGET_FS, segs,
                F_MIN_SEARCH_LO, F_MIN_SEARCH_HI,
                f_center=f_iapp_4f, mode="symmetric")
            f_min_sym_1f, _ = _f_min_strong_motion(
                d006_full, TARGET_FS, segs,
                F_MIN_SEARCH_LO, F_MIN_SEARCH_HI,
                f_center=f_iapp_1f, mode="symmetric")
            def _diff_pct(f_iapp, f_a, f_s):
                if (f_iapp is None or np.isnan(f_iapp) or f_iapp <= 0
                        or f_a is None or np.isnan(f_a)
                        or f_s is None or np.isnan(f_s)):
                    return float("nan")
                return (f_s - f_a) / f_iapp * 100.0
            f_min_diff_pct_4f = _diff_pct(f_iapp_4f, f_min_4f, f_min_sym_4f)
            f_min_diff_pct_1f = _diff_pct(f_iapp_1f, f_min_1f, f_min_sym_1f)
            # f_99app = post-DSM recovery, [t95, t99]
            f_99_4f, _ = _modal_in_segment(w10f_full, TARGET_FS,
                                            {"a": segs["t95"], "b": segs["t99"]},
                                            "a", "b", F_EMP_MIN, F_EMP_MAX)
            f_99_1f, _ = _modal_in_segment(d006_full, TARGET_FS,
                                            {"a": segs["t95"], "b": segs["t99"]},
                                            "a", "b", F_EMP_MIN, F_EMP_MAX)
            operational = {
                "f_iapp_4f": f_iapp_4f, "f_min_4f": f_min_4f, "f_99app_4f": f_99_4f,
                "f_iapp_1f": f_iapp_1f, "f_min_1f": f_min_1f, "f_99app_1f": f_99_1f,
                "f_min_sym_4f": f_min_sym_4f, "f_min_diff_pct_4f": f_min_diff_pct_4f,
                "f_min_sym_1f": f_min_sym_1f, "f_min_diff_pct_1f": f_min_diff_pct_1f,
                "t_first_arrival_sec": segs["t_first_sec"],
                "t5_sec": segs["t5_sec"], "t95_sec": segs["t95_sec"],
                "t99_sec": segs["t99_sec"], "arias_total": segs["arias_total"],
            }

            # ---- CWT of response signal (Todorovska 2007 / Astorga 2018) ----
            try:
                feat_4f = _cwt_w10f(w10f_full, TARGET_FS, pre_event_sec=PRE_EVENT_SEC)
                feat_1f = _cwt_w10f(d006_full, TARGET_FS, pre_event_sec=PRE_EVENT_SEC)
            except ValueError as exc:
                rows.append(_row(event, meta, comp, None, None, tf_peaks, None, operational, f"feat_fail:{exc}"))
                continue

            ops_4f = {
                "f_iapp":  f_iapp_4f, "f_min": f_min_4f, "f_99app": f_99_4f,
                "t_min": t_min_4f,
            }
            ops_1f = {
                "f_iapp":  f_iapp_1f, "f_min": f_min_1f, "f_99app": f_99_1f,
                "t_min": t_min_1f,
            }
            _plot_cwt_annotated(
                feat_4f, event_dir / f"cwt_4F_{comp}.png",
                title=f"{event.event_id} {comp}  W10F (4F)", segs=segs,
                ops=ops_4f, label_pre="W10F 4F", y_max=15.0,
            )
            _plot_cwt_annotated(
                feat_1f, event_dir / f"cwt_1F_{comp}.png",
                title=f"{event.event_id} {comp}  D006 (1F)", segs=segs,
                ops=ops_1f, label_pre="D006 1F", y_max=15.0,
            )

            # ---- FFT comparison ----
            fft_peaks = _plot_fft(
                event_dir / f"fft_{comp}.png", TARGET_FS, comp,
                [("NCREE", ncree_post), ("D006(1F)", d006_post), ("W10F(4F)", w10f_post)],
            )

            # ---- Raw waveforms (with pre-event window + Arias segments) ----
            _plot_raw(
                event_dir / f"raw_{comp}.png", TARGET_FS, comp,
                [
                    ("NCREE", t_axis, ncree_full),
                    ("D006(1F)", t_axis, d006_full),
                    ("W10F(4F)", t_axis, w10f_full),
                ],
                segments=segs,
            )

            rows.append(_row(event, meta, comp, feat_4f, feat_1f, tf_peaks, fft_peaks, operational, "OK"))
        return event.event_id, rows, None
    except Exception as exc:
        return event.event_id, rows, f"{type(exc).__name__}: {exc}"
    finally:
        gc.collect()


def _row(event, meta, comp, f4, f1, tf_peaks, fft_peaks, operational, status):
    def _g(k):
        v = meta.get(k) if meta else None
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else v
    def _fmt(v, spec=".4f"):
        if v is None: return ""
        try:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return ""
        except Exception:
            pass
        return f"{v:{spec}}"
    op = operational or {}
    tf4 = tf_peaks.get("NCREE → W10F (4F)", (None, None)) if tf_peaks else (None, None)
    tf1 = tf_peaks.get("NCREE → D006 (1F)", (None, None)) if tf_peaks else (None, None)
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
        # Legacy CWT(IRF) columns deprecated — replaced by CWT(response) plots
        # using operational f_iapp / f_min / f_99app per Astorga 2018.
        "f_init_4f":   "", "f_lowest_4f": "", "f_final_4f": "", "t_lowest_4f": "",
        "f_init_1f":   "", "f_lowest_1f": "", "f_final_1f": "", "t_lowest_1f": "",
        "tf_peak_freq_4f": f"{tf4[0]:.3f}" if tf4[0] is not None and not np.isnan(tf4[0]) else "",
        "tf_peak_amp_4f":  f"{tf4[1]:.4g}" if tf4[1] is not None and not np.isnan(tf4[1]) else "",
        "tf_peak_freq_1f": f"{tf1[0]:.3f}" if tf1[0] is not None and not np.isnan(tf1[0]) else "",
        "tf_peak_amp_1f":  f"{tf1[1]:.4g}" if tf1[1] is not None and not np.isnan(tf1[1]) else "",
        "tf_f1_4f": f"{tf_peaks['_f1_4f']:.3f}" if tf_peaks and not np.isnan(tf_peaks.get('_f1_4f', np.nan)) else "",
        "tf_f1_1f": f"{tf_peaks['_f1_1f']:.3f}" if tf_peaks and not np.isnan(tf_peaks.get('_f1_1f', np.nan)) else "",
        "f_empirical": f"{F_EMPIRICAL:.3f}",
        "f_emp_min":   f"{F_EMP_MIN:.3f}",
        "f_emp_max":   f"{F_EMP_MAX:.3f}",
        # New operational features (Astorga 2018 / Michel-Guéguen)
        "f_iapp_4f":   _fmt(op.get("f_iapp_4f"), ".3f"),
        "f_min_4f":    _fmt(op.get("f_min_4f"),  ".3f"),
        "f_99app_4f":  _fmt(op.get("f_99app_4f"),".3f"),
        "f_iapp_1f":   _fmt(op.get("f_iapp_1f"), ".3f"),
        "f_min_1f":    _fmt(op.get("f_min_1f"),  ".3f"),
        "f_99app_1f":  _fmt(op.get("f_99app_1f"),".3f"),
        "f_min_sym_4f":       _fmt(op.get("f_min_sym_4f"), ".3f"),
        "f_min_diff_pct_4f":  _fmt(op.get("f_min_diff_pct_4f"), ".2f"),
        "f_min_sym_1f":       _fmt(op.get("f_min_sym_1f"), ".3f"),
        "f_min_diff_pct_1f":  _fmt(op.get("f_min_diff_pct_1f"), ".2f"),
        "t_first_arrival_sec": _fmt(op.get("t_first_arrival_sec"), ".2f"),
        "t5_sec":  _fmt(op.get("t5_sec"),  ".2f"),
        "t95_sec": _fmt(op.get("t95_sec"), ".2f"),
        "t99_sec": _fmt(op.get("t99_sec"), ".2f"),
        "arias_total": _fmt(op.get("arias_total"), ".4g"),
        "damping_ratio_4f": _fmt(tf_peaks.get("_zeta_4f") if tf_peaks else None, ".4f"),
        "damping_ratio_1f": _fmt(tf_peaks.get("_zeta_1f") if tf_peaks else None, ".4f"),
        "fft_peak_ncree": f"{fft_peaks['NCREE']:.3f}" if fft_peaks else "",
        "fft_peak_d006":  f"{fft_peaks['D006(1F)']:.3f}" if fft_peaks else "",
        "fft_peak_w10f":  f"{fft_peaks['W10F(4F)']:.3f}" if fft_peaks else "",
        "fft_peak_cb_ncree": _fmt(fft_peaks.get("NCREE_cb") if fft_peaks else None, ".3f"),
        "fft_peak_cb_d006":  _fmt(fft_peaks.get("D006(1F)_cb") if fft_peaks else None, ".3f"),
        "fft_peak_cb_w10f":  _fmt(fft_peaks.get("W10F(4F)_cb") if fft_peaks else None, ".3f"),
        "segment_source": SEGMENT_SOURCE,
        "status": status or "OK",
    }


def _rolling_trend(times, values, window=30):
    """Return (t_sorted, rolling_median) — time-ordered rolling median over a
    fixed event window. Used as a smooth, outlier-robust trend line.
    """
    import pandas as pd
    # Use raw numpy arrays for index/values to avoid pandas auto-reindex
    # turning everything into NaN.
    t_arr = np.asarray(times.values if hasattr(times, "values") else times)
    v_arr = np.asarray(values.values if hasattr(values, "values") else values,
                       dtype=float)
    s = pd.Series(v_arr, index=t_arr).dropna().sort_index()
    if len(s) < max(5, window // 2):
        return s.index.values, s.values
    rolled = s.rolling(window=window, min_periods=max(3, window // 4),
                        center=True).median()
    return rolled.index.values, rolled.values


# Reference event for segmented analysis: 2024-04-03 M7.2 Hualien mainshock
# (UTC 2024-04-02 23:58:11). Per Astorga 2018 / Brenguier 2008 / Michel-Guéguen,
# split OLS at clear physical discontinuity rather than fitting through it.
MAINSHOCK_UTC = "2024-04-02 23:58:11"


def _linear_fit(times_pd, y_pd):
    """OLS linear regression y(t) with slope (per year), 95% CI, p-value.

    Returns dict with keys: tt (datetimes), yy (fit), ci_lo, ci_hi,
    slope_per_yr, p_value, n.  Or None if too few points.
    """
    import pandas as pd
    from scipy import stats
    t_arr = pd.to_datetime(times_pd).values
    y_arr = pd.to_numeric(y_pd, errors="coerce").values
    m = ~np.isnan(y_arr)
    if m.sum() < 10:
        return None
    t_used = t_arr[m]
    y_used = y_arr[m]
    # Convert datetimes -> years since first event (numeric)
    t0 = t_used.min()
    x = (t_used - t0).astype("timedelta64[s]").astype(float) / (365.25 * 24 * 3600)
    res = stats.linregress(x, y_used)
    # Fit + 95% CI envelope (mean response, not prediction interval)
    x_grid = np.linspace(x.min(), x.max(), 200)
    y_fit = res.intercept + res.slope * x_grid
    se = res.stderr  # SE of slope
    se_inter = res.intercept_stderr
    # Mean-response CI: se_y(x0) = sqrt(se_inter² + (x0·se_slope)²) [approx]
    se_y = np.sqrt(se_inter ** 2 + (x_grid * se) ** 2)
    tcrit = stats.t.ppf(0.975, df=len(x) - 2)
    ci_lo = y_fit - tcrit * se_y
    ci_hi = y_fit + tcrit * se_y
    tt = t0 + (x_grid * 365.25 * 24 * 3600 * 1e9).astype("timedelta64[ns]")
    return {
        "tt": tt, "yy": y_fit, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "slope_per_yr": float(res.slope),
        "p_value": float(res.pvalue),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue ** 2),
        "n": int(m.sum()),
    }


def _plot_summary_one(csv_path, out_dir, kind):
    """Produce THREE figures per kind ('cwt' or 'tf'):
       - <out_dir>/summary_<kind>_scatter.png : events only
       - <out_dir>/summary_<kind>_rolling.png : scatter + rolling median
       - <out_dir>/summary_<kind>_ols.png     : scatter + OLS linear fit + 95% CI
    All saved at 300 dpi; canvas widened to keep legends outside plotting area.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    df["t"] = pd.to_datetime(df["utc_time"])

    if kind == "cwt":
        series = [
            ("f_iapp_4f",  "tab:blue",   "f_iapp (pre-event ambient)"),
            ("f_min_4f",   "tab:orange", "f_min (coseismic, DSM)"),
            ("f_99app_4f", "tab:green",  "f_99app (post-DSM recovery)"),
        ]
        title_root = "4F Operational Modal Frequencies (NCREE→W10F)"
    else:
        series = [
            ("tf_peak_freq_4f", "black",     "TF peak (full band)"),
            ("tf_f1_4f",        "tab:purple","f₁ (code band)"),
        ]
        title_root = "4F Transfer-Function Peaks (NCREE→W10F)"

    trend_colors = {
        "tab:blue":   "#08306b", "tab:orange": "#7f2704",
        "tab:green":  "#00441b", "black":      "#d62728",
        "tab:purple": "#4a148c",
    }

    def _marker(col):
        if col == "tf_peak_freq_4f": return "x"
        if col == "tf_f1_4f":        return "*"
        return "o"

    def _scatter_layer(ax, sub):
        for col, color, label in series:
            y = pd.to_numeric(sub[col], errors="coerce")
            ax.scatter(sub["t"].values, y.values, c=color, s=22, alpha=0.45,
                        marker=_marker(col), label=label, zorder=3,
                        edgecolors="none")

    if kind == "cwt":
        caption = (
            "Operational modal-frequency definitions (after Astorga et al. 2018 BSSA;\n"
            "Michel & Guéguen 2010): f_iapp = pre-event ambient apparent frequency\n"
            "(STA/LTA pick); f_min = coseismic minimum during DSM (Arias 5–95%);\n"
            "f_99app = apparent frequency at 99% cumulative Arias intensity\n"
            "(post-DSM recovery). 'apparent' denotes the soil-structure system\n"
            "frequency, distinct from fixed-base modal frequency (Todorovska 2007)."
        )
    else:
        caption = (
            "TF peak: frequency at global max of |H(f)| from water-level\n"
            "deconvolution (Snieder & Şafak 2006); may capture higher modes when\n"
            "1st-mode response is weak. f₁ (code band): peak constrained to the\n"
            "Taiwan Seismic Code §2.6 empirical band [f_emp/1.4, f_emp×1.4]."
        )

    def _finalize(fig, axes, subtitle, out_path):
        for ax in axes:
            ax.grid(alpha=0.3)
            # Legend OUTSIDE plot area to avoid being blocked by lines
            leg = ax.legend(loc="center left",
                             bbox_to_anchor=(1.005, 0.5),
                             fontsize=8, framealpha=0.95,
                             borderaxespad=0)
            leg.set_zorder(20)
        axes[-1].set_xlabel("Event Time (UTC)")
        fig.suptitle(f"{title_root} — {subtitle} — 2023–2026", fontsize=12)
        # Caption box bottom-left, outside main axes
        fig.text(0.01, 0.005, caption, fontsize=7.5,
                  family="monospace", color="#333333",
                  bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#999999"))
        fig.tight_layout(rect=(0, 0.10, 0.78, 1.0))  # reserve right + bottom
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ---- Figure 1: scatter only ----
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].sort_values("t")
        _draw_empirical_band(ax, axis="y")
        _scatter_layer(ax, sub)
        ax.set_ylabel(f"{comp}  Freq (Hz)")
    _finalize(fig, axes, "Event Scatter",
               out_dir / f"summary_{kind}_scatter.png")

    # ---- Figure 2: scatter + rolling-median trend ----
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].sort_values("t")
        _draw_empirical_band(ax, axis="y")
        _scatter_layer(ax, sub)
        for col, color, label in series:
            y = pd.to_numeric(sub[col], errors="coerce")
            tt, yy = _rolling_trend(sub["t"], y, window=30)
            ax.plot(tt, yy, color=trend_colors[color], linewidth=2.6,
                    alpha=0.95, zorder=10, solid_capstyle="round",
                    label=f"{label} — rolling median (n=30)")
        ax.set_ylabel(f"{comp}  Freq (Hz)")
    _finalize(fig, axes, "Scatter + Rolling-Median Trend",
               out_dir / f"summary_{kind}_rolling.png")

    # ---- Figure 3: scatter + OLS linear regression + 95% CI ----
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].sort_values("t")
        _draw_empirical_band(ax, axis="y")
        _scatter_layer(ax, sub)
        for col, color, label in series:
            y = pd.to_numeric(sub[col], errors="coerce")
            lf = _linear_fit(sub["t"], y)
            if lf is None:
                continue
            tc = trend_colors[color]
            ax.fill_between(lf["tt"], lf["ci_lo"], lf["ci_hi"],
                            color=tc, alpha=0.18, zorder=5,
                            linewidth=0)
            slope_label = (f"{label} OLS: "
                            f"{lf['slope_per_yr']*1000:+.1f} mHz/yr "
                            f"(p={lf['p_value']:.2g}, R²={lf['r2']:.2f})")
            ax.plot(lf["tt"], lf["yy"], color=tc, linewidth=2.4,
                    linestyle="-", alpha=0.95, zorder=10,
                    label=slope_label)
        ax.set_ylabel(f"{comp}  Freq (Hz)")
    _finalize(fig, axes, "Scatter + OLS Linear Regression (95% CI)",
               out_dir / f"summary_{kind}_ols.png")

    # ---- Figure 4: scatter + SEGMENTED OLS at 2024-04-03 M7.2 mainshock ----
    cutoff = pd.Timestamp(MAINSHOCK_UTC)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, comp in zip(axes, ("E", "N", "Z")):
        sub = df[df["comp"] == comp].sort_values("t")
        _draw_empirical_band(ax, axis="y")
        _scatter_layer(ax, sub)
        # Vertical line marking the mainshock
        ax.axvline(cutoff, color="red", linestyle="-.", linewidth=1.4,
                   alpha=0.7, zorder=6,
                   label=f"M7.2 mainshock 2024-04-03")
        for col, color, label in series:
            tc = trend_colors[color]
            y_all = pd.to_numeric(sub[col], errors="coerce")
            pre_mask = sub["t"] < cutoff
            post_mask = sub["t"] >= cutoff
            # Pre-mainshock segment
            lf_pre = _linear_fit(sub.loc[pre_mask, "t"], y_all[pre_mask])
            if lf_pre is not None:
                ax.fill_between(lf_pre["tt"], lf_pre["ci_lo"], lf_pre["ci_hi"],
                                color=tc, alpha=0.15, zorder=5, linewidth=0)
                ax.plot(lf_pre["tt"], lf_pre["yy"], color=tc, linewidth=2.2,
                        linestyle="-", alpha=0.95, zorder=10,
                        label=(f"{label} PRE: {lf_pre['slope_per_yr']*1000:+.1f} mHz/yr "
                                f"(p={lf_pre['p_value']:.2g}, n={lf_pre['n']})"))
            # Post-mainshock segment
            lf_post = _linear_fit(sub.loc[post_mask, "t"], y_all[post_mask])
            if lf_post is not None:
                ax.fill_between(lf_post["tt"], lf_post["ci_lo"], lf_post["ci_hi"],
                                color=tc, alpha=0.15, zorder=5, linewidth=0)
                ax.plot(lf_post["tt"], lf_post["yy"], color=tc, linewidth=2.2,
                        linestyle="--", alpha=0.95, zorder=10,
                        label=(f"{label} POST: {lf_post['slope_per_yr']*1000:+.1f} mHz/yr "
                                f"(p={lf_post['p_value']:.2g}, n={lf_post['n']})"))
        ax.set_ylabel(f"{comp}  Freq (Hz)")
    _finalize(fig, axes, "Segmented OLS — Pre vs Post M7.2 Mainshock",
               out_dir / f"summary_{kind}_ols_segmented.png")


def plot_concept_diagram(event, out_path):
    """Produce an annotated concept figure illustrating f_iapp / f_min /
    f_99app on a real event (default: M7.2 mainshock).

    Top: W10F vertical-component acceleration with STA/LTA arrival + Arias 5/95/99%
         segment markers and the three measurement windows shaded.
    Bottom: 1st-mode instantaneous frequency (Hilbert IF on bandpass-filtered
         W10F-N) with f_iapp / f_min / f_99app annotated.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    streams = _load_traces(event)
    t_ref = streams["NCREE"]["Z"].stats.starttime
    fs = TARGET_FS
    comp = "N"
    w10f_tr = _resample_to(streams["W10F"][comp], fs)
    w10f_full = _slice_window(w10f_tr, t_ref, PRE_EVENT_SEC, POST_EVENT_SEC, fs)
    finite = ~np.isnan(w10f_full)
    if finite.any():
        w10f_full[finite] -= w10f_full[finite].mean()
    t_axis = np.arange(len(w10f_full)) / fs - PRE_EVENT_SEC

    segs = _event_segments(w10f_full, fs, pre_event_sec=PRE_EVENT_SEC)
    if segs is None:
        return
    f_iapp, _ = _modal_in_segment(w10f_full, fs,
                                    {"a": 0, "b": segs["t_first"]},
                                    "a", "b", F_EMP_MIN, F_EMP_MAX)
    f_min, t_min = _f_min_strong_motion(
        w10f_full, fs, segs, F_MIN_SEARCH_LO, F_MIN_SEARCH_HI, f_center=f_iapp)
    f_99, _ = _modal_in_segment(w10f_full, fs,
                                  {"a": segs["t95"], "b": segs["t99"]},
                                  "a", "b", F_EMP_MIN, F_EMP_MAX)

    # Hilbert IF time series for bottom panel (uses same bandpass as _f_min)
    if not np.isnan(f_iapp) and f_iapp > 0:
        lo = max(0.3, f_iapp / F_MIN_BAND_DOWN_ASYM); hi = min(fs/2 - 0.1, f_iapp * F_MIN_BAND_UP_ASYM)
        sos = butter(6, [lo, hi], btype="band", fs=fs, output="sos")
        bp = sosfiltfilt(sos, np.nan_to_num(w10f_full))
        analytic = hilbert(bp)
        phase = np.unwrap(np.angle(analytic))
        if_curve = np.diff(phase) / (2*np.pi) * fs
        if_curve = np.append(if_curve, if_curve[-1])
        if_curve = uniform_filter1d(if_curve, size=int(fs))
        env = np.abs(analytic)
    else:
        if_curve = np.full_like(w10f_full, np.nan)
        env = np.zeros_like(w10f_full)

    fig, (ax_a, ax_f) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    # ----- TOP: acceleration -----
    ax_a.plot(t_axis, w10f_full, color="black", linewidth=0.6, zorder=3)
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
    ax_a.set_ylabel("W10F (4F) N-axis\nAcceleration (gal)")
    ax_a.grid(alpha=0.3)
    ax_a.set_title(f"Operational Modal-Frequency Definitions — Example: {event.event_id} (M7.2 mainshock)",
                    fontsize=12)
    ax_a.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
                  fontsize=8, framealpha=0.95)

    # ----- BOTTOM: instantaneous frequency -----
    ax_f.plot(t_axis, if_curve, color="tab:orange", linewidth=0.8, alpha=0.6,
              label="Hilbert instantaneous freq (bandpass on f_iapp)")
    _draw_empirical_band(ax_f, axis="y")
    # Three measurement points
    pts = [
        ("f_iapp", segs["t_first_sec"]/2, f_iapp, "tab:blue"),
        ("f_min", t_min if t_min is not None else (segs["t5_sec"]+segs["t95_sec"])/2,
         f_min, "tab:orange"),
        ("f_99app", (segs["t95_sec"]+segs["t99_sec"])/2, f_99, "tab:green"),
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
    # Annotate softening + recovery arrows
    if not (np.isnan(f_iapp) or np.isnan(f_min)):
        ax_f.annotate("",
                       xy=(t_min if t_min else (segs["t5_sec"]+segs["t95_sec"])/2, f_min),
                       xytext=(segs["t_first_sec"]/2, f_iapp),
                       arrowprops=dict(arrowstyle="->", color="red", lw=1.8,
                                        alpha=0.6))
        drop_pct = (f_iapp - f_min) / f_iapp * 100
        midx = (segs["t_first_sec"]/2 + (segs["t5_sec"]+segs["t95_sec"])/2) / 2
        midy = (f_iapp + f_min) / 2
        ax_f.text(midx, midy + 0.05,
                   f"coseismic\nsoftening {drop_pct:.0f}%",
                   color="red", fontsize=9, fontweight="bold", ha="center")
    if not (np.isnan(f_99) or np.isnan(f_min) or np.isnan(f_iapp)):
        # Recovery indicator
        recovery = (f_99 - f_min) / (f_iapp - f_min + 1e-9) * 100 if f_iapp > f_min else 0
        ax_f.text(0.99, 0.05,
                   f"Recovery (f_99 vs pre-event): "
                   f"{(f_99/f_iapp-1)*100:+.0f}%",
                   transform=ax_f.transAxes, ha="right",
                   fontsize=9, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    ax_f.set_xlabel("Time (s) — t=0 at NCREE trigger")
    ax_f.set_ylabel("1st-mode\nFrequency (Hz)")
    ax_f.set_ylim(1.0, 3.5)
    ax_f.grid(alpha=0.3)
    ax_f.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
                  fontsize=8, framealpha=0.95)

    fig.tight_layout(rect=(0, 0, 0.78, 1.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_scatter(csv_path: Path, out_dir: Path):
    _plot_summary_one(csv_path, out_dir, kind="cwt")
    _plot_summary_one(csv_path, out_dir, kind="tf")


# ----------------- v5 reviewer additions (D006-specific) -----------------
def _plot_fmin_symmetric_vs_asymmetric_d006(csv_path: Path, out_dir: Path):
    """v5 reviewer §3: f_min softening % under asymmetric vs symmetric bandpass
    for D006 (1F and 4F separately). A persistent offset = estimator bias.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import json
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    if df.empty:
        return
    for col in ("f_iapp_4f", "f_min_4f", "f_min_sym_4f",
                "f_iapp_1f", "f_min_1f", "f_min_sym_1f"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    summary = {"comp": {}}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for floor_idx, (suffix, label) in enumerate((("4f", "4F (W10F)"),
                                                  ("1f", "1F (D006)"))):
        for col_idx, comp in enumerate(("E", "N", "Z")):
            ax = axes[floor_idx, col_idx]
            sub = df[df["comp"] == comp].dropna(
                subset=[f"f_iapp_{suffix}", f"f_min_{suffix}",
                        f"f_min_sym_{suffix}"])
            if len(sub) < 5:
                ax.text(0.5, 0.5, f"{label} {comp}: n={len(sub)}",
                        transform=ax.transAxes, ha="center")
                continue
            soft_asym = ((sub[f"f_iapp_{suffix}"] - sub[f"f_min_{suffix}"])
                          / sub[f"f_iapp_{suffix}"] * 100)
            soft_sym = ((sub[f"f_iapp_{suffix}"] - sub[f"f_min_sym_{suffix}"])
                         / sub[f"f_iapp_{suffix}"] * 100)
            med_diff = float((soft_asym - soft_sym).median())
            ax.scatter(soft_sym, soft_asym, s=14, alpha=0.5,
                       edgecolors="none")
            lim = max(float(soft_asym.max()), float(soft_sym.max()), 5.0)
            ax.plot([0, lim], [0, lim], color="red", linestyle="--",
                    linewidth=1, label="1:1")
            ax.set_xlabel("softening % (symmetric)")
            ax.set_ylabel("softening % (asymmetric)")
            ax.set_title(f"{label} {comp}  n={len(sub)}  "
                         f"med(asym-sym)={med_diff:+.1f} pp")
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right", fontsize=8)
            summary["comp"][f"{suffix}_{comp}"] = {
                "n": int(len(sub)),
                "median_asym_pct": float(soft_asym.median()),
                "median_sym_pct":  float(soft_sym.median()),
                "median_diff_pp":  med_diff,
            }
    fig.suptitle(
        "D006 — f_min estimator bias check: asymmetric vs symmetric bandpass "
        "(reviewer §3)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1.0, 0.96))
    fig.savefig(out_dir / "summary_fmin_symmetric_vs_asymmetric_D006.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "summary_fmin_symmetric_vs_asymmetric_D006.json").open(
            "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def _plot_damping_all_events_d006(main_csv_path: Path,
                                   interstory_csv_path: Path,
                                   out_dir: Path,
                                   n_boot: int = 2000):
    """v5 reviewer §4: D006 damping ratio across ALL events — single-station
    NCREE→4F (`damping_ratio_4f` from main CSV) vs Snieder-Safak inter-story
    1F→4F. Box-plot per axis + bootstrap median CI + 2-sample bootstrap of
    the median difference. Addresses the reviewer's complaint that the
    16% gap rested on a single event with no uncertainty.

    interstory_csv_path is the per-event log written by
    src.d006_interstory.run_interstory. If it does not exist, the function
    no-ops (so the v5 main can be re-run without rerunning interstory).
    """
    import json
    import matplotlib.pyplot as plt
    import pandas as pd

    if not main_csv_path.exists() or not interstory_csv_path.exists():
        return
    df_main = pd.read_csv(main_csv_path)
    df_main = df_main[df_main["status"] == "OK"].copy()
    df_main["zeta_single"] = pd.to_numeric(df_main.get("damping_ratio_4f"),
                                            errors="coerce")
    df_inter = pd.read_csv(interstory_csv_path)

    # v5 interstory CSV writes per-component zeta_inter_{E,N,Z} columns
    # (see src.d006_interstory._zeta_halfpower). Fall back to legacy single
    # column names if a downstream CSV variant supplies one.
    has_per_comp = all(f"zeta_inter_{c}" in df_inter.columns
                       for c in ("E", "N", "Z"))
    legacy_col = None
    if not has_per_comp:
        for cand in ("zeta_interstory", "damping_ratio", "zeta",
                     "damping_interstory"):
            if cand in df_inter.columns:
                legacy_col = cand
                break
        if legacy_col is None:
            return

    rng = np.random.default_rng(20260405)
    summary = {"n_boot": n_boot, "comp": {}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, comp in zip(axes, ("E", "N", "Z")):
        zs = df_main.loc[df_main["comp"] == comp, "zeta_single"].dropna().values
        if has_per_comp:
            zi = pd.to_numeric(df_inter[f"zeta_inter_{comp}"],
                               errors="coerce").dropna().values
        elif "comp" in df_inter.columns:
            zi = pd.to_numeric(df_inter.loc[df_inter["comp"] == comp, legacy_col],
                                errors="coerce").dropna().values
        else:
            zi = pd.to_numeric(df_inter[legacy_col],
                                errors="coerce").dropna().values
        if len(zs) < 5 or len(zi) < 5:
            ax.text(0.5, 0.5, f"{comp}: n_single={len(zs)}  n_inter={len(zi)}",
                    transform=ax.transAxes, ha="center")
            summary["comp"][comp] = {"status": "insufficient",
                                     "n_single": int(len(zs)),
                                     "n_inter": int(len(zi))}
            continue
        bp = ax.boxplot([zs, zi], labels=["single (NCREE->4F)",
                                          "interstory (1F->4F)"],
                        showfliers=False, widths=0.55, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#9ecae1", "#fdae6b"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        # Bootstrap median CIs
        boot_s = np.median(rng.choice(zs, size=(n_boot, len(zs)),
                                       replace=True), axis=1)
        boot_i = np.median(rng.choice(zi, size=(n_boot, len(zi)),
                                       replace=True), axis=1)
        diffs = boot_s - boot_i
        ci_s = np.percentile(boot_s, [2.5, 97.5])
        ci_i = np.percentile(boot_i, [2.5, 97.5])
        ci_d = np.percentile(diffs, [2.5, 97.5])
        m_s = float(np.median(zs)); m_i = float(np.median(zi))
        sig = "sig" if not (ci_d[0] <= 0 <= ci_d[1]) else "n.s."
        ax.set_ylabel("damping ratio ζ")
        ax.set_title(
            f"{comp}  n_s={len(zs)}  n_i={len(zi)}\n"
            f"med_s={m_s:.3f} [{ci_s[0]:.3f}, {ci_s[1]:.3f}]\n"
            f"med_i={m_i:.3f} [{ci_i[0]:.3f}, {ci_i[1]:.3f}]\n"
            f"Δ CI=[{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]  {sig}",
            fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        summary["comp"][comp] = {
            "n_single": int(len(zs)), "n_inter": int(len(zi)),
            "median_single": m_s, "median_single_ci95": [float(ci_s[0]), float(ci_s[1])],
            "median_inter":  m_i, "median_inter_ci95":  [float(ci_i[0]), float(ci_i[1])],
            "delta_single_minus_inter_ci95": [float(ci_d[0]), float(ci_d[1])],
            "significant_at_95pct": sig == "sig",
        }
    fig.suptitle(
        "D006 damping ratio — single-station NCREE->4F vs Snieder-Safak "
        "interstory 1F->4F (all events, bootstrap 95% CI)  [reviewer §4]",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1.0, 0.94))
    fig.savefig(out_dir / "damping_all_events.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "damping_all_events.json").open(
            "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def main():
    _setup_logging()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    events, skipped = build_event_list()
    events = [e for e in events if e.event_id[:4] >= YEAR_MIN]
    print(f"Filtered to {YEAR_MIN}+: {len(events)} events")

    with SKIP_LOG_PATH.open("w", encoding="utf-8") as f:
        for s in skipped:
            f.write(f"{s.event_id}\t{s.reason}\n")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        n_workers = max(1, (os.cpu_count() or 2) - 2)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(process_event, e) for e in events]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Events"):
                event_id, rows, err = fut.result()
                if err:
                    logging.error("Event %s failed: %s", event_id, err)
                for r in rows:
                    writer.writerow(r)

    plot_summary_scatter(CSV_PATH, OUTPUT_ROOT)

    # Concept figure: generated once for the 2024-04-03 M7.2 Hualien mainshock
    # only (illustrative figure for paper / report use).
    main_id = "20240402235754"
    m7_event = next((e for e in events if e.event_id == main_id), None)
    if m7_event is not None:
        try:
            plot_concept_diagram(m7_event, OUTPUT_ROOT / "concept_frequencies.png")
            print(f"Concept figure: {OUTPUT_ROOT / 'concept_frequencies.png'}")
        except Exception as exc:
            logging.error("concept figure failed: %s", exc)

    # §4.3 (reviewer): D006 has both 1F and 4F sensors — run the inter-story
    # Snieder-Şafak deconvolution to get an SSI-free f_fixed estimate. This
    # is the only NDHU station that supports this analysis.
    interstory_dir = OUTPUT_ROOT / "interstory"
    try:
        from src.d006_interstory import run_interstory
        run_interstory(out_root=interstory_dir)
    except Exception as exc:
        logging.error("inter-story deconvolution failed: %s", exc)

    # ---- v5 reviewer additions ----
    try:
        _plot_fmin_symmetric_vs_asymmetric_d006(CSV_PATH, OUTPUT_ROOT)
    except Exception as exc:
        logging.error("f_min sym/asym D006 figure failed: %s", exc)
    # The interstory summary CSV name is typically interstory_summary.csv.
    try:
        _plot_damping_all_events_d006(
            CSV_PATH, interstory_dir / "interstory_summary.csv", OUTPUT_ROOT,
        )
    except Exception as exc:
        logging.error("all-event damping comparison failed: %s", exc)

    print(f"\nOutputs: {OUTPUT_ROOT}\nCSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
