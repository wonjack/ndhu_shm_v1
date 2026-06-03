"""§2.3, §2.5, §2.6 — sensitivity sweep on the M7.2 mainshock event.

Runs a parameter grid on one event/component to show how robust the headline
metrics (TF_f1, f_min) are to the chosen processing parameters:

- ε : water-level deconvolution regularisation
       Snieder & Şafak 2006 used 0.10; we use 0.05. Reviewer asked for a
       sensitivity test in {0.01, 0.05, 0.10, 0.20}.
- ω₀ : Morlet centre frequency for CWT.
       Default `cmor1.5-1.0` → angular centre ≈ 2π ≈ 6.28. We test
       `cmor1.0-0.5`, `cmor1.5-1.0`, `cmor1.5-2.0` so ω₀ ≈ {π, 2π, 4π}.
- BW : f_min filter band as a multiplier of f_iapp.
       Default [f/1.7, f×1.15]. We test {[0.5, 1.1], [0.7, 1.15], [0.5, 1.2]}.

Scope (user spec): D006 (closest to NCREE, best SNR), N component, mainshock
event 20240402235754. One figure per parameter group.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pywt
from obspy import read
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.signal.windows import tukey
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline_common import (
    MAINSHOCK_EVENT_ID,
    PRE_EVENT_SEC,
    POST_EVENT_SEC,
    TARGET_FS,
    TUKEY_ALPHA,
    KO_BANDWIDTH,
    KO_LOG_BINS,
    _ko_smooth,
    _peak_freq,
    _resample_to,
    _slice_window,
    _event_segments,
)
from src.station_config import STATIONS, code_band


def _tapered(x, alpha=TUKEY_ALPHA):
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return x.copy()
    return (x - x.mean()) * tukey(len(x), alpha=alpha)


def _tf_with_eps(input_sig, output_sig, fs, epsilon):
    n = min(len(input_sig), len(output_sig))
    x = _tapered(input_sig[:n])
    y = _tapered(output_sig[:n])
    X = np.fft.rfft(x); Y = np.fft.rfft(y)
    power = np.abs(X) ** 2
    water = epsilon * power.max()
    denom = np.maximum(power, water)
    H = Y * np.conjugate(X) / denom
    freqs_lin = np.fft.rfftfreq(n, 1.0 / fs)
    return _ko_smooth(freqs_lin, np.abs(H))


def _cwt_with_wavelet(signal, fs, wavelet_str, freqs_target):
    sig = np.asarray(signal, dtype=float)
    scales = pywt.central_frequency(wavelet_str) * fs / freqs_target
    cwt_matrix, _ = pywt.cwt(sig, scales, wavelet_str,
                             sampling_period=1.0 / fs)
    return cwt_matrix


def _f_min_with_band(signal_full, fs, segments, fmin, fmax, f_iapp,
                     lo_mult, hi_mult):
    a, b = segments["t5"], segments["t95"]
    if b - a < int(2 * fs) or not np.isfinite(f_iapp) or f_iapp <= 0:
        return float("nan")
    lo = max(0.3, f_iapp * lo_mult)
    hi = min(fs / 2 - 0.1, f_iapp * hi_mult)
    if hi - lo < 0.3:
        return float("nan")
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
    if_seg = inst_freq[a:b]; env_seg = envelope[a:b]
    if env_seg.size == 0 or env_seg.max() <= 0:
        return float("nan")
    if_lower = max(lo * 1.02, fmin)
    if_upper = min(hi * 0.98, fmax * 1.05)
    valid = (env_seg > 0.5 * env_seg.max()) & (if_seg >= if_lower) & (if_seg <= if_upper)
    if valid.sum() < int(0.5 * fs):
        return float("nan")
    return float(np.median(if_seg[valid]))


def _load_streams_d006(event):
    """Reuse the legacy D006 event matcher (D006 + W10F + NCREE)."""
    out = {"NCREE": {}, "D006": {}}
    for c, p in event.ncree_paths.items():
        out["NCREE"][c] = read(str(p))[0]
    for c, p in event.palert_1f.items():       # 1F sensor for D006
        out["D006"][c] = read(str(p))[0]
    return out


def run_sweep(out_dir: Path | None = None,
              event_id: str = MAINSHOCK_EVENT_ID,
              comp: str = "N"):
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir) if out_dir is not None else Path(r"D:\PHD\project2026") / "sensitivity_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = STATIONS["D006"]
    matcher = importlib.import_module("src.matcher_ncree")
    events, _ = matcher.build_event_list()
    event = next((e for e in events if e.event_id == event_id), None)
    if event is None:
        raise RuntimeError(f"event {event_id} not found in NCREE matcher")

    streams = _load_streams_d006(event)
    t_ref = streams["NCREE"]["Z"].stats.starttime
    ncree_tr = _resample_to(streams["NCREE"][comp], TARGET_FS)
    struct_tr = _resample_to(streams["D006"][comp], TARGET_FS)

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

    f_emp, f_emp_min, f_emp_max = code_band(cfg)
    title_suffix = f"  event={event_id}  station=D006  comp={comp}"

    # -------------------- ε sweep --------------------
    eps_values = [0.01, 0.05, 0.10, 0.20]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for eps in eps_values:
        freqs, mag = _tf_with_eps(ncree_post, struct_post, TARGET_FS, eps)
        pf, _ = _peak_freq(freqs, mag, fmin=0.4, fmax=15.0)
        f1, _ = _peak_freq(freqs, mag, fmin=f_emp_min, fmax=f_emp_max)
        ax.semilogy(freqs, mag, linewidth=1.4,
                    label=f"ε = {eps:.2f}   TF_peak = {pf:.2f} Hz   TF_f1 = {f1:.2f} Hz")
    ax.axvspan(f_emp_min, f_emp_max, color="cyan", alpha=0.12, zorder=0)
    ax.axvline(f_emp, color="cyan", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"f_emp = {f_emp:.2f} Hz")
    ax.set_xlim(0, 20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H(f)|")
    ax.set_title("Sensitivity: water-level ε  (Snieder & Şafak 2006 default 0.10)" + title_suffix)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_eps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------- ω₀ (Morlet) sweep --------------------
    cwt_freqs = np.logspace(np.log10(0.4), np.log10(15.0), 120)
    morlet_set = [("cmor1.0-0.5", "narrow bw, ω₀ ≈ π (3.14)"),
                  ("cmor1.5-1.0", "DEFAULT, ω₀ ≈ 2π (6.28)"),
                  ("cmor1.5-2.0", "wide bw,  ω₀ ≈ 4π (12.57)")]
    fig, axes = plt.subplots(len(morlet_set), 1, figsize=(13, 11), sharex=True)
    for ax, (wav, lbl) in zip(axes, morlet_set):
        cwt = _cwt_with_wavelet(struct_full, TARGET_FS, wav, cwt_freqs)
        t_axis = np.arange(cwt.shape[1]) / TARGET_FS - PRE_EVENT_SEC
        ax.pcolormesh(t_axis, cwt_freqs, np.abs(cwt), shading="auto", cmap="viridis")
        ax.axhspan(f_emp_min, f_emp_max, color="cyan", alpha=0.20, zorder=2)
        ax.axhline(f_emp, color="cyan", linestyle="--", linewidth=0.8, alpha=0.9, zorder=3)
        ax.set_ylim(0.4, 15)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Morlet wavelet = {wav}   ({lbl})", fontsize=10)
    axes[-1].set_xlabel("Time (s) — t=0 at NCREE trigger")
    fig.suptitle("Sensitivity: Morlet ω₀" + title_suffix, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_morlet.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------- f_min filter bandwidth sweep --------------------
    segs = _event_segments(struct_full, TARGET_FS, pre_event_sec=PRE_EVENT_SEC)
    if segs is not None:
        # pre-event ambient f_iapp for centring the filter
        from src.pipeline_common import _modal_in_segment
        f_iapp, _ = _modal_in_segment(struct_full, TARGET_FS,
                                      {"a": 0, "b": segs["t_first"]},
                                      "a", "b", f_emp_min, f_emp_max)
        band_set = [(0.5, 1.10, "wide low / narrow high"),
                    (1.0 / 1.7, 1.15, "DEFAULT"),
                    (0.7, 1.15, "narrow"),
                    (0.5, 1.20, "wide both")]
        fig, ax = plt.subplots(figsize=(11, 5))
        labels, vals = [], []
        for lo, hi, lbl in band_set:
            f_min_val = _f_min_with_band(struct_full, TARGET_FS, segs,
                                         f_emp * 0.5, f_emp_max, f_iapp, lo, hi)
            labels.append(f"[{lo*f_iapp:.2f}, {hi*f_iapp:.2f}] Hz\n({lbl})")
            vals.append(f_min_val)
        bars = ax.bar(range(len(vals)), vals, color="tab:orange", alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{v:.2f}" if np.isfinite(v) else "NaN",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(f_iapp, color="tab:blue", linestyle="--", linewidth=1.0,
                   label=f"f_iapp = {f_iapp:.2f} Hz")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("f_min (Hz)")
        ax.set_title("Sensitivity: f_min Hilbert-IF filter band  "
                     "(multipliers × f_iapp)" + title_suffix)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "sensitivity_fmin_band.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"Sensitivity sweep written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_sweep()
