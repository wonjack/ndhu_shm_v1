from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace

from . import processor


def plot_raw_waveforms(stream: Stream, output_dir: Path, station_name: str) -> Path:
    traces = _select_components(stream, station_name)
    time_axes = [_time_axis(trace) for trace in traces]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, trace, time_axis in zip(axes, traces, time_axes):
        ax.plot(time_axis, trace.data, linewidth=0.8)
        ax.set_ylabel(f"{trace.stats.channel[-1]} Acceleration (gal)")
        ax.axvline(0.0, color="k", linewidth=0.5, alpha=0.6)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{station_name} Raw Waveforms")
    fig.tight_layout()

    out_path = output_dir / f"{station_name}_raw.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_cwt_with_ridge(
    cwt_matrix: np.ndarray,
    freqs: np.ndarray,
    time_axis: np.ndarray,
    ridge_frequencies: np.ndarray,
    freq_init: float,
    freq_lowest: float,
    freq_final: float,
    idx_init: int,
    idx_lowest: int,
    idx_final: int,
    output_path: Path,
    *,
    title: str,
    y_max: float = 20.0,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pcolormesh(time_axis, freqs, np.abs(cwt_matrix), shading="auto")
    ax.plot(time_axis, ridge_frequencies, color="white", linestyle="--", linewidth=1.2)

    ax.scatter(
        [time_axis[idx_init], time_axis[idx_lowest], time_axis[idx_final]],
        [freq_init, freq_lowest, freq_final],
        c=["tab:blue", "tab:orange", "tab:green"],
        s=25,
        zorder=3,
        label="Init/Lowest/Final",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, y_max)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_fft_spectra(stream: Stream, output_dir: Path, station_name: str) -> Path:
    traces = _select_components(stream, station_name)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, trace in zip(axes, traces):
        freqs, amps = _fft_spectrum(trace.data, trace.stats.sampling_rate)
        ax.plot(freqs, amps, linewidth=0.8)
        valid = freqs > 0
        peak_idx = int(np.argmax(amps[valid]))
        peak_idx = int(np.where(valid)[0][peak_idx])
        peak_freq = freqs[peak_idx]
        peak_amp = amps[peak_idx]
        ax.scatter([peak_freq], [peak_amp], color="red", s=18, zorder=3)
        ax.annotate(
            f"{peak_freq:.2f} Hz",
            xy=(peak_freq, peak_amp),
            xytext=(peak_freq, peak_amp * 0.9 + 1e-12),
            fontsize=8,
            color="red",
        )
        ax.set_ylabel(f"{trace.stats.channel[-1]} Amplitude (gal)")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"{station_name} FFT Spectra")
    fig.tight_layout()

    out_path = output_dir / f"{station_name}_fft.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_deconv_irf(
    irf_streams: Mapping[str, Tuple[np.ndarray, float]],
    output_dir: Path,
    label: str,
) -> Path:
    components = ["E", "N", "Z"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, component in zip(axes, components):
        if component not in irf_streams:
            raise ValueError(f"Missing IRF component {component}")
        irf, sampling_rate = irf_streams[component]
        time_axis = np.arange(len(irf), dtype=float) / sampling_rate
        time_axis -= processor.EVENT_ORIGIN_SEC
        ax.plot(time_axis, irf, linewidth=0.8)
        ax.set_ylabel(f"{component} IRF (unitless)")
        ax.axvline(0.0, color="k", linewidth=0.5, alpha=0.6)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{label} IRF")
    fig.tight_layout()

    out_path = output_dir / f"{label}_irf.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_trend_from_csv(csv_path: Path, output_path: Path) -> Path:
    records = _load_csv_records(csv_path)
    if not records:
        return output_path

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for idx, component in enumerate(["E", "N", "Z"]):
        comp_records = [r for r in records if r["Component"] == component]
        if not comp_records:
            continue
        comp_records = sorted(comp_records, key=lambda r: r["EventTime"])
        times = [r["EventTime"] for r in comp_records]
        freq_init = [r["Freq_Init"] for r in comp_records]
        freq_low = [r["Freq_Lowest"] for r in comp_records]
        freq_final = [r["Freq_Final"] for r in comp_records]
        axes[idx].scatter(times, freq_init, label="Init", s=10)
        axes[idx].scatter(times, freq_low, label="Lowest", s=10)
        axes[idx].scatter(times, freq_final, label="Final", s=10)
        axes[idx].set_ylabel("Frequency (Hz)")
        axes[idx].legend(fontsize=8)
    axes[-1].set_xlabel("Event Time")
    fig.suptitle("Frequency Trends")
    fig.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _select_components(stream: Stream, station_name: str) -> list[Trace]:
    traces: list[Trace] = []
    for component in ("E", "N", "Z"):
        trace = _find_trace(stream, station_name, component)
        traces.append(trace)
    return traces


def _find_trace(stream: Stream, station: str, component: str) -> Trace:
    for trace in stream:
        if trace.stats.station == station and trace.stats.channel.endswith(component):
            return trace
    raise ValueError(f"Missing trace for station={station} component={component}")


def _time_axis(trace: Trace) -> np.ndarray:
    n = trace.stats.npts
    fs = trace.stats.sampling_rate
    return np.arange(n, dtype=float) / fs - processor.EVENT_ORIGIN_SEC


def _fft_spectrum(signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(signal, dtype=float)
    n = data.size
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    amps = np.abs(np.fft.rfft(data)) / n
    return freqs, amps


def _load_csv_records(csv_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                event_time = datetime.strptime(row["Event_ID"], "%Y%m%d%H%M%S")
                records.append(
                    {
                        "EventTime": event_time,
                        "Component": row["Component"],
                        "Freq_Init": float(row["Freq_Init"]),
                        "Freq_Lowest": float(row["Freq_Lowest"]),
                        "Freq_Final": float(row["Freq_Final"]),
                    }
                )
            except (ValueError, KeyError):
                continue
    return records
