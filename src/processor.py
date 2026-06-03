from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pywt
from obspy import Stream, Trace
from scipy.signal import correlate


TARGET_FS = 100.0
FILTER_MIN_HZ = 0.1
FILTER_MAX_HZ = 20.0
FILTER_ORDER = 4
SYNC_MAX_LAG_SEC = 30.0
EVENT_ORIGIN_SEC = 120.0
WATERLEVEL_EPS = 0.05


def preprocess(stream: Stream) -> Stream:
    processed = stream.copy()
    processed.resample(TARGET_FS)
    processed.filter(
        "bandpass",
        freqmin=FILTER_MIN_HZ,
        freqmax=FILTER_MAX_HZ,
        corners=FILTER_ORDER,
        zerophase=True,
    )
    processed.detrend("linear")
    return processed


def synchronize(stream: Stream) -> float:
    cwa_z = _find_trace(stream, station="F068", channel_suffix="Z")
    palert_z = _find_trace(stream, station="D006", channel_suffix="Z")

    lag_seconds = _estimate_lag_seconds(cwa_z, palert_z)
    if abs(lag_seconds) > SYNC_MAX_LAG_SEC:
        raise ValueError(f"Lag too large: {lag_seconds:.3f}s")

    for trace in stream:
        trace.stats.starttime -= lag_seconds

    return lag_seconds


def analyze_core(
    input_trace: Trace, output_trace: Trace
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    int,
    int,
    int,
]:
    input_trace = input_trace.copy()
    output_trace = output_trace.copy()

    input_trace.stats.starttime -= EVENT_ORIGIN_SEC
    output_trace.stats.starttime -= EVENT_ORIGIN_SEC

    irf = compute_irf(input_trace, output_trace, epsilon=WATERLEVEL_EPS, shift_origin=False)
    cwt_matrix, freqs = compute_cwt(
        output_trace.data,
        output_trace.stats.sampling_rate,
        freqs=_default_cwt_freqs(),
    )
    ridge_freqs = compute_ridge_frequencies(cwt_matrix, freqs, min_freq=0.6)
    time_axis = _time_axis(output_trace)
    (
        freq_init,
        freq_lowest,
        freq_final,
        idx_init,
        idx_lowest,
        idx_final,
    ) = extract_feature_freqs(cwt_matrix, freqs, time_axis, min_freq=0.6)
    return (
        irf,
        cwt_matrix,
        freqs,
        ridge_freqs,
        freq_init,
        freq_lowest,
        freq_final,
        idx_init,
        idx_lowest,
        idx_final,
    )


def compute_irf(
    input_trace: Trace,
    output_trace: Trace,
    *,
    epsilon: float = WATERLEVEL_EPS,
    shift_origin: bool = True,
) -> np.ndarray:
    if shift_origin:
        input_trace = input_trace.copy()
        output_trace = output_trace.copy()
        input_trace.stats.starttime -= EVENT_ORIGIN_SEC
        output_trace.stats.starttime -= EVENT_ORIGIN_SEC
    return _water_level_deconvolution(
        input_trace.data, output_trace.data, epsilon=epsilon
    )


def compute_cwt(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    freqs: Iterable[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(signal, dtype=float)
    wavelet = "cmor1.5-1.0"
    use_freqs = np.asarray(list(freqs) if freqs is not None else _default_cwt_freqs())
    scales = pywt.central_frequency(wavelet) * sampling_rate / use_freqs
    cwt_matrix, _ = pywt.cwt(data, scales, wavelet, sampling_period=1.0 / sampling_rate)
    return cwt_matrix, use_freqs


def compute_ridge_frequencies(
    cwt_matrix: np.ndarray, freqs: np.ndarray, *, min_freq: float = 0.6
) -> np.ndarray:
    power = np.abs(cwt_matrix)
    valid = freqs >= min_freq
    if not np.any(valid):
        raise ValueError("No valid frequencies available after masking.")
    masked_power = power.copy()
    masked_power[~valid, :] = -np.inf
    ridge_idx = np.argmax(masked_power, axis=0)
    return freqs[ridge_idx]


def extract_feature_freqs(
    cwt_matrix: np.ndarray,
    freqs: np.ndarray,
    time_axis: np.ndarray,
    *,
    min_freq: float = 0.6,
) -> Tuple[float, float, float, int, int, int]:
    ridge_freqs = compute_ridge_frequencies(cwt_matrix, freqs, min_freq=min_freq)
    idx0 = int(np.argmin(np.abs(time_axis)))
    freq_init = float(ridge_freqs[idx0])
    power = np.abs(cwt_matrix)
    max_power = float(power.max())
    if max_power <= 0.0:
        raise ValueError("CWT power is zero; cannot apply energy gating.")
    threshold = 0.1 * max_power
    high_energy_mask = power.max(axis=0) >= threshold
    ridge_valid = ridge_freqs[idx0:]
    energy_valid = high_energy_mask[idx0:]
    candidate_mask = energy_valid
    if not np.any(candidate_mask):
        raise ValueError("No high-energy time points available after gating.")
    candidate_indices = np.flatnonzero(candidate_mask)
    rel_low_idx = int(np.argmin(ridge_valid[candidate_mask]))
    idx_lowest = idx0 + int(candidate_indices[rel_low_idx])
    freq_lowest = float(ridge_freqs[idx_lowest])
    idx_final = len(ridge_freqs) - 1
    freq_final = float(ridge_freqs[idx_final])
    return freq_init, freq_lowest, freq_final, idx0, idx_lowest, idx_final


def _time_axis(trace: Trace) -> np.ndarray:
    n = trace.stats.npts
    fs = trace.stats.sampling_rate
    return np.arange(n, dtype=float) / fs - EVENT_ORIGIN_SEC


def _find_trace(stream: Stream, station: str, channel_suffix: str) -> Trace:
    for trace in stream:
        stats = trace.stats
        if stats.station == station and stats.channel.endswith(channel_suffix):
            return trace
    raise ValueError(f"Missing trace for station={station} channel=*{channel_suffix}")


def _estimate_lag_seconds(cwa_z: Trace, palert_z: Trace) -> float:
    if cwa_z.stats.sampling_rate != palert_z.stats.sampling_rate:
        raise ValueError("Sampling rates do not match for lag estimation.")

    n = min(len(cwa_z.data), len(palert_z.data))
    data_a = np.asarray(cwa_z.data[:n], dtype=float)
    data_b = np.asarray(palert_z.data[:n], dtype=float)

    corr = correlate(data_b, data_a, mode="full")
    lag_index = int(np.argmax(corr)) - (n - 1)
    return lag_index / cwa_z.stats.sampling_rate


def _water_level_deconvolution(
    input_signal: np.ndarray, output_signal: np.ndarray, *, epsilon: float
) -> np.ndarray:
    n = min(len(input_signal), len(output_signal))
    x = np.asarray(input_signal[:n], dtype=float)
    y = np.asarray(output_signal[:n], dtype=float)

    x_f = np.fft.rfft(x)
    y_f = np.fft.rfft(y)
    power = np.abs(x_f) ** 2
    water_level = epsilon * power.max()
    denom = np.maximum(power, water_level)
    h_f = y_f * np.conjugate(x_f) / denom
    return np.fft.irfft(h_f, n=n)


def _default_cwt_freqs() -> np.ndarray:
    return np.linspace(FILTER_MIN_HZ, FILTER_MAX_HZ, 100)
