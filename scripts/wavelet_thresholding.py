#!/usr/bin/env python3
"""Wavelet-thresholding baseline (WPM) for the bias-variance study.

Walden/Percival/McCoy-style denoising of the log spectral estimate: subtract
the log-chi-squared mean offset ``digamma(K) - log(K)``, soft-threshold the
wavelet detail coefficients at the universal threshold (noise scale
``sqrt(trigamma(K))``), reconstruct, and exponentiate. ``K`` is the
estimate's equivalent number of degrees-of-freedom pairs, taken from the
``SpectralEstimate.dof`` field each estimator reports.

Reads the smoothing-arm realisations written by
``generate_realisations.py smooth`` and writes one thresholded stack per
estimator setting alongside them::

    <base>/{ar4,matern}_unbiased/quad_sim/{multi,lag}_wavelet/{NW}.npy
    <base>/{ar4,matern}_unbiased/welch_sim/welch_wavelet/{m}.npy

Estimates are computed on the full rfft grid (``drop_endpoints=False``) so
the wavelet transform sees the whole log spectrum, then trimmed to the
endpoint-dropped grid the rest of the study uses.

Requires PyWavelets (``pip install -e '.[scripts]'``).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pywt
from scipy.special import digamma, polygamma

from psdebias import lag_window, multitaper, welch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, leave=True):
        if desc:
            print(desc)
        return iterable


# =============================================================================
# Smoothing-arm study settings (must match generate_realisations.SMOOTH_CONFIG)
# =============================================================================

N_QUAD = 2 ** 13
NWS_QUAD = np.array([3, 4, 5, 6])
LAGS_QUAD = N_QUAD // NWS_QUAD
WINDOW = "blackman"

SEGMENT_LENGTH = 2 ** 12
N_SEGMENTS = 2 ** np.arange(3, 7)  # 8, 16, 32, 64
TAPER = np.hanning(SEGMENT_LENGTH)  # estimators normalize to unit energy


# =============================================================================
# WPM thresholding of the log spectrum
# =============================================================================


def _soft(coeffs, lam):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - lam, 0.0)


def _pad_reflect(y, n):
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)), mode="reflect")


def wpm_threshold_logspectrum(
    estimate,
    k_dof: float,
    *,
    wavelet: str = "sym4",
    levels: int | None = None,
    coarse_keep: int = 1,
):
    """Denoise a spectral estimate in the log domain; returns linear scale.

    ``log(estimate)`` is (approximately) the log spectrum plus log-chi-squared
    noise with mean ``digamma(K) - log(K)`` and variance ``trigamma(K)``. The
    detail coefficients are soft-thresholded at the universal threshold
    ``sigma * sqrt(2 log n)``; the coarsest ``coarse_keep`` detail levels are
    left untouched to protect the spectrum's broad shape.
    """
    n_freq = len(estimate)
    offset = digamma(k_dof) - np.log(k_dof)
    y = np.log(estimate) - offset
    sigma = np.sqrt(polygamma(1, k_dof))

    padded = _pad_reflect(y, 2 ** int(np.ceil(np.log2(n_freq))))
    if levels is None:
        levels = pywt.dwt_max_level(len(padded), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(padded, wavelet, level=levels, mode="periodization")

    lam = sigma * np.sqrt(2 * np.log(len(padded)))
    for j in range(1, len(coeffs) - coarse_keep):
        coeffs[j] = _soft(coeffs[j], lam)

    cleaned = pywt.waverec(coeffs, wavelet, mode="periodization")[:n_freq]
    return np.exp(cleaned + offset)


# =============================================================================
# Runners
# =============================================================================


def run_quad(base_path: str, process: str) -> None:
    base = os.path.join(base_path, f"{process}_unbiased", "quad_sim")
    samples = np.load(os.path.join(base, "samples", "samples.npy"))
    os.makedirs(os.path.join(base, "multi_wavelet"), exist_ok=True)
    os.makedirs(os.path.join(base, "lag_wavelet"), exist_ok=True)

    for nw, lag in zip(NWS_QUAD, LAGS_QUAD):
        nw, lag = int(nw), int(lag)
        multi_rows, lag_rows = [], []
        for x in tqdm(samples, desc=f"{process} wavelet NW={nw}", leave=False):
            est_multi = multitaper(x, nw=nw, drop_endpoints=False)
            est_lag = lag_window(x, window=WINDOW, lag=lag, drop_endpoints=False)
            # Threshold on the full rfft grid, then [1:-1] drops the DC/Nyquist
            # bins back to the endpoint-dropped grid the rest of the study uses.
            multi_rows.append(wpm_threshold_logspectrum(est_multi.psd, est_multi.dof)[1:-1])
            lag_rows.append(wpm_threshold_logspectrum(est_lag.psd, est_lag.dof)[1:-1])

        np.save(os.path.join(base, "multi_wavelet", f"{nw}.npy"), np.array(multi_rows))
        np.save(os.path.join(base, "lag_wavelet", f"{nw}.npy"), np.array(lag_rows))


def run_welch(base_path: str, process: str) -> None:
    base = os.path.join(base_path, f"{process}_unbiased", "welch_sim")
    os.makedirs(os.path.join(base, "welch_wavelet"), exist_ok=True)

    for m in N_SEGMENTS:
        m = int(m)
        samples = np.load(os.path.join(base, "samples", f"{m}.npy"))
        rows = []
        for x in tqdm(samples, desc=f"{process} wavelet m={m}", leave=False):
            est = welch(
                x, segment_length=SEGMENT_LENGTH, n_segments=m, step=SEGMENT_LENGTH,
                taper=TAPER, drop_endpoints=False,
            )
            rows.append(wpm_threshold_logspectrum(est.psd, est.dof)[1:-1])

        np.save(os.path.join(base, "welch_wavelet", f"{m}.npy"), np.array(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-path", type=str, default="./realisations")
    args = parser.parse_args()

    for process in ("ar4", "matern"):
        run_quad(args.base_path, process)
        run_welch(args.base_path, process)


if __name__ == "__main__":
    main()
