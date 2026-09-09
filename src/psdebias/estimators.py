"""Classical quadratic spectral estimators.

Every estimator returns a :class:`SpectralEstimate` named tuple
``(freqs, psd, kernel)`` where ``kernel`` is the one-sided bias sequence
``h[tau]`` (the effective lag window / taper autocorrelation) that
``psdebias.sampler``/``psdebias.dwelch`` need to build spectral-window-convolved
bases, and ``corr`` is the one-sided frequency-correlation sequence ``rho[d]``
that ``psdebias.dwelch`` needs for the generalised least squares of DQuad Eq. 12.
All four estimators apply consistent ``1/fs`` density scaling (the original
package omitted it for the periodogram only).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.fft
import scipy.linalg
import scipy.signal
from numpy import typing as npt

from psdebias.util import psd_from_acf, segment


class SpectralEstimate(NamedTuple):
    freqs: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]
    kernel: npt.NDArray[np.float64]

def _unit_taper(taper: npt.NDArray[np.float64] | None, n: int) -> npt.NDArray[np.float64]:
    taper = np.ones(n) if taper is None else np.asarray(taper, dtype=np.float64)
    if len(taper) != n:
        raise ValueError(f"taper length {len(taper)} != expected {n}")
    return taper / np.linalg.norm(taper)

def _taper_kernel(taper: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """One-sided autocorrelation of a unit-energy taper: h[0] = 1.

    This IS the bias description of the estimator: the expected estimate is
    the true spectrum convolved with the spectral window |H(w)|^2, and h[tau]
    is that window's inverse Fourier transform (Wiener-Khinchin). Multiplying
    a model spectrum's ACF by h in the lag domain (util.psd_from_acf) applies
    the window's blurring exactly — which is how splines.half_bases_biased
    builds the debiasing bases from this sequence.
    """
    n = len(taper)
    return np.correlate(taper, taper, mode="full")[n - 1 :]

def periodogram(
    x: npt.NDArray[np.float64],
    *,
    fs: float = 1.0,
    taper: npt.NDArray[np.float64] | None = None,
    drop_endpoints: bool = True,
) -> SpectralEstimate:
    """(Tapered) periodogram: ``|rfft(taper * x)|^2 / fs`` on the rfft grid."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    taper = _unit_taper(taper, n)
    freqs = scipy.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(scipy.fft.rfft(taper * x)) ** 2 / fs
    kernel = _taper_kernel(taper)
    # The zero and Nyquist bins are dropped by default: their sampling
    # distribution differs (chi^2 with 1 d.o.f. instead of 2, and DC absorbs
    # any nonzero mean), and the sampler's weighted regression assumes the
    # homogeneous interior bins.
    sl = slice(1, -1) if drop_endpoints else slice(None)
    # rho indexes bin SEPARATION, not position, so the leading len(freqs[sl])
    # entries cover every separation the retained grid can produce regardless of
    # which bins were dropped.
    return SpectralEstimate(freqs[sl], psd[sl], kernel)


def welch(
    x: npt.NDArray[np.float64],
    *,
    segment_length: int,
    n_segments: int,
    step: int,
    fs: float = 1.0,
    taper: npt.NDArray[np.float64] | None = None,
    drop_endpoints: bool = True,
) -> SpectralEstimate:
    """Welch's method: mean of tapered periodograms over overlapping segments."""
    x = np.asarray(x, dtype=np.float64)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if segment_length <= 0 or n_segments <= 0 or step <= 0:
        raise ValueError("segment_length, n_segments and step must be positive")
    taper = _unit_taper(taper, segment_length)
    segments = segment(x, segment_length, n_segments, step)
    freqs = scipy.fft.rfftfreq(segment_length, d=1.0 / fs)
    psd = np.mean(np.abs(scipy.fft.rfft(segments * taper, axis=1)) ** 2, axis=0) / fs
    kernel = _taper_kernel(taper)
    sl = slice(1, -1) if drop_endpoints else slice(None)
    return SpectralEstimate(freqs[sl], psd[sl], kernel)

def lag_window(
    x: npt.NDArray[np.float64],
    *,
    window: str = "bartlett",
    lag: int | None = None,
    fs: float = 1.0,
    drop_endpoints: bool = True,
) -> SpectralEstimate:
    """Lag-window (Blackman-Tukey) estimator.

    The biased sample ACF is windowed to ``lag`` lags and transformed on the
    ``n``-point rfft grid. ``kernel`` is the effective one-sided lag window
    ``w[tau] * (1 - tau/n)`` (window times the implicit triangular bias of the
    biased ACF), zero beyond ``lag``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if lag is None:
        lag = n // 10
    # lag < n/2 (not just < n): the negative-lag fold below places lags 0..lag at
    # buffer indices 0..lag and lags -lag..-1 at n-lag..n-1. Those supports overlap
    # once lag >= n/2, corrupting the ACF embedded on the circular rfft buffer.
    if not 0 < lag < n // 2:
        raise ValueError(f"lag must be in (0, {n // 2}), got {lag}")

    # Biased sample ACF (divide by n, not n-tau): the /n normalization is what
    # injects the triangular (1 - tau/n) bias folded into the kernel below.
    acf = np.correlate(x, x, mode="full") / n
    corr_one_sided = acf[n - 1 : n + lag]  # lags 0..lag

    win = scipy.signal.get_window(window, 2 * lag + 1, fftbins=False)
    win_one_sided = win[lag:]  # symmetric window, take the lag>=0 half
    windowed = corr_one_sided * win_one_sided

    # Lay the windowed one-sided ACF onto the n-point circular rfft buffer. A
    # real spectrum has a symmetric ACF, so negative lag -tau aliases to index
    # n - tau; padded[-lag:] = windowed[1:][::-1] mirrors lags 1..lag onto the
    # buffer's tail. Everything past +/-lag is zero (the window's finite support).
    padded = np.zeros(n)
    padded[: lag + 1] = windowed
    padded[-lag:] = windowed[1:][::-1]
    psd = scipy.fft.rfft(padded).real / fs

    # Effective one-sided lag window = chosen window times the biased-ACF's
    # implicit triangular taper (1 - tau/n); this is the bias sequence h[tau]
    # the debiasing bases convolve against, zero beyond lag.
    taus = np.arange(lag + 1)
    kernel = np.zeros(n)
    kernel[: lag + 1] = win_one_sided * (1.0 - taus / n)

    freqs = scipy.fft.rfftfreq(n, d=1.0 / fs)
    sl = slice(1, -1) if drop_endpoints else slice(None)
    # rho^(LW)(eta) ∝ sum_tau w^2[tau] (1 - |tau|/n) cos(2 pi eta tau): the
    # transform of the squared lag window carrying ONE triangular factor, which
    # is the fold-and-rfft psd_from_acf already performs for the bases. (The
    # general Eq. 8 route would instead need an O(n^3) eigendecomposition of Q,
    # which for a lag-window is full rank.)
    #
    # Why one factor and not two. The estimator is the quadratic form
    # A[s, t] = w[|s-t|] cos(2 pi eta (s-t)) / n -- the RAW window; the biased
    # ACF's triangular taper is not in A at all, it emerges from there being
    # n - |tau| pairs on each diagonal. So it enters the mean once (giving
    # `kernel`) and, via tr(A_i A_j) = sum over diagonals, exactly once again --
    # not squared. Using kernel**2 here overcounts it and is wrong by O(lag/n)
    # (~3% at lag = n/4). See docs/CHANGES.md §8; this departs from the note's
    # Eq. 17, which is the asymptotic form with no triangular factor at all.
    window_padded = np.zeros(n)
    window_padded[: lag + 1] = win_one_sided

    return SpectralEstimate(freqs[sl], psd[sl], kernel)


def multitaper(
    x: npt.NDArray[np.float64],
    *,
    nw: float = 3.5,
    n_tapers: int | None = None,
    fs: float = 1.0,
    drop_endpoints: bool = True,
) -> SpectralEstimate:
    """Multitaper estimator with DPSS tapers; ``kernel`` is the mean taper
    autocorrelation (the composite spectral window's ACF)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n_tapers is None:
        n_tapers = int(2 * nw)
    tapers = scipy.signal.windows.dpss(n, nw, Kmax=n_tapers)

    # Average the eigenspectra of the K DPSS-tapered records.
    psd = np.mean(np.abs(scipy.fft.rfft(tapers * x[None, :], axis=1)) ** 2, axis=0) / fs
    # Composite spectral window's bias sequence: mean of the per-taper
    # autocorrelations (each taper contributes |H_k(w)|^2; averaging the ACFs =
    # averaging the windows), keeping the one-sided lags 0..n-1.
    autos = np.array([np.correlate(t, t, mode="full") for t in tapers])
    kernel = autos.mean(axis=0)[n - 1 :]

    freqs = scipy.fft.rfftfreq(n, d=1.0 / fs)
    sl = slice(1, -1) if drop_endpoints else slice(None)
    return SpectralEstimate(freqs[sl], psd[sl], kernel,)
