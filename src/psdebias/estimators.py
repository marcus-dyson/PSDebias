"""Classical quadratic spectral estimators.

Every estimator returns a :class:`SpectralEstimate` named tuple
``(freqs, psd, kernel)`` where ``kernel`` is the one-sided bias sequence
``h[tau]`` (the effective lag window / taper autocorrelation) that
``psdebias.sampler``/``psdebias.dwelch`` need to build spectral-window-convolved
bases. All four estimators apply consistent ``1/fs`` density scaling (the
original package omitted it for the periodogram only).
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
import scipy.fft
import scipy.signal
from numpy import typing as npt
from scipy.linalg import toeplitz
from scipy.special import digamma, gammaln, polygamma

from psdebias.util import segment

# Effectively a struct to store everything. The correlation matrices are opt-in
# (``corr=True``): they are g x g, which is 134 MB apiece on the study's
# n = 2**13 grid, and the realisation sweeps only ever keep ``psd``.
class SpectralEstimate(NamedTuple):
    freqs: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]
    kernel: npt.NDArray[np.float64]
    dof: np.float64
    corr_mat: npt.NDArray[np.float64] | None = None
    log_corr_mat: npt.NDArray[np.float64] | None = None


def _unit_taper(taper: npt.NDArray[np.float64] | None, n: int) -> npt.NDArray[np.float64]:
    taper = np.ones(n) if taper is None else np.asarray(taper, dtype=np.float64)
    if len(taper) != n:
        raise ValueError(f"taper length {len(taper)} != expected {n}")
    return taper / np.linalg.norm(taper)

def _lag_window_dof(n: int, lag: int, window: str) -> float:
    """Effective DOF of a lag-window estimate: ``nu = 2 n / sum_tau w(tau)^2``."""
    w = scipy.signal.get_window(window, 2 * lag + 1, fftbins=False)
    return 1.0 * n / np.sum(w ** 2)

def _kibble_log_corr(
    rho: npt.NDArray[np.float64], alpha: float
) -> npt.NDArray[np.float64]:
    """Log-scale correlation implied by the linear-scale correlation ``rho``.

    A Welch or multitaper estimate is a positive quadratic form in a Gaussian
    record, so a pair of bins follows Kibble's bivariate gamma with shape
    ``alpha`` (this package's ``dof``) and correlation ``rho``. That law is a
    negative-binomial mixture -- ``X, Y | K ~ iid Gamma(alpha + K)`` with
    ``K ~ NB(alpha, rho)`` -- under which the logs are conditionally
    independent, so all their covariance rides on the mixing index:

        corr(log X, log Y) = Var_K[psi(alpha + K)] / psi'(alpha).

    The correction is not cosmetic at low dof: two-segment Welch with
    ``rho = 0.456`` has a log-correlation of 0.386, so the delta-method answer
    (log correlation = rho) is 18% high.
    """
    rho = np.asarray(rho, dtype=np.float64)
    trigamma = float(polygamma(1, alpha))
    out = np.empty_like(rho)
    for i, r in enumerate(rho.ravel()):
        s = abs(r)
        if s >= 1.0 - 1e-12:
            out.flat[i] = np.sign(r)
        elif s < 1e-6:
            # Leading term of the series: only K = 1 contributes at O(rho).
            out.flat[i] = r / (alpha * trigamma)
        else:
            # Truncate the negative binomial well past its upper tail.
            mean = alpha * s / (1.0 - s)
            sd = np.sqrt(alpha * s) / (1.0 - s)
            k = np.arange(int(mean + 12.0 * sd) + 50)
            log_p = (
                gammaln(alpha + k) - gammaln(alpha) - gammaln(k + 1)
                + k * np.log(s) + alpha * np.log1p(-s)
            )
            p = np.exp(log_p)
            p /= p.sum()
            psi = digamma(alpha + k)
            out.flat[i] = np.sign(r) * (p @ (psi - p @ psi) ** 2) / trigamma
    return out


def _lag_window_corr_mats(
    n: int, lag: int, win_one_sided: npt.NDArray[np.float64], g: int
) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Code that creates a toeplitz matrix for the linear scale and log scale of a given
    PSD estimator.

    Blackman-Tukey smooths the (asymptotically independent) periodogram with the
    spectral window W, so the covariance at bin offset d is W's autocorrelation
    -- whose inverse transform is the squared lag window. The biased ACF adds
    one factor of (1 - tau/n), not two: squaring it overstates the correlation
    (0.403 against an empirical 0.355 at n = 256, lag = 100, d = 2).

    Unlike Welch/multitaper this is not a positive quadratic form, and Kibble's
    bivariate gamma moves the log-scale correlation the wrong way there (0.736
    where the empirical value is 0.778). The delta method -- log correlation =
    linear correlation -- is the better approximation, so both returns are the
    same array.
    """
    taus = np.arange(lag + 1)
    seq = win_one_sided ** 2 * (1.0 - taus / n)
    # Same circular fold as the estimate itself: lags 0..lag at the head,
    # their mirror images at the tail.
    padded = np.zeros(n)
    padded[: lag + 1] = seq
    padded[-lag:] = seq[1:][::-1]
    row = scipy.fft.rfft(padded).real
    row /= row[0]
    mat = toeplitz(row[:g])
    return mat, mat


def _welch_corr_mats(
    segment_length: int,
    taper: npt.NDArray[np.float64],
    step: int,
    k: int,
    dof: float,
    g: int,
) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Code that creates a toeplitz matrix for the linear scale and log scale of a given
    PSD estimator.

    For a locally flat spectrum ``cov{I_m(f_j), I_m'(f_k)} = |C_s(f_j - f_k)|^2``
    where ``C_s(v) = sum_t h_t h_{t+s} exp(-2 pi i v t)`` is the transform of the
    taper's lagged product at segment shift ``s = (m - m') * step`` (the
    segment-offset phase cancels under the modulus). Summing over segment pairs
    gives the numerator below, whose ``d = 0`` value is exactly the ``denom`` of
    :func:`_welch_dof` -- dof and correlation share one account of the overlap.
    """
    num = np.zeros(segment_length // 2 + 1)
    for m in range(k):
        shift = m * step
        if shift >= segment_length:
            break
        prod = np.zeros(segment_length)
        prod[: segment_length - shift] = taper[: segment_length - shift] * taper[shift:]
        weight = k if m == 0 else 2 * (k - m)
        num += weight * np.abs(scipy.fft.rfft(prod)) ** 2
    row = num[:g] / num[0]
    return toeplitz(row), toeplitz(_kibble_log_corr(row, dof))


def _multitaper_corr_mats(
    tapers: npt.NDArray[np.float64], dof: float, g: int
) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Code that creates a toeplitz matrix for the linear scale and log scale of a given
    PSD estimator.

    Averaging K eigenspectra gives ``cov(d) = sum_{a,b} |V_ab(d)|^2 / K^2`` with
    ``V_ab`` the transform of the taper product ``v_a v_b``. At ``d = 0`` DPSS
    orthonormality collapses the double sum to K, recovering the 1/K variance
    that ``dof = n_tapers`` encodes.
    """
    n_tapers = len(tapers)
    num = np.zeros(tapers.shape[1] // 2 + 1)
    for a in range(n_tapers):
        for b in range(a, n_tapers):
            cross = np.abs(scipy.fft.rfft(tapers[a] * tapers[b])) ** 2
            num += cross if a == b else 2.0 * cross
    row = num[:g] / num[0]
    return toeplitz(row), toeplitz(_kibble_log_corr(row, dof))


def _welch_dof(n: int, segment_length: int, taper, step: int) -> float:
    """Effective DOF of a Welch estimate with overlapping tapered segments:
    ``nu = 2 K^2 / [K + 2 sum_m (K - m) c_m^2]`` where ``c_m`` is the taper
    autocorrelation at segment shift ``m * step`` (Welch 1967)."""
    if step < 1:
        raise ValueError("step must be >= 1")
    k = 1 + (n - segment_length) // step
    if k < 1:
        raise ValueError("segment longer than the record")

    taper = np.asarray(taper, dtype=np.float64)
    taper = taper / np.sqrt(np.sum(taper ** 2))  # unit energy => c_0 = 1
    c = np.zeros(k)
    for m in range(k):
        shift = m * step
        if shift < segment_length:
            c[m] = np.dot(taper[: segment_length - shift], taper[shift:])

    denom = k + 2.0 * np.sum((k - np.arange(1, k)) * c[1:] ** 2)
    return 1.0 * k ** 2 / denom

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
    # Periodagram has DOF = 1
    dof = 1
    return SpectralEstimate(freqs[sl], psd[sl], kernel, dof)


def welch(
    x: npt.NDArray[np.float64],
    *,
    segment_length: int,
    n_segments: int,
    step: int,
    fs: float = 1.0,
    taper: npt.NDArray[np.float64] | None = None,
    drop_endpoints: bool = True,
    corr: bool = True,
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

    # Degrees of Freedom
    dof = _welch_dof(segment_length + (n_segments - 1) * step, segment_length, taper, step)

    # n_segments (not the record length) is what segment() actually averaged.
    corr_mat, log_corr_mat = (
        _welch_corr_mats(segment_length, taper, step, n_segments, dof, len(freqs[sl]))
        if corr else (None, None)
    )

    return SpectralEstimate(freqs[sl], psd[sl], kernel, dof, corr_mat, log_corr_mat)


def lag_window(
    x: npt.NDArray[np.float64],
    *,
    window: str = "bartlett",
    lag: int | None = None,
    fs: float = 1.0,
    drop_endpoints: bool = True,
    corr: bool = True,
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

    # Lag-window dof
    dof = _lag_window_dof(n=len(x), lag=lag, window=window)

    corr_mat, log_corr_mat = (
        _lag_window_corr_mats(n, lag, win_one_sided, len(freqs[sl]))
        if corr else (None, None)
    )

    return SpectralEstimate(freqs[sl], psd[sl], kernel, dof, corr_mat, log_corr_mat)


def multitaper(
    x: npt.NDArray[np.float64],
    *,
    nw: float = 3.5,
    n_tapers: int | None = None,
    fs: float = 1.0,
    drop_endpoints: bool = True,
    corr: bool = True,
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

    # multitaper dof
    dof = n_tapers

    corr_mat, log_corr_mat = (
        _multitaper_corr_mats(tapers, dof, len(freqs[sl])) if corr else (None, None)
    )

    return SpectralEstimate(freqs[sl], psd[sl], kernel, dof, corr_mat, log_corr_mat)
