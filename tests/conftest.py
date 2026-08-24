"""Shared fixtures and pure-numpy/scipy reference implementations (oracles).

The oracles reproduce the original auto-speccy algorithms verbatim (including
the O(N) loop and uniform-grid dwelch assembly) so the rewritten, vectorized
code can be tested against them exactly where the math is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


# The paper's AR(4) process (Percival & Walden), recursion form
# X_t = sum_j phi_j X_{t-j} + eps_t  ->  characteristic polynomial [1, -phi...].
PAPER_PHI = np.array([2.7607, -3.8106, 2.6535, -0.9238])
PAPER_AR_POLY = np.concatenate(([1.0], -PAPER_PHI))


def original_get_exact_rfft_psd(acf: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Verbatim port of auto_speccy.util.get_exact_rfft_psd (O(N) loop)."""
    n = len(acf)
    biased = acf * kernel
    padded = np.zeros(n)
    padded[0] = biased[0]
    for tau in range(1, n):
        padded[tau] += biased[tau]
        padded[n - tau] += biased[tau]
    return np.real(np.fft.rfft(padded))


def reference_log_posterior(
    y: np.ndarray,
    x: np.ndarray,
    n_interior_active: int,
    n_interior_candidates: int,
    c: float,
    a_sigma: float,
    b_sigma: float,
    a_pi: float,
    b_pi: float,
) -> float:
    """Paper's log posterior (App. B) via scipy, no shortcuts: least squares by
    lstsq, Beta-Binomial prior over interior knot count."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    ess = float(y @ y - y @ (x @ beta))
    p = x.shape[1]
    g = len(y)
    log_lik = -(p / 2) * np.log(1 + c) - (g / 2 + a_sigma) * np.log(max(ess, 0.0) + 2 * b_sigma)
    log_prior = scipy.special.betaln(
        n_interior_active + a_pi, n_interior_candidates - n_interior_active + b_pi
    ) - scipy.special.betaln(a_pi, b_pi)
    return log_lik + log_prior


# ---------------------------------------------------------------------------
# Frequency-correlation oracles (DQuad Eq. 8/9; spectral-covariance note Eq. 10)
# ---------------------------------------------------------------------------
#
# Deliberately brute force: explicit double loops over taper pairs and an
# explicit complex exponential per lag, so nothing is shared with the batched
# rfft route in estimators.py. These are slow and that is fine -- they run on
# tiny inputs.


def reference_corr_multitaper(tapers: np.ndarray, etas: np.ndarray) -> np.ndarray:
    """rho(eta) = (1/K) ||H^T E_eta H||_F^2 by the direct O(n K^2) double loop.

    Note Eq. 23, i.e. DQuad Eq. 8 with d_k = 1/K and M = K. Normalised by its
    own value at eta = 0 so the result is a correlation.
    """
    k_tapers, n = tapers.shape
    t = np.arange(n)

    def at(eta: float) -> float:
        total = 0.0
        for j in range(k_tapers):
            for k in range(k_tapers):
                phi = np.sum(tapers[j] * tapers[k] * np.exp(-2j * np.pi * eta * t))
                total += abs(phi) ** 2
        return total / k_tapers

    rho = np.array([at(float(e)) for e in np.atleast_1d(etas)])
    return rho / at(0.0)


def reference_corr_welch(
    taper: np.ndarray, n_segments: int, step: int, etas: np.ndarray
) -> np.ndarray:
    """rho(eta) for Welch by the literal block-pair double sum of DQuad Eq. 8.

    Each block contributes a full-length taper h^m that is the segment taper
    shifted to offset m*step and zero-padded to the whole record, exactly as in
    the paper's Example 3; d_m = 1/M. The sum runs over ALL ordered pairs and
    the Fourier kernel uses the GLOBAL time index, so the Psi_{m-m'} collapse
    that estimators.py relies on (the block-index phase cancelling inside the
    modulus) is not assumed here -- it is what this oracle tests.
    """
    seg_len = len(taper)
    n = (n_segments - 1) * step + seg_len
    tapers = np.zeros((n_segments, n))
    for m in range(n_segments):
        tapers[m, m * step : m * step + seg_len] = taper
    t = np.arange(n)

    def at(eta: float) -> float:
        total = 0.0
        for j in range(n_segments):
            for k in range(n_segments):
                gamma = np.sum(tapers[j] * tapers[k] * np.exp(-2j * np.pi * eta * t))
                total += abs(gamma) ** 2
        return total / n_segments  # M * d_j * d_k = M / M^2 = 1/M

    rho = np.array([at(float(e)) for e in np.atleast_1d(etas)])
    return rho / at(0.0)


def reference_bandwidth_multitaper(tapers: np.ndarray) -> float:
    """Closed-form correlation area (DQuad Eq. 9, note Eq. 25).

    B = M sum_{j,k} d_j d_k sum_t h_j^2 h_k^2, which for d_k = 1/K collapses to
    (1/K) sum_t (sum_k h_{k,t}^2)^2. This is the integral of rho over the band,
    in normalised frequency -- no Fourier transform involved anywhere.
    """
    k_tapers = tapers.shape[0]
    return float(np.sum(np.sum(tapers**2, axis=0) ** 2) / k_tapers)


# ---------------------------------------------------------------------------
# Exact quadratic-form oracle (Isserlis; no asymptotics, no Monte Carlo)
# ---------------------------------------------------------------------------
#
# Every estimator here is quadratic in the data: bin k is I(f_k) = x' A_k x for
# a symmetric A_k that depends only on the estimator's design. For Gaussian x
# with covariance sigma^2 I, Isserlis' theorem gives the covariance of two
# quadratic forms exactly:
#
#     Cov(x' A x, x' B x) = 2 sigma^4 tr(A B).
#
# That is a *finite-sample* identity, so it pins rho against the definition of
# a correlation rather than against another rearrangement of DQuad Eq. 8 --
# which is all the reference_corr_* oracles above can do. The builders below
# transcribe each estimator's definition directly; nothing is shared with the
# batched-rfft route in estimators.py.


def _rank_one_bin(taper: np.ndarray, k: int, n_phase: int, offset: int, n: int) -> np.ndarray:
    """A for one tapered DFT bin: |sum_t h_t x_t e^{-2 pi i k t / n_phase}|^2.

    Splitting the modulus into real and imaginary parts gives cc' + ss'. The
    taper occupies ``n``-length rows starting at ``offset`` (Welch's blocks);
    the Fourier phase uses the LOCAL index, because each block is transformed on
    its own ``n_phase``-point grid.
    """
    j = np.arange(len(taper))
    cos_part = np.zeros(n)
    sin_part = np.zeros(n)
    cos_part[offset : offset + len(taper)] = taper * np.cos(2 * np.pi * k * j / n_phase)
    sin_part[offset : offset + len(taper)] = taper * np.sin(2 * np.pi * k * j / n_phase)
    return np.outer(cos_part, cos_part) + np.outer(sin_part, sin_part)


def quadratic_form_periodogram(taper: np.ndarray, k: int) -> np.ndarray:
    """A_k for the tapered periodogram. ``taper`` must already be unit-energy."""
    n = len(taper)
    return _rank_one_bin(taper, k, n, 0, n)


def quadratic_form_welch(
    taper: np.ndarray, k: int, n_segments: int, step: int
) -> np.ndarray:
    """A_k for Welch: the mean of the per-block periodogram forms.

    Each block's form is embedded in the full record at offset ``m * step``, so
    overlapping blocks share entries of A -- which is exactly the correlation
    between blocks that the Psi_d collapse in ``_welch_corr`` accounts for.
    """
    seg_len = len(taper)
    n = (n_segments - 1) * step + seg_len
    blocks = [_rank_one_bin(taper, k, seg_len, m * step, n) for m in range(n_segments)]
    return sum(blocks) / n_segments


def quadratic_form_multitaper(tapers: np.ndarray, k: int) -> np.ndarray:
    """A_k for the multitaper: the mean of the K eigenspectrum forms."""
    k_tapers, n = tapers.shape
    return sum(_rank_one_bin(t, k, n, 0, n) for t in tapers) / k_tapers


def quadratic_form_lag_window(window: np.ndarray, k: int, n: int) -> np.ndarray:
    """A_k for the lag-window estimator, from the RAW window (lags 0..lag).

    f(k) = sum_tau w[|tau|] rhat[tau] cos(2 pi k tau / n) with the biased ACF
    rhat[tau] = (1/n) sum_t x_t x_{t+tau}, so A[s, t] = w[|s-t|] cos(...) / n.
    Note what is *absent*: the (1 - |tau|/n) triangular factor is not in A at
    all. It appears in the moments because only n - |tau| pairs (s, t) sit on
    each diagonal -- once in the mean (hence ``kernel``) and, crucially, only
    once in tr(A_i A_j) as well.
    """
    lag = len(window) - 1
    sep = np.subtract.outer(np.arange(n), np.arange(n))
    weights = np.where(np.abs(sep) <= lag, window[np.abs(np.clip(sep, -lag, lag))], 0.0)
    return weights * np.cos(2 * np.pi * k * sep / n) / n


def exact_bin_covariance(a: np.ndarray, b: np.ndarray) -> float:
    """2 tr(A B), the exact covariance of two quadratic forms in unit-variance
    Gaussian noise. Both arguments are symmetric, so tr(A B) = sum(A * B)."""
    return float(2.0 * np.sum(a * b))


def reference_gls_dense(design, response, corr_circulant):
    """Closed-form GLS (DQuad Eq. 12) by dense linear algebra.

    Builds W explicitly from its first row and solves the normal equations with
    a dense inverse -- no FFT diagonalisation anywhere. This is the oracle for
    the O(n log n) circulant route in ``dwelch._apply_w_inverse``.

    ``design`` is Gamma^-1 B' and ``response`` is Gamma^-1 I, both already on
    the full two-sided Fourier grid; ``corr_circulant`` is W's first row.
    """
    n = len(corr_circulant)
    w = np.array([np.roll(corr_circulant, i) for i in range(n)])
    w_inv = np.linalg.inv(w)
    gram = design.T @ w_inv @ design
    moment = design.T @ w_inv @ response
    return np.linalg.solve(gram, moment)
