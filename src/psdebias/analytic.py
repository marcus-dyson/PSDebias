"""Closed-form spectra and autocovariances for AR(p) and Matern processes."""

from __future__ import annotations

import numpy as np
from numpy import typing as npt
from scipy.special import gamma as gamma_fn
from scipy.special import kv


def ar_spectrum(
    ff: npt.NDArray[np.float64],
    ar_poly: npt.NDArray[np.float64],
    sd: float,
    delta: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Theoretical PSD of an AR(p) process.

    ``ar_poly`` is the AR characteristic polynomial ``[1, a_1, ..., a_p]`` acting
    as ``X_t = -a_1 X_{t-1} - ... - a_p X_{t-p} + eps_t`` with
    ``eps_t ~ N(0, sd^2)``. In the paper's recursion ``X_t = sum phi_j X_{t-j} + eps``
    this means ``ar_poly = [1, -phi_1, ..., -phi_p]``.

    ``S(f) = sd^2 / |1 + sum_k a_k exp(-2 pi i k f delta)|^2``
    """
    ff = np.asarray(ff, dtype=np.float64) * delta
    ar_poly = np.asarray(ar_poly, dtype=np.float64)
    k = np.arange(1, len(ar_poly))
    # d[f] = 1 + sum_k a_k exp(-2 pi i k f): the (F x p-1) matrix of e^{-2 pi i k f}
    # (outer product of frequencies and lags) contracted with the AR coefficients
    # a_1..a_p, giving the frequency response of the AR filter per frequency.
    d = 1.0 + np.exp(-2j * np.pi * np.outer(ff, k)) @ ar_poly[1:]
    return sd**2 / np.abs(d) ** 2


def matern_acf(
    tau: npt.NDArray[np.float64],
    eta: float,
    alpha: float,
    lmbda: float,
    sigma: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Matern autocovariance at lags ``tau`` (smoothness ``nu = alpha - 1/2``),
    with optional additive white-noise variance ``sigma^2`` at lag zero."""
    tau = np.asarray(tau, dtype=np.float64)
    # Parameterization: eta^2 is the marginal variance, lmbda the inverse
    # length scale, and the smoothness enters as nu = alpha - 1/2 so that the
    # PSD decays like f^(-2 alpha) (matern_spectrum) — alpha = 1 is the
    # Ornstein-Uhlenbeck/AR(1)-like case, alpha -> inf the Gaussian kernel.
    # K_nu diverges at 0, so the exact limit eta^2 (+ white-noise sigma^2,
    # which only ever appears at lag 0) is patched in explicitly.
    nu = alpha - 0.5
    out = np.empty_like(tau)
    zero = tau == 0.0
    out[zero] = eta**2 + sigma**2
    tau_pos = np.abs(lmbda * tau[~zero])
    out[~zero] = 2 * eta**2 / (gamma_fn(nu) * 2**nu) * tau_pos**nu * kv(nu, tau_pos)
    return out


def matern_spectrum(
    ff: npt.NDArray[np.float64], eta: float, alpha: float, lmbda: float
) -> npt.NDArray[np.float64]:
    """Matern PSD: ``S(f) = C * ((2 pi f)^2 + lambda^2)^(-alpha)`` with the
    normalisation matching :func:`matern_acf`."""
    # c is the same normalization constant implicit in matern_acf: it makes the
    # PSD integrate to the marginal variance eta^2, so the acf/spectrum pair is
    # consistent (acf(0) = int S(f) df = eta^2). s folds it together with the
    # eta^2 * lmbda^(2 alpha - 1) scale of the (2 pi f)^2 + lmbda^2 kernel.
    c = gamma_fn(0.5) * gamma_fn(alpha - 0.5) / (2 * gamma_fn(alpha) * np.pi)
    s = eta**2 * lmbda ** (2 * alpha - 1) / c
    return s * ((2 * np.pi * np.asarray(ff, dtype=np.float64)) ** 2 + lmbda**2) ** (-alpha)
