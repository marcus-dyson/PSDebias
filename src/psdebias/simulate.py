"""Process simulators for validation studies: Matern (circulant embedding), AR(p)."""

from __future__ import annotations

import numpy as np
import scipy.signal
from numpy import typing as npt

from psdebias.analytic import matern_acf

def sample_matern(
    n_samples: int,
    n_time: int,
    eta: float,
    alpha: float,
    lmbda: float,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Draw independent Matern realisations via circulant embedding.

    The covariance matrix on a regular grid embeds in a circulant matrix whose
    eigenvalues are the DFT of ``[acf(0..n-1), acf(n-2..1)]``. With that spectrum
    ``lam``, ``y = ifft(sqrt(lam) * fft(z)).real`` for ``z ~ N(0, I)`` has exactly
    the target covariance on its first ``n_time`` entries (the transform matrix is
    real-symmetric because ``lam`` is symmetric, and it squares to the embedding).

    Returns shape ``(n_samples, n_time)``. Raises ``ValueError`` when the
    embedding is not positive semi-definite for the given parameters.
    """
    rng = np.random.default_rng() if rng is None else rng
    acf = matern_acf(np.arange(n_time, dtype=np.float64), eta, alpha, lmbda, sigma=sigma)
    # Embed the Toeplitz covariance in a circulant matrix C of size
    # m = 2(n-1): first row [r0..r_{n-1}, r_{n-2}..r_1]. Circulants are
    # diagonalized by the DFT, so eig(C) = fft(first row) — computed in
    # O(m log m) with no matrix ever formed.
    c = np.concatenate([acf, acf[-2:0:-1]])
    lam = np.fft.fft(c).real
    if np.any(lam < 0):
        raise ValueError("circulant embedding not positive definite for these parameters")

    # y = F^-1 diag(sqrt(lam)) F z is A z with A = C^{1/2}: A is real and
    # symmetric because lam is real and symmetric (c is real and palindromic),
    # and A @ A = C. Hence cov(y) = A cov(z) A' = C exactly — the first n_time
    # entries of each row have precisely the Matern covariance, and the
    # nominal .real strips only floating-point residue, not information.
    m = len(c)
    z = rng.standard_normal((n_samples, m))
    y = np.fft.ifft(np.sqrt(lam) * np.fft.fft(z, axis=1), axis=1).real
    return y[:, :n_time]


def sample_ar(
    n_samples: int,
    n_time: int,
    ar_poly: npt.NDArray[np.float64],
    sd: float = 1.0,
    burn_in: int | None = None,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Draw AR(p) realisations ``X_t = -a_1 X_{t-1} - ... - a_p X_{t-p} + eps_t``.

    ``ar_poly = [1, a_1, ..., a_p]`` is the same characteristic polynomial used by
    :func:`psdebias.analytic.ar_spectrum` (for the paper's ``X_t = sum phi_j X_{t-j}``
    recursion pass ``[1, -phi_1, ..., -phi_p]``). Replaces the original package's
    dependency on ``statsmodels.tsa.arima_process.arma_generate_sample``, which
    uses this identical convention.

    Returns shape ``(n_samples, n_time)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    ar_poly = np.asarray(ar_poly, dtype=np.float64)
    if ar_poly[0] != 1.0:
        raise ValueError("ar_poly must start with 1 (characteristic polynomial form)")
    if burn_in is None:
        burn_in = max(50 * (len(ar_poly) - 1), 500)
    eps = sd * rng.standard_normal((n_samples, n_time + burn_in))
    x = scipy.signal.lfilter([1.0], ar_poly, eps, axis=1)
    return x[:, burn_in:]
