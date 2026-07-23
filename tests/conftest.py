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
