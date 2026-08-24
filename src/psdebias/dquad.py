"""Fixed-knot debiased quadratic estimator (the DWelch / DQuad baseline).

Given a quadratic spectral estimate and the knot configuration ``gamma``,
:func:`dquad` solves the paper's Eq. 5 by weighted non-negative least squares
on spectral-window-convolved B0 bases, then evaluates the *unconvolved* bases
at the fitted coefficients to obtain the debiased spectrum.

NNLS enforces coefficient positivity directly here: a fixed mesh has no MH
sampler rejecting negative-coefficient configurations, so the constraint moves
into the solver instead.

Passing ``corr`` selects a different fit entirely -- DQuad Eq. 12, done as the
paper does it: closed-form generalised least squares on the full two-sided
Fourier grid, with the correlation matrix circulant and its inverse applied by
FFT. That path is *unconstrained*, because Eq. 12's solution is analytic; the
positivity constraint above applies only to the default path.
"""

from __future__ import annotations

import numpy as np
import scipy.fft
import scipy.optimize
from numpy import typing as npt

from psdebias import splines


def _check_inputs(estimate, freqs, gamma, kernel, n_time):
    estimate = np.asarray(estimate, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    gamma = np.ascontiguousarray(gamma, dtype=np.int8)
    kernel = np.asarray(kernel, dtype=np.float64)
    if len(estimate) != len(freqs):
        raise ValueError(f"estimate length {len(estimate)} != freqs length {len(freqs)}")
    if np.any(estimate <= 0):
        raise ValueError("estimate must be strictly positive (it is used as an inverse weight)")
    if len(kernel) != n_time:
        raise ValueError(f"kernel length {len(kernel)} != n_time {n_time}")

    return estimate, freqs, gamma, kernel


def dquad(
    estimate: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    *,
    gamma: npt.NDArray[np.int8],
    kernel: npt.NDArray[np.float64],
    n_time: int,
) -> npt.NDArray[np.float64]:
    """Debias with piecewise-constant (B0) bases.

    ``gamma`` indicates active interior knots among the candidates
    ``[0, freqs..., 0.5]`` (length ``len(freqs)``); ``kernel`` is the
    estimator's one-sided bias sequence (length ``n_time``).

    ``corr`` is the estimator's one-sided frequency-correlation sequence
    (``SpectralEstimate.corr``). Supplying it solves DQuad Eq. 12 in closed form
    on the two-sided grid instead of the diagonally weighted NNLS problem;
    omitting it leaves the previous behaviour untouched.

    The Eq. 12 path needs the DC and Nyquist bins in order to mirror onto the
    two-sided grid, so pass estimates built with ``drop_endpoints=False``. Its
    ``gamma`` therefore runs over ``freqs[1:-1]``, the interior candidates,
    since ``freqs[0] = 0`` and ``freqs[-1] = 1/2`` already sit on the ghosts.
    """
    estimate, freqs, gamma, kernel = _check_inputs(
        estimate, freqs, gamma, kernel, n_time
    )
    # Two accepted layouts, as in util.freq_slice: the endpoint-dropped grid
    # (ghost knots off-grid, added here) or the full one-sided grid (freqs[0] and
    # freqs[-1] ARE the ghosts, so the candidate list is freqs itself).
    knots = freqs if freqs[0] == 0.0 else np.concatenate(([0.0], freqs, [0.5]))
    biased = splines.basis_b0_biased(freqs, knots, gamma, n_time, kernel)

    x_biased = biased * (estimate**-1)[:, None]
    beta, _ = scipy.optimize.nnls(x_biased, np.ones(len(estimate)))

    output = splines.basis_b0(freqs, knots, gamma)
    return output @ beta
