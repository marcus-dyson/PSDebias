"""Fixed-knot debiased quadratic estimators (DWelch / DQuad baselines).

Given a quadratic spectral estimate and the knot configuration ``gamma``,
these solve the paper's Eq. 5 by weighted non-negative least squares on
spectral-window-convolved spline bases, then evaluate the *unconvolved*
bases at the fitted coefficients to obtain the debiased spectrum.

Unlike the original package, the B1 variant is assembled through the same
slope/pivot machinery as the adaptive sampler (:mod:`psdebias.splines`), so
it is exact for arbitrary nonuniform knot configurations — the original
``A[1:] + B[:-1]`` pairing (plus a clip-to-1 patch) was only correct when
the active knots were equally spaced.
"""

from __future__ import annotations

import numpy as np
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


def dwelch_b0(
    estimate: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    *,
    gamma: npt.NDArray[np.int8],
    kernel: npt.NDArray[np.float64],
    n_time: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Debias with piecewise-constant (B0) bases.

    ``gamma`` indicates active interior knots among the candidates
    ``[0, freqs..., 0.5]`` (length ``len(freqs)``); ``kernel`` is the
    estimator's one-sided bias sequence (length ``n_time``).
    """
    estimate, freqs, gamma, kernel = _check_inputs(estimate, freqs, gamma, kernel, n_time)
    knots = np.concatenate(([0.0], freqs, [0.5]))
    weights = estimate**-1

    x_biased = splines.basis_b0_biased(freqs, knots, gamma, n_time, kernel) * weights[:, None]
    beta, _ = scipy.optimize.nnls(x_biased, np.ones(len(estimate)))
    x_out = splines.basis_b0(freqs, knots, gamma)
    return freqs, x_out @ beta


def dwelch_b1(
    estimate: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    *,
    gamma: npt.NDArray[np.int8],
    kernel: npt.NDArray[np.float64],
    n_time: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Debias with piecewise-linear (B1) bases; exact for nonuniform knots."""
    estimate, freqs, gamma, kernel = _check_inputs(estimate, freqs, gamma, kernel, n_time)
    knots = np.concatenate(([0.0], freqs, [0.5]))
    weights = estimate**-1

    # Step 1 — fit on the BIASED bases: solve the paper's Eq. 5 with the
    # response divided through by the estimate (constant response 1, the 1/I
    # variance-stabilizing weight carried on the bases). NNLS enforces the
    # coefficient positivity here directly — a fixed mesh has no MH sampler
    # rejecting negative-coefficient configurations, so the constraint moves
    # into the solver instead.
    falling_b, rising_b = splines.half_bases_biased(freqs, knots, n_time, kernel)
    x_biased = splines.assemble_design(
        falling_b * weights[None, :], rising_b * weights[None, :],
        gamma, knots, splines.EMPTY_IDX,
    )
    beta, _ = scipy.optimize.nnls(x_biased, np.ones(len(estimate)))

    # Step 2 — evaluate the UNCONVOLVED bases at those coefficients. The
    # coefficients describe the spectrum before the window blurred it, so
    # reading them off the clean hats is what "inverts" the convolution.
    # Every grid frequency is a candidate knot here (knots = [0, freqs, 0.5]),
    # hence halve_idx covers the whole grid.
    falling_u, rising_u = splines.half_bases(freqs, knots)
    halve_idx = np.arange(len(freqs), dtype=np.int64)
    x_out = splines.assemble_design(falling_u, rising_u, gamma, knots, halve_idx)
    return freqs, x_out @ beta
