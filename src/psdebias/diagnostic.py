"""Debias-or-smooth diagnostics.

Two diagnostics live here:

* :func:`sign_diagnostic` — the paper's Sec. IV-C rule, kept verbatim for
  reproducibility: fit the all-knots configuration by unconstrained WLS on the
  window-convolved bases and recommend debiasing only if every coefficient is
  positive.
* :func:`regime_diagnostic` — the recommended replacement (used by
  ``fit_psd(mode="auto")``). Less conservative, and robust to the three
  mechanisms that make the sign rule cry wolf on small or noisy data:

  1. **Sampling noise.** With ~100 coefficients, some unconstrained estimates
     dip below zero by chance even when every true coefficient is positive.
  2. **Collinearity below the window bandwidth.** Knots spaced finer than the
     spectral window's bandwidth are unidentifiable through the blur — the
     convolved hats become nearly linearly dependent, coefficient variance
     explodes, and signs become meaningless.
  3. **Approximation error.** On sharp spectra a fixed B1 mesh's best
     weighted-L2 fit can genuinely need small negative coefficients near
     peaks; low noise makes these systematic negatives look "significant".

  The reframe behind the new rule: the debias sampler never needs the
  all-knots *unconstrained* fit to be positive — it needs *some well-fitting
  positive configuration to exist*, because it only ever visits positive
  configurations. Non-negative least squares at the full mesh tests exactly
  that: its fit-quality gap over unconstrained least squares measures what the
  positivity cone costs. A small gap means a positive blur-matched
  representation fits essentially as well as the best linear one — debiasing
  is operable and warranted. A large gap means positivity is fighting the
  data, the signature of an estimate whose blurring does not match the bases.

  Mechanism (2) is removed *by construction*: the diagnostic mesh is floored
  at the window's equivalent bandwidth (:func:`window_bandwidth`), the finest
  spacing the blur makes identifiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.stats
from numpy import typing as npt

from psdebias import splines


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def _validate(estimate, freqs, kernel, n_time):
    estimate = np.asarray(estimate, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if len(estimate) != len(freqs):
        raise ValueError(f"estimate length {len(estimate)} != freqs length {len(freqs)}")
    if np.any(estimate <= 0):
        raise ValueError("estimate must be strictly positive")
    if len(kernel) != n_time:
        raise ValueError(f"kernel length {len(kernel)} != n_time {n_time}")
    return estimate, freqs, kernel


def _full_mesh_design(estimate, freqs, kernel, n_time, knot_spacing, log_knots):
    """Weighted, window-convolved design with every candidate knot active —
    the debiasing regression both diagnostics probe."""
    knots, _, _ = splines.knot_grid(
        freqs, knot_spacing, log_knots=log_knots, ghost_knots=True
    )
    weights = estimate**-1
    falling_b, rising_b = splines.half_bases_biased(freqs, knots, n_time, kernel)
    gamma = np.ones(len(knots) - 2, dtype=np.int8)
    design = splines.assemble_design(
        falling_b * weights[None, :], rising_b * weights[None, :],
        gamma, knots, splines.EMPTY_IDX,
    )
    return design


def window_bandwidth(kernel: npt.NDArray[np.float64], n_time: int) -> float:
    """Equivalent bandwidth of the estimator's spectral window, in frequency
    bins of the ``n_time``-point grid.

    By Parseval on the window ``W`` (whose inverse Fourier transform is the
    one-sided ``kernel`` ``h``): ``int W = h[0]`` and
    ``int W^2 = h[0]^2 + 2 sum_{tau>=1} h[tau]^2``, so the equivalent width

        ``B_eq = (int W)^2 / int W^2``   (cycles/sample)

    is the width of the flat window with the same area and energy. Times
    ``n_time`` it is expressed in grid bins. Sanity anchors: the boxcar-taper
    (Fejer) window gives the textbook 1.50 bins; a multitaper window with
    time-bandwidth ``NW`` gives approximately its design width ``2 NW``.

    Interpretation for meshing: structure finer than ``B_eq`` cannot be told
    apart after blurring by ``W``, so basis functions narrower than this are
    nearly collinear in the convolved design.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if len(kernel) != n_time:
        raise ValueError(f"kernel length {len(kernel)} != n_time {n_time}")
    energy = kernel[0] ** 2 + 2.0 * np.sum(kernel[1:] ** 2)
    return float(n_time * kernel[0] ** 2 / energy)


# ---------------------------------------------------------------------------
# Paper Sec. IV-C sign rule (kept for reproducibility)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticResult:
    debias_recommended: bool
    beta: npt.NDArray[np.float64]
    n_negative: int


def sign_diagnostic(
    estimate: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    *,
    kernel: npt.NDArray[np.float64],
    n_time: int,
    knot_spacing: int = 4,
    log_knots: bool = False,
) -> DiagnosticResult:
    """The paper's Sec. IV-C rule, verbatim: all-knots unconstrained WLS on
    the window-convolved bases; debias only when every coefficient is
    positive.

    Mechanism: if the estimate is *less* blurred than its expectation
    (Regime 2), the window-convolved bases push too much power into the
    low-power tails and the least-squares fit can only compensate by turning
    coefficients negative. The rule reads that signature — but it also reacts
    to negatives from noise, collinearity, and approximation error, making it
    conservative on small data. Prefer :func:`regime_diagnostic` for
    decision-making; this function is kept as the paper's reference rule.
    """
    estimate, freqs, kernel = _validate(estimate, freqs, kernel, n_time)
    design = _full_mesh_design(estimate, freqs, kernel, n_time, knot_spacing, log_knots)
    beta, *_ = np.linalg.lstsq(design, np.ones(len(estimate)), rcond=None)
    n_negative = int(np.sum(beta <= 0))
    return DiagnosticResult(
        debias_recommended=n_negative == 0, beta=beta, n_negative=n_negative
    )


# ---------------------------------------------------------------------------
# Bandwidth-matched NNLS-gap diagnostic (recommended)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeDiagnostic:
    """Full evidence report of :func:`regime_diagnostic`.

    The decision is carried by ``gap_per_constraint`` alone; everything else
    is supporting evidence for the user's own judgment.
    """

    debias_recommended: bool
    gap_per_constraint: float
    n_constrained: int
    bandwidth_bins: float
    knot_spacing: int
    condition_number: float


def regime_diagnostic(
    estimate: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    *,
    kernel: npt.NDArray[np.float64],
    n_time: int,
    knot_spacing: int | None = None,
    log_knots: bool = False,
    gap_threshold: float = 4.0,
    alpha: float = 0.05,
) -> RegimeDiagnostic:
    """Should this estimate be debiased? Bandwidth-matched NNLS-gap rule.

    Pipeline:

    1. Mesh at the window's equivalent bandwidth (``knot_spacing=None``,
       the default, uses ``ceil(window_bandwidth(...))``) — the finest mesh
       the blur makes identifiable. Passing an explicitly finer spacing is
       honored but reintroduces the collinearity the default avoids.
    2. Unconstrained WLS and non-negative least squares of the debiasing
       regression at that mesh.
    3. **Decision**: debias iff the NNLS fit costs little over the
       unconstrained fit —

           ``gap_per_constraint = ((RSS_nnls - RSS_ols) / sigma_hat^2)
                                   / max(n_constrained, 1) <= gap_threshold``

       where ``n_constrained`` counts coefficients NNLS pinned at zero.
       Under a true nonnegative representation each active constraint costs
       O(sigma^2) (chi-bar-squared heuristic, ~chi2_1 per constraint), so
       values of a few are noise and large values mean positivity is fighting
       the data — the blur-mismatch signature. The default threshold 4.0 was
       calibrated on AR(4)/Matern/sunspot probe studies (true-Regime-1 cases
       reached at most ~3.6; deliberate mismatches score far higher).
    """
    estimate, freqs, kernel = _validate(estimate, freqs, kernel, n_time)

    bandwidth = window_bandwidth(kernel, n_time)
    if knot_spacing is None:
        knot_spacing = max(int(math.ceil(bandwidth)), 1)
    design = _full_mesh_design(estimate, freqs, kernel, n_time, knot_spacing, log_knots)
    y = np.ones(len(estimate))
    g, p = design.shape

    beta_ols, _, _, sv = np.linalg.lstsq(design, y, rcond=None)
    rss_ols = float(np.sum((y - design @ beta_ols) ** 2))
    sigma2 = rss_ols / max(g - p, 1)
    condition_number = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf

    beta_nnls, resid_norm = scipy.optimize.nnls(design, y)
    rss_nnls = float(resid_norm**2)
    n_constrained = int(np.sum(beta_nnls == 0))
    gap_per_constraint = float(
        (rss_nnls - rss_ols) / (sigma2 * max(n_constrained, 1))
    )

    return RegimeDiagnostic(
        debias_recommended=gap_per_constraint <= gap_threshold,
        gap_per_constraint=gap_per_constraint,
        n_constrained=n_constrained,
        bandwidth_bins=float(bandwidth),
        knot_spacing=int(knot_spacing),
        condition_number=condition_number,
    )
