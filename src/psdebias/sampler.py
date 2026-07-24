"""Adaptive Bayesian knot selection for spectral estimates (PSDebias).

Implements the paper's Algorithm 1: Metropolis-Hastings over the latent binary
knot vector gamma, with the regression coefficients and noise variance
integrated out in closed form (Zellner g-prior + inverse-gamma, Appendix B).
One class serves both regimes:

* ``mode="smooth"`` (variance-dominant): fit ``log(estimate)`` on unconvolved
  B1 bases with unit weights.
* ``mode="debias"`` (bias-dominant): fit the constant response ``1`` on
  spectral-window-convolved bases scaled by ``1/estimate`` (paper Sec. IV-B),
  with coefficients constrained positive; the debiased spectrum is read off
  the *unconvolved* bases.

The log posterior over configurations is (paper Appendix B, with ``g`` the
number of frequency bins, ``q`` active interior knots, ``p = q + 2`` design
columns, ``L`` interior candidates):

    log pi(gamma | Y) = -(p/2) log(1+c) - (g/2 + a_sigma) log(ESS + 2 b_sigma)
                        + betaln(q + a_pi, L - q + b_pi) - betaln(a_pi, b_pi)

The original auto-speccy implementation used the number of *candidate knots*
in place of ``g`` in the second exponent (and a slightly different Beta prior
in the sampler than in its MAP selector); both are corrected here — see
CHANGES.md.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy import typing as npt
from scipy.special import digamma

from psdebias import splines


# ---------------------------------------------------------------------------
# JIT-compiled numerical core
# ---------------------------------------------------------------------------

@njit(cache=True)
def _betaln(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

@njit(cache=True)
def _log_posterior(
    ess: float,
    q_interior: int,
    n_interior_candidates: int,
    g: int,
    c: float,
    a_sigma: float,
    b_sigma: float,
    a_pi: float,
    b_pi: float,
    log_beta_norm: float,
) -> float:
    # Term by term (paper Appendix B, p(Y|gamma) * p(gamma)):
    #
    #   -(p/2) log(1+c)
    #       Dimension penalty from integrating beta out under the Zellner
    #       g-prior. p = q_interior + 2 because the two edge half-hats are
    #       always in the model — every coefficient actually integrated out
    #       must be counted, not just the selectable ones. Larger c => flatter
    #       prior on beta => stiffer penalty per extra basis.
    #
    #   -(g/2 + a_sigma) log(ESS + 2 b_sigma)
    #       The evidence term after integrating sigma^2 out against
    #       IG(a_sigma, b_sigma). The exponent uses g = len(Y), the number of
    #       frequency bins. (The original package used the candidate-knot
    #       count here — solver.py:45 — which weakened the fit reward ~4x and
    #       disagreed with its own sigma^2 Gibbs step; see CHANGES.md 2.1.)
    #       ESS is clamped at 0 (it is a residual sum of squares, so negative
    #       values can only be floating-point noise) and b_sigma > 0 keeps the
    #       log finite even for a perfect fit.
    #
    #   betaln(q + a_pi, L - q + b_pi) - betaln(a_pi, b_pi)
    #       Beta-Binomial prior over WHICH interior candidates are active,
    #       marginalized over the inclusion probability pi ~ Beta(a_pi, b_pi).
    #       Counts only the L selectable interior knots (paper Eq. 10).
    #       a_pi = b_pi = 1 makes every model size a priori uniform.
    p = q_interior + 2  # coefficients integrated out (edge columns always active)
    residual = max(ess, 0.0) + 2.0 * b_sigma
    log_lik = -(p / 2.0) * np.log(1.0 + c) - (g / 2.0 + a_sigma) * np.log(residual)
    log_prior = (
        _betaln(q_interior + a_pi, n_interior_candidates - q_interior + b_pi)
        - log_beta_norm
    )
    return log_lik + log_prior


@njit(cache=True)
def _cholesky(a: npt.NDArray[np.float64], chol: npt.NDArray[np.float64]) -> bool:
    """Lower-triangular Cholesky of ``a`` into ``chol``. Returns False when a
    pivot falls below a relative floor (rank-deficient Gram matrix), which the
    sampler treats as proposal rejection. No exceptions"""
    p = a.shape[0]
    # Relative pivot floor: a pivot is the squared distance from column j to
    # the span of columns 0..j-1, so "pivot <= 1e-12 * largest diagonal" means
    # "this basis column is numerically a linear combination of the others"
    # (e.g. two knots so close their hats coincide on the grid). Scaling by
    # max(diag) makes the test invariant to the overall scale of the weights.
    max_diag = 0.0
    for j in range(p):
        if a[j, j] > max_diag:
            max_diag = a[j, j]
    tol = 1e-12 * max_diag
    for j in range(p):
        s = a[j, j]
        for k in range(j):
            s -= chol[j, k] * chol[j, k]
        if s <= tol or not np.isfinite(s):
            return False
        chol[j, j] = np.sqrt(s)
        for i in range(j + 1, p):
            t = a[i, j]
            for k in range(j):
                t -= chol[i, k] * chol[j, k]
            chol[i, j] = t / chol[j, j]
        for k in range(j + 1, p):
            chol[j, k] = 0.0
    return True


@njit(cache=True)
def _solve_lower(chol: npt.NDArray[np.float64], b: npt.NDArray[np.float64], p: int) -> None:
    """In-place forward substitution: b <- L^{-1} b (first p entries)."""
    for i in range(p):
        s = b[i]
        for k in range(i):
            s -= chol[i, k] * b[k]
        b[i] = s / chol[i, i]


@njit(cache=True)
def _solve_upper(chol: npt.NDArray[np.float64], b: npt.NDArray[np.float64], p: int) -> None:
    """In-place back substitution with L^T: b <- L^{-T} b (first p entries)."""
    for i in range(p - 1, -1, -1):
        s = b[i]
        for k in range(i + 1, p):
            s -= chol[k, i] * b[k]
        b[i] = s / chol[i, i]


@njit(cache=True)
def _least_squares(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    yty: float,
    beta: npt.NDArray[np.float64],
    chol: npt.NDArray[np.float64],
) -> tuple[bool, float]:
    """Least squares through the Gram matrix's Cholesky factor.

    Fills ``beta[:p]`` and ``chol[:p, :p]``; returns ``(ok, ess)``. Never forms
    an explicit inverse and never touches normal-equation solves outside the
    triangular factor (the original called ``solve``/``inv`` on X'X directly).
    """
    p = x.shape[1]
    gram = x.T @ x
    if not _cholesky(gram, chol):
        return False, np.inf
    xty = x.T @ y
    for i in range(p):
        beta[i] = xty[i]
    _solve_lower(chol, beta, p)
    _solve_upper(chol, beta, p)
    ess = yty
    for i in range(p):
        ess -= xty[i] * beta[i]
    return True, ess


@njit(cache=True)
def _mh_kernel(
    gamma0: npt.NDArray[np.int8],
    y: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    halve_idx: npt.NDArray[np.int64],
    log_u: npt.NDArray[np.float64],
    flip_idx: npt.NDArray[np.int64],
    flip_val: npt.NDArray[np.int8],
    chi2_draws: npt.NDArray[np.float64],
    z_draws: npt.NDArray[np.float64],
    c: float,
    a_sigma: float,
    b_sigma: float,
    a_pi: float,
    b_pi: float,
    warmup: int,
    n_iterations: int,
    thin: int,
    require_positive: bool,
):
    """Algorithm 1. Returns ``(gammas, betas, ess_kept, n_accept)``.

    Thinned states only are stored, and the pre-generated conditional-draw
    randoms (``chi2_draws``, ``z_draws``) are consumed per *kept* state, so
    memory is O(n_kept) rather than O(n_iterations).

    Design notes for readers
    ------------------------
    * All randomness arrives pre-generated as arrays (``log_u`` for the accept
      tests, ``flip_idx``/``flip_val`` for proposals, ``chi2_draws``/
      ``z_draws`` for the conditional posterior). Numba's own RNG cannot be
      seeded through a ``numpy.random.Generator``, so drawing everything in
      Python first is what makes runs exactly reproducible — and it costs
      nothing, since the arrays are consumed sequentially.
    * The chain's carried state is the septuple
      ``(gamma, design, chol, beta, ess, log_post, p)`` — everything needed
      to (a) score the next proposal incrementally and (b) draw coefficients
      when a state is stored. On acceptance the proposal's versions replace
      all of them; nothing is ever recomputed from scratch after init.
    * Output arrays are rectangular and NaN-padded (row i uses its first
      ``sum(gamma_i) + 2`` slots). Numba lists of ragged arrays would work but
      are slow and unbounded; the fixed layout also makes the memory bill
      explicit: ``n_kept * n_beta_draws * (L+2)`` doubles.
    * The proposal (set ``n_select`` random positions to fresh Bernoulli(1/2)
      values) is symmetric, so the MH ratio is just the posterior ratio.
      Note a proposal may equal the current state (flips can be no-ops); it
      is then accepted with probability 1, which is harmless.
    """
    n_interior = len(gamma0)
    g = len(y)
    p_max = n_interior + 2
    n_total = warmup + n_iterations
    n_kept = (n_iterations + thin - 1) // thin
    n_beta_draws = chi2_draws.shape[1]

    gammas = np.zeros((n_kept, n_interior), dtype=np.int8)
    betas = np.full((n_kept, n_beta_draws, p_max), np.nan)
    ess_kept = np.empty(n_kept)

    log_beta_norm = _betaln(a_pi, b_pi)
    yty = float(y @ y)

    gamma = gamma0.copy()
    design = splines.assemble_design(falling, rising, gamma, knots, halve_idx)
    beta = np.empty(p_max)
    chol = np.empty((p_max, p_max))
    p = design.shape[1]
    ok, ess = _least_squares(design, y, yty, beta[:p], chol[:p, :p])
    if not ok:
        raise ValueError("initial knot configuration gives a singular design matrix")
    q = int(np.sum(gamma))
    log_post = _log_posterior(ess, q, n_interior, g, c, a_sigma, b_sigma, a_pi, b_pi, log_beta_norm)

    beta_new = np.empty(p_max)
    chol_new = np.empty((p_max, p_max))
    work = np.empty(p_max)
    n_accept = 0

    for i in range(n_total):
        gamma_new = gamma.copy()
        for j in range(flip_idx.shape[1]):
            gamma_new[flip_idx[i, j]] = flip_val[i, j]

        design_new = splines.update_design(
            falling, rising, gamma_new, gamma, design, knots, halve_idx
        )
        p_new = design_new.shape[1]
        ok, ess_new = _least_squares(design_new, y, yty, beta_new[:p_new], chol_new[:p_new, :p_new])

        # Accept rule: reject outright if the Gram matrix was singular (ok is
        # False) or, in debias mode, if any least-squares coefficient is
        # non-positive — that truncation keeps the chain inside configurations
        # whose debiased spectrum is positive (paper Algorithm 1, line 5).
        # Otherwise standard Metropolis on the log scale:
        # accept iff min(0, delta) > log U, folded into two comparisons here.
        accept = False
        if ok:
            positive = True
            if require_positive:
                for j in range(p_new):
                    if beta_new[j] <= 0.0:
                        positive = False
                        break
            if positive:
                q_new = int(np.sum(gamma_new))
                log_post_new = _log_posterior(
                    ess_new, q_new, n_interior, g, c, a_sigma, b_sigma, a_pi, b_pi, log_beta_norm
                )
                delta = log_post_new - log_post
                if delta >= 0.0 or delta > log_u[i]:
                    accept = True

        if accept:
            gamma = gamma_new
            design = design_new
            ess = ess_new
            log_post = log_post_new
            p = p_new
            for j in range(p):
                beta[j] = beta_new[j]
                for k in range(p):
                    chol[j, k] = chol_new[j, k]
            n_accept += 1

        if i >= warmup and (i - warmup) % thin == 0:
            j_keep = (i - warmup) // thin
            gammas[j_keep] = gamma
            ess_kept[j_keep] = ess
            # Conditional (sigma^2, beta) draws for the CURRENT state (paper
            # App. B). Marginally beta | Y, gamma is multivariate Student-t;
            # it is sampled here by composition:
            #   sigma^2 = (ESS + 2 b_sigma) / chi2_nu        (sigma^2 ~ IG)
            #   beta    = beta_hat + sqrt(sigma^2 c/(1+c)) L^{-T} z, z ~ N(0,I)
            # where L is the Cholesky factor of X'X carried in the chain
            # state, so (L^{-T} z) has covariance (X'X)^{-1} — one triangular
            # solve per draw, no matrix inversion. c/(1+c) is the g-prior
            # shrinkage of the posterior covariance toward beta_hat.
            m = max(ess, 0.0) + 2.0 * b_sigma
            shrink = c / (1.0 + c)
            for d in range(n_beta_draws):
                scale = np.sqrt(m / chi2_draws[j_keep, d] * shrink)
                for jj in range(p):
                    work[jj] = z_draws[j_keep, d, jj]
                _solve_upper(chol, work, p)
                for jj in range(p):
                    betas[j_keep, d, jj] = beta[jj] + scale * work[jj]

    return gammas, betas, ess_kept, n_accept


@njit(cache=True)
def _predictive_pass(
    gammas: npt.NDArray[np.int8],
    betas: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    halve_idx: npt.NDArray[np.int64],
    exponentiate: bool,
    store: bool,
    offset: np.float64,
):
    """Posterior-predictive evaluation on the *unconvolved* bases.

    Walks the kept configurations with incremental design updates, transforms
    each stored coefficient draw to the frequency grid (exponentiating per draw
    in smooth mode so results are always linear-scale PSD), and accumulates
    streaming mean/variance. When ``store`` the full draw matrix
    ``(n_kept * n_beta_draws, F)`` is also returned (empty otherwise).
    """
    n_kept, n_beta_draws, _ = betas.shape
    n_freq = falling.shape[1]

    total = np.zeros(n_freq)
    total_sq = np.zeros(n_freq)
    n_draws_total = n_kept * n_beta_draws
    stored = np.empty((n_draws_total if store else 0, n_freq))

    design = splines.assemble_design(falling, rising, gammas[0], knots, halve_idx)
    gamma_prev = gammas[0]
    for i in range(n_kept):
        # Consecutive kept states differ by at most thin * n_select flips, so
        # the incremental update still reuses most columns here.
        design = splines.update_design(
            falling, rising, gammas[i], gamma_prev, design, knots, halve_idx
        )
        gamma_prev = gammas[i]
        p = design.shape[1]
        for d in range(n_beta_draws):
            curve = design @ betas[i, d, :p]
            # Smooth mode fits log S(f). Accumulate on the LOG scale and
            # exponentiate the mean (plug-in): exp(E[log S]) = S. The mean of
            # exp(curve) would instead be S * exp(sigma^2/2), inflated upward by
            # Jensen's inequality — the bias the log-mean offset unmasks.
            total += curve
            total_sq += curve * curve
            if store:
                if exponentiate:
                    # strip the sampling offset so predictive draws share the
                    # PSD scale of `mean` (= their geometric mean, exp of E[log])
                    stored[i * n_beta_draws + d] = np.exp(curve - offset)
                else:
                    stored[i * n_beta_draws + d] = curve

    mu = total / n_draws_total
    var = total_sq / n_draws_total - mu * mu
    if exponentiate:
        # exp of the log-scale posterior mean; map the log-scale variance to a
        # linear-scale sd via the lognormal relation sd = mean * sqrt(e^var - 1).

        # Take offset off mean - Undoing the correction
        mean = np.exp(mu - offset)
        std = mean * np.sqrt(np.maximum(np.exp(var) - 1.0, 0.0))
    else:
        mean = mu
        std = np.sqrt(np.maximum(var, 0.0)) # No negatives
    return mean, std, stored


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Posterior summary of a PSDebias fit. ``mean``/``std``/``posterior_predictive``
    are always linear-scale PSD.

    In debias mode ``mean`` is the posterior mean of the (linear) spectrum. In
    smooth mode the fit is on ``log S(f)``, and ``mean`` is the exponentiated
    posterior mean of that log-spectrum — ``exp(E[log S])``, the plug-in point
    estimate (a geometric mean/median), not the posterior mean of ``S`` itself.
    Taking the arithmetic mean of the exponentiated draws would instead give
    ``S * exp(sigma^2/2)``, inflated upward by Jensen's inequality. ``std`` is
    the matching linear-scale sd (lognormal-mapped in smooth mode), and
    ``posterior_predictive`` still holds the individual exponentiated draws."""

    freqs: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    mode: str
    expected_knots: float
    acceptance_rate: float
    gammas: npt.NDArray[np.int8]
    betas: npt.NDArray[np.float64]
    ess: npt.NDArray[np.float64]
    posterior_predictive: npt.NDArray[np.float64] | None


class PSDebias:
    """Adaptive smoothing/debiasing of a quadratic spectral estimate.

    Parameters
    ----------
    estimate : linear-scale spectral estimate on ``freqs``.
    freqs : normalized frequency grid in (0, 0.5) — the endpoint-dropped rfft
        grid of an ``n_time``-point segment. Work in normalized frequency
        (``fs = 1``); rescale externally for physical units.
    mode : ``"debias"`` (bias-dominant regime, needs ``kernel`` and ``n_time``)
        or ``"smooth"`` (variance-dominant regime).
    kernel : one-sided bias sequence ``h[tau]`` of the estimator (returned by
        every ``psdebias.estimators`` function), length ``n_time``.
    n_time : segment length whose rfft grid ``freqs`` lives on.
    knot_spacing : one candidate knot per this many frequency bins.
    log_knots : geometric candidate spacing (resolves low-frequency detail).
    rng : ``numpy.random.Generator`` for reproducibility.
    """

    def __init__(
        self,
        estimate: npt.NDArray[np.float64],
        freqs: npt.NDArray[np.float64],
        *,
        mode: str,
        kernel: npt.NDArray[np.float64] | None = None,
        dof: np.float64 | None = None, # Need dof for log scale correction
        n_time: int | None = None,
        knot_spacing: int = 4,
        log_knots: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        if mode not in ("debias", "smooth"):
            raise ValueError(f"mode must be 'debias' or 'smooth', got {mode!r}")
        estimate = np.asarray(estimate, dtype=np.float64)
        freqs = np.asarray(freqs, dtype=np.float64)
        if len(estimate) != len(freqs):
            raise ValueError(f"estimate length {len(estimate)} != freqs length {len(freqs)}")
        if np.any(estimate <= 0):
            raise ValueError("estimate must be strictly positive")
        if freqs[0] <= 0.0 or freqs[-1] >= 0.5:
            raise ValueError(
                "freqs must lie strictly inside (0, 0.5): pass the endpoint-dropped "
                "rfft grid in normalized frequency (fs = 1)"
            )
        self.dof = dof
        self.mode = mode
        self.freqs = freqs
        self.estimate = estimate
        self.rng = np.random.default_rng() if rng is None else rng

        # One kernel, two regressions. Everything mode-specific is decided
        # right here and handed to the same _mh_kernel:
        #
        #                       debias                     smooth
        #   knots            [0, freqs[pos], 0.5]       freqs[pos]
        #   response Y       ones(g)                    log(estimate)
        #   fitting bases    window-convolved x (1/I)   direct evaluations
        #   halve (fit)      none (smooth PSD curves)   interior candidates
        #   positivity       required on beta_hat       not needed (exp > 0)
        #   output bases     unconvolved                unconvolved
        #   output transform none                       exp per draw
        #
        # Debias: the paper's Eq. 5 regression I(w) ~ sum beta_i (B_i * H_n)(w)
        # is divided through by I(w) — response becomes the constant 1 and the
        # 1/I weighting rides on the bases. That stabilizes the variance
        # (periodogram-type estimates have sd proportional to their mean) while
        # keeping the linear scale that links the spectrum to the ACVS.
        # Smooth: log I(w) has approximately constant variance already, so the
        # regression is ordinary least squares on plain B1 hats.
        if mode == "debias":
            if kernel is None or n_time is None:
                raise ValueError("mode='debias' requires kernel and n_time")
            kernel = np.asarray(kernel, dtype=np.float64)
            self.knots, self._interior_idx, self._halve_out = splines.knot_grid(
                freqs, knot_spacing, log_knots=log_knots, ghost_knots=True
            )
            weights = estimate**-1
            falling_b, rising_b = splines.half_bases_biased(freqs, self.knots, n_time, kernel)
            self._falling_fit = falling_b * weights[None, :]
            self._rising_fit = rising_b * weights[None, :]
            self._halve_fit = splines.EMPTY_IDX
            self.response = np.ones(len(estimate))
            self._require_positive = True
        else:
            self.knots, self._interior_idx, self._halve_out = splines.knot_grid(
                freqs, knot_spacing, log_knots=log_knots, ghost_knots=False
            )
            self._falling_fit, self._rising_fit = splines.half_bases(freqs, self.knots)
            self._halve_fit = self._halve_out
            # Add in Correction (Need dof):
            if self.dof is None:
                raise ValueError("DOF required for smoothing")
            offset = np.log(self.dof) - digamma(self.dof)
            self.response = np.log(estimate) + offset
            self._require_positive = False

        self._falling_out, self._rising_out = splines.half_bases(freqs, self.knots)
        self.n_interior = len(self.knots) - 2
        self._hyper: dict[str, float] = {}
        self.result: FitResult | None = None

    def _prior_initial_gamma(
        self, a_pi: float, b_pi: float, max_attempts: int = 1000
    ) -> npt.NDArray[np.int8]:
        """Draw an initial knot configuration from the Beta-Bernoulli prior:
        ``pi ~ Beta(a_pi, b_pi)``, ``gamma_i ~ Bernoulli(pi)``, resampling both
        each attempt.

        In debias mode the chain's target is truncated to configurations with
        all-positive least-squares coefficients, so the initial state must
        satisfy that too — otherwise the sampler can start outside its own
        support and freeze (every proposal from a dense negative-coefficient
        state gets rejected). Attempts are retried until the unconstrained WLS
        fit is all-positive; typically a handful suffice because small ``pi``
        draws produce sparse, wide-hat configurations.
        """
        yty = float(self.response @ self.response)
        for _ in range(max_attempts):
            pi0 = self.rng.beta(a_pi, b_pi)
            gamma = (self.rng.uniform(size=self.n_interior) < pi0).astype(np.int8)

            design = splines.assemble_design(
                self._falling_fit, self._rising_fit, gamma, self.knots, self._halve_fit
            )
            p = design.shape[1]
            beta = np.empty(p)
            chol = np.empty((p,p))

            ok, _ = _least_squares(design, self.response, yty, beta,chol)

            if not ok:
                continue
            if not self._require_positive or np.all(beta > 0):
                return gamma

        raise ValueError(
            f"no initial knot configuration with all-positive coefficients found "
            f"in {max_attempts} prior draws — debiasing is likely not appropriate "
            f"for this estimate; run regime_diagnostic() or use mode='smooth'"
        )

    def sample(
        self,
        *,
        n_iterations: int = 50_000,
        warmup: int = 30_000,
        thin: int = 10,
        n_select: int = 2,
        n_beta_draws: int = 50,
        a_pi: float = 1.0,
        b_pi: float = 1.0,
        c: float | None = None,
        a_sigma: float = 1e-10,
        b_sigma: float = 1e-3,
        store_predictive: bool = False,
        initial_gamma: npt.NDArray[np.int8] | None = None,
    ) -> FitResult:
        """Run the sampler and return a :class:`FitResult`.

        ``c`` is the g-prior scale; the paper's choice ``c = g`` (the number of
        frequency bins) is the default. ``a_pi``/``b_pi`` parameterize the
        Beta-Binomial prior on the number of active knots (1, 1 = uniform).
        ``initial_gamma`` overrides the default prior-sampled starting
        configuration (in debias mode, supply one whose weighted least-squares
        coefficients are positive, or the chain may never move).
        """
        if c is None:
            c = float(len(self.response))
        rng = self.rng
        n_total = warmup + n_iterations
        n_kept = (n_iterations + thin - 1) // thin
        g = len(self.response)

        if initial_gamma is not None:
            gamma0 = np.ascontiguousarray(initial_gamma, dtype=np.int8)
            if gamma0.shape != (self.n_interior,):
                raise ValueError(
                    f"initial_gamma must have shape ({self.n_interior},), got {gamma0.shape}"
                )
        else:
            gamma0 = self._prior_initial_gamma(a_pi, b_pi)

        log_u = np.log(rng.uniform(size=n_total))
        flip_idx = rng.integers(0, self.n_interior, size=(n_total, n_select))
        flip_val = rng.integers(0, 2, size=(n_total, n_select)).astype(np.int8)
        nu = g + 2 * a_sigma
        chi2_draws = rng.chisquare(nu, size=(n_kept, n_beta_draws))
        z_draws = rng.standard_normal((n_kept, n_beta_draws, self.n_interior + 2))

        gammas, betas, ess, n_accept = _mh_kernel(
            gamma0,
            self.response,
            self._falling_fit,
            self._rising_fit,
            self.knots,
            self._halve_fit,
            log_u,
            flip_idx,
            flip_val,
            chi2_draws,
            z_draws,
            float(c),
            float(a_sigma),
            float(b_sigma),
            float(a_pi),
            float(b_pi),
            warmup,
            n_iterations,
            thin,
            self._require_positive,
        )

        if n_accept == 0:
            warnings.warn(
                "MCMC chain never accepted a proposal — every stored sample is the "
                "initial configuration and the results are unreliable. Check "
                "regime_diagnostic() and consider mode='smooth'.",
                RuntimeWarning,
                stacklevel=2,
            )

        offset = (np.log(self.dof) - digamma(self.dof)) if self.mode == "smooth" else 0.0
        mean, std, stored = _predictive_pass(
            gammas,
            betas,
            self._falling_out,
            self._rising_out,
            self.knots,
            self._halve_out,
            self.mode == "smooth",
            store_predictive,
            offset=offset
        )

        self._hyper = {"a_pi": a_pi, "b_pi": b_pi, "c": float(c),
                       "a_sigma": a_sigma, "b_sigma": b_sigma}
        self.result = FitResult(
            freqs=self.freqs,
            mean=mean,
            std=std,
            mode=self.mode,
            expected_knots=float(np.mean(np.sum(gammas, axis=1))),
            acceptance_rate=n_accept / n_total,
            gammas=gammas,
            betas=betas,
            ess=ess,
            posterior_predictive=stored if store_predictive else None,
        )
        return self.result

    def map_estimate(self) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
        """Most probable sampled configuration and its spectrum estimate.

        Scores every kept ``(gamma, ESS)`` with the *same* compiled log
        posterior the chain targets (the original package used two different
        formulas for sampling and mode selection). The returned curve averages
        the stored coefficient draws of the winning configuration on the
        unconvolved bases (exponentiated per draw in smooth mode).
        """
        if self.result is None:
            raise RuntimeError("call sample() first")
        h = self._hyper
        log_beta_norm = _betaln(h["a_pi"], h["b_pi"])
        g = len(self.response)
        best, best_lp = 0, -np.inf
        for i in range(len(self.result.gammas)):
            lp = _log_posterior(
                self.result.ess[i],
                int(np.sum(self.result.gammas[i])),
                self.n_interior,
                g,
                h["c"],
                h["a_sigma"],
                h["b_sigma"],
                h["a_pi"],
                h["b_pi"],
                log_beta_norm,
            )
            if lp > best_lp:
                best, best_lp = i, lp
        gamma_map = self.result.gammas[best]
        design = splines.assemble_design(
            self._falling_out, self._rising_out, gamma_map, self.knots, self._halve_out
        )
        p = design.shape[1]
        curves = self.result.betas[best, :, :p] @ design.T
        if self.mode == "smooth":
            # Taking away offset
            offset = np.log(self.dof) - digamma(self.dof)
            curves = np.exp(curves - offset)
        return gamma_map, curves.mean(axis=0)

    def design_matrix(self, gamma: npt.NDArray[np.int8], *, biased: bool) -> npt.NDArray[np.float64]:
        """Design matrix for a given configuration (debugging/diagnostics).
        ``biased=True`` returns the fitting bases (weighted, window-convolved in
        debias mode); ``biased=False`` the unconvolved output bases."""
        gamma = np.ascontiguousarray(gamma, dtype=np.int8)
        if biased:
            return splines.assemble_design(
                self._falling_fit, self._rising_fit, gamma, self.knots, self._halve_fit
            )
        return splines.assemble_design(
            self._falling_out, self._rising_out, gamma, self.knots, self._halve_out
        )


def fit_psd(
    estimate: SpectralEstimate,
    *,
    n_time: int | None = None,
    mode: str = "auto",
    knot_spacing: int = 4,
    log_knots: bool = False,
    rng: np.random.Generator | None = None,
    **sample_kwargs,
) -> FitResult:
    """One-call PSDebias: pick the regime, sample, return the fit.

    ``mode="auto"`` runs :func:`psdebias.diagnostic.regime_diagnostic` — the
    bandwidth-matched NNLS-gap rule (does a positive, blur-matched
    representation fit essentially as well as the best unconstrained one?) —
    and dispatches to ``"debias"`` or ``"smooth"`` accordingly. The diagnostic
    chooses its own mesh from the window bandwidth, independent of the
    sampler's ``knot_spacing``. Requires ``kernel`` and ``n_time`` unless
    ``mode="smooth"``.
    """
    from psdebias.diagnostic import regime_diagnostic

    # take out items
    freqs, psd, kernel, dof = estimate

    if mode == "auto":
        if kernel is None or n_time is None:
            raise ValueError("mode='auto' requires kernel and n_time for the regime diagnostic")
        diag = regime_diagnostic(
            psd, freqs, kernel=kernel, n_time=n_time, log_knots=log_knots,
        )
        mode = "debias" if diag.debias_recommended else "smooth"

    sampler = PSDebias(
        psd, freqs, mode=mode, kernel=kernel,dof=dof, n_time=n_time,
        knot_spacing=knot_spacing, log_knots=log_knots, rng=rng,
    )
    return sampler.sample(**sample_kwargs)
