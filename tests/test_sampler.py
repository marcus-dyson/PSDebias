from pathlib import Path

import numpy as np
import pytest

from psdebias.estimators import multitaper, welch
from psdebias.sampler import (
    PSDebias,
    _betaln,
    _cholesky,
    _least_squares,
    _log_posterior,
    _mh_kernel,
)
from psdebias.simulate import sample_ar
from psdebias.splines import EMPTY_IDX, assemble_design, half_bases, knot_grid

from conftest import PAPER_AR_POLY, reference_log_posterior


@pytest.fixture(scope="module")
def welch_ar4():
    rng = np.random.default_rng(7)
    l, m, s = 512, 24, 512
    x = sample_ar(1, m * l, PAPER_AR_POLY, rng=rng)[0]
    return welch(x, segment_length=l, n_segments=m, step=s)


class TestNumericalCore:
    def test_betaln_vs_scipy(self, rng):
        import scipy.special

        for _ in range(20):
            a, b = rng.uniform(0.1, 50, 2)
            np.testing.assert_allclose(_betaln(a, b), scipy.special.betaln(a, b), rtol=1e-12)

    def test_least_squares_vs_lstsq(self, rng):
        for _ in range(20):
            n, p = 100, 7
            x = rng.standard_normal((n, p))
            y = rng.standard_normal(n)
            beta = np.empty(p)
            chol = np.empty((p, p))
            ok, ess = _least_squares(x, y, float(y @ y), beta, chol)
            assert ok
            expected, res, *_ = np.linalg.lstsq(x, y, rcond=None)
            np.testing.assert_allclose(beta, expected, rtol=1e-9)
            np.testing.assert_allclose(ess, res[0], rtol=1e-9)
            # chol is the lower Cholesky factor of X'X
            np.testing.assert_allclose(chol @ chol.T, x.T @ x, rtol=1e-9)

    def test_cholesky_flags_rank_deficiency(self, rng):
        x = rng.standard_normal((50, 4))
        x = np.column_stack([x, x[:, 0]])  # duplicate column
        gram = x.T @ x
        chol = np.empty((5, 5))
        assert not _cholesky(gram, chol)

    def test_log_posterior_vs_reference(self, rng):
        # Full pipeline comparison on random regression problems
        for _ in range(10):
            n, n_candidates = 80, 20
            q = int(rng.integers(0, n_candidates))
            p = q + 2
            x = rng.standard_normal((n, p))
            y = rng.standard_normal(n)
            c, a_sigma, b_sigma, a_pi, b_pi = 80.0, 1e-10, 1e-3, 1.0, 1.0

            beta = np.empty(p)
            chol = np.empty((p, p))
            ok, ess = _least_squares(x, y, float(y @ y), beta, chol)
            assert ok
            got = _log_posterior(
                ess, q, n_candidates, n, c, a_sigma, b_sigma, a_pi, b_pi,
                _betaln(a_pi, b_pi),
            )
            expected = reference_log_posterior(
                y, x, q, n_candidates, c, a_sigma, b_sigma, a_pi, b_pi
            )
            np.testing.assert_allclose(got, expected, rtol=1e-10)


class TestConditionalDraws:
    def test_moments_for_fixed_configuration(self, welch_ar4):
        # Freeze gamma by proposing no-op flips; the kernel then draws
        # (sigma^2, beta) from the conditional posterior of one configuration.
        freqs, psd, _, _ = welch_ar4
        y = np.log(psd)
        knots, _, halve = knot_grid(freqs, 16)
        falling, rising = half_bases(freqs, knots)
        n_interior = len(knots) - 2
        gamma0 = np.ones(n_interior, dtype=np.int8)

        rng = np.random.default_rng(3)
        n_iter, n_draws = 400, 100
        g = len(y)
        c, a_sigma, b_sigma = float(g), 1e-10, 1e-3
        nu = g + 2 * a_sigma

        gammas, betas, ess_kept, _ = _mh_kernel(
            gamma0, y, falling, rising, knots, halve,
            np.log(rng.uniform(size=n_iter)),
            np.zeros((n_iter, 2), dtype=np.int64),
            np.repeat(gamma0[0], 2 * n_iter).reshape(n_iter, 2).astype(np.int8),
            rng.chisquare(nu, size=(n_iter, n_draws)),
            rng.standard_normal((n_iter, n_draws, n_interior + 2)),
            c, a_sigma, b_sigma, 1.0, 1.0,
            0, n_iter, 1, False,
        )
        assert np.all(gammas == 1)

        design = assemble_design(falling, rising, gamma0, knots, halve)
        p = design.shape[1]
        beta_hat, *_ = np.linalg.lstsq(design, y, rcond=None)
        ess = float(y @ y - y @ (design @ beta_hat))
        m = ess + 2 * b_sigma
        cov_expected = (m / (nu - 2)) * (c / (1 + c)) * np.linalg.inv(design.T @ design)

        draws = betas[:, :, :p].reshape(-1, p)
        np.testing.assert_allclose(
            draws.mean(axis=0), beta_hat,
            atol=6 * np.sqrt(np.diag(cov_expected) / len(draws)).max(),
        )
        # elementwise sampling standard error of a covariance estimate:
        # SE(c_ij) ~ sqrt((c_ii c_jj + c_ij^2) / n)
        d = np.diag(cov_expected)
        se = np.sqrt((np.outer(d, d) + cov_expected**2) / len(draws))
        assert np.all(np.abs(np.cov(draws.T) - cov_expected) < 6 * se + 0.15 * np.abs(cov_expected))
        np.testing.assert_allclose(ess_kept, ess, rtol=1e-9)


class TestPSDebias:
    def test_determinism(self, welch_ar4):
        freqs, psd, kernel, dof = welch_ar4
        kwargs = dict(n_iterations=2000, warmup=500, thin=5, n_beta_draws=5)
        results = []
        for _ in range(2):
            sampler = PSDebias(
                psd, freqs, mode="smooth", dof=dof, knot_spacing=16,
                rng=np.random.default_rng(42),
            )
            results.append(sampler.sample(**kwargs))
        np.testing.assert_array_equal(results[0].gammas, results[1].gammas)
        np.testing.assert_array_equal(results[0].betas, results[1].betas)
        np.testing.assert_array_equal(results[0].mean, results[1].mean)

    def test_map_estimate_uses_kernel_posterior(self, welch_ar4):
        freqs, psd, kernel, dof = welch_ar4
        sampler = PSDebias(
            psd, freqs, mode="smooth", dof=dof, knot_spacing=16, rng=np.random.default_rng(2)
        )
        res = sampler.sample(n_iterations=2000, warmup=500, thin=5, n_beta_draws=5)
        gamma_map, curve = sampler.map_estimate()

        # Re-score every stored configuration with the pure-scipy reference and
        # check the argmax matches (guards sampler/selector formula drift)
        scores = []
        for i in range(len(res.gammas)):
            design = sampler.design_matrix(res.gammas[i], biased=False)
            scores.append(
                reference_log_posterior(
                    sampler.response, design, int(res.gammas[i].sum()),
                    sampler.n_interior, sampler._hyper["c"], 1e-10, 1e-3, 1.0, 1.0,
                )
            )
        np.testing.assert_array_equal(gamma_map, res.gammas[int(np.argmax(scores))])
        assert np.all(curve > 0)
        assert curve.shape == freqs.shape

    def test_debias_mode_positive_coefficients(self, welch_ar4):
        freqs, psd, kernel, _ = welch_ar4
        sampler = PSDebias(
            psd, freqs, mode="debias", kernel=kernel, n_time=512,
            knot_spacing=16, rng=np.random.default_rng(11),
        )
        res = sampler.sample(n_iterations=3000, warmup=1000, thin=5, n_beta_draws=5)
        assert np.all(res.mean > 0)
        assert 0.001 < res.acceptance_rate < 0.9
        # NaN padding: every row has exactly q+2 finite coefficient entries
        for i in range(0, len(res.gammas), 50):
            p = res.gammas[i].sum() + 2
            assert np.all(np.isfinite(res.betas[i, :, :p]))
            assert np.all(np.isnan(res.betas[i, :, p:]))

    def test_store_predictive(self, welch_ar4):
        freqs, psd, _, dof = welch_ar4
        sampler = PSDebias(
            psd, freqs, mode="smooth", dof=dof, knot_spacing=16, rng=np.random.default_rng(5)
        )
        res = sampler.sample(
            n_iterations=500, warmup=100, thin=5, n_beta_draws=4, store_predictive=True
        )
        assert res.posterior_predictive is not None
        n_kept = (500 + 4) // 5
        assert res.posterior_predictive.shape == (n_kept * 4, len(freqs))
        np.testing.assert_allclose(
            res.posterior_predictive.mean(axis=0), res.mean, rtol=1e-10
        )

    def test_initial_gamma_override(self, welch_ar4):
        freqs, psd, _, dof = welch_ar4
        sampler = PSDebias(
            psd, freqs, mode="smooth", dof=dof, knot_spacing=16, rng=np.random.default_rng(9)
        )
        gamma0 = np.zeros(sampler.n_interior, dtype=np.int8)
        res = sampler.sample(
            n_iterations=200, warmup=50, thin=5, n_beta_draws=3, initial_gamma=gamma0
        )
        assert res.gammas.shape[1] == sampler.n_interior
        with pytest.raises(ValueError, match="initial_gamma"):
            sampler.sample(
                n_iterations=100, warmup=10, thin=5, n_beta_draws=3,
                initial_gamma=np.zeros(3, dtype=np.int8),
            )

    def test_input_validation(self, welch_ar4):
        freqs, psd, kernel, _ = welch_ar4
        with pytest.raises(ValueError, match="mode"):
            PSDebias(psd, freqs, mode="banana")
        with pytest.raises(ValueError, match="requires kernel"):
            PSDebias(psd, freqs, mode="debias")
        with pytest.raises(ValueError, match="strictly positive"):
            PSDebias(psd - psd.max(), freqs, mode="smooth")
        with pytest.raises(ValueError, match="inside"):
            PSDebias(psd, freqs + 0.5, mode="smooth")


SUNSPOT_CSV = Path(__file__).parent.parent / "data" / "SN_y_tot_V2.0.csv"


@pytest.fixture(scope="module")
def sunspot_sampler():
    x = np.genfromtxt(SUNSPOT_CSV, delimiter=";").T[1]
    freqs, psd, kernel, _ = multitaper(x, nw=3.5)
    return PSDebias(
        psd, freqs, mode="debias", kernel=kernel, n_time=len(x),
        knot_spacing=2, rng=np.random.default_rng(1),
    )


@pytest.mark.skipif(not SUNSPOT_CSV.exists(), reason="sunspot dataset not present")
class TestDebiasInitialization:
    """Regression tests for the frozen-chain bug found on the sunspot dataset:
    an initial configuration outside the positivity support (e.g. all knots
    active with negative WLS coefficients) freezes the debias chain."""

    def _wls_beta(self, sampler, gamma):
        design = sampler.design_matrix(gamma, biased=True)
        beta, *_ = np.linalg.lstsq(design, sampler.response, rcond=None)
        return beta

    def test_scenario_is_nontrivial(self, sunspot_sampler):
        # the all-knots fit must have negative coefficients, otherwise this
        # scenario would not exercise the initialization at all
        gamma_all = np.ones(sunspot_sampler.n_interior, dtype=np.int8)
        assert np.any(self._wls_beta(sunspot_sampler, gamma_all) <= 0)

    def test_prior_init_is_all_positive(self, sunspot_sampler):
        gamma0 = sunspot_sampler._prior_initial_gamma(1.0, 1.0)
        assert gamma0.shape == (sunspot_sampler.n_interior,)
        assert np.all(self._wls_beta(sunspot_sampler, gamma0) > 0)

    def test_prior_init_deterministic_under_seed(self):
        x = np.genfromtxt(SUNSPOT_CSV, delimiter=";").T[1]
        freqs, psd, kernel, _ = multitaper(x, nw=3.5)
        draws = []
        for _ in range(2):
            s = PSDebias(
                psd, freqs, mode="debias", kernel=kernel, n_time=len(x),
                knot_spacing=2, rng=np.random.default_rng(123),
            )
            draws.append(s._prior_initial_gamma(1.0, 1.0))
        np.testing.assert_array_equal(draws[0], draws[1])

    def test_chain_moves_on_sunspot(self, sunspot_sampler):
        res = sunspot_sampler.sample(
            n_iterations=4000, warmup=2000, thin=10, n_beta_draws=5
        )
        assert res.acceptance_rate > 0.05
        assert len(np.unique(res.gammas, axis=0)) > 10
        assert np.all(np.isfinite(res.mean))
