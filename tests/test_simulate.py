import numpy as np
import pytest

from psdebias.analytic import ar_spectrum, matern_acf
from psdebias.simulate import sample_ar, sample_matern

from conftest import PAPER_AR_POLY


class TestSampleMatern:
    @pytest.mark.parametrize("alpha", [1.0, 1.5])
    def test_study_grid_embedding_positive(self, alpha, rng):
        # The simulation study's parameters must not raise
        x = sample_matern(2, 2**11, eta=1.0, alpha=alpha, lmbda=0.1, rng=rng)
        assert x.shape == (2, 2**11)
        assert np.all(np.isfinite(x))

    def test_empirical_acf_matches_theory(self, rng):
        eta, alpha, lmbda = 1.0, 1.5, 0.3
        n_samples, n_time = 4000, 256
        x = sample_matern(n_samples, n_time, eta, alpha, lmbda, rng=rng)
        target = matern_acf(np.arange(11, dtype=float), eta, alpha, lmbda)
        empirical = np.array(
            [np.mean(x[:, : n_time - k] * x[:, k:]) for k in range(11)]
        )
        np.testing.assert_allclose(empirical, target, atol=4 * eta**2 / np.sqrt(n_samples))

    def test_deterministic_under_seed(self):
        a = sample_matern(3, 64, 1.0, 1.5, 0.3, rng=np.random.default_rng(7))
        b = sample_matern(3, 64, 1.0, 1.5, 0.3, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


class TestSampleAr:
    def test_shape(self, rng):
        x = sample_ar(3, 500, [1.0, -0.5], rng=rng)
        assert x.shape == (3, 500)

    def test_requires_monic_polynomial(self, rng):
        with pytest.raises(ValueError, match="must start with 1"):
            sample_ar(1, 100, [0.9, -0.5], rng=rng)

    def test_ar1_variance_and_lag1(self, rng):
        phi, sd = 0.7, 1.0
        x = sample_ar(2000, 300, [1.0, -phi], sd=sd, rng=rng)
        var_target = sd**2 / (1 - phi**2)
        np.testing.assert_allclose(np.var(x), var_target, rtol=0.05)
        lag1 = np.mean(x[:, :-1] * x[:, 1:])
        np.testing.assert_allclose(lag1, phi * var_target, rtol=0.05)

    def test_paper_ar4_variance_matches_spectrum_integral(self, rng):
        # var = 2 * integral_0^{1/2} S(f) df
        ff = np.linspace(0, 0.5, 200001)
        var_target = 2 * np.trapezoid(ar_spectrum(ff, PAPER_AR_POLY, 1.0), ff)
        x = sample_ar(400, 4096, PAPER_AR_POLY, rng=rng)
        np.testing.assert_allclose(np.var(x), var_target, rtol=0.1)
