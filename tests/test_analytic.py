import numpy as np
import pytest

from psdebias.analytic import ar_spectrum, matern_acf, matern_spectrum

from conftest import PAPER_AR_POLY


class TestArSpectrum:
    def test_white_noise_flat(self):
        ff = np.linspace(0.01, 0.49, 50)
        np.testing.assert_allclose(ar_spectrum(ff, [1.0], 2.0), 4.0)

    def test_integrates_to_variance_ar1(self):
        # AR(1): X_t = phi X_{t-1} + eps, var = sd^2 / (1 - phi^2)
        phi, sd = 0.6, 1.5
        ff = np.linspace(0, 0.5, 20001)
        s = ar_spectrum(ff, [1.0, -phi], sd)
        # two-sided spectrum: variance = 2 * integral over (0, 1/2)
        var = 2 * np.trapezoid(s, ff)
        np.testing.assert_allclose(var, sd**2 / (1 - phi**2), rtol=1e-4)

    def test_paper_ar4_positive_and_peaked(self):
        ff = np.linspace(0.001, 0.499, 999)
        s = ar_spectrum(ff, PAPER_AR_POLY, 1.0)
        assert np.all(s > 0)
        # Percival & Walden AR(4) has its twin peaks near f ~ 0.11-0.14
        assert 0.08 < ff[np.argmax(s)] < 0.16

    def test_matches_original_loop_formula(self, rng):
        # original implementation: d = 1 + sum a_k exp(-2i pi k f), psd = sd^2/|d|^2
        ff = rng.uniform(0, 0.5, 17)
        poly = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
        d = np.ones(len(ff), dtype=complex)
        for k in range(1, len(poly)):
            d += poly[k] * np.exp(-2j * np.pi * k * ff)
        np.testing.assert_allclose(ar_spectrum(ff, poly, 1.3), 1.3**2 / np.abs(d) ** 2)


class TestMatern:
    def test_acf_lag_zero_is_variance(self):
        acf = matern_acf(np.arange(10.0), eta=1.0, alpha=1.5, lmbda=0.1, sigma=0.5)
        np.testing.assert_allclose(acf[0], 1.0**2 + 0.5**2)

    def test_acf_decreasing(self):
        acf = matern_acf(np.arange(100.0), eta=1.0, alpha=1.5, lmbda=0.1)
        assert np.all(np.diff(acf) < 0)

    def test_spectrum_vs_acf_via_fourier(self):
        # S(f) should be approximately the Fourier transform of the ACF
        eta, alpha, lmbda = 1.0, 1.5, 0.5
        n = 2**14
        acf = matern_acf(np.arange(n, dtype=float), eta, alpha, lmbda)
        folded = acf.copy()
        folded[1:] += acf[:0:-1]
        s_grid = np.fft.rfft(folded).real
        ff = np.fft.rfftfreq(n)
        s_true = matern_spectrum(ff[1:200], eta, alpha, lmbda)
        np.testing.assert_allclose(s_grid[1:200], s_true, rtol=0.02)

    @pytest.mark.parametrize("alpha", [1.0, 1.5])
    def test_study_parameters_finite(self, alpha):
        acf = matern_acf(np.arange(2**11, dtype=float), 1.0, alpha, 0.1)
        assert np.all(np.isfinite(acf))
