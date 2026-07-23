"""End-to-end integration: PSDebias recovers analytic spectra and beats the
raw estimator it starts from. Asserts against analytic truth only — the fixed
posterior (see CHANGES.md) intentionally differs from the original package's
sampling distribution, so original outputs are not comparison targets."""

from pathlib import Path

import numpy as np
import pytest

from psdebias.analytic import ar_spectrum, matern_spectrum
from psdebias.estimators import multitaper, welch
from psdebias.sampler import fit_psd
from psdebias.simulate import sample_ar, sample_matern

from conftest import PAPER_AR_POLY

SUNSPOT_CSV = Path(__file__).parent.parent / "data" / "SN_y_tot_V2.0.csv"


def log_mse(estimate, truth):
    return float(np.mean((np.log(estimate) - np.log(truth)) ** 2))


@pytest.mark.slow
class TestEndToEnd:
    def test_ar4_debias_beats_raw_welch(self):
        rng = np.random.default_rng(2024)
        l, m = 1024, 32
        x = sample_ar(1, m * l, PAPER_AR_POLY, rng=rng)[0]
        est = welch(x, segment_length=l, n_segments=m, step=l)
        freqs = est.freqs
        truth = ar_spectrum(freqs, PAPER_AR_POLY, 1.0)

        # dense candidate mesh (~128 knots) as in the paper's studies; the
        # sampler prunes it to order-20 active bases
        fit = fit_psd(
            est, n_time=l, mode="debias", knot_spacing=4,
            rng=np.random.default_rng(1),
            n_iterations=20_000, warmup=10_000, thin=10, n_beta_draws=20,
        )
        assert np.all(fit.mean > 0)
        assert 0.01 < fit.acceptance_rate < 0.6
        assert 10 < fit.expected_knots < 60
        assert log_mse(fit.mean, truth) < 0.6 * log_mse(est.psd, truth)
        # the AR(4) twin-peak region must survive debiasing
        peak_region = (freqs > 0.08) & (freqs < 0.16)
        assert fit.mean[peak_region].max() > 0.5 * truth[peak_region].max()

    def test_ar4_mode_auto_dispatches_debias(self):
        # On a coarser diagnostic mesh this scenario passes the sign check
        rng = np.random.default_rng(2024)
        l, m = 1024, 32
        x = sample_ar(1, m * l, PAPER_AR_POLY, rng=rng)[0]
        est = welch(x, segment_length=l, n_segments=m, step=l)
        fit = fit_psd(
            est, n_time=l, mode="auto", knot_spacing=16,
            rng=rng, n_iterations=5_000, warmup=2_000, thin=10, n_beta_draws=10,
        )
        assert fit.mode == "debias"
        assert np.all(fit.mean > 0)

    def test_matern_smooth_beats_raw_multitaper(self):
        rng = np.random.default_rng(2025)
        n = 2**13
        x = sample_matern(1, n, 1.0, 1.0, 0.1, rng=rng)[0]
        est = multitaper(x, nw=5.0)
        freqs = est.freqs
        truth = matern_spectrum(freqs, 1.0, 1.0, 0.1)

        fit = fit_psd(
            est, n_time=n, mode="auto", knot_spacing=16,
            rng=rng, n_iterations=20_000, warmup=10_000, thin=10, n_beta_draws=20,
        )
        # With the bandwidth-matched regime diagnostic this dispatches
        # "debias" (the multitaper blur genuinely matches the bases); either
        # regime must improve on the raw multitaper against analytic truth.
        assert np.all(fit.mean > 0)
        assert log_mse(fit.mean, truth) < log_mse(est.psd, truth)

    @pytest.mark.skipif(not SUNSPOT_CSV.exists(), reason="sunspot dataset not present")
    def test_sunspot_forced_debias_regression(self):
        # Regression for the frozen-chain bug: forced debiasing of the annual
        # sunspot multitaper estimate (small N, dense candidate mesh) must mix
        # and track the estimate. The broken all-ones initialization gave
        # acceptance 0.0 and log-MSE ~221 here.
        x = np.genfromtxt(SUNSPOT_CSV, delimiter=";").T[1]
        est = multitaper(x, nw=3.5)
        fit = fit_psd(
            est, n_time=len(x), mode="auto",
            knot_spacing=2, rng=np.random.default_rng(1),
            n_iterations=20_000, warmup=10_000, thin=10, n_beta_draws=20,
        )
        # the regime diagnostic must dispatch the paper's choice here
        assert fit.mode == "debias"
        assert fit.acceptance_rate > 0.05
        assert len(np.unique(fit.gammas, axis=0)) > 10
        assert np.all(np.isfinite(fit.mean))
        assert log_mse(np.maximum(fit.mean, 1e-12), est.psd) < 1.0

    def test_uncertainty_bands_calibrated_roughly(self):
        # The posterior std should not collapse to zero nor exceed the signal scale
        rng = np.random.default_rng(7)
        l, m = 512, 24
        x = sample_ar(1, m * l, PAPER_AR_POLY, rng=rng)[0]
        est = welch(x, segment_length=l, n_segments=m, step=l)
        fit = fit_psd(
            est, n_time=l, mode="smooth", knot_spacing=8,
            rng=rng, n_iterations=10_000, warmup=5_000, thin=10, n_beta_draws=20,
        )
        assert np.all(fit.std > 0)
        assert np.median(fit.std / fit.mean) < 1.0
