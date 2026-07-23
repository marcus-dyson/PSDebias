from pathlib import Path

import numpy as np
import pytest

from psdebias.analytic import ar_spectrum
from psdebias.diagnostic import regime_diagnostic, sign_diagnostic, window_bandwidth
from psdebias.estimators import multitaper, welch
from psdebias.simulate import sample_ar, sample_matern

from conftest import PAPER_AR_POLY

SUNSPOT_CSV = Path(__file__).parent.parent / "data" / "SN_y_tot_V2.0.csv"


class TestWindowBandwidth:
    def test_fejer_textbook_value(self):
        # boxcar taper -> Fejer window -> equivalent bandwidth 1.5 bins
        l = 1024
        taper = np.ones(l) / np.sqrt(l)
        kernel = np.correlate(taper, taper, mode="full")[l - 1 :]
        np.testing.assert_allclose(window_bandwidth(kernel, l), 1.5, rtol=1e-2)

    @pytest.mark.parametrize("nw", [3.5, 5.0])
    def test_multitaper_close_to_design_width(self, nw, rng):
        n = 2**12
        _, _, kernel, _ = multitaper(rng.standard_normal(n), nw=nw)
        b = window_bandwidth(kernel, n)
        assert abs(b - 2 * nw) / (2 * nw) < 0.2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="kernel length"):
            window_bandwidth(np.ones(10), 20)


class TestSignDiagnostic:
    """The paper's Sec. IV-C rule, kept verbatim (reference behavior)."""

    def test_bias_dominant_recommends_debias(self):
        rng = np.random.default_rng(1)
        l, m = 1024, 32
        x = sample_ar(1, m * l, PAPER_AR_POLY, rng=rng)[0]
        freqs, psd, kernel, _ = welch(x, segment_length=l, n_segments=m, step=l)
        d = sign_diagnostic(psd, freqs, kernel=kernel, n_time=l, knot_spacing=16)
        assert d.debias_recommended
        assert d.n_negative == 0

    def test_variance_dominant_recommends_smooth(self):
        rng = np.random.default_rng(1)
        x = sample_matern(1, 2**13, 1.0, 1.0, 0.1, rng=rng)[0]
        freqs, psd, kernel, _ = multitaper(x, nw=5.0)
        d = sign_diagnostic(psd, freqs, kernel=kernel, n_time=len(x), knot_spacing=16)
        assert not d.debias_recommended
        assert d.n_negative > 0

    def test_input_validation(self):
        freqs = np.fft.rfftfreq(64)[1:-1]
        psd = np.ones(len(freqs))
        kernel = np.zeros(64)
        kernel[0] = 1.0
        with pytest.raises(ValueError, match="length"):
            sign_diagnostic(psd[:-1], freqs, kernel=kernel, n_time=64)
        with pytest.raises(ValueError, match="strictly positive"):
            sign_diagnostic(psd - 1.0, freqs, kernel=kernel, n_time=64)


class TestRegimeDiagnostic:
    @pytest.mark.parametrize("m", [16, 32, 64])
    def test_ar4_regime1_recommends_debias(self, m):
        # The sign rule false-negatives on some of these (noise/approximation
        # negatives at fine meshes); the bandwidth-matched gap rule must not.
        rng = np.random.default_rng(1)
        l = 1024
        x = sample_ar(1, m * l, PAPER_AR_POLY, rng=rng)[0]
        freqs, psd, kernel, _ = welch(x, segment_length=l, n_segments=m, step=l)
        d = regime_diagnostic(psd, freqs, kernel=kernel, n_time=l)
        assert d.debias_recommended
        assert d.gap_per_constraint <= 4.0
        assert np.all(d.beta_nnls >= 0)
        assert d.condition_number > 0
        assert d.knot_spacing == int(np.ceil(d.bandwidth_bins))

    def test_deterministic_blur_mismatch_rejected(self):
        # An UNBLURRED "estimate" (the analytic spectrum itself) presented
        # with a heavy-blur kernel: the convolved bases are far more blurred
        # than the data, positivity fights it, the gap must reject. No RNG.
        l = 256
        freqs = np.fft.rfftfreq(l)[1:-1]
        truth = ar_spectrum(freqs, PAPER_AR_POLY, 1.0)
        taper = np.ones(l) / np.sqrt(l)
        kernel = np.correlate(taper, taper, mode="full")[l - 1 :]
        d = regime_diagnostic(truth, freqs, kernel=kernel, n_time=l)
        assert not d.debias_recommended
        assert d.gap_per_constraint > 4.0

    def test_noise_negatives_discounted(self):
        # Find a Regime-1 realization whose forced-fine-mesh fit shows raw
        # negative coefficients (the sign rule would cry wolf), then check the
        # bandwidth-corrected t-test discounts most of them and the
        # bandwidth-matched auto mesh still recommends debiasing.
        l, m = 1024, 32
        found = False
        for seed in range(20):
            rng = np.random.default_rng(seed)
            x = sample_ar(1, m * l, PAPER_AR_POLY, rng=rng)[0]
            freqs, psd, kernel, _ = welch(x, segment_length=l, n_segments=m, step=l)
            d_fine = regime_diagnostic(psd, freqs, kernel=kernel, n_time=l, knot_spacing=4)
            n_raw = int((d_fine.beta_ols <= 0).sum())
            if n_raw > 0:
                found = True
                assert not d_fine.sign_rule_recommended
                assert d_fine.n_significant_negative < n_raw
                d_auto = regime_diagnostic(psd, freqs, kernel=kernel, n_time=l)
                assert d_auto.debias_recommended
                break
        assert found, "no realization with raw negative coefficients in 20 seeds"

    def test_report_fields_coherent(self, rng):
        l, m = 512, 24
        x = sample_ar(1, m * l, PAPER_AR_POLY, rng=rng)[0]
        freqs, psd, kernel, _ = welch(x, segment_length=l, n_segments=m, step=l)
        d = regime_diagnostic(psd, freqs, kernel=kernel, n_time=l)
        assert d.t_threshold > 0
        assert np.isfinite(d.min_t)
        assert d.residual_inflation > 0
        assert len(d.beta_ols) == len(d.beta_nnls)
        assert d.bandwidth_bins > 0


@pytest.mark.skipif(not SUNSPOT_CSV.exists(), reason="sunspot dataset not present")
class TestSunspotRegression:
    """The motivating case: the paper debiases sunspots; the sign rule said
    smooth; the new rule must say debias."""

    @pytest.mark.parametrize("nw", [3.5, 5.0])
    def test_recommends_debias_where_sign_rule_failed(self, nw):
        x = np.genfromtxt(SUNSPOT_CSV, delimiter=";").T[1]
        freqs, psd, kernel, _ = multitaper(x, nw=nw)
        d = regime_diagnostic(psd, freqs, kernel=kernel, n_time=len(x))
        assert d.debias_recommended
        assert d.gap_per_constraint < 1.0  # decisively, per probe values
        assert not d.sign_rule_recommended  # the improvement this pins
