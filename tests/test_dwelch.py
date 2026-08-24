import numpy as np
import pytest

from psdebias.analytic import ar_spectrum
from psdebias.dwelch import dquad
from psdebias.estimators import welch
from psdebias.simulate import sample_ar

from conftest import PAPER_AR_POLY


@pytest.fixture(scope="module")
def welch_estimate():
    rng = np.random.default_rng(99)
    l, m, s = 256, 30, 128
    x = sample_ar(1, (m - 1) * s + l, PAPER_AR_POLY, rng=rng)[0]
    freqs, psd, kernel, *_ = welch(x, segment_length=l, n_segments=m, step=s)
    return freqs, psd, kernel, l


class TestDquad:
    def test_runs_and_positive(self, welch_estimate):
        freqs, psd, kernel, l = welch_estimate
        gamma = np.zeros(len(freqs), dtype=np.int8)
        gamma[::8] = 1
        debiased = dquad(psd, freqs, gamma=gamma, kernel=kernel, n_time=l)
        assert debiased.shape == freqs.shape
        assert np.all(debiased >= 0)

    def test_flat_spectrum_recovered(self):
        # White noise: E[welch] is flat; debiasing should return ~flat spectrum
        rng = np.random.default_rng(5)
        l, m, s = 128, 200, 64
        x = rng.standard_normal((m - 1) * s + l)
        freqs, psd, kernel, *_ = welch(x, segment_length=l, n_segments=m, step=s)
        gamma = np.zeros(len(freqs), dtype=np.int8)
        gamma[::10] = 1
        debiased = dquad(psd, freqs, gamma=gamma, kernel=kernel, n_time=l)
        # interior blocks tight; the first block borders the DC ghost knot and
        # carries more estimation noise, so it only gets a loose bound
        np.testing.assert_allclose(debiased[10:], 1.0, rtol=0.15)
        np.testing.assert_allclose(debiased[:10], 1.0, rtol=0.35)

    def test_reduces_tail_error_vs_raw_welch(self, welch_estimate):
        freqs, psd, kernel, l = welch_estimate
        gamma = np.zeros(len(freqs), dtype=np.int8)
        gamma[::8] = 1
        debiased = dquad(psd, freqs, gamma=gamma, kernel=kernel, n_time=l)
        truth = ar_spectrum(freqs, PAPER_AR_POLY, 1.0)
        # debiasing should reduce log-domain error vs the raw Welch estimate
        # in the strongly biased tail region
        tail = freqs > 0.3
        eps = 1e-12
        err_debiased = np.mean((np.log(debiased[tail] + eps) - np.log(truth[tail])) ** 2)
        err_welch = np.mean((np.log(psd[tail]) - np.log(truth[tail])) ** 2)
        assert err_debiased < err_welch

    def test_rejects_bad_inputs(self, welch_estimate):
        freqs, psd, kernel, l = welch_estimate
        gamma = np.zeros(len(freqs), dtype=np.int8)
        with pytest.raises(ValueError, match="positive"):
            dquad(psd - psd.max(), freqs, gamma=gamma, kernel=kernel, n_time=l)
        with pytest.raises(ValueError, match="kernel length"):
            dquad(psd, freqs, gamma=gamma, kernel=kernel[:-1], n_time=l)
