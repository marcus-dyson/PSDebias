import numpy as np
import pytest
import scipy.optimize

from psdebias.analytic import ar_spectrum
from psdebias.dwelch import dwelch_b0, dwelch_b1
from psdebias.estimators import welch
from psdebias.simulate import sample_ar
from psdebias.splines import acf_falling, acf_rising
from psdebias.util import psd_from_acf

from conftest import PAPER_AR_POLY


def original_dwelch_b1(estimate, freqs, gamma, kernel, n_time):
    """Oracle: the original auto-speccy dwelch_b1 construction — per-ACTIVE-segment
    half splines, adjacent-pair addition, clip-to-1 double-count patch."""
    knots = np.concatenate(([0.0], freqs, [0.5]))
    mask = np.ones(len(knots), dtype=bool)
    mask[1:-1] = gamma.astype(bool)
    active = knots[mask]
    lowers, uppers = active[:-1], active[1:]
    k = len(active) - 1
    weights = estimate**-1

    full = n_time // 2 + 1
    sl = slice(1, -1) if len(freqs) == full - 2 else slice(None)

    fall_b = np.array([psd_from_acf(acf_falling(n_time, lowers[i], uppers[i]), kernel)[sl] for i in range(k)])
    rise_b = np.array([psd_from_acf(acf_rising(n_time, lowers[i], uppers[i]), kernel)[sl] for i in range(k)])
    x_biased = np.vstack((fall_b[0], fall_b[1:] + rise_b[:-1], rise_b[-1])).T * weights[:, None]

    def fall(lo, hi):
        return np.where((freqs >= lo) & (freqs <= hi), (hi - freqs) / (hi - lo), 0.0)

    def rise(lo, hi):
        return np.where((freqs >= lo) & (freqs <= hi), (freqs - lo) / (hi - lo), 0.0)

    fall_u = np.array([fall(lowers[i], uppers[i]) for i in range(k)])
    rise_u = np.array([rise(lowers[i], uppers[i]) for i in range(k)])
    x_out = np.vstack((fall_u[0], fall_u[1:] + rise_u[:-1], rise_u[-1])).T
    x_out = np.clip(x_out, 0.0, 1.0)

    beta, _ = scipy.optimize.nnls(x_biased, np.ones(len(estimate)))
    return x_out @ beta


@pytest.fixture(scope="module")
def welch_estimate():
    rng = np.random.default_rng(99)
    l, m, s = 256, 30, 128
    x = sample_ar(1, (m - 1) * s + l, PAPER_AR_POLY, rng=rng)[0]
    freqs, psd, kernel, _ = welch(x, segment_length=l, n_segments=m, step=s)
    return freqs, psd, kernel, l


class TestDwelchB1:
    def test_matches_original_construction_uniform(self, welch_estimate):
        freqs, psd, kernel, l = welch_estimate
        gamma = np.zeros(len(freqs), dtype=np.int8)
        gamma[::8] = 1
        _, got = dwelch_b1(psd, freqs, gamma=gamma, kernel=kernel, n_time=l)
        expected = original_dwelch_b1(psd, freqs, gamma, kernel, l)
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)

    def test_matches_original_construction_nonuniform(self, welch_estimate, rng):
        freqs, psd, kernel, l = welch_estimate
        gamma = (rng.uniform(size=len(freqs)) < 0.05).astype(np.int8)
        gamma[10] = 1  # ensure at least one interior knot
        _, got = dwelch_b1(psd, freqs, gamma=gamma, kernel=kernel, n_time=l)
        expected = original_dwelch_b1(psd, freqs, gamma, kernel, l)
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)

    def test_output_positive_and_tracks_truth(self, welch_estimate):
        freqs, psd, kernel, l = welch_estimate
        gamma = np.zeros(len(freqs), dtype=np.int8)
        gamma[::8] = 1
        _, debiased = dwelch_b1(psd, freqs, gamma=gamma, kernel=kernel, n_time=l)
        assert np.all(debiased >= 0)
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
            dwelch_b1(psd - psd.max(), freqs, gamma=gamma, kernel=kernel, n_time=l)
        with pytest.raises(ValueError, match="kernel length"):
            dwelch_b1(psd, freqs, gamma=gamma, kernel=kernel[:-1], n_time=l)


class TestDwelchB0:
    def test_runs_and_positive(self, welch_estimate):
        freqs, psd, kernel, l = welch_estimate
        gamma = np.zeros(len(freqs), dtype=np.int8)
        gamma[::8] = 1
        _, debiased = dwelch_b0(psd, freqs, gamma=gamma, kernel=kernel, n_time=l)
        assert debiased.shape == freqs.shape
        assert np.all(debiased >= 0)

    def test_flat_spectrum_recovered(self):
        # White noise: E[welch] is flat; debiasing should return ~flat spectrum
        rng = np.random.default_rng(5)
        l, m, s = 128, 200, 64
        x = rng.standard_normal((m - 1) * s + l)
        freqs, psd, kernel, _ = welch(x, segment_length=l, n_segments=m, step=s)
        gamma = np.zeros(len(freqs), dtype=np.int8)
        gamma[::10] = 1
        _, debiased = dwelch_b0(psd, freqs, gamma=gamma, kernel=kernel, n_time=l)
        # interior blocks tight; the first block borders the DC ghost knot and
        # carries more estimation noise, so it only gets a loose bound
        np.testing.assert_allclose(debiased[10:], 1.0, rtol=0.15)
        np.testing.assert_allclose(debiased[:10], 1.0, rtol=0.35)
