import numpy as np
import pytest
import scipy.linalg
import scipy.signal

from psdebias.estimators import lag_window, multitaper, periodogram, welch

from conftest import (
    exact_bin_covariance,
    quadratic_form_lag_window,
    quadratic_form_multitaper,
    quadratic_form_periodogram,
    quadratic_form_welch,
    reference_bandwidth_multitaper,
    reference_corr_multitaper,
    reference_corr_welch,
)


@pytest.fixture
def white_noise(rng):
    return rng.standard_normal(2**12)


def total_power(freqs, psd, fs=1.0):
    """Approximate variance from a one-sided PSD (doubling interior bins)."""
    df = fs / (2 * (len(freqs) + 1)) if freqs[0] > 0 else fs / (2 * (len(freqs) - 1))
    return 2 * np.sum(psd) * df


class TestPeriodogram:
    def test_parseval_white_noise(self, white_noise):
        f, psd, _, *_ = periodogram(white_noise)
        np.testing.assert_allclose(total_power(f, psd), np.var(white_noise), rtol=0.05)

    def test_fs_scaling(self, white_noise):
        _, psd1, _, *_ = periodogram(white_noise, fs=1.0)
        f10, psd10, _, *_ = periodogram(white_noise, fs=10.0)
        np.testing.assert_allclose(psd10, psd1 / 10.0)
        assert f10[-1] < 5.0  # Nyquist = fs/2

    def test_kernel_unit_lag0(self, white_noise, rng):
        _, _, kernel, *_ = periodogram(white_noise, taper=scipy.signal.windows.hann(len(white_noise)))
        np.testing.assert_allclose(kernel[0], 1.0, rtol=1e-12)
        assert len(kernel) == len(white_noise)

    def test_drop_endpoints(self, white_noise):
        f_all, psd_all, _, *_ = periodogram(white_noise, drop_endpoints=False)
        f_int, psd_int, _, *_ = periodogram(white_noise, drop_endpoints=True)
        assert f_all[0] == 0.0
        np.testing.assert_array_equal(f_int, f_all[1:-1])
        np.testing.assert_array_equal(psd_int, psd_all[1:-1])


class TestWelch:
    def test_vs_scipy(self, white_noise):
        l, m, s = 256, 30, 128
        taper = scipy.signal.windows.hann(l)
        f, psd, _, *_ = welch(
            white_noise, segment_length=l, n_segments=m, step=s, taper=taper,
            drop_endpoints=False,
        )
        f_sp, psd_sp = scipy.signal.welch(
            white_noise[: (m - 1) * s + l],
            fs=1.0,
            window=taper,
            nperseg=l,
            noverlap=l - s,
            detrend=False,
            scaling="density",
            return_onesided=True,
        )
        np.testing.assert_allclose(f, f_sp)
        # scipy doubles interior bins for the one-sided density; ours is the
        # raw mean periodogram (two-sided convention on the one-sided grid)
        np.testing.assert_allclose(2 * psd[1:-1], psd_sp[1:-1], rtol=1e-10)

    def test_parseval_white_noise(self, white_noise):
        f, psd, _, *_ = welch(white_noise, segment_length=256, n_segments=30, step=128)
        np.testing.assert_allclose(total_power(f, psd), np.var(white_noise), rtol=0.1)

    def test_too_short_raises(self, rng):
        with pytest.raises(ValueError, match="too short"):
            welch(rng.standard_normal(100), segment_length=64, n_segments=10, step=32)

    def test_invalid_params_raise(self, white_noise):
        with pytest.raises(ValueError):
            welch(white_noise, segment_length=64, n_segments=4, step=32, fs=-1.0)
        with pytest.raises(ValueError):
            welch(white_noise, segment_length=0, n_segments=4, step=32)


class TestLagWindow:
    def test_parseval_white_noise(self, white_noise):
        f, psd, _, *_ = lag_window(white_noise, window="bartlett", lag=256)
        np.testing.assert_allclose(total_power(f, psd), np.var(white_noise), rtol=0.1)

    def test_kernel_shape_and_support(self, white_noise):
        n = len(white_noise)
        _, _, kernel, *_ = lag_window(white_noise, window="bartlett", lag=100)
        assert len(kernel) == n
        assert np.all(kernel[101:] == 0)
        assert kernel[0] == 1.0

    def test_invalid_lag_raises(self, white_noise):
        with pytest.raises(ValueError, match="lag"):
            lag_window(white_noise, lag=len(white_noise) + 1)


class TestMultitaper:
    def test_parseval_white_noise(self, white_noise):
        f, psd, _, *_ = multitaper(white_noise, nw=4.0)
        np.testing.assert_allclose(total_power(f, psd), np.var(white_noise), rtol=0.1)

    def test_kernel_unit_lag0(self, white_noise):
        _, _, kernel, *_ = multitaper(white_noise, nw=4.0)
        np.testing.assert_allclose(kernel[0], 1.0, rtol=1e-10)

    def test_default_taper_count(self, white_noise):
        # K = 2*NW default should not raise and gives smooth positive psd
        _, psd, _, *_ = multitaper(white_noise, nw=3.5)
        assert np.all(psd > 0)


class TestCorrelation:
    """The frequency-correlation sequence rho (DQuad Eq. 8, note Eq. 10).

    rho[d] is the correlation between two spectral estimates d frequency bins
    apart. It depends only on the separation, so the full correlation matrix of
    the paper's Eq. 12 is toeplitz(rho).
    """

    @staticmethod
    def area(corr, n_time):
        """Integral of rho over the band, from the FULL one-sided sequence.

        rho's inverse transform has support |tau| <= n-1, so sampling on the
        n-point grid and summing is exact, not a quadrature approximation.
        """
        return (corr[0] + 2 * corr[1:-1].sum() + corr[-1]) / n_time

    def test_rectangular_periodogram_is_identity(self, rng):
        # h_t^2 = 1/n makes H_2 the Dirichlet kernel, which vanishes at every
        # nonzero Fourier-grid separation -- so the bins are exactly uncorrelated
        # and the GLS of Eq. 12 collapses to the existing diagonal weighting.
        est = periodogram(rng.standard_normal(512))
        expected = np.zeros(len(est.freqs))
        expected[0] = 1.0
        np.testing.assert_allclose(est.corr, expected, atol=1e-12)

    def test_unit_diagonal(self, rng):
        x = rng.standard_normal(512)
        hann = scipy.signal.windows.hann(512, sym=False)
        for est in (
            periodogram(x, taper=hann),
            welch(x, segment_length=128, n_segments=7, step=64),
            lag_window(x, window="bartlett", lag=32),
            multitaper(x, nw=3.5),
        ):
            np.testing.assert_allclose(est.corr[0], 1.0, rtol=1e-12)

    def test_toeplitz_is_psd(self, rng):
        # rho is a positive-definite function of the separation (Bochner: it is
        # the transform of a nonnegative weighted sum of |Gamma|^2), so sampling
        # it on a uniform lattice must give a PSD matrix.
        x = rng.standard_normal(512)
        hann = scipy.signal.windows.hann(512, sym=False)
        for est in (
            periodogram(x, taper=hann),
            welch(x, segment_length=128, n_segments=7, step=64),
            lag_window(x, window="bartlett", lag=32),
            multitaper(x, nw=3.5),
        ):
            w = scipy.linalg.toeplitz(est.corr)
            np.testing.assert_allclose(w, w.T, rtol=1e-12)
            assert np.linalg.eigvalsh(w).min() >= -1e-10

    def test_length_matches_freqs(self, rng):
        x = rng.standard_normal(512)
        for drop in (True, False):
            for est in (
                periodogram(x, drop_endpoints=drop),
                welch(x, segment_length=128, n_segments=7, step=64, drop_endpoints=drop),
                lag_window(x, window="bartlett", lag=32, drop_endpoints=drop),
                multitaper(x, nw=3.5, drop_endpoints=drop),
            ):
                assert est.corr.shape == est.freqs.shape

    def test_multitaper_matches_direct_oracle(self, rng):
        n, nw, k_tapers = 64, 3.0, 5
        est = multitaper(rng.standard_normal(n), nw=nw, n_tapers=k_tapers, drop_endpoints=False)
        tapers = scipy.signal.windows.dpss(n, nw, Kmax=k_tapers)
        seps = np.arange(8)
        expected = reference_corr_multitaper(tapers, seps / n)
        np.testing.assert_allclose(est.corr[seps], expected, rtol=1e-10, atol=1e-12)

    def test_welch_matches_block_oracle(self, rng):
        seg_len, n_segments, step = 16, 4, 8
        n = (n_segments - 1) * step + seg_len
        est = welch(
            rng.standard_normal(n),
            segment_length=seg_len, n_segments=n_segments, step=step,
            drop_endpoints=False,
        )
        taper = np.ones(seg_len) / np.sqrt(seg_len)
        seps = np.arange(seg_len // 2 + 1)
        expected = reference_corr_welch(taper, n_segments, step, seps / seg_len)
        np.testing.assert_allclose(est.corr, expected, rtol=1e-10, atol=1e-12)

    def test_multitaper_bandwidth_identity(self, rng):
        # Correlation area == resolution bandwidth (DQuad Eq. 9 / note Prop. 2).
        # The closed form never touches a Fourier transform, so this pins the
        # normalisation and the overall scale of rho independently.
        n, nw, k_tapers = 256, 4.0, 8
        est = multitaper(rng.standard_normal(n), nw=nw, n_tapers=k_tapers, drop_endpoints=False)
        tapers = scipy.signal.windows.dpss(n, nw, Kmax=k_tapers)
        np.testing.assert_allclose(
            self.area(est.corr, n), reference_bandwidth_multitaper(tapers), rtol=1e-10
        )

    @pytest.mark.parametrize("window,bandwidth", [("bartlett", 1.5), ("parzen", 1.854)])
    def test_lag_window_bandwidth_closed_form(self, rng, window, bandwidth):
        # Note Table 1: B_W = 1.5/m for Bartlett, 1.854/m for Parzen. Published
        # constants, derived in the continuum; lag << n keeps the biased-ACF
        # (1 - tau/n) factor folded into `kernel` from moving them much.
        n, lag = 1024, 32
        est = lag_window(rng.standard_normal(n), window=window, lag=lag, drop_endpoints=False)
        np.testing.assert_allclose(self.area(est.corr, n), bandwidth / lag, rtol=2e-2)

    # ------------------------------------------------------------------
    # Exact quadratic-form checks (conftest: Isserlis, finite-sample)
    # ------------------------------------------------------------------
    #
    # The oracles above are independent implementations of DQuad Eq. 8, so they
    # pin the algebra but assume the formula. These pin the formula itself:
    # Cov(I_i, I_j) = 2 tr(A_i A_j) exactly, and the estimators claim that
    # covariance is c (U(|i - j|) + U(i + j)) where U is rho's even periodic
    # extension. The U(i + j) term is the one the Toeplitz storage drops.

    @staticmethod
    def assert_exact_quadratic(corr, n_grid, forms, bins):
        def u(m):
            m %= n_grid
            return corr[min(m, n_grid - m)]

        tr = np.array([[np.sum(a * b) for b in forms] for a in forms])
        pred = np.array([[u(abs(i - j)) + u(i + j) for j in bins] for i in bins])
        # rho is normalised to rho[0] = 1, so a single overall scale is free.
        scale = np.sum(tr * pred) / np.sum(pred * pred)
        np.testing.assert_allclose(tr, scale * pred, rtol=1e-10, atol=1e-12 * tr.max())

    BINS = [0, 1, 2, 3, 5, 8, 13, 21, 31, 32]  # includes DC and Nyquist

    @pytest.mark.parametrize("tapered", [False, True])
    def test_periodogram_matches_exact_quadratic_form(self, rng, tapered):
        n = 64
        taper = scipy.signal.windows.hann(n, sym=False) if tapered else None
        est = periodogram(rng.standard_normal(n), taper=taper, drop_endpoints=False)
        h = np.ones(n) if taper is None else taper.copy()
        h /= np.linalg.norm(h)
        forms = [quadratic_form_periodogram(h, k) for k in self.BINS]
        self.assert_exact_quadratic(est.corr, n, forms, self.BINS)

    @pytest.mark.parametrize("tapered", [False, True])
    def test_welch_matches_exact_quadratic_form(self, rng, tapered):
        seg_len, n_segments, step = 16, 4, 8
        n = (n_segments - 1) * step + seg_len
        taper = scipy.signal.windows.hann(seg_len, sym=False) if tapered else None
        est = welch(
            rng.standard_normal(n),
            segment_length=seg_len, n_segments=n_segments, step=step,
            taper=taper, drop_endpoints=False,
        )
        h = np.ones(seg_len) if taper is None else taper.copy()
        h /= np.linalg.norm(h)
        bins = list(range(seg_len // 2 + 1))
        forms = [quadratic_form_welch(h, k, n_segments, step) for k in bins]
        # rho lives on the SEGMENT grid: separations are multiples of 1/seg_len.
        self.assert_exact_quadratic(est.corr, seg_len, forms, bins)

    def test_multitaper_matches_exact_quadratic_form(self, rng):
        n, nw, k_tapers = 64, 3.0, 5
        est = multitaper(rng.standard_normal(n), nw=nw, n_tapers=k_tapers, drop_endpoints=False)
        tapers = scipy.signal.windows.dpss(n, nw, Kmax=k_tapers)
        forms = [quadratic_form_multitaper(tapers, k) for k in self.BINS]
        self.assert_exact_quadratic(est.corr, n, forms, self.BINS)

    @pytest.mark.parametrize("window", ["bartlett", "parzen"])
    @pytest.mark.parametrize("lag", [4, 8, 16])
    def test_lag_window_matches_exact_quadratic_form(self, rng, window, lag):
        # The factor this catches is (1 - tau/n), so it only shows up once
        # lag/n is not tiny -- hence lags up to n/4 here.
        n = 64
        est = lag_window(rng.standard_normal(n), window=window, lag=lag, drop_endpoints=False)
        win = scipy.signal.get_window(window, 2 * lag + 1, fftbins=False)[lag:]
        forms = [quadratic_form_lag_window(win, k, n) for k in self.BINS]
        self.assert_exact_quadratic(est.corr, n, forms, self.BINS)

    def test_sum_frequency_term_is_the_toeplitz_error(self, rng):
        # Storing rho as a sequence asserts the correlation depends only on the
        # separation. It does not: the exact covariance carries a U(i + j) term
        # too. It is negligible mid-band (where the spectral window's mainlobe
        # is nowhere near f_i + f_j) and material only when both bins crowd DC
        # or Nyquist -- which is the whole justification for toeplitz(rho).
        n = 64
        taper = scipy.signal.windows.hann(n, sym=False)
        est = periodogram(rng.standard_normal(n), taper=taper, drop_endpoints=False)
        h = taper / np.linalg.norm(taper)

        def exact_corr(i, j):
            a, b = quadratic_form_periodogram(h, i), quadratic_form_periodogram(h, j)
            return exact_bin_covariance(a, b) / np.sqrt(
                exact_bin_covariance(a, a) * exact_bin_covariance(b, b)
            )

        for i, j in [(10, 10), (10, 11), (10, 12), (20, 22)]:
            np.testing.assert_allclose(exact_corr(i, j), est.corr[j - i], atol=1e-12)
        # Adjacent to DC the dropped term is not small: bins 0 and 1 are far
        # more correlated than toeplitz(rho) says.
        assert exact_corr(0, 1) - est.corr[1] > 0.05

    def test_tapered_periodogram_matches_single_taper_oracle(self, rng):
        # A one-taper multitaper IS the tapered periodogram, so the K = 1 case
        # of the direct Eq. 8 double loop covers the gap left by the
        # rectangular-only identity test above.
        n = 64
        taper = scipy.signal.windows.hann(n, sym=False)
        est = periodogram(rng.standard_normal(n), taper=taper, drop_endpoints=False)
        h = (taper / np.linalg.norm(taper))[None, :]
        seps = np.arange(10)
        expected = reference_corr_multitaper(h, seps / n)
        np.testing.assert_allclose(est.corr[seps], expected, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("step", [8, 16, 24])  # overlapping, abutting, gapped
    def test_welch_tapered_and_disjoint_match_block_oracle(self, rng, step):
        # step >= segment_length takes the `overlap <= 0` break in _welch_corr,
        # which the rectangular overlapping case never exercises.
        seg_len, n_segments = 16, 4
        n = (n_segments - 1) * step + seg_len
        taper = scipy.signal.windows.hann(seg_len, sym=False)
        est = welch(
            rng.standard_normal(n),
            segment_length=seg_len, n_segments=n_segments, step=step,
            taper=taper, drop_endpoints=False,
        )
        h = taper / np.linalg.norm(taper)
        seps = np.arange(seg_len // 2 + 1)
        expected = reference_corr_welch(h, n_segments, step, seps / seg_len)
        np.testing.assert_allclose(est.corr, expected, rtol=1e-10, atol=1e-12)
