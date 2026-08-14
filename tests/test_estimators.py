import numpy as np
import pytest
import scipy.signal

from psdebias.estimators import lag_window, multitaper, periodogram, welch


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
