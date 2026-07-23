import numpy as np
import pytest

from psdebias.util import freq_slice, get_nfreq, psd_from_acf, segment

from conftest import original_get_exact_rfft_psd


class TestGetNfreq:
    def test_even(self):
        assert get_nfreq(8) == 3  # bins 1,2,3 (0 and Nyquist=4 dropped)

    def test_odd(self):
        assert get_nfreq(9) == 4

    def test_matches_dropped_endpoint_grid_even(self):
        # For even n the endpoint-dropped rfft grid has full - 2 bins.
        # (For odd n the last rfft bin is not Nyquist, so get_nfreq counts it.)
        for n in [4, 16, 256, 1024]:
            full = n // 2 + 1
            assert get_nfreq(n) == full - 2

    def test_odd_counts_all_positive_bins(self):
        for n in [5, 17, 1023]:
            assert get_nfreq(n) == (n - 1) // 2


class TestSegment:
    def test_shape_and_content(self):
        x = np.arange(20.0)
        s = segment(x, segment_length=8, n_segments=4, step=4)
        assert s.shape == (4, 8)
        np.testing.assert_array_equal(s[0], x[0:8])
        np.testing.assert_array_equal(s[3], x[12:20])

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            segment(np.arange(10.0), segment_length=8, n_segments=4, step=4)


class TestFreqSlice:
    def test_full_grid(self):
        assert freq_slice(16, 9) == slice(None)

    def test_dropped_endpoints(self):
        assert freq_slice(16, 7) == slice(1, -1)

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            freq_slice(16, 5)


class TestPsdFromAcf:
    @pytest.mark.parametrize("n", [8, 9, 64, 129])
    def test_matches_original_loop_exactly(self, rng, n):
        acf = rng.standard_normal(n)
        kernel = rng.standard_normal(n)
        expected = original_get_exact_rfft_psd(acf, kernel)
        np.testing.assert_array_equal(psd_from_acf(acf, kernel), expected)

    def test_no_kernel_is_identity_weighting(self, rng):
        acf = rng.standard_normal(32)
        np.testing.assert_array_equal(
            psd_from_acf(acf), psd_from_acf(acf, np.ones(32))
        )

    def test_batched_equals_per_row(self, rng):
        acfs = rng.standard_normal((5, 64))
        kernel = rng.standard_normal(64)
        batched = psd_from_acf(acfs, kernel)
        for i in range(5):
            np.testing.assert_allclose(
                batched[i], psd_from_acf(acfs[i], kernel), rtol=1e-14
            )

    def test_does_not_mutate_input(self, rng):
        acf = rng.standard_normal(16)
        acf_copy = acf.copy()
        psd_from_acf(acf, np.ones(16))
        np.testing.assert_array_equal(acf, acf_copy)
