import numpy as np
import pytest
import scipy.signal

from psdebias.analytic import ar_spectrum
from psdebias.dwelch import dquad
from psdebias.estimators import lag_window, multitaper, welch
from psdebias import splines
from psdebias.simulate import sample_ar

from conftest import PAPER_AR_POLY, reference_gls_dense

# The DQuad Eq. 12 GLS path was reverted: `dquad` assumes independence in
# frequency again, so `_full_grid_index` and the `corr=` argument are gone. The
# tests below are kept verbatim rather than deleted -- they are the spec the
# path has to satisfy when the correlation work lands. They skip as a group on
# the absence of the private helper, so they wake up on their own.
try:
    from psdebias.dwelch import _full_grid_index
except ImportError:  # pragma: no cover - depends on which revision is checked out
    _full_grid_index = None

requires_gls = pytest.mark.skipif(
    _full_grid_index is None,
    reason="dquad's GLS path is reverted; see docs/CHANGES.md §8",
)


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


def log_mse(estimate, truth):
    return np.mean((np.log(np.maximum(estimate, 1e-12)) - np.log(truth)) ** 2)


@requires_gls
class TestDquadGLS:
    """dquad with the correlation matrix: DQuad Eq. 12, done as the paper does.

    Closed-form generalised least squares on the FULL two-sided Fourier grid,
    with W circulant and W^-1 applied by FFT. Unconstrained -- Eq. 12's solution
    is analytic, so there is no positivity constraint on this path.
    """

    @staticmethod
    def full_grid_welch(seed, taper=None):
        """A Welch estimate keeping DC and Nyquist, which the GLS path needs."""
        l, m, s = 256, 30, 128
        x = sample_ar(1, (m - 1) * s + l, PAPER_AR_POLY, rng=np.random.default_rng(seed))[0]
        return welch(x, segment_length=l, n_segments=m, step=s, taper=taper,
                     drop_endpoints=False), l

    def test_matches_dense_gls_oracle(self):
        """The FFT inverse == a dense inverse of the same circulant W."""
        est, l = self.full_grid_welch(21, scipy.signal.windows.hann(256, sym=False))
        gamma = np.zeros(len(est.freqs) - 2, dtype=np.int8)
        gamma[::8] = 1

        # Rebuild the paper's system independently and solve it densely.
        idx = _full_grid_index(l)
        design = splines.basis_b0_biased(
            est.freqs, est.freqs, gamma, l, est.kernel
        )[idx] / est.psd[idx][:, None]
        theta = reference_gls_dense(design, np.ones(l), est.corr[idx])
        expected = splines.basis_b0(est.freqs, est.freqs, gamma) @ theta

        got = dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel,
                    n_time=l, corr=est.corr)
        # Interior bins only: this pins the SOLVER (FFT inverse == dense
        # inverse). The ghost-knot endpoint correction is a property of the
        # output design, covered by the flat-spectrum test below, and restating
        # it here would just duplicate the implementation.
        np.testing.assert_allclose(got[1:-1], expected[1:-1], rtol=1e-8)

    def test_gls_runs_on_tapered_welch(self):
        est, l = self.full_grid_welch(11, scipy.signal.windows.hann(256, sym=False))
        assert est.corr[1] > 0.1  # guard: this configuration is genuinely correlated
        gamma = np.zeros(len(est.freqs) - 2, dtype=np.int8)
        gamma[::8] = 1
        debiased = dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel,
                         n_time=l, corr=est.corr)
        assert debiased.shape == est.freqs.shape
        assert np.all(np.isfinite(debiased))

    def test_gls_reduces_tail_error_vs_raw_welch(self):
        est, l = self.full_grid_welch(12, scipy.signal.windows.hann(256, sym=False))
        gamma = np.zeros(len(est.freqs) - 2, dtype=np.int8)
        gamma[::8] = 1
        debiased = dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel,
                         n_time=l, corr=est.corr)
        truth = ar_spectrum(est.freqs, PAPER_AR_POLY, 1.0)
        tail = est.freqs > 0.3
        assert log_mse(debiased[tail], truth[tail]) < log_mse(est.psd[tail], truth[tail])

    def test_flat_spectrum_recovered_including_endpoints(self):
        """The ghost knots now sit ON the grid, and that has a sharp edge.

        b0_box returns 1/2 exactly on a box boundary, which is correct where two
        boxes share an interior knot and each supplies half. At freqs[0] = 0 and
        freqs[-1] = 1/2 there is no neighbouring box to supply the other half,
        so the output design's rows sum to 1/2 and the debiased spectrum comes
        out at exactly half value in those two bins. The fitting design is
        unaffected -- basis_b0_biased goes through acf_box and psd_from_acf,
        which evaluate no boundaries -- so this is the B0 counterpart of the
        halve_idx split the sampler keeps between its two designs.
        """
        rng = np.random.default_rng(5)
        l, m, s = 128, 200, 64
        x = rng.standard_normal((m - 1) * s + l)
        est = welch(x, segment_length=l, n_segments=m, step=s, drop_endpoints=False)
        gamma = np.zeros(len(est.freqs) - 2, dtype=np.int8)
        gamma[::10] = 1
        debiased = dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel,
                         n_time=l, corr=est.corr)
        np.testing.assert_allclose(debiased[[0, -1]], 1.0, rtol=0.35)

    def test_requires_full_grid(self, welch_estimate):
        # The GLS path mirrors onto the two-sided grid, so it needs the DC and
        # Nyquist bins the other paths drop -- they cannot be reconstructed.
        freqs, psd, kernel, l = welch_estimate
        gamma = np.zeros(len(freqs), dtype=np.int8)
        corr = np.zeros(len(freqs))
        corr[0] = 1.0
        with pytest.raises(ValueError, match="drop_endpoints"):
            dquad(psd, freqs, gamma=gamma, kernel=kernel, n_time=l, corr=corr)

    def test_singular_correlation_matrix_raises(self):
        """Eq. 12 is undefined when W is singular, and it often is.

        The paper assumes W positive definite. For a lag-window it is not: the
        circulant's smallest eigenvalue is ~ -2e-15, so W^-1 divides by zero and
        the closed form yields NaN. Raising beats returning NaN silently -- this
        is a diagnostic, not a statistical intervention, and the message has to
        say which estimator setting caused it.
        """
        n, lag = 2048, 256
        x = sample_ar(1, n, PAPER_AR_POLY, rng=np.random.default_rng(7))[0]
        est = lag_window(x, window="bartlett", lag=lag, drop_endpoints=False)
        gamma = np.zeros(len(est.freqs) - 2, dtype=np.int8)
        gamma[::16] = 1
        with pytest.raises(ValueError, match="singular"):
            dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel,
                  n_time=n, corr=est.corr)

    def test_unconstrained_solution_may_go_negative(self):
        """Eq. 12 has no positivity constraint, and it shows.

        Multitaper W is well conditioned (cond ~1e3), so the solve succeeds --
        but the unconstrained coefficients put a large part of the band below
        zero. That is the paper's estimator behaving as specified, not a bug;
        pinned here so the behaviour is not mistaken for one later.
        """
        n = 2048
        x = sample_ar(1, n, PAPER_AR_POLY, rng=np.random.default_rng(300))[0]
        est = multitaper(x, nw=4.0, n_tapers=8, drop_endpoints=False)
        gamma = np.zeros(len(est.freqs) - 2, dtype=np.int8)
        gamma[::16] = 1
        debiased = dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel,
                         n_time=n, corr=est.corr)
        assert np.all(np.isfinite(debiased))
        assert np.any(debiased < 0)

    def test_rejects_bad_corr(self):
        est, l = self.full_grid_welch(13)
        gamma = np.zeros(len(est.freqs) - 2, dtype=np.int8)
        with pytest.raises(ValueError, match="corr"):
            dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel, n_time=l,
                  corr=est.corr[:-1])
        with pytest.raises(ValueError, match="corr"):
            dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel, n_time=l,
                  corr=np.diag(est.corr))
