import numpy as np
import pytest
import scipy.integrate

from psdebias.splines import (
    EMPTY_IDX,
    acf_box,
    acf_falling,
    acf_rising,
    assemble_design,
    b0_box,
    b1_falling,
    b1_rising,
    half_bases,
    half_bases_biased,
    knot_grid,
)


def brute_force_acf(f, lo, hi, n):
    """R(tau) = 2 * integral_lo^hi f(x) cos(2 pi x tau) dx by adaptive quadrature."""
    out = np.empty(n)
    for tau in range(n):
        val, _ = scipy.integrate.quad(
            lambda x: 2 * f(x) * np.cos(2 * np.pi * x * tau), lo, hi, limit=200
        )
        out[tau] = val
    return out


class TestPrimitiveEvaluation:
    def test_b1_ramps_boundary_inclusive(self):
        x = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        f = b1_falling(x, 0.1, 0.3)
        r = b1_rising(x, 0.1, 0.3)
        np.testing.assert_allclose(f, [0.0, 1.0, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(r, [0.0, 0.0, 0.5, 1.0, 0.0])

    def test_b0_box_half_at_boundary(self):
        x = np.array([0.05, 0.1, 0.2, 0.3, 0.35])
        np.testing.assert_allclose(b0_box(x, 0.1, 0.3), [0.0, 0.5, 1.0, 0.5, 0.0])


class TestClosedFormAcfs:
    @pytest.mark.parametrize("lo,hi", [(0.05, 0.12), (0.0, 0.25), (0.3, 0.5), (0.111, 0.113)])
    def test_rising_vs_quadrature(self, lo, hi):
        expected = brute_force_acf(lambda x: (x - lo) / (hi - lo), lo, hi, 21)
        np.testing.assert_allclose(acf_rising(21, lo, hi), expected, rtol=1e-8, atol=1e-12)

    @pytest.mark.parametrize("lo,hi", [(0.05, 0.12), (0.0, 0.25), (0.3, 0.5)])
    def test_falling_vs_quadrature(self, lo, hi):
        expected = brute_force_acf(lambda x: (hi - x) / (hi - lo), lo, hi, 21)
        np.testing.assert_allclose(acf_falling(21, lo, hi), expected, rtol=1e-8, atol=1e-12)

    @pytest.mark.parametrize("lo,hi", [(0.05, 0.12), (0.0, 0.25), (0.2, 0.5)])
    def test_box_vs_quadrature(self, lo, hi):
        expected = brute_force_acf(lambda x: 1.0, lo, hi, 21)
        np.testing.assert_allclose(acf_box(21, lo, hi), expected, rtol=1e-8, atol=1e-12)

    def test_rising_plus_falling_is_box(self):
        # The two ramps on a segment sum to the indicator, so must their ACFs
        np.testing.assert_allclose(
            acf_rising(50, 0.1, 0.3) + acf_falling(50, 0.1, 0.3),
            acf_box(50, 0.1, 0.3),
            rtol=1e-10, atol=1e-14,
        )


def make_grid(n_time=256, n_candidates=9):
    """Endpoint-dropped rfft grid plus a candidate knot set including ghosts."""
    freqs = np.fft.rfftfreq(n_time)[1:-1]
    knot_pos = np.linspace(0, len(freqs) - 1, n_candidates, dtype=int)
    knots = np.concatenate(([0.0], freqs[knot_pos], [0.5]))
    halve_idx = knot_pos.astype(np.int64)  # interior candidates = on-grid freqs
    return freqs, knots, halve_idx


def direct_hat_design(freqs, knots, gamma):
    """Exact B1 design by direct evaluation on the active (nonuniform) knots."""
    mask = np.ones(len(knots), dtype=bool)
    mask[1:-1] = gamma.astype(bool)
    active = knots[mask]
    p = len(active)
    cols = []
    for b in range(p):
        col = np.zeros(len(freqs))
        if b > 0:
            seg = (freqs >= active[b - 1]) & (freqs <= active[b])
            col[seg] = (freqs[seg] - active[b - 1]) / (active[b] - active[b - 1])
        if b < p - 1:
            seg = (freqs > active[b]) & (freqs <= active[b + 1])
            col[seg] = (active[b + 1] - freqs[seg]) / (active[b + 1] - active[b])
        if b == 0:
            seg = (freqs >= active[0]) & (freqs <= active[1])
            col[seg] = (active[1] - freqs[seg]) / (active[1] - active[0])
        if b == p - 1:
            seg = (freqs >= active[p - 2]) & (freqs <= active[p - 1])
            col[seg] = (freqs[seg] - active[p - 2]) / (active[p - 1] - active[p - 2])
        cols.append(col)
    return np.array(cols).T


class TestAssembleDesign:
    @pytest.mark.parametrize("gamma_pattern", ["all", "none", "alternating", "random"])
    def test_unbiased_equals_direct_evaluation(self, gamma_pattern, rng):
        freqs, knots, halve_idx = make_grid()
        n_int = len(knots) - 2
        gamma = {
            "all": np.ones(n_int, dtype=np.int8),
            "none": np.zeros(n_int, dtype=np.int8),
            "alternating": (np.arange(n_int) % 2).astype(np.int8),
            "random": rng.integers(0, 2, n_int).astype(np.int8),
        }[gamma_pattern]
        falling, rising = half_bases(freqs, knots)
        got = assemble_design(falling, rising, gamma, knots, halve_idx)
        expected = direct_hat_design(freqs, knots, gamma)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)

    def test_partition_of_unity(self, rng):
        freqs, knots, halve_idx = make_grid()
        gamma = rng.integers(0, 2, len(knots) - 2).astype(np.int8)
        falling, rising = half_bases(freqs, knots)
        design = assemble_design(falling, rising, gamma, knots, halve_idx)
        inside = (freqs >= knots[0]) & (freqs <= knots[-1])
        np.testing.assert_allclose(design[inside].sum(axis=1), 1.0, rtol=1e-12)

    def test_biased_column_vs_numerical_convolution(self):
        # Paper Eq. 5: biased basis = (B_j * H_n)(omega) with H_n the taper's
        # spectral window. Cross-check one interior hat against dense numerical
        # convolution using a boxcar taper (Fejer window).
        n_time = 64
        freqs = np.fft.rfftfreq(n_time)[1:-1]
        knots = np.concatenate(([0.0], freqs[[3, 10, 17, 24]], [0.5]))
        gamma = np.ones(4, dtype=np.int8)
        taper = np.ones(n_time) / np.sqrt(n_time)
        kernel = np.correlate(taper, taper, mode="full")[n_time - 1 :]

        falling_b, rising_b = half_bases_biased(freqs, knots, n_time, kernel)
        design_b = assemble_design(falling_b, rising_b, gamma, knots, EMPTY_IDX)

        # dense two-sided grid on [-1/2, 1/2)
        m = 1 << 14
        x = np.linspace(-0.5, 0.5, m, endpoint=False)
        window = np.abs(np.exp(-2j * np.pi * np.outer(x, np.arange(n_time))) @ taper) ** 2

        active = knots[np.concatenate(([True], gamma.astype(bool), [True]))]
        for col, centre in [(1, 1), (3, 3)]:  # two interior hats
            hat = np.interp(
                np.abs(x),
                [active[centre - 1], active[centre], active[centre + 1]],
                [0.0, 1.0, 0.0],
                left=0.0, right=0.0,
            )
            for j_freq in [2, 9, 20]:
                # circular convolution integral at data frequency freqs[j_freq]
                shift = np.abs(((x - freqs[j_freq] + 0.5) % 1.0) - 0.5)
                win_shifted = np.interp(shift, x[m // 2 :], window[m // 2 :])
                expected = np.trapezoid(hat * win_shifted, x)
                np.testing.assert_allclose(
                    design_b[j_freq, col], expected, rtol=5e-3, atol=1e-6
                )


class TestPartitionOfUnity:
    """B1 hats over the active knots sum to 1 at every grid point.

    This is what makes the constant vector lie exactly in the design's column
    span, so adding a constant to the response shifts the fit by exactly that
    constant and changes nothing else. A former log-scale offset correction
    relied on this and consequently cancelled itself out; pin the property so
    any future basis change surfaces loudly rather than silently.
    """

    def test_columns_sum_to_one(self, rng):
        n = 256
        freqs = np.arange(1, n // 2) / n
        knots, halve = knot_grid(freqs, 8)
        falling, rising = half_bases(freqs, knots)
        for _ in range(50):
            gamma = (rng.uniform(size=len(knots) - 2) < rng.uniform(0.05, 0.95)).astype(np.int8)
            design = assemble_design(falling, rising, gamma, knots, halve)
            np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-12)
