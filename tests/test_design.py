"""The critical gate: incremental design updates must equal full assembly."""

import numpy as np
import pytest

from psdebias.splines import (
    EMPTY_IDX,
    assemble_design,
    half_bases,
    half_bases_biased,
    update_design,
)


@pytest.fixture(scope="module")
def setup():
    n_time = 128
    freqs = np.fft.rfftfreq(n_time)[1:-1]
    knot_pos = np.linspace(0, len(freqs) - 1, 13, dtype=int)
    knots = np.concatenate(([0.0], freqs[knot_pos], [0.5]))
    halve_idx = knot_pos.astype(np.int64)
    taper = np.ones(n_time) / np.sqrt(n_time)
    kernel = np.correlate(taper, taper, mode="full")[n_time - 1 :]
    unbiased = half_bases(freqs, knots)
    biased = half_bases_biased(freqs, knots, n_time, kernel)
    return knots, halve_idx, unbiased, biased


def random_flip(gamma, rng, n_select=2):
    new = gamma.copy()
    idx = rng.integers(0, len(gamma), n_select)
    new[idx] = rng.integers(0, 2, n_select).astype(np.int8)
    return new


@pytest.mark.parametrize("mode", ["unbiased", "biased"])
def test_incremental_equals_full_random_walk(setup, mode, rng):
    knots, halve_idx, unbiased, biased = setup
    falling, rising = unbiased if mode == "unbiased" else biased
    hidx = halve_idx if mode == "unbiased" else EMPTY_IDX
    n_int = len(knots) - 2

    gamma_old = rng.integers(0, 2, n_int).astype(np.int8)
    design_old = assemble_design(falling, rising, gamma_old, knots, hidx)
    for _ in range(200):
        gamma_new = random_flip(gamma_old, rng)
        got = update_design(falling, rising, gamma_new, gamma_old, design_old, knots, hidx)
        expected = assemble_design(falling, rising, gamma_new, knots, hidx)
        np.testing.assert_array_equal(got, expected)
        gamma_old, design_old = gamma_new, got  # walk on, reusing incremental state


@pytest.mark.parametrize("mode", ["unbiased", "biased"])
@pytest.mark.parametrize(
    "old_pattern,new_pattern",
    [
        ("zeros", "ones"),
        ("ones", "zeros"),
        ("zeros", "first"),
        ("zeros", "last"),
        ("ones", "first"),
        ("first", "last"),
    ],
)
def test_incremental_edge_configurations(setup, mode, old_pattern, new_pattern):
    knots, halve_idx, unbiased, biased = setup
    falling, rising = unbiased if mode == "unbiased" else biased
    hidx = halve_idx if mode == "unbiased" else EMPTY_IDX
    n_int = len(knots) - 2

    def pattern(name):
        g = np.zeros(n_int, dtype=np.int8)
        if name == "ones":
            g[:] = 1
        elif name == "first":
            g[0] = 1
        elif name == "last":
            g[-1] = 1
        return g

    gamma_old, gamma_new = pattern(old_pattern), pattern(new_pattern)
    design_old = assemble_design(falling, rising, gamma_old, knots, hidx)
    got = update_design(falling, rising, gamma_new, gamma_old, design_old, knots, hidx)
    expected = assemble_design(falling, rising, gamma_new, knots, hidx)
    np.testing.assert_array_equal(got, expected)


def test_identity_update_is_pure_copy(setup, rng):
    knots, halve_idx, (falling, rising), _ = setup
    gamma = rng.integers(0, 2, len(knots) - 2).astype(np.int8)
    design = assemble_design(falling, rising, gamma, knots, halve_idx)
    got = update_design(falling, rising, gamma, gamma.copy(), design, knots, halve_idx)
    np.testing.assert_array_equal(got, design)
