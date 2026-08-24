"""B-spline bases on the frequency domain, unbiased and spectral-window-biased.

The design-matrix machinery follows the paper's Appendix A: every nonuniform
B1 hat is a linear combination of per-candidate-segment *linear primitives*
(a falling and a rising ramp on each segment between consecutive candidate
knots). The primitives have closed-form inverse Fourier transforms, so their
convolution with the estimator's spectral window is paid once (one batched FFT
in :func:`half_bases_biased`) and every design matrix afterwards is assembled
by pure linear combination — including the incremental per-MCMC-step update
:func:`update_design`.

Conventions
-----------
* ``knots``: all K candidate knot locations (sorted, including both ends).
* ``gamma``: int8 indicator of the K-2 *interior* candidates; the two end
  knots are always active.
* ``falling[s] / rising[s]``: the primitive pair on candidate segment
  ``[knots[s], knots[s+1]]`` — either direct grid evaluations (`half_bases`)
  or spectral-window-convolved PSDs (`half_bases_biased`), shape (K-1, F).
* ``halve_idx``: grid indices coinciding with interior candidate knots. The
  direct-evaluation primitives are boundary-inclusive, so at exactly these
  points two adjacent segments both contribute the full hat value; assembled
  columns are halved there, after which the design equals the exact B1
  evaluation everywhere. Pass an empty array for biased primitives (the
  FFT-derived bases have no such double counting). This single argument
  replaces the original package's four separate ``odd_from_even_*`` functions.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy import typing as npt

from psdebias.util import freq_slice, psd_from_acf

EMPTY_IDX = np.empty(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Candidate knot grids
# ---------------------------------------------------------------------------

def knot_grid(
    freqs: npt.NDArray[np.float64],
    knot_spacing: int,
    *,
    log_knots: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Candidate knot set for the adaptive sampler.

    Selects ``len(freqs) // knot_spacing`` grid positions (uniformly, or
    geometrically when ``log_knots`` to resolve low-frequency structure). The
    candidate set is ``[0.0, freqs[pos], 0.5]``: every selected frequency is an
    interior candidate, and the two off-grid ghost knots let the spectral-window
    convolution wrap the full domain.

    The spectral window redistributes power across the whole of [0, 1/2] (its
    tails wrap around DC and Nyquist), so the basis must span the full interval
    even though the data grid excludes the endpoints. The ghost knots sit OFF
    the data grid at exactly 0 and 1/2 and are never selectable — they are the
    fixed ends.

    Returns ``(knots, halve_idx)`` where ``halve_idx`` holds the grid indices of
    the interior candidates, ready to pass to :func:`assemble_design` for
    direct-evaluation bases. They are carried around as *integer positions into
    freqs* rather than knot values so that downstream code never has to
    rediscover "which grid point is a knot" by comparing floats — the original
    package did that with searchsorted on float equality, which only worked by
    luck of shared construction.
    """
    n = len(freqs)
    n_target = max(n // knot_spacing, 1)
    if log_knots:
        pos = np.unique(np.geomspace(1, n - 1, n_target).astype(np.int64))
    else:
        pos = np.unique(np.linspace(0, n - 1, n_target, dtype=np.int64))
    knots = np.concatenate(([0.0], freqs[pos], [0.5]))
    return knots, pos


# ---------------------------------------------------------------------------
# Direct-evaluation primitives
# ---------------------------------------------------------------------------

def b1_falling(
    x: npt.NDArray[np.float64], lo: float, hi: float
) -> npt.NDArray[np.float64]:
    """Falling ramp: 1 at ``lo``, 0 at ``hi``, zero outside (boundary-inclusive)."""
    return np.where((x >= lo) & (x <= hi), (hi - x) / (hi - lo), 0.0)


def b1_rising(
    x: npt.NDArray[np.float64], lo: float, hi: float
) -> npt.NDArray[np.float64]:
    """Rising ramp: 0 at ``lo``, 1 at ``hi``, zero outside (boundary-inclusive)."""
    return np.where((x >= lo) & (x <= hi), (x - lo) / (hi - lo), 0.0)


def b0_box(
    x: npt.NDArray[np.float64], lo: float, hi: float
) -> npt.NDArray[np.float64]:
    """Indicator of ``(lo, hi)`` with value 1/2 exactly on the boundary."""
    inside = ((x > lo) & (x < hi)).astype(np.float64)
    return inside + 0.5 * ((x == lo) | (x == hi))


# ---------------------------------------------------------------------------
# Closed-form ACFs of the primitives (paper Eq. 8)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _ramp_acf(n: int, lo: float, hi: float, m: float, b: float) -> npt.NDArray[np.float64]:
    """Length-``n`` inverse Fourier transform (at integer lags) of the linear
    segment ``f(x) = m*x + b`` supported on ``[lo, hi]``, viewed as one side of
    a symmetric two-sided spectrum:

    ``R(tau) = 2 * integral_lo^hi (m x + b) cos(2 pi x tau) dx``

    tau = 0 integrates directly; tau >= 1 by parts, written with ``sinc``
    (``x sin(2 pi x tau) / (2 pi tau) = x^2 sinc(2 x tau)``).
    """
    out = np.empty(n)
    # tau = 0: R(0) = 2 * int (m x + b) dx = m (hi^2 - lo^2) + 2 b (hi - lo).
    # Special-cased because the tau >= 1 closed form divides by tau.
    out[0] = m * (hi * hi - lo * lo) + 2.0 * b * (hi - lo)
    tau = np.arange(1, n)
    # tau >= 1: two textbook integrals, written with np.sinc (sin(pi z)/(pi z)):
    #   int cos(2 pi x tau) dx   = sin(2 pi x tau) / (2 pi tau)      = x sinc(2 x tau)
    #   int x cos(2 pi x tau) dx = x sin(2 pi x tau) / (2 pi tau)
    #                              + cos(2 pi x tau) / (2 pi tau)^2   (by parts)
    #                            = x^2 sinc(2 x tau) + cos(2 pi x tau)/(4 pi^2 tau^2)
    # Both evaluated at hi minus lo, times the leading 2 (two-sided spectrum).
    out[1:] = 2.0 * m * (
        hi * hi * np.sinc(2.0 * hi * tau)
        - lo * lo * np.sinc(2.0 * lo * tau)
        + (np.cos(2.0 * np.pi * hi * tau) - np.cos(2.0 * np.pi * lo * tau))
        / (4.0 * np.pi**2 * tau**2)
    ) + 2.0 * b * (hi * np.sinc(2.0 * hi * tau) - lo * np.sinc(2.0 * lo * tau))
    return out


@njit(cache=True)
def acf_falling(n: int, lo: float, hi: float) -> npt.NDArray[np.float64]:
    """ACF of the falling ramp on ``[lo, hi]``."""
    w = hi - lo
    return _ramp_acf(n, lo, hi, -1.0 / w, hi / w)


@njit(cache=True)
def acf_rising(n: int, lo: float, hi: float) -> npt.NDArray[np.float64]:
    """ACF of the rising ramp on ``[lo, hi]``."""
    w = hi - lo
    return _ramp_acf(n, lo, hi, 1.0 / w, -lo / w)


@njit(cache=True)
def acf_box(n: int, lo: float, hi: float) -> npt.NDArray[np.float64]:
    """ACF of the box on ``[lo, hi]``: ``2 w sinc(w tau) cos(2 pi c tau)``."""
    w, c = hi - lo, 0.5 * (hi + lo)
    tau = np.arange(n)
    return 2.0 * w * np.sinc(w * tau) * np.cos(2.0 * np.pi * c * tau)


# ---------------------------------------------------------------------------
# Half-spline (primitive) stacks per candidate segment
# ---------------------------------------------------------------------------

def half_bases(
    freqs: npt.NDArray[np.float64], knots: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Direct grid evaluations of the primitive pair on every candidate segment.

    Returns ``(falling, rising)``, each of shape ``(K-1, len(freqs))``.
    """
    falling = np.array([b1_falling(freqs, knots[s], knots[s + 1]) for s in range(len(knots) - 1)])
    rising = np.array([b1_rising(freqs, knots[s], knots[s + 1]) for s in range(len(knots) - 1)])
    return falling, rising


def half_bases_biased(
    freqs: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    n_time: int,
    kernel: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Spectral-window-convolved primitives on every candidate segment.

    Builds the closed-form ACF of each primitive, applies the estimator's bias
    kernel (taper autocorrelation) and transforms all of them onto the data's
    ``n_time``-point rfft grid in one batched FFT. Returns ``(falling, rising)``
    of shape ``(K-1, len(freqs))``, aligned with ``freqs`` via
    :func:`psdebias.util.freq_slice` (raises if the grids are incompatible).
    """
    if len(kernel) != n_time:
        raise ValueError(f"kernel length {len(kernel)} != n_time {n_time}")
    n_seg = len(knots) - 1
    # This is the expensive step of the whole method, and it runs exactly once
    # per PSDebias construction: closed-form ACFs for all 2(K-1) primitives,
    # multiplied by the taper autocorrelation (= convolution with the spectral
    # window, see psd_from_acf), transformed in ONE batched rfft. Afterwards
    # every design matrix — including each of the ~10^4-10^5 MCMC proposals —
    # is just linear combinations of these rows. This is the paper's
    # Appendix A "convolution paid once" construction.
    acfs = np.empty((2, n_seg, n_time))
    for s in range(n_seg):
        acfs[0, s] = acf_falling(n_time, knots[s], knots[s + 1])
        acfs[1, s] = acf_rising(n_time, knots[s], knots[s + 1])
    psd = psd_from_acf(acfs, kernel)
    sl = freq_slice(n_time, len(freqs))
    return psd[0][:, sl].copy(), psd[1][:, sl].copy()


# ---------------------------------------------------------------------------
# B0 bases (used by dquad)
# ---------------------------------------------------------------------------

def basis_b0(
    freqs: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.int8],
) -> npt.NDArray[np.float64]:
    """B0 design matrix: one box column per *active* segment, shape (F, q+1)."""
    active = _active_knots(knots, gamma)
    cols = [b0_box(freqs, active[i], active[i + 1]) for i in range(len(active) - 1)]
    return np.array(cols).T


def basis_b0_biased(
    freqs: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.int8],
    n_time: int,
    kernel: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Spectral-window-convolved B0 design matrix, shape (F, q+1)."""
    if len(kernel) != n_time:
        raise ValueError(f"kernel length {len(kernel)} != n_time {n_time}")
    active = _active_knots(knots, gamma)
    acfs = np.array([acf_box(n_time, active[i], active[i + 1]) for i in range(len(active) - 1)])
    psd = psd_from_acf(acfs, kernel)
    sl = freq_slice(n_time, len(freqs))
    return psd[:, sl].T.copy()


def _active_knots(
    knots: npt.NDArray[np.float64], gamma: npt.NDArray[np.int8]
) -> npt.NDArray[np.float64]:
    if len(gamma) != len(knots) - 2:
        raise ValueError(f"gamma length {len(gamma)} != n interior candidates {len(knots) - 2}")
    mask = np.ones(len(knots), dtype=bool)
    mask[1:-1] = np.asarray(gamma, dtype=bool)
    return knots[mask]


# ---------------------------------------------------------------------------
# B1 design-matrix assembly (paper Appendix A)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _active_index(gamma: npt.NDArray[np.int8], n_knots: int) -> npt.NDArray[np.int64]:
    """Indices of active knots in the candidate grid (both ends always active)."""
    idx = np.empty(n_knots, dtype=np.int64)
    idx[0] = 0
    p = 1
    for i in range(len(gamma)):
        if gamma[i]:
            idx[p] = i + 1
            p += 1
    idx[p] = n_knots - 1
    return idx[: p + 1]


@njit(cache=True)
def _hat_column(
    out: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    seg_start: int,
    seg_stop: int,
    slope: float,
    pivot: float,
) -> None:
    """Accumulate one linear piece of a hat, ``h(x) = slope * (x - pivot)`` over
    candidate segments ``[seg_start, seg_stop)``: on each segment the piece is
    the interpolation ``h(left) * falling + h(right) * rising``.

    Why this works: each side of a B1 hat is a straight line, and a straight
    line restricted to one candidate segment [knots[s], knots[s+1]] is exactly
    the linear interpolation between its endpoint values,

        h(x) = h(knots[s]) * falling_s(x) + h(knots[s+1]) * rising_s(x),

    because falling_s is 1 at the left end / 0 at the right and rising_s is the
    reverse. The (slope, pivot) pair encodes the line: a rising side from
    active knot a to c is h(x) = (x-a)/(c-a) -> slope=1/(c-a), pivot=a; a
    falling side from c to d is h(x) = (d-x)/(d-c) -> slope=-1/(d-c), pivot=d.
    Crucially the SAME weights apply whether falling/rising hold direct grid
    evaluations or their spectral-window-convolved PSDs — convolution is
    linear, so convolving the primitives and combining is identical to
    combining and then convolving. That single fact is what lets the sampler
    reuse one precomputed primitive table for every proposal.
    """
    for s in range(seg_start, seg_stop):
        cl = slope * (knots[s] - pivot)
        cu = slope * (knots[s + 1] - pivot)
        for f in range(out.shape[0]):
            out[f] += cl * falling[s, f] + cu * rising[s, f]


@njit(cache=True)
def _compute_column(
    out: npt.NDArray[np.float64],
    b: int,
    p: int,
    idx: npt.NDArray[np.int64],
    active: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    halve_idx: npt.NDArray[np.int64],
) -> None:
    """Fill column ``b`` of the design (of ``p`` total): edge half-hats at
    ``b == 0`` / ``b == p-1``, full hats otherwise; then apply the
    double-counting halving at ``halve_idx``.

    The halve_idx correction, worked through
    ----------------------------------------
    The primitives are boundary-inclusive: at a grid point x that IS a
    candidate boundary knots[s], both rising_{s-1}(x) = 1 and falling_s(x) = 1.
    A hat h spanning that boundary therefore receives its value twice:

        segment s-1 contributes  h(knots[s]) * rising_{s-1}(x) = h(x)
        segment s   contributes  h(knots[s]) * falling_s(x)    = h(x)

    e.g. candidate knots [.1, .2, .3], hat rising over both segments with
    h(.2) = 0.5: at the grid point x = .2 the sum gives 0.5 + 0.5 = 1.0, twice
    the true value. At every non-boundary grid point exactly one segment
    covers x and the interpolation is single-counted. Halving the rows at the
    interior candidate positions (halve_idx) therefore makes the assembled
    matrix equal the EXACT B1 evaluation everywhere — including at inactive
    candidates sitting mid-hat (tests/test_splines.py proves this equality).
    Biased primitives are smooth PSD curves with no boundary evaluation
    involved, so no double count exists and callers pass an empty halve_idx.
    """
    out[:] = 0.0
    if b == 0:  # left-edge half-hat: falls from 1 at active[0] to 0 at active[1]
        _hat_column(out, falling, rising, knots, idx[0], idx[1],
                    -1.0 / (active[1] - active[0]), active[1])
    elif b == p - 1:  # right-edge half-hat: rises from 0 at active[p-2] to 1
        _hat_column(out, falling, rising, knots, idx[p - 2], idx[p - 1],
                    1.0 / (active[p - 1] - active[p - 2]), active[p - 2])
    else:  # interior hat centred on active[b]: rising side then falling side
        _hat_column(out, falling, rising, knots, idx[b - 1], idx[b],
                    1.0 / (active[b] - active[b - 1]), active[b - 1])
        _hat_column(out, falling, rising, knots, idx[b], idx[b + 1],
                    -1.0 / (active[b + 1] - active[b]), active[b + 1])
    for j in range(len(halve_idx)):
        out[halve_idx[j]] *= 0.5


@njit(cache=True)
def assemble_design(
    falling: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.int8],
    knots: npt.NDArray[np.float64],
    halve_idx: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Build the full B1 design matrix, shape ``(F, q+2)``: one column per
    active knot (interior hats plus the two edge half-hats)."""
    idx = _active_index(gamma, len(knots))
    active = knots[idx]
    p = len(idx)
    n_freq = falling.shape[1]
    design = np.zeros((n_freq, p))
    col = np.empty(n_freq)
    for b in range(p):
        _compute_column(col, b, p, idx, active, falling, rising, knots, halve_idx)
        design[:, b] = col
    return design


@njit(cache=True)
def update_design(
    falling: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    gamma_new: npt.NDArray[np.int8],
    gamma_old: npt.NDArray[np.int8],
    design_old: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    halve_idx: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Incremental :func:`assemble_design`, reusing unchanged columns.

    A hat column is identified by the candidate-grid indices of its
    ``(left neighbour, centre, right neighbour)`` active knots; edge columns
    use sentinels (left edge: right = -1; right edge: left = -2, both outside
    the valid index range). Columns whose triple is unchanged between
    ``gamma_old`` and ``gamma_new`` are copied from ``design_old``; only the
    rest are recomputed. MCMC proposals flip a couple of knots per step, so
    nearly everything is copied. The O(p^2) triple scan is faster in practice
    than hashing for the p ~ 10-30 columns this sampler sees.
    """
    n_knots = len(knots)
    idx_new = _active_index(gamma_new, n_knots)
    idx_old = _active_index(gamma_old, n_knots)
    p_new = len(idx_new)
    p_old = len(idx_old)
    n_freq = falling.shape[1]

    # A hat column is fully determined by WHERE its three defining active knots
    # sit on the candidate grid — left neighbour, centre, right neighbour —
    # because those fix both its support and its two slopes. Flipping a knot
    # elsewhere in gamma cannot change such a column, so matching triples can
    # be copied verbatim. Edge half-hats have only two defining knots; the
    # missing side is encoded with sentinels chosen outside the valid index
    # range [0, n_knots-1] so an edge key can never collide with an interior
    # key: left edge -> (0, first-interior, -1), right edge ->
    # (-2, last-interior, n_knots-1).
    old_key = np.empty((p_old, 3), dtype=np.int64)
    for b in range(p_old):
        old_key[b, 0] = -2 if b == p_old - 1 else (0 if b == 0 else idx_old[b - 1])
        old_key[b, 1] = idx_old[1] if b == 0 else (idx_old[p_old - 2] if b == p_old - 1 else idx_old[b])
        old_key[b, 2] = -1 if b == 0 else (n_knots - 1 if b == p_old - 1 else idx_old[b + 1])

    active_new = knots[idx_new]
    design = np.empty((n_freq, p_new))
    col = np.empty(n_freq)
    for b in range(p_new):
        if b == 0:
            ka, kb, kc = np.int64(0), idx_new[1], np.int64(-1)
        elif b == p_new - 1:
            ka, kb, kc = np.int64(-2), idx_new[p_new - 2], np.int64(n_knots - 1)
        else:
            ka, kb, kc = idx_new[b - 1], idx_new[b], idx_new[b + 1]

        # Linear scan over old columns. p is ~10-30 here, so the O(p^2) total
        # scan is a few hundred integer compares — cheaper in practice (and in
        # compiled code) than building any hash structure per proposal.
        found = False
        for b_old in range(p_old):
            if old_key[b_old, 0] == ka and old_key[b_old, 1] == kb and old_key[b_old, 2] == kc:
                design[:, b] = design_old[:, b_old]
                found = True
                break
        if not found:
            # Only columns whose knot neighbourhood changed reach here — with
            # n_select=2 flips per proposal that is O(1) columns, so the
            # per-iteration cost is a couple of _compute_column calls plus
            # copies, not a full (F x p) rebuild.
            _compute_column(col, b, p_new, idx_new, active_new, falling, rising, knots, halve_idx)
            design[:, b] = col
    return design
