"""Shared numerical helpers: frequency-grid bookkeeping, segmentation, ACF -> PSD."""

from __future__ import annotations

import numpy as np
import scipy.fft
from numpy import typing as npt


def get_nfreq(n: int) -> int:
    """Number of positive rfft frequencies for an ``n``-point signal, excluding
    the zero and Nyquist bins (the bins dropped by ``drop_endpoints=True``)."""
    return (n - 1) // 2 if n % 2 else n // 2 - 1


def segment(
    x: npt.NDArray[np.float64], segment_length: int, n_segments: int, step: int
) -> npt.NDArray[np.float64]:
    """Split ``x`` into ``n_segments`` overlapping segments of ``segment_length``,
    starting every ``step`` samples. Returns shape ``(n_segments, segment_length)``."""
    required = (n_segments - 1) * step + segment_length
    if len(x) < required:
        raise ValueError(
            f"signal of length {len(x)} too short for {n_segments} segments of "
            f"length {segment_length} at step {step} (needs {required})"
        )
    idx = step * np.arange(n_segments)[:, None] + np.arange(segment_length)[None, :]
    return x[idx]


def freq_slice(n_time: int, n_freq: int) -> slice:
    """Slice selecting ``n_freq`` bins out of the full ``n_time``-point rfft grid.

    Accepts exactly two layouts: the full one-sided grid, or the grid with the
    zero and Nyquist endpoints dropped. Anything else is an error (the original
    implementation silently guessed ``slice(1, 1 + n_freq)`` here, which could
    misalign the basis with the data grid without warning).
    """
    full = n_time // 2 + 1
    if n_freq == full:
        return slice(None)
    if n_freq == full - 2:
        return slice(1, -1)
    raise ValueError(
        f"n_freq={n_freq} does not match an rfft grid of an {n_time}-point signal "
        f"(expected {full} bins, or {full - 2} with endpoints dropped)"
    )


def psd_from_acf(
    acf: npt.NDArray[np.float64], kernel: npt.NDArray[np.float64] | None = None
) -> npt.NDArray[np.float64]:
    """Exact one-sided PSD of a (possibly kernel-biased) one-sided ACF.

    ``acf`` holds lags ``0..N-1`` along the last axis; leading axes are batched.
    ``kernel`` (same length ``N``) is the taper-autocorrelation bias sequence and
    is applied multiplicatively before transforming. The negative lags are folded
    onto the length-``N`` circular buffer (lag ``-tau`` adds at index ``N - tau``)
    so the result lands exactly on the ``N``-point rfft grid of the data.
    """
    # Multiplying the ACF by the taper autocorrelation in the lag domain IS the
    # convolution with the spectral window in the frequency domain (the two are
    # a Fourier pair) — this is where "the convolution is paid once" happens.
    biased = acf * kernel if kernel is not None else np.asarray(acf, dtype=np.float64)

    # The rfft of a length-N sequence treats it as one period of an N-periodic
    # signal, so the negative lag -tau is indistinguishable from lag N - tau.
    # A real spectrum has a symmetric ACF (r[-tau] = r[tau]); folding adds each
    # positive lag onto its alias:
    #     folded[0]   = b[0]
    #     folded[t]   = b[t] + b[N - t],   t = 1..N-1
    # b[..., :0:-1] is exactly [b[N-1], b[N-2], ..., b[1]], so the one-liner
    # below implements that sum for every batched row at once. The result of
    # rfft(folded) is then the PSD evaluated *exactly* on the same N-point rfft
    # grid as the data's periodogram — no interpolation, no truncation error.
    folded = biased.copy()
    folded[..., 1:] += biased[..., :0:-1]
    return scipy.fft.rfft(folded, axis=-1).real
