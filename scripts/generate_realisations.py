#!/usr/bin/env python3
"""Bias-variance study for PSDebias: generate realisations, estimates, and fits.

Draws AR(4) and Matern realisations, estimates their spectra with Welch /
lag-window / multitaper, and runs the adaptive :class:`psdebias.PSDebias`
sampler on each, caching everything under ``realisations/`` so
``notebooks/bias-variance.ipynb`` can measure empirical bias and variance.
One script, two arms::

    python generate_realisations.py debias    # bias-dominant arm  -> *_debiasing/
    python generate_realisations.py smooth    # variance-dominant  -> *_unbiased/

The debias arm runs the sampler in ``mode="debias"`` and also computes the
fixed-mesh ``dwelch_b0`` baselines; the smooth arm runs ``mode="smooth"``
with no closed-form debiasing. ``--n-samples/--n-iterations/--warmup`` exist
for smoke runs; ``--screen`` (debias arm only) redraws each realisation until
``psdebias.regime_diagnostic`` recommends debiasing for the estimator that
will consume it, so screened data lies inside the regime the debias sampler
assumes.

Output tree (``<suffix>`` is ``_debiasing`` or ``_unbiased``)::

    realisations/
    |-- ar4<suffix>/
    |   |-- welch_sim/  (samples, welch, [dwelch,] adapt_welch)
    |   `-- quad_sim/   (samples, lag-window, multitaper, [dlag, dmulti,]
    |                    adapt_lag, adapt_multi)
    `-- matern<suffix>/ (same structure)

In the debias arm the quad samples are per-setting (``samples/lag_NW{n}.npy``,
``samples/multi_NW{n}.npy``) so each can be screened against its own
estimator; the smooth arm keeps one shared ``samples/samples.npy`` (which
``wavelet_thresholding.py`` also reads). Adaptive CSVs hold one row per
realisation, ``[E[#knots]] + mean_S(f) + std_S(f)``, appended and flushed as
they finish so interrupted runs resume from the last complete row.

Differences from the retired ``auto_speccy`` scripts this file replaces:

* the corrected psdebias posterior (docs/CHANGES.md sec. 2) -- regenerated
  results intentionally differ from historical runs;
* CSVs are linear-scale ``S(f)`` in both arms (smooth-mode draws are
  exponentiated inside the sampler);
* per-realisation seeded generators instead of one global ``np.random.seed``,
  so a resumed CSV re-derives the same stream for every row;
* Welch segments step by exactly ``L`` (v1 stepped ``L - 1``);
* the dwelch/dquad mesh step equals the arm's sampler ``knot_spacing``
  (the old independent res0/res1 grids are gone).
"""

from __future__ import annotations

import argparse
import csv
import os
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy import typing as npt

from psdebias import (
    PSDebias,
    SpectralEstimate,
    dwelch_b0,
    lag_window,
    multitaper,
    regime_diagnostic,
    sample_ar,
    sample_matern,
    welch,
)

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a script-only nicety, not a psdebias dependency
    def tqdm(iterable, desc=None, leave=True):
        if desc:
            print(desc)
        return iterable


# =============================================================================
# Configuration
# =============================================================================


@dataclass(eq=False)
class Config:
    """All knobs for one arm of the study."""

    arm: str                    # "debias" | "smooth"
    process_suffix: str         # "_debiasing" | "_unbiased"
    compute_debiasing: bool
    n_samples: int = 2 #

    # Welch arm of the study
    segment_length: int = 1024
    n_segments: npt.NDArray[np.int64] = field(
        default_factory=lambda: 2 ** np.arange(3, 7)  # 8, 16, 32, 64
    )
    taper: npt.NDArray[np.float64] | None = None  # None -> rectangular
    knot_spacing_welch: int = 4

    # Quadratic (lag-window / multitaper) arm of the study
    n_time_quad: int = 2 ** 11
    nws_quad: npt.NDArray[np.int64] = field(default_factory=lambda: np.array([3, 4, 6, 8]))
    lags_quad: npt.NDArray[np.int64] = field(
        default_factory=lambda: 2 ** 11 // np.array([3, 4, 8, 16])
    )
    window: str = "bartlett"
    knot_spacing_quad: int = 8

    # Sampler settings (shared)
    n_iterations: int = 5_000
    warmup: int = 3_000
    thin: int = 10
    n_beta_draws: int = 50
    a_pi: float = 1.0
    b_pi: float = 1.0

    # Processes
    ar_poly: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, -2.7607, 3.8106, -2.6535, 0.9238])
    )
    matern_alpha: float = 1.5
    matern_lambda: float = 0.1
    matern_eta: float = 1.0
    # Matern draws are generated at least this long and truncated: the
    # circulant embedding's finite grid distorts the longest-lag covariances,
    # so the kept segment comes from a much longer realisation.
    matern_draw_length: int = 2 ** 18
    ar_burn_in: int = 1000

    seed: int = 42
    screen: bool = False
    screen_max_attempts: int = 100
    base_path: str = "./realisations"


DEBIAS_CONFIG = dict(
    arm="debias",
    process_suffix="_debiasing",
    compute_debiasing=True,
    segment_length=1024,
    taper=None,  # rectangular
    knot_spacing_welch=4,
    n_time_quad=2 ** 11,
    nws_quad=np.array([3, 4, 6, 8]),
    lags_quad=2 ** 11 // np.array([3, 4, 8, 16]),
    window="bartlett",
    knot_spacing_quad=8,
)

SMOOTH_CONFIG = dict(
    arm="smooth",
    process_suffix="_unbiased",
    compute_debiasing=False,
    segment_length=2 ** 12,
    taper=np.hanning(2 ** 12),  # estimators normalize to unit energy
    knot_spacing_welch=16,
    n_time_quad=2 ** 13,
    nws_quad=np.array([3, 4, 5, 6]),
    lags_quad=2 ** 13 // np.array([3, 4, 5, 6]),
    window="blackman",
    knot_spacing_quad=32,
)


# =============================================================================
# RNG: one deterministic stream per (purpose, setting, realisation, attempt)
# =============================================================================


def stream_rng(cfg: Config, tag: str, *keys: int) -> np.random.Generator:
    """Independent generator keyed by a string tag plus integer indices.

    Every random draw in the script comes from one of these, so a resumed run
    (or a screening redraw) reproduces exactly the stream it would have used
    in a single uninterrupted run.
    """
    return np.random.default_rng([cfg.seed, zlib.crc32(tag.encode()), *keys])


# =============================================================================
# Sample generation (with optional debias-regime screening)
# =============================================================================


def draw_process(cfg: Config, process: str, length: int, rng: np.random.Generator):
    """One realisation of ``process`` ("ar4" | "matern") of ``length`` samples.

    Draws from the same stream are prefix-nested: a shorter request consumes a
    prefix of the same underlying noise, so callers that reuse one stream tag
    across sample lengths (the Welch arm's segment counts) get truncations of
    a common realisation rather than independent redraws.
    """
    if process == "ar4":
        return sample_ar(1, length, cfg.ar_poly, burn_in=cfg.ar_burn_in, rng=rng)[0]
    draw_length = max(length, cfg.matern_draw_length)
    return sample_matern(
        1, draw_length, cfg.matern_eta, cfg.matern_alpha, cfg.matern_lambda, rng=rng
    )[0, :length]


def build_samples(
    cfg: Config,
    process: str,
    length: int,
    tag: str,
    passes_screen: Callable[[npt.NDArray[np.float64]], bool] | None,
):
    """Draw ``cfg.n_samples`` realisations, redrawing each until it passes.

    ``passes_screen`` is None when screening is off (every draw accepted on
    attempt 0). With screening on, each realisation is redrawn from a fresh
    per-attempt stream until the regime diagnostic recommends debiasing.
    """
    rows = []
    for i in range(cfg.n_samples):
        for attempt in range(cfg.screen_max_attempts):
            x = draw_process(cfg, process, length, stream_rng(cfg, tag, i, attempt))
            if passes_screen is None or passes_screen(x):
                rows.append(x)
                break
        else:
            raise RuntimeError(
                f"screening exhausted {cfg.screen_max_attempts} attempts for "
                f"{tag} realisation {i}: the regime diagnostic keeps rejecting "
                f"debiasing for this process/estimator combination"
            )
    return np.array(rows)


def debias_screen(cfg: Config, estimator: Callable, n_time: int):
    """Predicate: does ``regime_diagnostic`` recommend debiasing this sample's
    estimate? Uses the same kernel/n_time the sampler will later receive."""
    if not cfg.screen:
        return None

    def passes(x) -> bool:
        est = estimator(x)
        diag = regime_diagnostic(est.psd, est.freqs, kernel=est.kernel, n_time=n_time)
        return diag.debias_recommended

    return passes


# =============================================================================
# Estimators for this study (normalized frequency, endpoint-dropped grid)
# =============================================================================


def welch_estimate(cfg: Config, x, m: int) -> SpectralEstimate:
    return welch(
        x, segment_length=cfg.segment_length, n_segments=m,
        step=cfg.segment_length, taper=cfg.taper,
    )


def lag_estimate(cfg: Config, x, lag: int) -> SpectralEstimate:
    return lag_window(x, window=cfg.window, lag=lag)


def multi_estimate(cfg: Config, x, nw: int) -> SpectralEstimate:
    return multitaper(x, nw=nw)


def load_or_stack(path: str, estimator: Callable, samples) -> tuple[SpectralEstimate, npt.NDArray]:
    """Cached stack of per-sample PSDs plus the first full estimate (whose
    freqs/kernel are identical across samples). When the stack is already on
    disk only the first sample is re-estimated."""
    first = estimator(samples[0])
    if os.path.exists(path):
        return first, np.load(path)
    psds = np.array([first.psd] + [estimator(x).psd for x in samples[1:]])
    np.save(path, psds)
    return first, psds


# =============================================================================
# Fixed-mesh debiasing baseline (DWelch / DQuad)
# =============================================================================


def dwelch_gamma(n_freqs: int, step: int) -> npt.NDArray[np.int8]:
    """Uniform interior mesh with one active knot every ``step`` bins."""
    gamma = np.zeros(n_freqs, dtype=np.int8)
    gamma[step:-step:step] = 1
    return gamma


def debias_fixed_mesh(psds, freqs, kernel, n_time: int, step: int):
    """``dwelch_b0`` each row; keep only converged, strictly-positive results."""
    gamma = dwelch_gamma(len(freqs), step)
    results = []
    for psd in psds:
        try:
            results.append(dwelch_b0(psd, freqs, gamma=gamma, kernel=kernel, n_time=n_time)[1])
        except RuntimeError:  # nnls failed to converge
            continue
    debiased = np.array(results)
    if debiased.size == 0:
        return debiased
    return debiased[np.all(debiased > 0, axis=1)]


# =============================================================================
# Adaptive PSDebias fits
# =============================================================================


def run_adaptive(
    cfg: Config,
    est: SpectralEstimate,
    *,
    n_time: int,
    n_series: int,
    knot_spacing: int,
    rng: np.random.Generator,
):
    """Fit one estimate with the arm's sampler mode; return (mean, std, E[#knots]).

    ``c`` is the g-prior scale: the study follows the original scripts in
    using the time-series length rather than the paper's default ``c = g``
    (docs/CHANGES.md sec. 6).
    """
    if cfg.arm == "debias":
        sampler = PSDebias(
            est.psd, est.freqs, mode="debias", kernel=est.kernel, n_time=n_time,
            knot_spacing=knot_spacing, rng=rng,
        )
    else:
        sampler = PSDebias(
            est.psd, est.freqs, mode="smooth", dof=est.dof, knot_spacing=knot_spacing, rng=rng,
        )
    fit = sampler.sample(
        n_iterations=cfg.n_iterations, warmup=cfg.warmup, thin=cfg.thin,
        n_beta_draws=cfg.n_beta_draws, a_pi=cfg.a_pi, b_pi=cfg.b_pi,
        c=float(n_series),
    )
    return fit.mean, fit.std, fit.expected_knots


# =============================================================================
# Caching / CSV helpers
# =============================================================================


def load_or_compute(path: str, compute: Callable, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    result = compute(*args, **kwargs)
    np.save(path, result)
    return result


def append_adaptive_csv(csv_path: str, n_samples: int, row_fn: Callable, desc: str) -> None:
    """One row per realisation, resuming from the current line count; each row
    is flushed so the file can be tailed (and the run interrupted) safely."""
    start_i = 0
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            start_i = sum(1 for _ in f)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        for i in tqdm(range(start_i, n_samples), desc=desc, leave=False):
            mean, std, expected_knots = row_fn(i)
            writer.writerow([expected_knots] + list(mean) + list(std))
            f.flush()


# =============================================================================
# Simulation runners
# =============================================================================


def _sim_dir(cfg: Config, process: str, sim: str) -> str:
    return os.path.join(cfg.base_path, f"{process}{cfg.process_suffix}", sim)


def create_directory_structure(cfg: Config) -> None:
    debias = cfg.compute_debiasing
    welch_subdirs = ["samples", "welch"] + (["dwelch"] if debias else []) + ["adapt_welch"]
    quad_subdirs = (
        ["samples", "lag-window", "multitaper"]
        + (["dlag", "dmulti"] if debias else [])
        + ["adapt_lag", "adapt_multi"]
    )
    for process in ("ar4", "matern"):
        for sim, subdirs in (("welch_sim", welch_subdirs), ("quad_sim", quad_subdirs)):
            for subdir in subdirs:
                os.makedirs(os.path.join(_sim_dir(cfg, process, sim), subdir), exist_ok=True)


def run_welch_simulations(cfg: Config) -> None:
    for m in tqdm(cfg.n_segments, desc="Welch: samples/estimates"):
        m = int(m)
        length = m * cfg.segment_length
        for process in ("ar4", "matern"):
            base = _sim_dir(cfg, process, "welch_sim")
            screen = debias_screen(
                cfg, lambda x, m=m: welch_estimate(cfg, x, m), cfg.segment_length
            )
            # One stream tag across segment counts: unscreened samples at
            # different m are prefix-nested truncations of shared realisations
            # (see draw_process), matching the original study's long-draw cache.
            samples = load_or_compute(
                os.path.join(base, "samples", f"{m}.npy"),
                build_samples, cfg, process, length, f"welch/{process}", screen,
            )
            first, psds = load_or_stack(
                os.path.join(base, "welch", f"{m}.npy"),
                lambda x, m=m: welch_estimate(cfg, x, m), samples,
            )

            if cfg.compute_debiasing:
                load_or_compute(
                    os.path.join(base, "dwelch", f"{m}.npy"),
                    debias_fixed_mesh,
                    psds, first.freqs, first.kernel, cfg.segment_length,
                    cfg.knot_spacing_welch,
                )

    for m in tqdm(cfg.n_segments, desc="Welch: adaptive fits"):
        m = int(m)
        for process in ("ar4", "matern"):
            base = _sim_dir(cfg, process, "welch_sim")
            samples = np.load(os.path.join(base, "samples", f"{m}.npy"))

            def row(i: int, m=m, samples=samples, process=process):
                est = welch_estimate(cfg, samples[i], m)
                return run_adaptive(
                    cfg, est, n_time=cfg.segment_length, n_series=len(samples[i]),
                    knot_spacing=cfg.knot_spacing_welch,
                    rng=stream_rng(cfg, f"adapt_welch/{process}/m{m}", i),
                )

            append_adaptive_csv(
                os.path.join(base, "adapt_welch", f"{m}.csv"),
                cfg.n_samples, row, f"{process} adapt_welch({cfg.arm}) m={m}",
            )


def _quad_variants(cfg: Config):
    """The two quad estimators: (name, sample basename, estimator factory)."""
    return (
        ("lag", lambda lag: (lambda x: lag_estimate(cfg, x, lag)), "lag-window", "dlag"),
        ("multi", lambda nw: (lambda x: multi_estimate(cfg, x, nw)), "multitaper", "dmulti"),
    )


def quad_samples_path(cfg: Config, base: str, name: str, nw: int) -> str:
    """Debias arm: per-setting samples (each screened against its own
    estimator). Smooth arm: one shared file for both estimators."""
    if cfg.arm == "debias":
        return os.path.join(base, "samples", f"{name}_NW{nw}.npy")
    return os.path.join(base, "samples", "samples.npy")


def run_quad_simulations(cfg: Config) -> None:
    settings = list(zip(cfg.nws_quad, cfg.lags_quad))

    for nw, lag in tqdm(settings, desc="Quad: samples/estimates"):
        nw, lag = int(nw), int(lag)
        for process in ("ar4", "matern"):
            base = _sim_dir(cfg, process, "quad_sim")
            for name, make_estimator, est_dir, debias_dir in _quad_variants(cfg):
                setting = lag if name == "lag" else nw
                estimator = make_estimator(setting)
                screen = debias_screen(cfg, estimator, cfg.n_time_quad)
                samples = load_or_compute(
                    quad_samples_path(cfg, base, name, nw),
                    build_samples, cfg, process, cfg.n_time_quad,
                    f"quad/{process}/{name}_NW{nw}" if cfg.arm == "debias"
                    else f"quad/{process}/shared",
                    screen,
                )

                first, psds = load_or_stack(
                    os.path.join(base, est_dir, f"NW{nw}.npy"), estimator, samples
                )

                if cfg.compute_debiasing:
                    load_or_compute(
                        os.path.join(base, debias_dir, f"NW{nw}.npy"),
                        debias_fixed_mesh,
                        psds, first.freqs, first.kernel, cfg.n_time_quad,
                        cfg.knot_spacing_quad,
                    )

    for nw, lag in tqdm(settings, desc="Quad: adaptive fits"):
        nw, lag = int(nw), int(lag)
        for process in ("ar4", "matern"):
            base = _sim_dir(cfg, process, "quad_sim")
            for name, make_estimator, _, _ in _quad_variants(cfg):
                setting = lag if name == "lag" else nw
                estimator = make_estimator(setting)
                samples = np.load(quad_samples_path(cfg, base, name, nw))

                def row(i: int, estimator=estimator, samples=samples,
                        name=name, nw=nw, process=process):
                    est = estimator(samples[i])
                    return run_adaptive(
                        cfg, est, n_time=cfg.n_time_quad, n_series=cfg.n_time_quad,
                        knot_spacing=cfg.knot_spacing_quad,
                        rng=stream_rng(cfg, f"adapt_{name}/{process}/NW{nw}", i),
                    )

                append_adaptive_csv(
                    os.path.join(base, f"adapt_{name}", f"NW{nw}.csv"),
                    cfg.n_samples, row, f"{process} adapt_{name}({cfg.arm}) NW={nw}",
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm", choices=("debias", "smooth"),
                        help="which arm of the study to generate")
    parser.add_argument("--screen", action="store_true",
                        help="debias arm only: redraw each realisation until "
                             "regime_diagnostic recommends debiasing its estimate")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--base-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.screen and args.arm != "debias":
        parser.error("--screen only applies to the debias arm")

    cfg = Config(**(DEBIAS_CONFIG if args.arm == "debias" else SMOOTH_CONFIG))
    cfg.screen = args.screen
    for name in ("n_samples", "n_iterations", "warmup", "base_path", "seed"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    create_directory_structure(cfg)
    run_welch_simulations(cfg)
    run_quad_simulations(cfg)


if __name__ == "__main__":
    main()
