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
for smoke runs.

Output tree (``<suffix>`` is ``_debiasing`` or ``_unbiased``)::

    realisations/
    |-- ar4<suffix>/
    |   |-- welch_sim/  (samples, welch, [dwelch,] adapt_welch)
    |   `-- quad_sim/   (samples, lag-window, multitaper, [dlag, dmulti,]
    |                    adapt_lag, adapt_multi)
    `-- matern<suffix>/ (same structure)

In the debias arm the quad samples are per-setting (``samples/lag_NW{n}.npy``,
``samples/multi_NW{n}.npy``), one file per estimator; the smooth arm keeps
one shared ``samples/samples.npy`` (which
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
import multiprocessing as mp
import os
import sys
import zlib
from dataclasses import dataclass, field
from typing import Callable

# Pin every BLAS/Numba backend to one thread BEFORE numpy imports: each fit is
# single-threaded and we fan out one worker process per core, so library-level
# threading would only oversubscribe (P workers x P threads). setdefault leaves
# any explicit user override intact.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from numpy import typing as npt

from psdebias import (
    PSDebias,
    SpectralEstimate,
    dwelch_b0,
    lag_window,
    multitaper,
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
    n_samples: int = 200 #

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
    n_iterations: int = 50_000
    warmup: int = 30_000
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
# RNG: one deterministic stream per (purpose, setting, realisation)
# =============================================================================


def stream_rng(cfg: Config, tag: str, *keys: int) -> np.random.Generator:
    """Independent generator keyed by a string tag plus integer indices.

    Every random draw in the script comes from one of these, so a resumed run
    reproduces exactly the stream it would have used in a single uninterrupted
    run.
    """
    return np.random.default_rng([cfg.seed, zlib.crc32(tag.encode()), *keys])


# =============================================================================
# Sample generation
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


def build_samples(cfg: Config, process: str, length: int, tag: str):
    """Draw ``cfg.n_samples`` realisations of ``process``.

    The trailing 0 in each stream key is the retired screening attempt index,
    kept so cached realisations still reproduce bit-for-bit.
    """
    return np.array([
        draw_process(cfg, process, length, stream_rng(cfg, tag, i, 0))
        for i in range(cfg.n_samples)
    ])


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
            dof=est.dof, knot_spacing=knot_spacing, rng=rng,
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


@dataclass
class RowJob:
    """Everything one adaptive CSV needs to fit row ``i``. Held in a module
    global and inherited by forked workers (copy-on-write) so only the integer
    index is dispatched — the ``estimator`` closure and ``samples`` array never
    have to pickle."""

    cfg: Config
    samples: npt.NDArray[np.float64]
    estimator: Callable            # x -> SpectralEstimate
    tag: str                       # stream_rng tag, also the row's seed key
    n_time: int
    n_series: int
    knot_spacing: int


# Set by _run_adaptive_csv just before the pool forks; read by _row_worker.
_ROW_JOB: RowJob | None = None


def _row_worker(i: int):
    """Fit realisation ``i`` of the current job. Returns
    ``(i, expected_knots, mean, std, err)``; on a fit that raises ``ValueError``
    (debias exhaustion / input validation) returns a NaN sentinel row so the run
    neither crashes nor deadlocks on resume (the deterministic seed would re-hit
    the same failure)."""
    job = _ROW_JOB
    est = job.estimator(job.samples[i])
    try:
        mean, std, expected_knots = run_adaptive(
            job.cfg, est, n_time=job.n_time, n_series=job.n_series,
            knot_spacing=job.knot_spacing,
            rng=stream_rng(job.cfg, job.tag, i),
        )
        return i, expected_knots, mean, std, None
    except ValueError as e:
        nan = np.full(len(est.freqs), np.nan)
        return i, float("nan"), nan, nan, str(e)


def _run_adaptive_csv(
    csv_path: str, n_samples: int, job: RowJob, desc: str, workers: int
) -> None:
    """One row per realisation, fanned out over ``workers`` processes. Resumes
    from the current line count; ``imap`` keeps rows in ``i`` order so the
    resume-by-line-count and the deterministic per-row seed are preserved, and
    each row is flushed so the file can be tailed and the run interrupted."""
    global _ROW_JOB
    start_i = 0
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            start_i = sum(1 for _ in f)
    if start_i >= n_samples:
        return

    _ROW_JOB = job
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            def emit(result):
                i, expected_knots, mean, std, err = result
                if err is not None:
                    print(f"[warn] {desc} row {i} failed: {err}", file=sys.stderr)
                writer.writerow([expected_knots] + list(mean) + list(std))
                f.flush()

            todo = range(start_i, n_samples)
            if workers == 1:
                for i in tqdm(todo, desc=desc, leave=False):
                    emit(_row_worker(i))
            else:
                ctx = mp.get_context("fork")
                with ctx.Pool(workers) as pool:
                    for result in tqdm(
                        pool.imap(_row_worker, todo),
                        total=n_samples - start_i, desc=desc, leave=False,
                    ):
                        emit(result)
    finally:
        _ROW_JOB = None


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


def run_welch_simulations(cfg: Config, workers: int) -> None:
    for m in tqdm(cfg.n_segments, desc="Welch: samples/estimates"):
        m = int(m)
        length = m * cfg.segment_length
        for process in ("ar4", "matern"):
            base = _sim_dir(cfg, process, "welch_sim")
            # One stream tag across segment counts: samples at different m are
            # prefix-nested truncations of shared realisations (see
            # draw_process), matching the original study's long-draw cache.
            samples = load_or_compute(
                os.path.join(base, "samples", f"{m}.npy"),
                build_samples, cfg, process, length, f"welch/{process}",
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

            job = RowJob(
                cfg=cfg, samples=samples,
                estimator=lambda x, m=m: welch_estimate(cfg, x, m),
                tag=f"adapt_welch/{process}/m{m}",
                n_time=cfg.segment_length, n_series=int(samples.shape[1]),
                knot_spacing=cfg.knot_spacing_welch,
            )
            _run_adaptive_csv(
                os.path.join(base, "adapt_welch", f"{m}.csv"),
                cfg.n_samples, job, f"{process} adapt_welch({cfg.arm}) m={m}", workers,
            )


def _quad_variants(cfg: Config):
    """The two quad estimators: (name, sample basename, estimator factory)."""
    return (
        ("lag", lambda lag: (lambda x: lag_estimate(cfg, x, lag)), "lag-window", "dlag"),
        ("multi", lambda nw: (lambda x: multi_estimate(cfg, x, nw)), "multitaper", "dmulti"),
    )


def quad_samples_path(cfg: Config, base: str, name: str, nw: int) -> str:
    """Debias arm: per-setting samples, one file per estimator. Smooth arm:
    one shared file for both estimators."""
    if cfg.arm == "debias":
        return os.path.join(base, "samples", f"{name}_NW{nw}.npy")
    return os.path.join(base, "samples", "samples.npy")


def run_quad_simulations(cfg: Config, workers: int) -> None:
    settings = list(zip(cfg.nws_quad, cfg.lags_quad))

    for nw, lag in tqdm(settings, desc="Quad: samples/estimates"):
        nw, lag = int(nw), int(lag)
        for process in ("ar4", "matern"):
            base = _sim_dir(cfg, process, "quad_sim")
            for name, make_estimator, est_dir, debias_dir in _quad_variants(cfg):
                setting = lag if name == "lag" else nw
                estimator = make_estimator(setting)
                samples = load_or_compute(
                    quad_samples_path(cfg, base, name, nw),
                    build_samples, cfg, process, cfg.n_time_quad,
                    f"quad/{process}/{name}_NW{nw}" if cfg.arm == "debias"
                    else f"quad/{process}/shared",
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

                job = RowJob(
                    cfg=cfg, samples=samples, estimator=estimator,
                    tag=f"adapt_{name}/{process}/NW{nw}",
                    n_time=cfg.n_time_quad, n_series=cfg.n_time_quad,
                    knot_spacing=cfg.knot_spacing_quad,
                )
                _run_adaptive_csv(
                    os.path.join(base, f"adapt_{name}", f"NW{nw}.csv"),
                    cfg.n_samples, job, f"{process} adapt_{name}({cfg.arm}) NW={nw}", workers,
                )


def _warmup_numba() -> None:
    """Compile the ``@njit(cache=True)`` kernels once in the parent so forked
    workers load them from the on-disk cache instead of racing to compile on the
    first CSV. Best-effort: a tiny smooth-mode fit exercises the shared kernels
    and never exhausts its init search."""
    try:
        n = 128
        freqs = np.arange(1, n // 2) / n
        tmp = Config(**SMOOTH_CONFIG)
        tmp.n_iterations, tmp.warmup, tmp.thin, tmp.n_beta_draws = 2, 1, 1, 2
        est = SpectralEstimate(freqs, np.ones(len(freqs)) + 0.1, np.zeros(n), np.float64(8.0))
        run_adaptive(tmp, est, n_time=n, n_series=n, knot_spacing=4,
                     rng=np.random.default_rng(0))
    except Exception:  # warmup is an optimization, never fatal
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm", choices=("debias", "smooth"),
                        help="which arm of the study to generate")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker processes for the adaptive fits "
                             "(default: all cores; 1 runs serially in-process)")
    parser.add_argument("--base-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = Config(**(DEBIAS_CONFIG if args.arm == "debias" else SMOOTH_CONFIG))
    for name in ("n_samples", "n_iterations", "warmup", "base_path", "seed"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    workers = max(1, workers)

    create_directory_structure(cfg)
    if workers > 1:
        _warmup_numba()
    run_welch_simulations(cfg, workers)
    run_quad_simulations(cfg, workers)


if __name__ == "__main__":
    main()
