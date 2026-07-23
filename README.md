# psdebias — Adaptive Smoothing of Quadratic Spectral Estimators

`psdebias` implements **PSDebias** (Dyson, Astfalck, Cripps & Stemler,
*"Adaptive Smoothing of Quadratic Spectral Estimators"*): a Bayesian adaptive
smoothing framework for the general class of quadratic spectral estimators —
periodogram, Welch, multitaper and lag-window — in which the number, location
and width of B1-spline bases are learned from the data by Metropolis-Hastings
over a latent binary knot vector.

Two regimes, one machinery:

* **Variance-dominant** estimates are *smoothed* on the log scale, where their
  variance is approximately constant.
* **Bias-dominant** estimates are *debiased* on the linear scale: the spline
  bases are convolved with the estimator's known spectral window, the
  regression inverts the blurring, and evaluating the unconvolved bases at the
  fitted coefficients recovers the underlying spectrum.

A data-driven **sign diagnostic** (paper Sec. IV-C) decides which regime is
warranted before any sampling.

This package is a ground-up rewrite of the original `auto-speccy` research
code with corrected statistics, stabilized numerics, ~65x lower sampler
memory, and a full test suite. **See [CHANGES.md](CHANGES.md) for every
difference against the original, and [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a
guided tour of how the implementation works.**

## Installation

```console
pip install -e '.[test]'      # from this directory; Python >= 3.10
pytest                        # fast tests; `pytest -m ""` adds slow end-to-end runs
```

Dependencies: `numpy`, `scipy`, `numba`.

## Quick start

```python
import numpy as np
from psdebias import welch, fit_psd, ar_spectrum, sample_ar

# --- simulate the paper's AR(4) process -------------------------------------
ar_poly = np.array([1.0, -2.7607, 3.8106, -2.6535, 0.9238])  # [1, -phi...]
x = sample_ar(1, 32 * 1024, ar_poly, rng=np.random.default_rng(0))[0]

# --- any quadratic estimator; note the third output: the bias kernel --------
freqs, psd, kernel = welch(x, segment_length=1024, n_segments=32, step=1024)

# --- the full PSDebias workflow in one call ---------------------------------
fit = fit_psd(
    psd, freqs,
    kernel=kernel, n_time=1024,   # the estimator's window, for the diagnostic/debias
    mode="auto",                  # sign diagnostic picks "debias" or "smooth"
    knot_spacing=4,               # one candidate knot per 4 frequency bins
    rng=np.random.default_rng(1),
    n_iterations=50_000, warmup=30_000,
)

fit.mean          # posterior-mean PSD (always linear scale)
fit.std           # pointwise posterior std
fit.mode          # which regime the diagnostic chose
fit.expected_knots, fit.acceptance_rate
```

Lower-level control:

```python
from psdebias import PSDebias, regime_diagnostic

diag = regime_diagnostic(psd, freqs, kernel=kernel, n_time=1024)
print(diag.debias_recommended, diag.gap_per_constraint, diag.bandwidth_bins)

sampler = PSDebias(psd, freqs, mode="debias", kernel=kernel, n_time=1024,
                   knot_spacing=4, rng=np.random.default_rng(2))
result = sampler.sample(n_iterations=50_000, warmup=30_000, thin=10)
gamma_map, psd_map = sampler.map_estimate()   # most probable configuration
```

Fixed-mesh debiasing (the DWelch/DQuad baselines):

```python
from psdebias import dwelch_b1
gamma = np.zeros(len(freqs), dtype=np.int8)
gamma[::10] = 1                                   # a uniform mesh
_, debiased = dwelch_b1(psd, freqs, gamma=gamma, kernel=kernel, n_time=1024)
```

## API

| symbol | purpose |
|---|---|
| `periodogram, welch, lag_window, multitaper` | classical estimators; each returns `SpectralEstimate(freqs, psd, kernel)` where `kernel` is the one-sided bias sequence `h[tau]` the debiasing machinery needs |
| `fit_psd` | diagnostic + regime dispatch + sampling + posterior summary in one call |
| `PSDebias` | the sampler class (`sample`, `map_estimate`, `design_matrix`) |
| `regime_diagnostic`, `window_bandwidth` | recommended debias-or-smooth rule (bandwidth-matched NNLS-gap; used by `mode="auto"`) |
| `sign_diagnostic` | paper Sec. IV-C sign rule (kept as the reference behavior) |
| `dwelch_b0, dwelch_b1` | fixed-knot debiasing (B0 / B1 bases, arbitrary nonuniform knots) |
| `ar_spectrum, matern_acf, matern_spectrum` | closed-form validation targets |
| `sample_ar, sample_matern` | seeded process simulators (AR via `lfilter`, Matern via circulant embedding) |

Work in **normalized frequency** (`fs = 1`, grid strictly inside (0, 0.5) —
i.e. the endpoint-dropped rfft grid the estimators return): the debias mode's
closed-form basis ACFs are derived on that grid. Rescale results to physical
units afterwards.

## How it works (paper -> code)

| paper | code |
|---|---|
| Eq. 3 linear model per configuration `gamma` | `sampler._least_squares` (Cholesky of the Gram matrix) |
| Eq. 6/7 `B0`/`B1` bases from the latent vector | `splines.assemble_design` / `update_design` |
| Appendix A: truncated linear primitives, convolution paid once via FFT | `splines.acf_rising/acf_falling` + `half_bases_biased` (one batched `rfft`) |
| Appendix B: marginal posterior over `gamma`, conjugate `(sigma^2, beta)` draws | `sampler._log_posterior`, `_mh_kernel` |
| Algorithm 1 (MH with subset flips, positivity in debias mode) | `sampler._mh_kernel` |
| Sec. IV-A smoothing (log domain) | `PSDebias(mode="smooth")` |
| Sec. IV-B debiasing (linear domain, weighted) | `PSDebias(mode="debias")` |
| Sec. IV-C sign diagnostic | `diagnostic.sign_diagnostic`, `fit_psd(mode="auto")` |

Per-iteration cost is O(N log N)-equivalent: the spectral-window convolutions
are precomputed once, each MH step reassembles only the design columns whose
knot neighbourhood changed (`update_design`), and each proposal is scored by a
single small Cholesky solve.

## Reproduction scripts

The paper's bias-variance study lives outside the library
(`pip install -e '.[scripts]'` for its extras — `tqdm`, `PyWavelets`,
`matplotlib`, `scienceplots`):

* `scripts/generate_realisations.py {debias,smooth}` — draws AR(4)/Matern
  realisations, runs the classical estimators, the fixed-mesh `dwelch_b0`
  baseline (debias arm) and the adaptive sampler, caching everything under
  `scripts/realisations/`. `--screen` (debias arm) redraws each realisation
  until `regime_diagnostic` endorses debiasing its estimate;
  `--n-samples/--n-iterations/--warmup` shrink it for smoke runs.
* `scripts/wavelet_thresholding.py` — WPM wavelet-thresholding baseline over
  the smoothing-arm realisations.
* `notebooks/bias-variance.ipynb` — empirical MSE/bias/variance comparison of
  all streams against the closed-form spectra.

## Testing

```console
pytest            # unit tests (~10 s)
pytest -m ""      # + slow end-to-end AR(4)/Matern studies (~4 min)
```

Highlights: bit-for-bit equality of incremental vs full design assembly over
random MCMC-like walks; closed-form basis ACFs vs adaptive quadrature; biased
bases vs dense numerical convolution with the spectral window; the JIT
posterior vs an independent scipy reference; conditional-draw moments vs the
analytic Student-t posterior; determinism under a fixed seed; and end-to-end
mean-square-error improvement over the raw estimators on analytic spectra.

## License

MIT.
