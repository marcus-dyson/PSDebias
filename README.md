# PSDebias — Adaptive Debiasing of Quadratic Spectral Estimators

`psdebias` implements **PSDebias** (Dyson, Astfalck, Cripps & Stemler,
*"Adaptive Smoothing of Quadratic Spectral Estimators"*): a Bayesian adaptive
debiasing framework for the general class of quadratic spectral estimators —
periodogram, Welch, multitaper and lag-window — in which the number, location
and width of B1-spline bases are learned from the data by Metropolis-Hastings
over a latent binary knot vector.

Estimates are debiased on the linear scale: the spline bases are convolved
with the estimator's known spectral window, the regression inverts the
blurring, and evaluating the unconvolved bases at the fitted coefficients
recovers the underlying spectrum.

## Installation

```console
pip install -e .   # from this directory; Python >= 3.10
```

Dependencies: `numpy`, `scipy`, `numba`.

## Quick start

```python
import numpy as np
from psdebias import welch, fit_psd, ar_spectrum, sample_ar

# --- simulate the paper's AR(4) process -------------------------------------
ar_poly = np.array([1.0, -2.7607, 3.8106, -2.6535, 0.9238])  # [1, -phi...]
x = sample_ar(1, 32 * 1024, ar_poly, rng=np.random.default_rng(0))[0]

# --- any quadratic estimator; carries the bias kernel ------------------------
est = welch(x, segment_length=1024, n_segments=32, step=1024)

# --- the full PSDebias workflow in one call ---------------------------------
fit = fit_psd(
    est,
    n_time=1024,                  # the estimator's window, for the debiasing
    knot_spacing=4,               # one candidate knot per 4 frequency bins
    rng=np.random.default_rng(1),
    n_iterations=50_000, warmup=30_000,
)

fit.mean          # posterior-mean PSD (linear scale)
fit.std           # pointwise posterior std
fit.expected_knots, fit.acceptance_rate
```

Lower-level control:

```python
from psdebias import PSDebias

sampler = PSDebias(est.psd, est.freqs, kernel=est.kernel, n_time=1024,
                   knot_spacing=4, rng=np.random.default_rng(2))
result = sampler.sample(n_iterations=50_000, warmup=30_000, thin=10)
gamma_map, psd_map = sampler.map_estimate()   # most probable configuration
```

Fixed-mesh debiasing (the DWelch/DQuad baselines):

```python
from psdebias import dquad
gamma = np.zeros(len(est.freqs), dtype=np.int8)
gamma[::10] = 1                                   # a uniform mesh
debiased = dquad(est.psd, est.freqs, gamma=gamma, kernel=est.kernel,
                 n_time=1024)
```

`dquad` also accepts the full one-sided grid (`drop_endpoints=False`), where
`freqs[0] = 0` and `freqs[-1] = 1/2` are themselves the ghost knots and `gamma`
runs over the interior candidates `freqs[1:-1]`.

## API

| symbol | purpose |
|---|---|
| `periodogram, welch, lag_window, multitaper` | classical estimators; each returns `SpectralEstimate(freqs, psd, kernel)`, where `kernel` is the one-sided bias sequence `h[tau]` the debiasing machinery needs |
| `fit_psd` | sampling + posterior summary in one call |
| `PSDebias` | the sampler class (`sample`, `map_estimate`, `design_matrix`) |
| `dquad` | fixed-knot debiasing on B0 bases, arbitrary nonuniform knots (the DWelch/DQuad baseline) |
| `ar_spectrum, matern_acf, matern_spectrum` | closed-form validation targets |
| `sample_ar, sample_matern` | seeded process simulators (AR via `lfilter`, Matern via circulant embedding) |

Work in **normalized frequency** (`fs = 1`, grid strictly inside (0, 0.5) —
i.e. the endpoint-dropped rfft grid the estimators return): the closed-form
basis ACFs are derived on that grid. Rescale results to physical units
afterwards.

## How it works (paper -> code)

| paper | code |
|---|---|
| Eq. 3 linear model per configuration `gamma` | `sampler._least_squares` (Cholesky of the Gram matrix) |
| Eq. 5 debiasing regression (linear domain, weighted by `1/I`) | `PSDebias.__init__` |
| Eq. 6/7 `B0`/`B1` bases from the latent vector | `splines.assemble_design` / `update_design` |
| Appendix A: truncated linear primitives, convolution paid once via FFT | `splines.acf_rising/acf_falling` + `half_bases_biased` (one batched `rfft`) |
| Appendix B: marginal posterior over `gamma`, conjugate `(sigma^2, beta)` draws | `sampler._log_posterior`, `_mh_kernel` |
| Algorithm 1 (MH with subset flips, coefficient positivity) | `sampler._mh_kernel` |

Per-iteration cost is O(N log N)-equivalent: the spectral-window convolutions
are precomputed once, each MH step reassembles only the design columns whose
knot neighbourhood changed (`update_design`), and each proposal is scored by a
single small Cholesky solve.

## Reproduction scripts

The paper's bias-variance study lives outside the library
(`pip install -e '.[scripts]'` for its extras — `tqdm`, `matplotlib`,
`scienceplots`):

* `scripts/generate_realisations.py` — draws AR(4)/Matern realisations, runs
  the classical estimators, the fixed-mesh `dquad` baseline and the
  adaptive sampler, caching everything under `scripts/realisations/`.
  `--n-samples/--n-iterations/--warmup` shrink it for smoke runs.
* `notebooks/bias-variance.ipynb` — empirical MSE/bias/variance comparison of
  all streams against the closed-form spectra.
* `notebooks/fig_gen.ipynb` — regenerates the paper's method figures (AR(4)
  debiasing against the uniform-mesh baselines, sunspot application) into
  `figs/paper/`.
* `notebooks/demo.ipynb` — a short walk through the API on one realisation.

## License

MIT.
