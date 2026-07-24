# psdebias vs. auto-speccy — every change, and why

`psdebias` is a ground-up rewrite of `~/Desktop/Auto-Speccy` (`auto_speccy`),
the reference implementation of *"Adaptive Smoothing of Quadratic Spectral
Estimators"* (Dyson, Astfalck, Cripps & Stemler). The rewrite fixes two
statistical defects where the code diverged from the paper, replaces the
numerically fragile linear algebra, cuts sampler memory by ~65x, implements
the paper's Sec. IV-C diagnostic (previously missing from the library), and
retires all dead or broken code. This document lists every change against the
original, with file/line references into `auto_speccy` and the behavioral
consequences.

Everything here is verified by the test suite (`pytest -m ""` runs all 104+
tests including the slow end-to-end studies). Where the mathematics is
unchanged, the new code is tested for exact or near-exact agreement against
verbatim ports of the original algorithms (see `tests/conftest.py` and the
oracles inside individual test files).

---

## 1. Symbol map

| auto-speccy | psdebias | category |
|---|---|---|
| `solver.MH` | `sampler.PSDebias` | renamed + fixed |
| `MH(Biasing=True/False)` | `PSDebias(mode="debias"/"smooth")` | renamed |
| `MH(Basis_Resolution=, n_ts=, h_conv=)` | `PSDebias(knot_spacing=, n_time=, kernel=)` | renamed |
| `MH.sample(n_gibbs_steps=, thinning=, api=, bpi=, asig=, bsig=)` | `PSDebias.sample(n_beta_draws=, thin=, a_pi=, b_pi=, a_sigma=, b_sigma=)` | renamed |
| `MH.get_mode_gamma` + `get_biased_mode_estimate` | `PSDebias.map_estimate` | merged + fixed |
| `MH.get_estimates` | `FitResult.mean/std/expected_knots` | replaced |
| `MH.get_{biased,unbiased}_design_matrix` | `PSDebias.design_matrix(gamma, biased=)` | merged |
| — | `sampler.fit_psd(mode="auto")` | **new** |
| — | `diagnostic.regime_diagnostic` | **new** (bandwidth-matched NNLS-gap rule) |
| `solver.posterior` (njit) | `sampler._log_posterior` | fixed |
| `solver._mcmc_sampling_loop` | `sampler._mh_kernel` | rewritten |
| `splines.odd_from_even_biased` / `..._unbiased` | `splines.assemble_design(..., halve_idx)` | unified (4 -> 2 fns) |
| `splines.odd_from_even_{biased,unbiased}_incremental` | `splines.update_design(..., halve_idx)` | unified |
| `splines.b1_spline_lower/upper` | `splines.b1_falling/b1_rising` | renamed |
| `splines.zero_val_r/l` + `acf_non_zero_r/l` + `biased_i_r/l` | `splines.acf_rising/acf_falling` (+ shared `_ramp_acf`) | merged |
| `splines.gen_basis_b1` / `gen_basis_b1_biased` | `splines.half_bases` / `half_bases_biased` | rewritten (batched) |
| `splines.gen_basis_b0` / `gen_basis_b0_biased` | `splines.basis_b0` / `basis_b0_biased` | rewritten (batched) |
| — | `splines.knot_grid` | **new** (was inline in `MH.__init__`) |
| `psd_est.periodogram/welch/lag_window/multitaper` | `estimators.*` (same names) | fixed scaling, keyword API |
| `dwelch.dwelch_b0/b1` | `dwelch.dwelch_b0/b1` | rebuilt on shared assembly |
| `util.get_exact_rfft_psd` | `util.psd_from_acf` | vectorized + batched |
| `util.split` | `util.segment` | vectorized |
| `util.get_nfreq` | `util.get_nfreq` | kept |
| `util.sample_matern` | `simulate.sample_matern` | moved, vectorized, seeded |
| — | `simulate.sample_ar` | **new** (drops `statsmodels`) |
| `analytic.ar_spectrum/matern_acf/matern_spectrum` | `analytic.*` | kept (vectorized) |
| `probfuncs.py` | *(removed)* | dead duplicate |
| `plotting.py` | *(removed)* | broken (NameError) |
| `util.bochner` | *(removed)* | dead |

## 2. Statistical fixes (results **intentionally differ** from the original)

### 2.1 Marginal-likelihood exponent used the wrong "sample size"

`solver.py:45` scored configurations with

```
BIC = -(q/2)·log(1+c) - (K/2 + a_sigma)·log(ESS + 2·b_sigma)      # original
```

where `K = len(interval)` is the number of **candidate knots** (~129 in a
typical run). The paper (Appendix B) derives

```
p(Y|gamma) ∝ (1+c)^(-p/2) · (ESS + 2·b_sigma)^(-(g/2 + a_sigma))   # paper
```

with `g = len(Y)` the number of **frequency bins** (~511). The original's own
sigma^2 Gibbs step (`nu = len(estimate) + 2·asig`, `solver.py:508`) and its MAP
selector (`solver.py:631`) both used `g` — so the chain sampled one posterior
while the reported mode maximized another. The rewrite uses `g` everywhere
(`sampler.py::_log_posterior`).

**Consequence:** the residual term is penalized ~4x more strongly in the
exponent, so the corrected posterior concentrates harder on configurations
that actually fit. Acceptance rates and knot counts differ from historical
runs; end-to-end tests therefore assert against analytic truth, never against
original outputs. On the paper's AR(4) study the corrected sampler recovers
the twin peaks with ~20 active bases and roughly halves the log-domain MSE of
the raw Welch estimate (`tests/test_end_to_end.py`).

### 2.2 Inconsistent Beta-Binomial prior

The sampler evaluated `betaln(q + a_pi, K + 2 - q + b_pi)` with `q = Σγ + 2`
(ghost/end knots counted as "active choices", `solver.py:43,100`), while the
MAP selector evaluated `betaln(q + a_pi, K - q + b_pi)` (`solver.py:636`).
Neither matches the paper's Eq. 10, `B(q_γ + a_π, L - q_γ + b_π)` with `q_γ`
counting only the `L` *selectable interior* knots. The rewrite uses the paper
form, and `PSDebias.map_estimate` scores stored samples with the **same
compiled function** the chain targets, so this class of drift cannot recur
(guarded by `tests/test_sampler.py::test_map_estimate_uses_kernel_posterior`).

The `(1+c)` exponent keeps `p = q_γ + 2`: that is the number of coefficients
actually integrated out, since the two edge half-hats are always in the model.

## 3. Numerical and engineering changes (same math, better behavior)

### 3.1 Linear algebra: no normal-equation solves, no explicit inverses

Original: `solve(XᵀX, Xᵀy)` inside the JIT loop (`solver.py:92-94,129-131`),
`inv(XtX) @ X.T @ Y` at initialization (`solver.py:482`), and singularity
handled by `try/except` around `np.linalg` calls inside `@njit`
(`solver.py:128-149`). The rewrite factors the Gram matrix once per proposal
with a hand-rolled Cholesky that **returns a success flag** instead of
raising (`sampler.py::_cholesky`); β̂, ESS and the posterior coefficient draws
all reuse that factor through triangular solves (`_least_squares`,
`_solve_lower/_solve_upper`). A failed factorization is a clean proposal
rejection. Besides removing the exception machinery from the hot loop, this
keeps the kernel free of LAPACK calls and exception handling — the two things
Numba supports least reliably. Verified against `np.linalg.lstsq` to 1e-9 and benchmarked at
parity with the original kernel (~24k iterations/sec on this machine for the
511-bin, 65-candidate configuration).

### 3.2 Sampler memory: ~65x smaller at paper-scale settings

Original `sample()` pre-generated conditional-draw randoms for **every**
post-warmup iteration — `normal_sample` alone is
`n_iterations x n_gibbs_steps x K` (`solver.py:509`), ~2.6 GB at the paper's
50k x 50 x 129 — stored β draws for every iteration, and only then thinned
(`solver.py:549`). The rewrite thins **inside the kernel**: only every
`thin`-th state is stored, and χ²/normal draws are generated for kept states
only. Outputs are NaN-padded rectangular arrays (`(n_kept, n_draws, L+2)`),
which also removes Numba reflected-list overhead from the loop.

### 3.3 Basis construction: batched FFTs, vectorized fold

`util.get_exact_rfft_psd` built the symmetric ACF buffer with a per-lag
Python `for` loop (`util.py:79-81`) and was called once per half-spline
(~260 non-JIT Python calls per basis build). `util.psd_from_acf` vectorizes
the fold and accepts arbitrary leading batch axes, so `half_bases_biased`
transforms **all** primitives in a single `scipy.fft.rfft` call. Exactness vs
the original loop is tested bit-for-bit
(`tests/test_util.py::test_matches_original_loop_exactly`).

### 3.4 One design-matrix assembler instead of four

`odd_from_even_biased`, `odd_from_even_unbiased` and their two `_incremental`
twins (~470 lines, `splines.py:357-825`) collapse into `assemble_design` and
`update_design`. The only real difference between the biased and unbiased
paths — halving direct-evaluation bases at grid points that coincide with
interior candidate knots, where the boundary-inclusive primitives double
count — becomes an explicit `halve_idx` argument (empty for biased bases).
The incremental (ka, kb, kc) column-identity scheme is preserved unchanged.
`tests/test_design.py` walks 200 random flip pairs plus edge configurations
and requires bit-for-bit equality between incremental and full assembly, and
`tests/test_splines.py` proves the assembled unbiased design equals direct
B1-hat evaluation exactly (uniform *and* nonuniform knots) and cross-checks
the biased bases against dense numerical convolution with the spectral window.

### 3.5 Integer knot indices end float matching

The original recovered knot grid positions by `np.searchsorted(domain,
Interval[1:-1])` — float equality between two arrays that merely *should*
share a grid (`splines.py:526,721`) — and evaluated `b0_spline` boundaries
with `==` on floats. `splines.knot_grid` now returns the integer grid indices
of the interior candidates alongside the knot values, and those indices flow
to every consumer (`halve_idx`), so no float comparison decides anything
structural.

### 3.6 Strict frequency-grid validation

`gen_basis_b1_biased` silently guessed `slice(1, 1 + nfreq)` when the
estimate length matched neither the full nor the endpoint-dropped rfft grid
(`splines.py:343`) — a silent misalignment waiting to happen.
`util.freq_slice` accepts exactly the two valid layouts and raises otherwise.

### 3.7 Initialization: prior-sampled, with a working failure guard

Both modes initialize from the Beta-Bernoulli prior (`pi ~ Beta(a_pi, b_pi)`,
`gamma_i ~ Bernoulli(pi)`, drawn from the injected generator). In debias mode
draws are retried until the unconstrained weighted least-squares fit has
all-positive coefficients — the chain's target is truncated to that region,
so the start must lie inside it. This matches the original's strategy
(`solver.py:462-489`) but fixes its never-firing exhaustion guard
(`elif attempt == max_attempts` inside `range(max_attempts)`) and removes the
redundant first draw that was immediately overwritten (`solver.py:443-448`).
On exhaustion the new code raises with a pointer to `regime_diagnostic` /
`mode="smooth"`.

*History note:* the first cut of this rewrite instead started debias chains
deterministically at the all-knots configuration. That was a regression,
caught on the annual sunspot dataset: when the all-knots fit has negative
coefficients the chain starts outside the positivity support and can never
accept a proposal (acceptance 0.0, output garbage). The prior-sampled
initialization restores the original's correct behavior; a regression test
pins it (`tests/test_end_to_end.py::test_sunspot_forced_debias_regression`),
`sample(initial_gamma=...)` allows explicit overrides, and the sampler now
emits a `RuntimeWarning` whenever a chain finishes without accepting a single
proposal instead of returning frozen samples silently.

### 3.8 Reproducibility

All randomness flows from one injectable `numpy.random.Generator` (`rng=`
parameters throughout). The original mixed global `np.random.*` state with
`scipy.stats.rvs` calls, so no run was reproducible. Determinism under a
fixed seed is now a test.

## 4. New features

* **`regime_diagnostic` + `window_bandwidth`** (`diagnostic.py`): the
  debias-or-smooth rule `fit_psd(mode="auto")` uses. It replaces the paper's
  Sec. IV-C sign rule (all-knots WLS on the window-convolved bases; debias
  only when every coefficient is positive), which false-negatives on small or
  noisy data (noise-driven negative coefficients; collinearity of
  window-convolved bases below the window bandwidth; approximation-error
  negatives on sharp spectra). The new rule (a) computes the spectral
  window's equivalent bandwidth from the kernel by Parseval and floors the
  diagnostic mesh there — the finest spacing the blur makes identifiable —
  and (b) decides by the non-negative-least-squares fit gap: debias iff a
  *positive* blur-matched representation fits essentially as well as the
  best unconstrained one (which is all the positivity-truncated sampler
  needs). `RegimeDiagnostic` reports the decision plus `gap_per_constraint`,
  `n_constrained`, `bandwidth_bins`, `knot_spacing`, and `condition_number`.
  On the calibration probes it recovers the paper's intended verdict in
  every bias-dominant case, including the sunspot series where the sign
  rule refused to debias. **Note the semantics of `fit_psd(mode="auto")`**:
  less conservative than the paper's sign rule; blur-consistent
  variance-dominant scenarios (e.g. Matern under multitaper) dispatch to
  debias, where the sampler's own model selection provides the smoothing.
* **`fit_psd(estimate, freqs, kernel=..., n_time=..., mode="auto")`**: the
  paper's full workflow in one call — diagnostic, regime dispatch, sampling,
  posterior summary.
* **Linear-scale results always** (`FitResult.mean/std/posterior_predictive`):
  smooth-mode draws are exponentiated per draw inside the predictive pass.
  The original returned log-scale posterior draws in log mode and left the
  exponentiation (and the Jensen-inequality subtlety of exponentiating a mean)
  to the caller.
* **`sample_ar`** (`simulate.py`): AR(p) simulation via `scipy.signal.lfilter`
  with the same polynomial convention as `analytic.ar_spectrum`, replacing the
  `statsmodels` dependency used by the original's scripts.
* **Streaming posterior summaries**: `store_predictive=False` (default)
  computes mean/std in one pass without materializing the
  `(n_kept·n_draws, F)` draw matrix; opt in to keep the draws.

## 5. Removed

* `probfuncs.py` — pure-Python duplicates of the JIT posterior that the live
  code never called (and that carried the Section 2.1 confusion in its
  parameter naming). Its role — an independent reference implementation —
  now belongs to the test suite (`tests/conftest.py::reference_log_posterior`).
* `plotting.py` — raised `NameError` on any call (`np`, `plt` and three
  module globals undefined). Plotting stays in notebooks; the library no
  longer imports (or depends on) `matplotlib`.
* `util.bochner` — dead code superseded by `get_exact_rfft_psd`; it also
  mutated its input array in place.
* `welch(nfft=)` — accepted and ignored.
* `__init__.__all__` entries `"MCMC"` and `"daniell_lag_window"` — neither
  existed; `from auto_speccy import *` raised `AttributeError`. The new
  `__all__` is exact.
* Unused `MH` attributes (`sigma2s`, `mean`, `mode`, `std`, `probs`), unused
  imports, the stale 3-vs-4-value return annotation on the kernel, and the
  `tqdm` progress dependency.

## 6. Behavioral notes (deliberate, visible differences)

* **`periodogram` now divides by `fs`** like the other three estimators; the
  original omitted the scaling only there. All four estimators satisfy a
  Parseval check in the tests.
* **Estimators return `SpectralEstimate(freqs, psd, kernel)`** — a NamedTuple
  (tuple-unpacking compatible with the original's 3-tuples), with the bias
  kernel documented as a first-class output rather than an undocumented third
  element (the original README still described 2-tuples).
* **`c` defaults to `len(estimate)`** (the paper's `c = g`). The original had
  no default and its simulation scripts passed the *time-series* length.
  Pass `c=` explicitly to reproduce script-style behavior.
* **dwelch API**: knot configuration is `gamma` over the interior candidates
  `[0, freqs, 0.5]`, plus explicit `kernel`/`n_time` keywords. The B1
  assembly now routes through the same `assemble_design` machinery as the
  sampler. Note: the original's `A[1:] + B[:-1]` + `clip(0,1)` construction
  was *verified correct* for nonuniform knots during this rewrite (tests
  compare both paths on uniform and nonuniform configurations to 1e-8) — the
  change is unification and removal of the clip patch, not a bug fix.
* **The sampler validates its domain**: `freqs` must be the endpoint-dropped
  rfft grid in normalized frequency (strictly inside (0, 0.5)). The original
  accepted any domain and failed obscurely later.

## 7. Packaging

* `requires-python >= 3.10` (the original declared `>= 3.8` while using PEP
  604 unions and builtin generics that need 3.10+).
* Dependencies cut from 6 to 3: `numpy`, `scipy`, `numba` (dropped
  `matplotlib`, `statsmodels`, `tqdm`). Floors: `numpy>=1.24`, `scipy>=1.10`,
  `numba>=0.61` (first release supporting Python 3.13, which this machine
  runs).
* Real test suite (`tests/`, 104+ tests; `pytest` runs fast tests by default,
  `pytest -m ""` includes the slow end-to-end studies). The original's README
  advertised a `tests/` directory and an `IMPROVEMENTS.md` that did not exist.
