# psdebias vs. auto-speccy — every change, and why

`psdebias` is a ground-up rewrite of `~/Desktop/Auto-Speccy` (`auto_speccy`),
the reference implementation of *"Adaptive Smoothing of Quadratic Spectral
Estimators"* (Dyson, Astfalck, Cripps & Stemler). The rewrite fixes two
statistical defects where the code diverged from the paper, replaces the
numerically fragile linear algebra, cuts sampler memory by ~65x, and
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
| `MH(Biasing=True)` | `PSDebias` | renamed; the `Biasing=False` smoothing path is gone (sec. 5b) |
| `MH(Basis_Resolution=, n_ts=, h_conv=)` | `PSDebias(knot_spacing=, n_time=, kernel=)` | renamed |
| `MH.sample(n_gibbs_steps=, thinning=, api=, bpi=, asig=, bsig=)` | `PSDebias.sample(n_beta_draws=, thin=, a_pi=, b_pi=, a_sigma=, b_sigma=)` | renamed |
| `MH.get_mode_gamma` + `get_biased_mode_estimate` | `PSDebias.map_estimate` | merged + fixed |
| `MH.get_estimates` | `FitResult.mean/std/expected_knots` | replaced |
| `MH.get_{biased,unbiased}_design_matrix` | `PSDebias.design_matrix(gamma, biased=)` | merged |
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
| `dwelch.dwelch_b0/b1` | `dwelch.dquad` | rebuilt on shared assembly; B0 kept and renamed, B1 dropped (sec. 5b) |
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
On exhaustion the new code raises with a note that debiasing is likely not
appropriate for the estimate.

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

* **`fit_psd(estimate, n_time=...)`**: the full workflow in one call —
  sampler construction, sampling, posterior summary.
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

## 5b. Removed after the initial rewrite

* **Smooth mode (`mode="smooth"`) and the whole mode-dispatch machinery.**
  The package now serves only the bias-dominant regime, so `PSDebias` takes
  no `mode`; `kernel` and `n_time` are required. `FitResult.mode` is gone,
  and `_predictive_pass` no longer carries the log-scale/`exp` branch or the
  lognormal standard-deviation mapping. `_mh_kernel` lost its
  `require_positive` flag — coefficient positivity is now always enforced.

* **The `dof` field on `SpectralEstimate`, with `_welch_dof` and
  `_lag_window_dof`.** These existed only to build the smooth-mode offset
  `log(dof) - digamma(dof)`, which corrected the downward bias of
  `E[log I(f)]`.

  That offset was a **numerical no-op**, and its removal changes no output.
  The B1 hats over the active knots are a partition of unity, so the constant
  vector lies exactly in the design's column span (verified to 2.2e-16 over
  random configurations; pinned by
  `tests/test_splines.py::TestPartitionOfUnity`). Adding a constant to the
  response therefore shifted the fitted values by exactly that constant and
  left ESS, the posterior variance and the coefficient draws untouched — and
  the code then subtracted the same constant back at output
  (`exp(mu - offset)`). The correction cancelled itself.

* **`corr` parameters and the correlation-matrix machinery** on `welch`,
  `lag_window` and `multitaper` (`corr_mat`, `log_corr_mat`,
  `_kibble_log_corr`, `_lag_window_corr_mats`): accepted and ignored.

* **`mode="auto"` and the regime diagnostic** (`diagnostic.regime_diagnostic`,
  the bandwidth-matched NNLS-gap rule). The module was already absent; the
  stale `mode="auto"` end-to-end tests and the ~220-line HOW_IT_WORKS section
  documenting it have now been removed too.

* **`splines.knot_grid(ghost_knots=)`** — always `True` without smooth mode.
  It now returns `(knots, halve_idx)` rather than the same interior-index
  array under two names.

* **`dwelch_b0` renamed to `dquad`, and `dwelch_b1` removed.** The B1
  fixed-mesh variant had no caller in the library, the study scripts or the
  notebooks. `dquad` returns the debiased spectrum alone, not `(freqs, psd)`
  — `freqs` was the caller's own input echoed back. The auto-speccy
  construction oracle that cross-checked the B1 assembly
  (`tests/test_dwelch.py::original_dwelch_b1`) went with it; the equivalent
  guarantee for the *sampler's* B1 path is still pinned by
  `test_splines.py::test_unbiased_equals_direct_evaluation` and
  `test_design.py`.

* **`trace/` and the generated HTML copies of the docs** (`README.html`,
  `docs/*.html`).

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

## 8. Frequency-correlation matrix restored, and `dquad` gains DQuad Eq. 12

The original's correlation-matrix parameters were *accepted and ignored*, and
§5b removed them. They are now back, computed for real, and actually used.

### `SpectralEstimate` gains a fourth field, `corr`

The one-sided frequency-correlation sequence `rho[d]` between spectral estimates
`d` bins apart, alongside `freqs`, `psd` and `kernel`. It is DQuad Eq. 8 /
spectral-covariance note Eq. 10, specialised per estimator:

| estimator | `rho` | source |
|---|---|---|
| `periodogram` | `abs(rfft(taper**2))**2` | `H_2(eta)`, note Part I |
| `welch` | `sum_d (2 - delta_d0)(1 - d/M) abs(rfft(p_d))**2` | note Eq. 31 |
| `lag_window` | `psd_from_acf(kernel * win)` | corrected, see below |
| `multitaper` | `sum_{j,k} abs(rfft(h_j h_k))**2` | note Eq. 23 |

all normalised to `rho[0] = 1`. Stored as a sequence, not a matrix: `rho`
depends only on separation, so the length-g form is lossless while the dense one
is 8 MB at g = 1023 and 2 GB for a multitaper on a 32k-point series.
`.corr_matrix()` builds the dense form on demand. The lag-window case reuses
`util.psd_from_acf` verbatim; the rest are one batched `rfft` of short product
sequences. The general Eq. 8 route would need an `O(n^3)` eigendecomposition of
`Q` for a lag-window (`K = n` there, paper sec. 3.2), so the note's
per-estimator closed forms are what make this cheap.

### A third corrected defect: the lag-window `rho` (2026-08-24)

`lag_window` originally computed `rho` as `psd_from_acf(kernel**2)`, i.e. the
transform of `w^2[tau] (1 - tau/n)^2`. That is wrong by one triangular factor.

The estimator is the quadratic form `A[s, t] = w[|s-t|] cos(2 pi eta (s-t)) / n`
built from the **raw** window — the biased ACF's `(1 - |tau|/n)` taper is not in
`A` at all. It appears in the moments only because `n - |tau|` pairs `(s, t)`
sit on each diagonal: once in `E[I]` (which is what `kernel` is, and `kernel` is
unchanged), and once more in

```
Cov(I_i, I_j) = 2 tr(A_i A_j) = ( D(i - j) + D(i + j) ) / 2,
D(delta) = (1/n^2) sum_tau (n - |tau|) w^2[tau] cos(2 pi delta tau / n),
```

so the difference term carries `w^2 (1 - |tau|/n)` — first power. The corrected
code forms exactly that as `kernel * win`.

Error size, `n = 64`, max deviation of the full sequence from `D`:

| window | lag = 4 | lag = 8 | lag = 16 |
|---|---|---|---|
| bartlett | 8.8e-3 | 1.6e-2 | 3.2e-2 |
| parzen | 6.7e-3 | 1.1e-2 | 2.1e-2 |

With `kernel * win` the deviation is `<= 4.4e-16` in every case. The error is
`O(lag/n)`, which is why `test_lag_window_bandwidth_closed_form` (`lag = n/32`,
`rtol = 2e-2`) could not see it, and why it reaches a few percent at the
`lag = n/10` default.

This also means the code no longer follows the companion note's Eq. 17, which is
`sum_tau w^2[tau] cos(...)` with no triangular factor at all — the asymptotic
`lag/n -> 0` limit of the above. The exact finite-sample form is used instead;
the old source comment quoted Eq. 17 while the code did something third.

Only `lag_window(...).corr` changes. `periodogram`, `welch` and `multitaper` are
bit-identical, and were verified to be exactly the difference term of
`2 tr(A_i A_j)` (to 1e-15 relative) by the same oracle.

### The oracle that found it

`tests/conftest.py` gained `quadratic_form_{periodogram,welch,multitaper,
lag_window}` and `exact_bin_covariance`. Every estimator here is quadratic in
the data, so for Gaussian `x` Isserlis gives the covariance of two bins exactly,
with no asymptotics and no Monte Carlo. The new
`TestCorrelation::test_*_matches_exact_quadratic_form` assert

```
tr(A_i A_j) == c ( U(|i - j|) + U(i + j) )
```

for all four estimators, `U` being `rho`'s even `n`-periodic extension. The
pre-existing `reference_corr_*` oracles are independent implementations of the
*same* DQuad Eq. 8, so they pin the algebra but assume the formula; these pin
the formula against the definition of a correlation.

The `U(i + j)` term is what `toeplitz(rho)` drops — it is what makes the true
correlation matrix non-Toeplitz. It is negligible mid-band and material only
when both bins crowd DC or Nyquist;
`test_sum_frequency_term_is_the_toeplitz_error` pins both halves of that.

### `dquad` gains `corr=`, which selects DQuad Eq. 12

Implemented as the paper specifies it, not adapted:

* **Full two-sided Fourier grid.** `w_k = 2 pi k / n`, `k = -floor(n/2) ..
  ceil(n/2) - 1`. Reached by re-indexing the one-sided arrays, since both the
  estimate and the (real, even) bases take the same value at `+f` and `-f`.
* **W circulant, inverted by FFT** — the paper's sec. 3.2 `O(n log n)` claim.
  `W^-1` is never formed as a matrix.
* **Closed form, unconstrained.** `(X' W^-1 X) theta = X' W^-1 1`, solved
  directly. No NNLS on this path.

`Gamma^-1` was already the `1/estimate` weighting, so it is only `W^-1` that is
new. **`corr=None` is the default and reproduces the previous output
bit-for-bit** (verified by `assert_array_equal` on a fixed-seed fit); NNLS and
the positivity constraint still govern that path.

Because the Eq. 12 path mirrors onto the two-sided grid it needs the DC and
Nyquist bins, so it requires estimates built with `drop_endpoints=False` and
raises otherwise — those bins cannot be reconstructed once dropped. Its `gamma`
accordingly runs over `freqs[1:-1]`, since `freqs[0] = 0` and `freqs[-1] = 1/2`
already sit on the ghost knots.

One note on the paper's wording: it states `V` is circulant, but `V = Gamma W
Gamma` with `Gamma_ii = I(w_i)` varying is a diagonal-scaled circulant. The
algorithm is unaffected — `V^-1 = Gamma^-1 W^-1 Gamma^-1` and the `Gamma`
factors fold into the design and response — but the FFT-inverted object is `W`.

### A defect the full grid exposed

`splines.b0_box` returns 1/2 exactly on a box boundary, which is what makes two
boxes sharing an interior knot sum to 1 there. On the two-sided grid the ghost
knots **are** grid points, and at `f = 0` and `f = 1/2` no neighbouring box
supplies the other half — so the debiased spectrum came out at exactly half
value in those two bins. Corrected in `dquad`, and only for the output design:
`basis_b0_biased` goes through `acf_box` and `psd_from_acf`, which evaluate no
boundaries. This is the B0 counterpart of the `EMPTY_IDX` / `halve_idx` split
the sampler keeps between its two designs. Pinned by
`test_dwelch.py::TestDquadGLS::test_flat_spectrum_recovered_including_endpoints`.

### Where Eq. 12 holds up, and where it does not

Median log-MSE over 6 AR(4) realisations, B0 mesh every 16 bins (4 for Welch),
against the diagonally weighted NNLS fit:

| estimator | `lam_min(W)` | cond | WLS | Eq. 12 | negative bins |
|---|---|---|---|---|---|
| welch, rect, 50%, `L=512` | 6.7e-1 | 2.0 | 6.455 | 6.612 | 0 |
| welch, Hann, 50%, `L=512` | 1.6e-1 | 13 | 0.041 | 0.054 | 0 |
| multitaper `nw=4 K=8` | 7.9e-3 | 1.0e3 | 43.72 | 198.7 | 326 of 1023 |
| lag-window, Bartlett `lag=n/8` | **-1.8e-15** | 7e15 | 4.908 | **undefined** | — |

Welch — the paper's own primary case — is well conditioned and behaves. The
other two are the limits of Eq. 12 as written:

* **Multitaper solves but goes negative** across a third of the band. Eq. 12 is
  unconstrained, so this is the estimator behaving as specified, not a failure.
  Pinned by `test_unconstrained_solution_may_go_negative`.
* **Lag-window is singular.** Eq. 12 assumes W positive definite; a Bartlett lag
  window's correlation is wide enough that the circulant's smallest eigenvalue
  lands at -1.8e-15, and dividing by it makes the closed form NaN. `dquad` now
  raises with the eigenvalues named rather than returning NaN. That guard is a
  diagnostic and changes no result Eq. 12 actually defines.

The paper's own simulation avoids both: a modified Daniell of width `M = 2^5` at
`n = 2^14` is far narrower than the grid, and `S = O(n^(1/3))` caps the bases at
~25. Reproducing the paper's settings rather than this package's defaults is the
way to see Eq. 12 at its best.

Rectangular tapers additionally give `W = I` exactly — `h_t^2 = 1/n` makes `H_2`
the Dirichlet kernel, which vanishes at every nonzero Fourier-grid separation.
`dquad` in `scripts/generate_realisations.py` is deliberately left on the
diagonal weighting so the cached `scripts/realisations/` tree stays comparable.

`PSDebias` is untouched: the sampler still uses the diagonal weighting.
