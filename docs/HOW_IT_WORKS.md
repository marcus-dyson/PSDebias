# How psdebias works — a guided tour of the code

This document walks through the implementation for someone who knows the
paper (*Adaptive Smoothing of Quadratic Spectral Estimators*) but has not
read this codebase. It explains what each piece does, why it is built the way
it is, and where the paper's equations live in the source. Inline comments in
the source cover the same ground at the line level; this is the map.

Contents:

1. [The 30-second picture](#1-the-30-second-picture)
2. [The regression](#2-the-regression)
3. [Building the bases](#3-building-the-bases)
4. [The sampler](#4-the-sampler)
5. [Outputs](#5-outputs)
6. [Numerical engineering appendix](#6-numerical-engineering-appendix)
7. [Suggested reading order and paper cross-reference](#7-suggested-reading-order-and-paper-cross-reference)
8. [Where the tests pin behavior](#8-where-the-tests-pin-behavior)
9. [The bias-variance study (scripts and notebook)](#9-the-bias-variance-study-scripts-and-notebook)

---

## 1. The 30-second picture

```
 time series x
      |
      v
 estimators.py  ->  (freqs, psd, kernel)          kernel = taper autocorrelation
      |                                            = the estimator's bias, as a sequence
      v
 sampler.PSDebias(...)           builds bases ONCE   (splines.half_bases[_biased])
      .sample()                  runs MH over knot configurations (_mh_kernel)
      |
      v
 FitResult: mean, std, gammas, betas, acceptance_rate    (linear-scale PSD)
 PSDebias.map_estimate(): most probable single configuration
```

File map (all under `src/psdebias/`):

| file | one line |
|---|---|
| `estimators.py` | periodogram / Welch / lag-window / multitaper; each also returns its bias `kernel` |
| `util.py` | `psd_from_acf` (lag-domain -> rfft-grid transform, the workhorse), grid helpers |
| `splines.py` | everything about bases: candidate knots, primitives, closed-form ACFs, design-matrix assembly + incremental update |
| `sampler.py` | the Numba MH kernel, the `PSDebias` class, `fit_psd` |
| `dwelch.py` | fixed-mesh debiasing baseline `dquad` (NNLS instead of a sampler) |
| `analytic.py`, `simulate.py` | closed-form spectra and process simulators for validation |

## 2. The regression

Everything the sampler does is Bayesian variable selection on **one** linear
model, `Y = X_gamma beta + eps` (paper Eq. 3). What `Y` and `X` are is decided
once, in `PSDebias.__init__`:

| | |
|---|---|
| knots | `[0, freqs[pos], 0.5]` (ghosts, off-grid) |
| response `Y` | `ones(g)` |
| fitting bases | window-convolved, scaled by `1/estimate` |
| positivity of `beta_hat` | required (truncated posterior) |
| output bases | unconvolved B1 |

Two things worth internalizing:

* **Why the response is `ones`.** The paper's Eq. 5 regression is
  `I(w) ~ sum_i beta_i (B_i * H_n)(w)`. Dividing both sides by `I(w)` makes
  the response the constant 1 and moves the `1/I` factor onto the bases. That
  is precisely the variance-stabilizing weighting (periodogram-type estimates
  have standard deviation proportional to their mean), implemented by
  multiplying the biased primitive rows by `weights = 1/estimate`.
* **Why ghost knots at 0 and 0.5.** The spectral window's tails wrap power
  around DC and Nyquist. The basis must be able to place power there even
  though those bins are dropped from the data grid, so `splines.knot_grid`
  appends two off-grid knots at exactly 0 and 1/2. They are always active —
  only the on-grid candidates are selectable.

## 3. Building the bases

### 3.1 Candidate knots vs active knots

`splines.knot_grid` picks ~`len(freqs)/knot_spacing` **candidate** positions
(uniform, or geometric with `log_knots=True`). The latent vector `gamma`
(one 0/1 per *interior* candidate) says which are **active**; the two end
knots are always active. Candidates are carried as *integer grid positions*
(`interior_idx`), never rediscovered by comparing floats.

### 3.2 Primitives: two ramps per candidate segment

For every candidate segment `[knots[s], knots[s+1]]` there are two linear
primitives: `falling_s` (1 at the left end, 0 at the right) and `rising_s`
(the reverse). Any B1 hat on any *active* knot triple is a linear combination
of these — that is the whole trick. Two primitive tables exist:

* `half_bases` — the ramps evaluated directly on the frequency grid;
* `half_bases_biased` — each ramp's **spectral-window-convolved PSD**.

### 3.3 The convolution, paid once

`half_bases_biased` is the only expensive computation in the package, and it
runs once per `PSDebias` construction:

1. `acf_rising/acf_falling` give each ramp's inverse Fourier transform at
   integer lags **in closed form** (paper Eq. 8; the `sinc`/`cos` expressions
   come from integrating `(m x + b) cos(2 pi x tau)` by parts — derivation in
   the `_ramp_acf` comments).
2. Multiplying that ACF by the estimator's `kernel` (the taper
   autocorrelation) **is** the convolution with the spectral window — the two
   operations are a Fourier pair.
3. `util.psd_from_acf` folds negative lags onto the circular lag buffer
   (`r[-tau]` aliases to index `N - tau`) and takes one batched `rfft`,
   landing every primitive **exactly** on the data's frequency grid. No
   interpolation, no truncation error; `tests/test_util.py` checks the fold
   is bit-for-bit the original package's loop, and `tests/test_splines.py`
   checks the result against brute-force numerical convolution.

After this step, every design matrix ever needed — including one per MCMC
proposal — is assembled by linear combination alone. Convolution is linear,
so "combine convolved primitives" equals "convolve the combined hat".

### 3.4 Assembling a design matrix: slope and pivot

`assemble_design` builds one column per active knot. Each side of a hat is a
line `h(x) = slope * (x - pivot)`; restricted to one candidate segment, a
line is exactly `h(left) * falling_s + h(right) * rising_s`. So a column is a
short sum over the candidate segments its hat spans:

```
active knots a < c < d   (hat centred at c, possibly spanning
                          several candidate segments per side)
rising side  (a -> c): slope =  1/(c-a), pivot = a
falling side (c -> d): slope = -1/(d-c), pivot = d
edge half-hats: one side only
```

Worked micro-example: active knots `.1 < .2 < .4` with an inactive candidate
at `.3`. The hat at `.2` falls over two candidate segments `[.2,.3]` and
`[.3,.4]`; on `[.3,.4]` its contribution is
`h(.3)*falling + h(.4)*rising = 0.5*falling + 0*rising` — the correct linear
piece, using only precomputed `[.3,.4]` primitives.

### 3.5 The `halve_idx` correction

The direct-evaluation primitives are boundary-inclusive: at a grid point that
*is* a candidate boundary, `rising_{s-1}` and `falling_s` are both 1, so the
assembly counts the hat's value twice there (and only there). Halving the
assembled rows at the interior-candidate grid positions (`halve_idx`) makes
the design equal the **exact** B1 evaluation everywhere —
`tests/test_splines.py::test_unbiased_equals_direct_evaluation` asserts this
for uniform and random nonuniform configurations. Biased primitives are
smooth curves with no boundary evaluation, so they take an empty `halve_idx`.
This one argument is what replaced four near-duplicate functions in the
original package.

### 3.6 Incremental updates

`update_design` gives each column an identity: the candidate-grid indices of
its (left neighbour, centre, right neighbour) active knots, with sentinels
for the two edge columns. A proposal flips `n_select = 2` knots, so almost
every column's triple — hence the column itself — is unchanged and gets
copied; only O(1) columns are recomputed. This is what keeps the per-proposal
cost at "a couple of column builds + a small least squares" instead of a full
rebuild. `tests/test_design.py` walks 200 random flip pairs and demands
bit-for-bit equality with full assembly.

## 4. The sampler

### 4.1 The target (paper Appendix B)

For a configuration `gamma` with `q` active interior knots, `p = q + 2`
design columns, and `g = len(Y)` frequency bins:

```
log pi(gamma | Y) =  -(p/2) log(1+c)                        dimension penalty (g-prior)
                     -(g/2 + a_sigma) log(ESS + 2 b_sigma)  evidence / fit reward
                     + betaln(q + a_pi, L - q + b_pi)       Beta-Binomial prior
                     - betaln(a_pi, b_pi)
```

with `ESS = Y'Y - Y'X beta_hat`. Hyperparameter intuition:

* `c` (default `g`, the paper's choice): bigger `c` = flatter coefficient
  prior = each extra basis must "pay" more evidence. It also shrinks the
  coefficient draws by `c/(1+c)` (barely, at `c ~ 500`).
* `a_pi, b_pi`: prior on how many knots you expect; `(1,1)` is uniform over
  model size.
* `a_sigma, b_sigma`: near-flat IG prior on the noise variance; `b_sigma`'s
  practical job is keeping `log(ESS + 2 b_sigma)` finite for perfect fits.

Everything is computed by `_log_posterior` — one compiled function used by
*both* the chain and `map_estimate`, so the sampler and the mode selector can
never disagree (the original package had two divergent implementations; see
CHANGES.md sections 2.1–2.2 for what was wrong and why results shifted).

### 4.2 The MH loop (`_mh_kernel`), annotated

```
state: (gamma, design X, chol(X'X), beta_hat, ESS, log_post, p)

for i in warmup + n_iterations:
    gamma' = gamma with n_select positions set to fresh Bernoulli(1/2)   # symmetric proposal
    X'     = update_design(...)                # copies unchanged columns
    solve  = _least_squares(X', Y)             # Gram -> Cholesky -> beta_hat', ESS'
    reject if: Cholesky failed (singular)      # flag, not exception
            or any beta_hat' <= 0              # positivity truncation, Alg. 1 line 5
    else Metropolis: accept iff min(0, dlog_post) > log U[i]
    on accept: state <- primed versions        # nothing recomputed from scratch

    every thin-th post-warmup iteration:
        store (gamma, ESS)
        draw n_beta_draws of (sigma^2, beta):
            sigma^2 = (ESS + 2 b_sigma) / chi2_{g + 2 a_sigma}
            beta    = beta_hat + sqrt(sigma^2 * c/(1+c)) * L^{-T} z
```

The conditional draw is the paper's multivariate-t coefficient posterior
sampled by composition: `sigma^2` from its inverse-gamma conditional, then
`beta | sigma^2` Gaussian. `L^{-T} z` (one triangular solve, no inversion)
has covariance `(X'X)^{-1}`. `tests/test_sampler.py::TestConditionalDraws`
freezes the chain on one configuration and verifies the empirical mean and
covariance against the analytic values.

Mechanics worth knowing:

* **All randomness is pre-generated** in `sample()` from the single injected
  `numpy.random.Generator` and consumed sequentially by the kernel. That is
  what makes runs bit-reproducible under a seed (Numba's internal RNG cannot
  be driven by a `Generator`).
* **Thinning happens inside the kernel.** Only every `thin`-th state is
  stored, and the conditional-draw randoms exist only for kept states. At
  paper-scale settings this is the difference between ~40 MB and ~2.6 GB.
* **Storage is rectangular and NaN-padded**: `betas[i, d, :p_i]` are real,
  the rest NaN, `p_i = gammas[i].sum() + 2`. No ragged lists in compiled code.

### 4.3 Initialization (and the sunspot lesson)

In debias mode the posterior is **truncated** to configurations whose
least-squares coefficients are all positive. The chain must therefore *start*
inside that region: from a dense configuration with negative coefficients, a
two-knot flip essentially never lands all-positive, so every proposal is
rejected and the chain freezes at its start forever. This is not
hypothetical — an earlier version of this package initialized at
"all knots active" and froze exactly this way on the annual sunspot dataset
(acceptance 0.0; the regression test
`test_end_to_end.py::test_sunspot_forced_debias_regression` pins the fix).

`_prior_initial_gamma` therefore samples the start from the prior itself —
`pi ~ Beta(a_pi, b_pi)`, `gamma_i ~ Bernoulli(pi)` — retrying until the
unconstrained WLS fit is all-positive. Small `pi` draws give sparse, wide-hat
configurations, which pass easily; on sunspot data this needs a handful of
attempts. If a chain still finishes without a single acceptance, `sample()`
now emits a `RuntimeWarning` rather than returning frozen samples silently.

## 5. Outputs

`_predictive_pass` converts stored `(gamma, beta)` samples into spectra:

* Evaluation always uses the **unconvolved** bases — the coefficients
  describe the spectrum *before* the window blurred it; reading them off
  clean hats is the debiasing.
* Mean and standard deviation accumulate streamingly; the full
  `(n_kept * n_beta_draws, F)` draw matrix is materialized only with
  `store_predictive=True`.
* `map_estimate()` re-scores every stored `(gamma, ESS)` with the same
  `_log_posterior` as the chain and returns the winner's averaged curve —
  the paper's "Adapt Mode" estimate, vs. the posterior mean's "Adapt Mean".

## 6. Numerical engineering appendix

**Why a hand-rolled Cholesky returning a flag.** The kernel needs to treat a
singular Gram matrix (duplicate/degenerate knot placements) as an ordinary
proposal rejection. `try/except` around LAPACK inside `@njit` is fragile and
slow, so `_cholesky` is ~30 lines of scalar code that returns `False` on a
failed pivot (relative floor `1e-12 * max diag`). β̂, ESS, and the
coefficient draws all reuse the factor through `_solve_lower/_solve_upper`.
Verified against `np.linalg.lstsq` to 1e-9; at p ~ 10–30 the O(p^3) cost is
microseconds.

**Numba constraints that shaped the kernel:** no exceptions in the hot loop,
no reflected/typed lists (rectangular NaN-padded outputs instead), randoms
pre-generated (seedable), contiguous arrays throughout. The scalar loops look
old-fashioned but compile to the same SIMD the vector forms would.

**Cost model per iteration:** one incremental design update (O(1) column
builds of length F) + one Gram build (O(F p^2), the dominant term) + one
Cholesky (O(p^3)). Measured ~24k iterations/sec at F=511, L=63 on this
machine — parity with the original package's kernel, with ~65x less RNG/
storage memory (see CHANGES.md 3.2).

**Where the time really goes end-to-end:** basis precomputation is one
batched FFT (milliseconds); the MH loop is seconds; the predictive pass over
kept states is comparable to the loop when `store_predictive=True`.

## 7. Suggested reading order and paper cross-reference

Reading order that builds up dependencies naturally:

1. `util.py::psd_from_acf` (10 lines that everything else leans on)
2. `estimators.py::_taper_kernel` + one estimator (what `kernel` means)
3. `splines.py` top-of-module docstring, then `knot_grid`, `_ramp_acf`,
   `half_bases_biased`, `_hat_column`, `_compute_column`, `assemble_design`,
   `update_design`
4. `sampler.py::_log_posterior`, `_least_squares`, then `_mh_kernel`
   top-to-bottom, then `PSDebias.__init__`/`sample`
5. `dwelch.py` (short, reuses everything above)

| paper | code |
|---|---|
| Eq. 1–2 (bias = convolution with `H_n`) | `estimators._taper_kernel`, applied in `util.psd_from_acf` |
| Eq. 3 (linear model per `gamma`) | `sampler._least_squares` |
| Eq. 4–5 (spline family, convolved fit) | `PSDebias.__init__`; `dwelch.py` |
| Eq. 6–7 (B0/B1 from latent `gamma`) | `splines.assemble_design` (+ `basis_b0*`) |
| Eq. 8 (primitive inverse FTs) | `splines._ramp_acf`, `acf_rising/falling/box` |
| Eq. 9 (chained linear combinations) | `splines._hat_column` slope/pivot weights |
| Eq. 10 (Beta-Binomial prior) | `sampler._log_posterior` prior term |
| App. B marginal likelihood | `sampler._log_posterior` likelihood terms |
| App. B multivariate-t coefficients | conditional draws in `_mh_kernel` |
| Algorithm 1 | `sampler._mh_kernel` |
| Sec. IV-B debiasing | `PSDebias` |

## 8. Where the tests pin behavior

Use these as the safety net when modifying code — each one guards a specific
invariant:

| test | invariant |
|---|---|
| `test_util.py::test_matches_original_loop_exactly` | `psd_from_acf` == original auto-speccy loop, bit-for-bit |
| `test_splines.py::test_*_vs_quadrature` | closed-form ACFs == numerical integrals (1e-8) |
| `test_splines.py::test_unbiased_equals_direct_evaluation` | assembled design == exact B1 evaluation (incl. `halve_idx`) |
| `test_splines.py::test_biased_column_vs_numerical_convolution` | biased bases == dense convolution with the spectral window |
| `test_splines.py::TestPartitionOfUnity` | B1 columns sum to 1 at every grid point |
| `test_design.py::test_incremental_equals_full_random_walk` | `update_design` == `assemble_design`, bit-for-bit, 200 random flips |
| `test_sampler.py::test_log_posterior_vs_reference` | compiled posterior == independent scipy implementation |
| `test_sampler.py::TestConditionalDraws` | draw moments == analytic Student-t posterior |
| `test_sampler.py::test_determinism` | identical seeds give identical `FitResult`s |
| `test_sampler.py::test_map_estimate_uses_kernel_posterior` | MAP selector and chain share one posterior |
| `test_sampler.py::TestDebiasInitialization` | debias init lands inside the positivity support (sunspot data) |
| `test_dwelch.py::TestDquad` | fixed-mesh `dquad` recovers a flat spectrum and beats raw Welch in the biased tail |
| `test_end_to_end.py` (slow) | MSE improvement vs raw estimators on analytic truth; sunspot chain mixes |

## 9. The bias-variance study (scripts and notebook)

Everything outside `src/` reproduces the paper's simulation study: how much
does adaptive debiasing improve on the raw estimators and the fixed-mesh
baseline? Two pieces, run in order:

```
 scripts/generate_realisations.py     -> realisations/{ar4,matern}_debiasing/
 notebooks/bias-variance.ipynb        -> MSE/Bias/Var tables + figures
```

These need the script extras: `pip install -e '.[scripts]'` (tqdm,
matplotlib, scienceplots — none are library dependencies).

### 9.1 The configuration

All knobs live in the single `Config` dataclass in `generate_realisations.py`:

| | |
|---|---|
| Welch | L=1024, rectangular taper, few long segments |
| lag-window | bartlett, NW ∈ {3,4,6,8}, N=2^11 |
| knot spacing | 4 (Welch) / 8 (quad) |
| fixed-mesh baseline | `dquad` at the same mesh |

Short segments and rectangular tapers make the estimates bias-dominant —
heavy spectral leakage, where the window blur is the story. Four pipeline
stages run per (process, estimator, setting) cell:

1. **samples** — 200 realisations of AR(4) (paper coefficients) and Matern
   (alpha=1.5, lambda=0.1), cached as `.npy`;
2. **estimates** — the classical estimator per realisation
   (`welch/{m}.npy`, `lag-window/NW{n}.npy`, `multitaper/NW{n}.npy`);
3. **fixed-mesh debias** — `dquad` on a uniform mesh with one knot per
   `knot_spacing` bins, i.e. *the same mesh the sampler selects from*, so
   Dquad-vs-PSDebias isolates the value of adaptive knot selection rather
   than mesh resolution. Realisations whose NNLS fails or whose debiased
   spectrum is not strictly positive are dropped, so these stacks can have
   fewer (even zero) rows;
4. **adaptive fits** — one `PSDebias` chain per realisation (50k iterations,
   30k warmup, thin 10, `c = n` the series length as in the original
   scripts), summarised to one CSV row:
   `[E[#knots]] + posterior mean S(f) + posterior std S(f)`, linear scale.

Stages 1–3 cache whole arrays (`load_or_compute`: file exists = skip).
Stage 4 appends to its CSV row by row and flushes after each, so an
interrupted run resumes exactly where it stopped — the resume position is
just the current line count.

### 9.2 Reproducibility: one RNG stream per cell

There is no global seed. Every random draw comes from
`stream_rng(cfg, tag, i, attempt)` =
`np.random.default_rng([seed, crc32(tag), i, attempt])` — a deterministic
stream per (purpose, realisation, attempt). Two consequences:

* **Resume-safe**: a resumed CSV re-derives the identical stream for every
  row, so an interrupted-and-resumed run is bit-identical to an
  uninterrupted one (verified during development by truncating a CSV and
  diffing the regenerated row).
* **Prefix-nested Welch samples**: the Welch sample tag omits the segment
  count, so the m=8 and m=64 samples consume the same underlying noise —
  shorter records are truncations of the same realisations, not independent
  redraws. That reproduces the original study's shared-long-draw design and
  removes cross-m Monte Carlo noise from the "MSE vs number of segments"
  curves.

### 9.3 The notebook

`notebooks/bias-variance.ipynb` is one loader/stats/plot pipeline driven by
a `METHODS` config table (file patterns, x-axis) crossed with signal —
every (estimator × process) combination the original notebook duplicated by
hand. Per realisation and stream it computes, in the
log domain against the closed-form truth,

```
MSE  = mean_f (log Ŝ - log S)²      Bias = mean_f (log Ŝ - log S)
Var  = MSE - Bias²
```

then reports each stream's MSE / Bias² / Var as ratios to the raw estimator
(< 1 = improvement) and plots mean ± std bands across realisations. Two
details worth knowing:

* **Everything is logged uniformly at stats time.** All on-disk streams are
  linear-scale S(f) — raw estimates, adaptive CSV means, Dquad — so the
  loader applies no per-stream log juggling.
* **Dquad stacks are ragged.** The positivity filter of stage 3 means a
  setting can have any number of surviving rows, including zero; the stats
  keep per-setting lists instead of one rectangular array and report NaN
  where nothing survived (rendered as gaps in the plots).

The final rank bar chart is hand-recorded from the paper's write-up, not
derived from the arrays — it is labeled as such in the notebook.

### 9.4 Differences from the retired auto-speccy scripts

The scripts this study replaces lived in the original repository and used
the old `MH` sampler. Regenerated numbers will not match historical runs,
deliberately:

| change | why |
|---|---|
| corrected posterior (CHANGES.md sec. 2) | the chain now targets the paper's marginal likelihood |
| Welch step = L | v1 stepped L−1, an accidental 1-sample overlap |
| Dquad mesh = sampler `knot_spacing` | the res0/res1 grids were never actually distinct in v1's output; only the spacing-matched comparison is of interest |
| per-stream seeded RNG | the originals mixed global `np.random` state; no run was reproducible |
