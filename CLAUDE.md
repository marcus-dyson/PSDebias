# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

There is no bare `python` on PATH. Use the venv interpreter explicitly:

```bash
.venv/bin/python -c "import psdebias; print(psdebias.__version__)"
```

`psdebias` is installed editable from `src/psdebias`, so source edits take effect without reinstalling.

## Commands

Fast tests (~2 s; `addopts = "-m 'not slow'"` deselects the end-to-end suite):

```bash
.venv/bin/python -m pytest -q
```

Everything, including the 4 slow end-to-end MCMC studies (~5 min):

```bash
.venv/bin/python -m pytest -m "" -q
```

A single test:

```bash
.venv/bin/python -m pytest -q "tests/test_sampler.py::TestPSDebias::test_determinism"
```

The bias-variance study (needs `pip install -e '.[scripts]'`). Always point `--base-path` somewhere scratch when smoke-testing — the default writes into the cached `scripts/realisations/` tree:

```bash
.venv/bin/python scripts/generate_realisations.py --n-samples 2 --n-iterations 200 --warmup 50 --workers 1 --base-path /tmp/smoke
```

Debugging the numba kernels — set this **before** importing `psdebias`, otherwise the JIT-compiled internals are opaque to `pdb` and tracebacks:

```bash
NUMBA_DISABLE_JIT=1 .venv/bin/python your_script.py
```

## Architecture

The method: a quadratic spectral estimator blurs the true spectrum by convolving it with the estimator's spectral window. That window is fully described by a one-sided **bias kernel** `h[tau]` (a taper autocorrelation). PSDebias fits B1-spline bases *that have been convolved with the same window* to the blurred estimate, then evaluates the **unconvolved** bases at the fitted coefficients — which inverts the blur. Which knots are active is learned by Metropolis-Hastings over a latent binary vector `gamma`, with the regression coefficients and noise variance integrated out in closed form.

Data flow: `estimators.py` → `(freqs, psd, kernel)` → `splines.py` builds bases → `sampler.py` runs MH → `FitResult`.

There is only one regime. Smooth/log-scale mode, `mode="auto"`, and the regime diagnostic were all removed (see `docs/CHANGES.md` §5b); `PSDebias` takes no `mode`, and `kernel`/`n_time` are required.

### The load-bearing ideas

**The convolution is paid once.** `splines.half_bases_biased` builds the closed-form ACF of every linear primitive, multiplies by `kernel` in the lag domain (which *is* frequency-domain convolution with the spectral window — `util.psd_from_acf`), and transforms all of them in a single batched `rfft`. Every design matrix afterwards, including each of the ~10⁵ MCMC proposals, is a pure linear combination of those precomputed rows. Understanding this is the difference between reading `splines.py` as reasonable and reading it as baroque.

**Primitives, not hats.** Bases are never built directly. Each candidate segment carries a `falling`/`rising` ramp pair, shape `(K-1, F)`. A hat column is assembled from them by `_hat_column` using a `(slope, pivot)` encoding of each straight-line piece. Because convolution is linear, the *same* weights apply whether the primitives hold direct grid evaluations or their window-convolved PSDs — that single fact is what lets one precomputed table serve every proposal.

**Two design matrices, and they are not interchangeable.** `PSDebias` carries `_falling_fit`/`_rising_fit` (window-convolved, scaled by `1/estimate`) and `_falling_out`/`_rising_out` (unconvolved). The chain's ESS comes from the *fit* design; the spectrum is read off the *output* design. Anything re-deriving an ESS or a posterior score must use `design_matrix(gamma, biased=True)`. A test that got this wrong survived for a long time because the two designs happened to be identical in the deleted smooth mode.

**`halve_idx` differs between the two.** Direct-evaluation primitives are boundary-inclusive, so at a grid point that coincides with a candidate knot, two adjacent segments each contribute the full hat value. Halving those rows makes the assembly equal the exact B1 evaluation. Convolved primitives are smooth PSD curves with no boundary evaluation, so they take `splines.EMPTY_IDX`. Passing the wrong one is silent and wrong.

**Ghost knots.** `knot_grid` returns candidates `[0.0, freqs[pos], 0.5]`. The window's tails wrap power around DC and Nyquist, so the basis must span all of [0, 1/2] even though those bins are dropped from the data grid. The two ghosts sit off-grid and are always active; only the on-grid candidates are selectable.

**Positivity truncation.** The chain's target is restricted to configurations whose least-squares coefficients are all positive. This means the *initial* state must already be inside the support, or every proposal is rejected and the chain freezes — a real bug once hit on the sunspot series. `_prior_initial_gamma` resamples from the Beta-Bernoulli prior until it finds a valid start; `tests/test_sampler.py::TestDebiasInitialization` guards this.

**Reproducibility.** Numba's RNG cannot be driven by a `numpy.random.Generator`, so *all* randomness is pre-generated as arrays in `sample()` and consumed sequentially by `_mh_kernel`. This is why runs are bit-reproducible under a seed, and why the kernel's signature is so wide. Do not introduce RNG calls inside the compiled kernel.

**Storage layout.** `betas` is a rectangular NaN-padded `(n_kept, n_beta_draws, L+2)` array; row `i` uses its first `gammas[i].sum() + 2` slots. Thinning happens inside the kernel and conditional-draw randoms exist only for kept states — at paper-scale settings that is ~40 MB instead of ~2.6 GB.

**Partition of unity.** B1 hats over the active knots sum to 1 at every grid point, so the constant vector lies exactly in the design's column span. Consequence: adding a constant to the response shifts the fitted values by exactly that constant and changes nothing else — a former log-scale offset correction cancelled itself out for precisely this reason. Pinned by `tests/test_splines.py::TestPartitionOfUnity`.

### Conventions

Work in **normalized frequency** (`fs = 1`). `freqs` must be the endpoint-dropped rfft grid, strictly inside (0, 0.5) — the closed-form basis ACFs are derived on that grid, and `PSDebias` rejects anything else. `util.freq_slice` errors rather than guessing an alignment.

`c` (the g-prior scale) defaults to `g`, the number of frequency bins, per the paper. The study script deliberately passes `c = n_series` instead. Both are defensible; the divergence is documented in `docs/CHANGES.md` §6 and is easy to trip over when reproducing figures.

## Testing approach

Tests are oracle-based rather than snapshot-based, and that is the point — most of them re-derive the answer by an independent route. `tests/conftest.py` holds verbatim ports of the original `auto-speccy` algorithms plus a pure-scipy `reference_log_posterior`. Closed-form ACFs are checked against adaptive quadrature, biased bases against dense numerical convolution, the JIT posterior against scipy, conditional-draw moments against the analytic Student-t, and `update_design` against `assemble_design` bit-for-bit over random MCMC-like walks. `docs/HOW_IT_WORKS.md` §8 maps each test to the invariant it guards.

When changing anything in the numerical core, the meaningful check is not "tests pass" but "output is unchanged": build a fixed-seed fit before the change and `assert_array_equal` against it after.

## Documentation

`docs/CHANGES.md` is an audit trail against the original `auto-speccy` implementation, including two corrected statistical defects that make results intentionally differ from historical runs. Add to it rather than editing its history. `docs/HOW_IT_WORKS.md` is the guided tour. `docs/CODE_FEEDBACK.md` is a point-in-time review, not reference documentation.

Source comments explain *why* at high density (~34% of lines). This is deliberate and load-bearing — the derivations live next to the code they justify. Do not strip them in the name of concision.
