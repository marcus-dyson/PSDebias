# Code Feedback

A review of the `psdebias` source (`src/psdebias/`) and the study scripts
(`scripts/`), covering what the code does well and a short list of issues worth
attention. Nothing here has been applied to the source — it is observation only.

## Summary

This is a strong, carefully engineered codebase. The numerical core is
mathematically precise, well-tested, and unusually well documented: derivations
and design rationale are explained inline rather than merely restated. The notes
below are refinements, not blockers.

## Strengths

- **Documentation.** Every module and non-trivial function carries a docstring
  that explains the *why* (the paper section, the identity being used, the
  deviation from the original `auto_speccy`), not just the *what*. `splines.py`,
  `sampler.py`, and `util.py` are exemplary — e.g. the "convolution paid once"
  explanation in `splines.half_bases_biased` and the halve-idx double-count
  derivation in `splines._compute_column`.

- **Numerical care.** Least squares goes through a Cholesky factor of the Gram
  matrix with no explicit inverse (`sampler._least_squares`); rank deficiency is
  caught by a *relative* pivot floor (`sampler._cholesky`); ESS is clamped at 0
  before the log; posterior-predictive mean/variance is accumulated streaming
  (`sampler._predictive_pass`).

- **Reproducibility.** All randomness is pre-generated into arrays in Python and
  fed to the numba kernel, precisely because numba's RNG cannot be seeded from a
  NumPy `Generator` (`sampler._mh_kernel`). The study scripts extend this with
  per-`(purpose, setting, realisation, attempt)` keyed streams
  (`generate_realisations.stream_rng`) and resumable, flushed CSVs, so an
  interrupted run re-derives the identical stream.

- **Performance.** The expensive spectral-window convolution is done once as a
  single batched FFT, after which every design matrix — including each MCMC
  proposal — is a linear combination of the precomputed primitives
  (`splines`, `sampler`). Proposals reuse unchanged columns via
  `splines.update_design`, and the hot loops are numba-JIT compiled.

- **Correctness fixes over the original**, each documented and tied to
  `docs/CHANGES.md`: the posterior uses the bin count `g` rather than the
  candidate-knot count; nonuniform B1 assembly is exact rather than valid only
  for equally spaced knots; `util.freq_slice` errors instead of silently
  guessing an alignment; consistent `1/fs` density scaling across all four
  estimators.

- **Clear structure.** Sensible module separation (analytic / simulate /
  estimators / splines / sampler / diagnostic / dwelch / util), typed
  signatures throughout, and `NamedTuple` / `dataclass` result types.

## Issues and suggestions

1. **`estimators.lag_window` — edge case for large `lag` (fixed).** The negative-
   lag fold writes `padded[:lag+1]` and `padded[-lag:]`. These two regions
   overlap once `lag >= n/2` (since `n - lag <= lag`), so the second assignment
   corrupts the first and the resulting PSD is wrong. The guard previously only
   checked `0 < lag < n`; it now enforces the true precondition `0 < lag < n/2`.
   Latent in practice (`lag` defaults to `n // 10`), but the guard now rejects
   the corrupting band instead of silently producing garbage.

2. **`sampler.fit_psd(mode="auto")` recomputes the biased basis FFT.**
   `regime_diagnostic` builds the window-convolved bases (`half_bases_biased`)
   to score its bandwidth-matched mesh, then `PSDebias.__init__` builds them
   again for the sampler's mesh. The two meshes differ, so the work is not
   literally reusable, but the batched-FFT cost is paid twice per auto-fit.
   Worth noting if auto-mode fitting ever becomes a bottleneck.

3. **`c` (g-prior scale) convention diverges from the paper.** The sampler
   default is `c = len(response)` = number of bins `g` (the paper's choice), but
   the study scripts pass `c = n_series` (`run_adaptive`, documented in
   `CHANGES.md` §6). Both are defensible; the risk is that someone reproducing
   paper figures from the scripts gets different shrinkage without noticing. A
   one-line reminder at the call site (or a named constant) would help.

4. **Memory scaling of stored draws.** `sampler` stores `betas` as a
   `n_kept x n_beta_draws x (L+2)` NaN-padded array of doubles. This is
   documented in `_mh_kernel`, but it is worth stating as a concrete limit:
   with a fine candidate mesh (large `L`) and many `n_beta_draws`, this array
   dominates the memory bill. A downstream user tightening `knot_spacing`
   should expect quadratic-ish growth here.

5. **`sampler._mh_kernel` size and testability.** It is a ~150-line numba
   kernel threading many mutable buffers (`beta`, `chol`, `design`, `work`, …)
   through the chain state. The invariants are well documented, but the accept
   step and the conditional `(sigma^2, beta)` draw are hard to exercise in
   isolation. If this file grows, consider factoring the conditional-draw block
   into a separately testable `@njit` helper.

6. **Minor wording.** The comment at `estimators.py:22`
   ("This gets enumerated when output") reads awkwardly and does not add much
   over the `NamedTuple` declaration it annotates.

## Closing

The hard parts — the marginalized posterior, the exact nonuniform spline
assembly, the reproducible numba MCMC — are done well and defended in the
comments. The items above are edge-case hardening and ergonomics, not
corrections to the method.
