import os
os.environ["NUMBA_DISABLE_JIT"] = "1"   # so the tree includes kernel internals

import numpy as np
from psdebias import welch, sample_ar
from psdebias.sampler import PSDebias

ar_poly = np.array([1.0, -2.7607, 3.8106, -2.6535, 0.9238])
x = sample_ar(1, 32 * 1024, ar_poly, rng=np.random.default_rng(0))[0]
freqs, psd, kernel, _ = welch(x, segment_length=1024, n_segments=32, step=1024)

s = PSDebias(psd, freqs, mode="debias", kernel=kernel, n_time=1024,
             knot_spacing=16, rng=np.random.default_rng(1))
fit = s.sample(n_iterations=60, warmup=20, thin=5, n_beta_draws=3)
