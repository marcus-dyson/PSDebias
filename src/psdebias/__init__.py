"""psdebias — adaptive debiasing of quadratic spectral estimators.

Implements PSDebias (Dyson, Astfalck, Cripps & Stemler, "Adaptive Smoothing of
Quadratic Spectral Estimators"): Bayesian B1-spline knot selection over
spectral estimates, debiasing bias-dominant estimates through
spectral-window-convolved bases.
"""

from psdebias.analytic import ar_spectrum, matern_acf, matern_spectrum
from psdebias.dwelch import dquad
from psdebias.estimators import (
    SpectralEstimate,
    lag_window,
    multitaper,
    periodogram,
    welch,
)
from psdebias.sampler import FitResult, PSDebias, fit_psd
from psdebias.simulate import sample_ar, sample_matern

__version__ = "0.1.0"

__all__ = [
    "FitResult",
    "PSDebias",
    "SpectralEstimate",
    "ar_spectrum",
    "dquad",
    "fit_psd",
    "lag_window",
    "matern_acf",
    "matern_spectrum",
    "multitaper",
    "periodogram",
    "sample_ar",
    "sample_matern",
    "welch",
]
