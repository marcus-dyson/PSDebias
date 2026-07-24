"""psdebias — adaptive smoothing and debiasing of quadratic spectral estimators.

Implements PSDebias (Dyson, Astfalck, Cripps & Stemler, "Adaptive Smoothing of
Quadratic Spectral Estimators"): Bayesian B1-spline knot selection over
spectral estimates, smoothing variance-dominant estimates in the log domain
and debiasing bias-dominant estimates through spectral-window-convolved bases,
with the regime chosen by a data-driven regime diagnostic.
"""

from psdebias.analytic import ar_spectrum, matern_acf, matern_spectrum
from psdebias.diagnostic import (
    RegimeDiagnostic,
    regime_diagnostic,
    window_bandwidth,
)
from psdebias.dwelch import dwelch_b0, dwelch_b1
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
    "RegimeDiagnostic",
    "SpectralEstimate",
    "ar_spectrum",
    "dwelch_b0",
    "dwelch_b1",
    "fit_psd",
    "lag_window",
    "matern_acf",
    "matern_spectrum",
    "multitaper",
    "periodogram",
    "regime_diagnostic",
    "sample_ar",
    "sample_matern",
    "welch",
    "window_bandwidth",
]
