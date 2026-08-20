"""Channel-aware LiRA utilities for continuous stochastic model outputs.

The binary pilot models counts directly.  Circuit-level QNN serving instead
returns a probability vector after a classical head, so this module models the
relationship between exact and stochastic true-class log odds.  The initial
model is deliberately auditable: an affine mean with Gaussian residual noise.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Optional

import numpy as np

from .core import LatentDistributions, fit_latent_distributions, latent_lira_score


_MIN_SCALE = 1e-6


def _normal_logpdf(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = (value - mean) / std
    return -0.5 * standardized**2 - np.log(std) - 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True)
class AffineGaussianChannel:
    """Continuous serving channel ``Y = intercept + slope * Z + Normal(0, scale)``."""

    intercept: float
    slope: float
    scale: float

    def __post_init__(self) -> None:
        values = (float(self.intercept), float(self.slope), float(self.scale))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Channel parameters must be finite")
        if self.slope <= 0.0:
            raise ValueError("Channel slope must be positive")
        if self.scale <= 0.0:
            raise ValueError("Channel scale must be positive")

    @classmethod
    def fit(
        cls,
        exact_log_odds: Any,
        observed_log_odds: Any,
        *,
        min_scale: float = 1e-3,
        min_slope: float = 1e-4,
    ) -> "AffineGaussianChannel":
        """Fit the channel by ordinary least squares on paired public outputs."""
        exact = np.asarray(exact_log_odds, dtype=np.float64).reshape(-1)
        observed = np.asarray(observed_log_odds, dtype=np.float64).reshape(-1)
        if exact.shape != observed.shape or len(exact) < 3:
            raise ValueError("At least three aligned exact/observed pairs are required")
        if not np.isfinite(exact).all() or not np.isfinite(observed).all():
            raise ValueError("Channel calibration pairs must be finite")
        centered = exact - exact.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= _MIN_SCALE:
            raise ValueError("Exact calibration outputs have no usable variation")
        slope = max(float(np.dot(centered, observed - observed.mean()) / denominator), min_slope)
        intercept = float(observed.mean() - slope * exact.mean())
        residual = observed - (intercept + slope * exact)
        degrees_of_freedom = max(len(residual) - 2, 1)
        scale = max(float(np.sqrt(np.dot(residual, residual) / degrees_of_freedom)), min_scale)
        return cls(intercept=intercept, slope=slope, scale=scale)

    def mean(self, exact_log_odds: Any) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(exact_log_odds, dtype=np.float64)

    def invert_mean(self, observed_log_odds: Any) -> np.ndarray:
        observed = np.asarray(observed_log_odds, dtype=np.float64)
        return (observed - self.intercept) / self.slope

    def sample(self, exact_log_odds: Any, rng: np.random.Generator) -> np.ndarray:
        mean = self.mean(exact_log_odds)
        return rng.normal(mean, self.scale, size=mean.shape)


def affine_channel_lira_score(
    observed_log_odds: Any,
    model: LatentDistributions,
    channel: AffineGaussianChannel,
) -> np.ndarray:
    """Analytic LiRA score after marginalizing an affine Gaussian channel."""
    observed = np.asarray(observed_log_odds, dtype=np.float64)
    if observed.shape != model.mean_in.shape:
        raise ValueError("Observed scores and latent distributions must align")
    mean_in = channel.mean(model.mean_in)
    mean_out = channel.mean(model.mean_out)
    std_in = np.sqrt((channel.slope * model.std_in) ** 2 + channel.scale**2)
    std_out = np.sqrt((channel.slope * model.std_out) ** 2 + channel.scale**2)
    return _normal_logpdf(observed, mean_in, std_in) - _normal_logpdf(
        observed, mean_out, std_out
    )


def deconvolved_continuous_lira_score(
    observed_log_odds: Any,
    model: LatentDistributions,
    channel: AffineGaussianChannel,
) -> np.ndarray:
    """Invert the channel mean, but ignore residual serving uncertainty."""
    return latent_lira_score(channel.invert_mean(observed_log_odds), model)


def empirical_channel_lira_score(
    observed_log_odds: Any,
    reference_scores: Any,
    inclusion: Any,
    channel: AffineGaussianChannel,
) -> np.ndarray:
    """Marginalize the channel over the empirical IN/OUT reference mixtures."""
    observed = np.asarray(observed_log_odds, dtype=np.float64)
    reference = np.asarray(reference_scores, dtype=np.float64)
    inclusion = np.asarray(inclusion, dtype=bool)
    if reference.ndim != 2 or inclusion.shape != reference.shape:
        raise ValueError("reference_scores and inclusion must be [reference, candidate]")
    if observed.shape != (reference.shape[1],):
        raise ValueError("One observed score is required per candidate")
    n_in = inclusion.sum(axis=0)
    n_out = (~inclusion).sum(axis=0)
    if np.any(n_in == 0) or np.any(n_out == 0):
        raise ValueError("Every candidate needs IN and OUT references")
    component_logpdf = _normal_logpdf(
        observed[None, :], channel.mean(reference), channel.scale
    )
    log_in = np.full_like(component_logpdf, -np.inf)
    log_out = np.full_like(component_logpdf, -np.inf)
    log_in[inclusion] = component_logpdf[inclusion]
    log_out[~inclusion] = component_logpdf[~inclusion]

    # The reference design need not have the same IN count for every candidate.
    maximum_in = np.max(log_in, axis=0)
    maximum_out = np.max(log_out, axis=0)
    in_sum = np.sum(np.exp(log_in - maximum_in[None, :]), axis=0)
    out_sum = np.sum(np.exp(log_out - maximum_out[None, :]), axis=0)
    empirical_in = maximum_in + np.log(in_sum) - np.log(n_in)
    empirical_out = maximum_out + np.log(out_sum) - np.log(n_out)
    return empirical_in - empirical_out


def fit_noise_augmented_distributions(
    reference_scores: Any,
    inclusion: Any,
    channel: AffineGaussianChannel,
    *,
    draws: int,
    rng: np.random.Generator,
    variance_shrinkage: float = 0.15,
) -> LatentDistributions:
    """Simulate noisy reference outputs, then fit ordinary Gaussian LiRA.

    This is the explicit noise-augmentation baseline.  It spends ``draws``
    channel simulations per reference output, whereas analytic ChannelLiRA fits
    the latent bank once and composes it with the channel likelihood.
    """
    reference = np.asarray(reference_scores, dtype=np.float64)
    inclusion = np.asarray(inclusion, dtype=bool)
    if reference.ndim != 2 or inclusion.shape != reference.shape:
        raise ValueError("reference_scores and inclusion must be [reference, candidate]")
    if int(draws) < 1:
        raise ValueError("draws must be positive")
    mean = channel.mean(reference)
    augmented = rng.normal(
        mean[None, :, :],
        channel.scale,
        size=(int(draws), *reference.shape),
    ).reshape(int(draws) * reference.shape[0], reference.shape[1])
    augmented_inclusion = np.tile(inclusion, (int(draws), 1))
    return fit_latent_distributions(
        augmented,
        augmented_inclusion,
        variance_shrinkage=variance_shrinkage,
    )


def balanced_reference_subset(inclusion: Any, count: int) -> np.ndarray:
    """Choose a deterministic reference subset maximizing per-record balance.

    Exact balance is preferred, then the largest worst-case number of IN and OUT
    observations.  Exhaustive search is inexpensive for the retained 16-model
    banks and avoids a hidden dependence on arbitrary filename order.
    """
    inclusion = np.asarray(inclusion, dtype=bool)
    if inclusion.ndim != 2:
        raise ValueError("inclusion must be [reference, candidate]")
    n_references = inclusion.shape[0]
    count = int(count)
    if count < 4 or count > n_references:
        raise ValueError("Reference count must be between four and the bank size")
    best_indices: Optional[tuple[int, ...]] = None
    best_score: Optional[tuple[float, float]] = None
    target = count / 2.0
    for indices in itertools.combinations(range(n_references), count):
        n_in = inclusion[list(indices)].sum(axis=0)
        worst_class_count = float(min(n_in.min(), (count - n_in).min()))
        total_imbalance = float(np.abs(n_in - target).sum())
        score = (worst_class_count, -total_imbalance)
        if best_score is None or score > best_score:
            best_score = score
            best_indices = indices
    assert best_indices is not None and best_score is not None
    if best_score[0] < 2:
        raise ValueError("No subset gives every candidate two IN and two OUT references")
    return np.asarray(best_indices, dtype=np.int64)


def channel_diagnostics(
    exact_log_odds: Any,
    observed_log_odds: Any,
    channel: AffineGaussianChannel,
) -> dict[str, float]:
    """Return transparent residual diagnostics for a fitted channel."""
    exact = np.asarray(exact_log_odds, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed_log_odds, dtype=np.float64).reshape(-1)
    if exact.shape != observed.shape or not len(exact):
        raise ValueError("Diagnostic pairs must be non-empty and aligned")
    residual = observed - channel.mean(exact)
    rmse = float(np.sqrt(np.mean(residual**2)))
    centered_observed = observed - observed.mean()
    denominator = float(np.dot(centered_observed, centered_observed))
    r_squared = 1.0 - float(np.dot(residual, residual)) / denominator if denominator > 0 else float("nan")
    residual_mean = float(residual.mean())
    residual_std = float(residual.std())
    if residual_std > 0.0:
        standardized = (residual - residual_mean) / residual_std
        skew = float(np.mean(standardized**3))
        excess_kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        skew = excess_kurtosis = float("nan")
    return {
        "rmse": rmse,
        "r_squared": r_squared,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_skew": skew,
        "residual_excess_kurtosis": excess_kurtosis,
        "coverage_90pct": float(np.mean(np.abs(residual) <= 1.6448536269514722 * channel.scale)),
    }
