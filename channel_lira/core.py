"""Core likelihoods for membership inference through a binary serving channel.

The latent score is a true-class log-odds value.  Reference-model variation is
modeled in that latent space, while finite-shot and readout randomness are
modeled by a separate binary channel.  This separation is the minimal
ChannelLiRA construction described in the project proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_EPS = 1e-12


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> np.floating:
    implementation = getattr(np, "trapezoid", None)
    if implementation is not None:
        return implementation(values, coordinates)
    return np.trapz(values, coordinates)


def sigmoid(value: Any) -> np.ndarray:
    """Numerically stable logistic transform."""
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def logit(probability: Any) -> np.ndarray:
    """Numerically stable log-odds transform."""
    probability = np.clip(np.asarray(probability, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(probability) - np.log1p(-probability)


@dataclass(frozen=True)
class BinaryChannel:
    """Classical confusion channel applied to a latent success probability.

    ``false_positive`` is P(observed success | latent failure), and
    ``false_negative`` is P(observed failure | latent success).  Symmetric
    readout error ``e`` is represented by setting both values to ``e``.
    """

    false_positive: float = 0.0
    false_negative: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("false_positive", self.false_positive),
            ("false_negative", self.false_negative),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1); got {value}")
        if self.false_positive + self.false_negative >= 1.0:
            raise ValueError("Channel must be informative: fp + fn must be below one")

    @classmethod
    def symmetric(cls, error: float) -> "BinaryChannel":
        return cls(false_positive=float(error), false_negative=float(error))

    @property
    def slope(self) -> float:
        return 1.0 - self.false_positive - self.false_negative

    def apply(self, latent_probability: Any) -> np.ndarray:
        probability = np.asarray(latent_probability, dtype=np.float64)
        observed = self.false_positive + self.slope * probability
        return np.clip(observed, _EPS, 1.0 - _EPS)

    def invert(self, observed_probability: Any) -> np.ndarray:
        observed = np.asarray(observed_probability, dtype=np.float64)
        latent = (observed - self.false_positive) / self.slope
        return np.clip(latent, _EPS, 1.0 - _EPS)


@dataclass(frozen=True)
class LatentDistributions:
    """Per-candidate Gaussian approximations to latent IN and OUT scores."""

    mean_in: np.ndarray
    std_in: np.ndarray
    mean_out: np.ndarray
    std_out: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (self.mean_in, self.std_in, self.mean_out, self.std_out)
        )
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("All latent-distribution arrays must have the same shape")
        if arrays[0].ndim != 1:
            raise ValueError("Latent-distribution arrays must be one-dimensional")
        if np.any(arrays[1] <= 0.0) or np.any(arrays[3] <= 0.0):
            raise ValueError("Latent standard deviations must be positive")
        for name, array in zip(
            ("mean_in", "std_in", "mean_out", "std_out"), arrays
        ):
            object.__setattr__(self, name, array)

    def repeated(self, repeats: int) -> "LatentDistributions":
        if repeats < 1:
            raise ValueError("repeats must be positive")
        return LatentDistributions(
            mean_in=np.tile(self.mean_in, repeats),
            std_in=np.tile(self.std_in, repeats),
            mean_out=np.tile(self.mean_out, repeats),
            std_out=np.tile(self.std_out, repeats),
        )


def fit_latent_distributions(
    scores: Any,
    inclusion: Any,
    *,
    min_scale: float = 1e-3,
    variance_shrinkage: float = 0.15,
) -> LatentDistributions:
    """Fit robust per-candidate IN/OUT Gaussians from a reference bank.

    A small shrinkage toward the bank-wide within-candidate variance stabilizes
    the eight-IN/eight-OUT pilot design without mixing candidate locations.
    """
    scores = np.asarray(scores, dtype=np.float64)
    inclusion = np.asarray(inclusion, dtype=bool)
    if scores.ndim != 2 or inclusion.shape != scores.shape:
        raise ValueError("scores and inclusion must have shape [reference, candidate]")
    if not 0.0 <= variance_shrinkage <= 1.0:
        raise ValueError("variance_shrinkage must be in [0, 1]")
    n_candidates = scores.shape[1]
    in_values = []
    out_values = []
    for candidate in range(n_candidates):
        candidate_in = scores[inclusion[:, candidate], candidate]
        candidate_out = scores[~inclusion[:, candidate], candidate]
        if len(candidate_in) < 2 or len(candidate_out) < 2:
            raise ValueError("Every candidate needs at least two IN and two OUT references")
        in_values.append(candidate_in)
        out_values.append(candidate_out)
    in_array = np.stack(in_values)
    out_array = np.stack(out_values)
    mean_in = np.median(in_array, axis=1)
    mean_out = np.median(out_array, axis=1)
    var_in = np.var(in_array, axis=1, ddof=1)
    var_out = np.var(out_array, axis=1, ddof=1)
    pooled_in = float(np.median(var_in))
    pooled_out = float(np.median(var_out))
    weight = float(variance_shrinkage)
    std_in = np.sqrt((1.0 - weight) * var_in + weight * pooled_in)
    std_out = np.sqrt((1.0 - weight) * var_out + weight * pooled_out)
    return LatentDistributions(
        mean_in=mean_in,
        std_in=np.maximum(std_in, min_scale),
        mean_out=mean_out,
        std_out=np.maximum(std_out, min_scale),
    )


def _normal_logpdf(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    variance_term = (value - mean) / std
    return -0.5 * variance_term**2 - np.log(std) - 0.5 * math.log(2.0 * math.pi)


def latent_lira_score(observed_log_odds: Any, model: LatentDistributions) -> np.ndarray:
    observed = np.asarray(observed_log_odds, dtype=np.float64)
    if observed.shape != model.mean_in.shape:
        raise ValueError("Observed scores and latent distributions must align")
    return _normal_logpdf(observed, model.mean_in, model.std_in) - _normal_logpdf(
        observed, model.mean_out, model.std_out
    )


def naive_lira_score(counts: Any, shots: int, model: LatentDistributions) -> np.ndarray:
    """Ordinary LiRA after averaging stochastic outputs.

    The Jeffreys-smoothed shot frequency is treated as if it were an exact
    latent model output.  This intentionally conflates the serving channel and
    reference-model variation, making it the relevant baseline.
    """
    counts = _validate_counts(counts, shots, model.mean_in.shape)
    observed = (counts + 0.5) / (float(shots) + 1.0)
    return latent_lira_score(logit(observed), model)


def deconvolved_lira_score(
    counts: Any,
    shots: int,
    model: LatentDistributions,
    channel: BinaryChannel,
) -> np.ndarray:
    """Repeated-mean LiRA with a known channel inversion but no shot uncertainty."""
    counts = _validate_counts(counts, shots, model.mean_in.shape)
    observed = (counts + 0.5) / (float(shots) + 1.0)
    return latent_lira_score(logit(channel.invert(observed)), model)


def _validate_counts(counts: Any, shots: int, expected_shape: tuple[int, ...]) -> np.ndarray:
    if int(shots) < 1:
        raise ValueError("shots must be positive")
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != expected_shape:
        raise ValueError(f"Counts have shape {counts.shape}; expected {expected_shape}")
    if np.any(counts < 0) or np.any(counts > shots):
        raise ValueError("Counts must lie between zero and shots")
    return counts


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis)
    )


def _channel_log_likelihood(
    counts: np.ndarray,
    shots: int,
    mean: np.ndarray,
    std: np.ndarray,
    channel: BinaryChannel,
    quadrature_order: int,
) -> np.ndarray:
    if quadrature_order < 16:
        raise ValueError("quadrature_order must be at least sixteen")

    # Fixed Gauss-Hermite nodes become inaccurate as the binomial likelihood gets
    # much narrower than the latent Gaussian.  Locate each posterior mode first,
    # then use Gauss-Legendre nodes over a candidate-specific local interval.  This
    # remains stable from a handful to thousands of shots.
    lower_prior = mean - 10.0 * std
    upper_prior = mean + 10.0 * std
    smoothed_frequency = (counts + 0.5) / (float(shots) + 1.0)
    mode = np.clip(logit(channel.invert(smoothed_frequency)), lower_prior, upper_prior)
    variance = std**2

    def derivatives(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        latent_probability = sigmoid(value)
        observed_probability = channel.apply(latent_probability)
        first_probability = channel.slope * latent_probability * (1.0 - latent_probability)
        second_probability = first_probability * (1.0 - 2.0 * latent_probability)
        denominator = observed_probability * (1.0 - observed_probability)
        residual = counts - shots * observed_probability
        ratio = first_probability / denominator
        ratio_derivative = (
            second_probability / denominator
            - first_probability**2 * (1.0 - 2.0 * observed_probability) / denominator**2
        )
        gradient = -(value - mean) / variance + residual * ratio
        hessian = (
            -1.0 / variance
            - shots * first_probability * ratio
            + residual * ratio_derivative
        )
        return gradient, hessian

    for _ in range(16):
        gradient, hessian = derivatives(mode)
        safe_hessian = np.where(hessian < -1e-10, hessian, -1.0 / variance)
        step = np.clip(gradient / safe_hessian, -2.0 * std, 2.0 * std)
        updated = np.clip(mode - step, lower_prior, upper_prior)
        if float(np.max(np.abs(updated - mode))) < 1e-9:
            mode = updated
            break
        mode = updated

    def log_integrand(value: np.ndarray) -> np.ndarray:
        observed_probability = channel.apply(sigmoid(value))
        log_prior = (
            -0.5 * ((value - mean) / std) ** 2
            - np.log(std)
            - 0.5 * math.log(2.0 * math.pi)
        )
        return (
            log_prior
            + counts * np.log(observed_probability)
            + (shots - counts) * np.log1p(-observed_probability)
        )

    # Guard Newton against rare boundary/non-concave starts by comparing a few
    # deterministic candidates and retaining the best posterior point.
    candidates = np.stack((mode, mean, lower_prior, upper_prior), axis=1)
    candidate_log = np.stack(
        [log_integrand(candidates[:, index]) for index in range(candidates.shape[1])],
        axis=1,
    )
    mode = candidates[np.arange(len(mean)), np.argmax(candidate_log, axis=1)]
    _, hessian = derivatives(mode)
    information = np.maximum(-hessian, 0.01 / variance)
    posterior_scale = 1.0 / np.sqrt(information)
    lower = np.maximum(lower_prior, mode - 12.0 * posterior_scale)
    upper = np.minimum(upper_prior, mode + 12.0 * posterior_scale)

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    latent_scores = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    observed_probability = channel.apply(sigmoid(latent_scores))
    log_prior = (
        -0.5 * ((latent_scores - mean[:, None]) / std[:, None]) ** 2
        - np.log(std[:, None])
        - 0.5 * math.log(2.0 * math.pi)
    )
    log_kernel = counts[:, None] * np.log(observed_probability) + (
        shots - counts[:, None]
    ) * np.log1p(-observed_probability)
    log_weights = np.log(weights[None, :]) + np.log(half_width[:, None])
    # The binomial coefficient is identical under IN and OUT and cancels in the LLR.
    return _logsumexp(log_prior + log_kernel + log_weights, axis=1)


def channel_lira_score(
    counts: Any,
    shots: int,
    model: LatentDistributions,
    channel: BinaryChannel,
    *,
    quadrature_order: int = 32,
) -> np.ndarray:
    """Channel-marginalized IN/OUT log-likelihood ratio."""
    counts = _validate_counts(counts, shots, model.mean_in.shape)
    log_in = _channel_log_likelihood(
        counts,
        shots,
        model.mean_in,
        model.std_in,
        channel,
        quadrature_order,
    )
    log_out = _channel_log_likelihood(
        counts,
        shots,
        model.mean_out,
        model.std_out,
        channel,
        quadrature_order,
    )
    return log_in - log_out


def _roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Both member and nonmember examples are required")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    group_end = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tpr = np.r_[0.0, cumulative_tp[group_end] / positives]
    fpr = np.r_[0.0, cumulative_fp[group_end] / negatives]
    return fpr, tpr


def attack_metrics(labels: Any, scores: Any) -> dict[str, float]:
    """Return AUC, attack advantage, and resolvable low-FPR operating points."""
    fpr, tpr = _roc_points(np.asarray(labels), np.asarray(scores))
    metrics = {
        "auc": float(_trapezoid(tpr, fpr)),
        "advantage": float(np.max(tpr - fpr)),
    }
    for alpha, name in ((0.001, "tpr_at_0_1pct_fpr"), (0.01, "tpr_at_1pct_fpr"), (0.05, "tpr_at_5pct_fpr")):
        valid = fpr <= alpha + 1e-15
        metrics[name] = float(np.max(tpr[valid]))
        metrics[name.replace("tpr", "attained_fpr")] = float(fpr[valid][np.argmax(tpr[valid])])
    return metrics
