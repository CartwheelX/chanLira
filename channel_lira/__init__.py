"""Channel-aware likelihood-ratio attacks for stochastic model serving."""

from .core import (
    BinaryChannel,
    LatentDistributions,
    attack_metrics,
    channel_lira_score,
    deconvolved_lira_score,
    fit_latent_distributions,
    logit,
    naive_lira_score,
    sigmoid,
)
from .continuous import (
    AffineGaussianChannel,
    affine_channel_lira_score,
    balanced_reference_subset,
    channel_diagnostics,
    deconvolved_continuous_lira_score,
    empirical_channel_lira_score,
    fit_noise_augmented_distributions,
)

__all__ = [
    "BinaryChannel",
    "LatentDistributions",
    "attack_metrics",
    "channel_lira_score",
    "deconvolved_lira_score",
    "fit_latent_distributions",
    "logit",
    "naive_lira_score",
    "sigmoid",
    "AffineGaussianChannel",
    "affine_channel_lira_score",
    "balanced_reference_subset",
    "channel_diagnostics",
    "deconvolved_continuous_lira_score",
    "empirical_channel_lira_score",
    "fit_noise_augmented_distributions",
]
