from __future__ import annotations

import unittest

import numpy as np

from channel_lira.continuous import (
    AffineGaussianChannel,
    affine_channel_lira_score,
    balanced_reference_subset,
    deconvolved_continuous_lira_score,
    empirical_channel_lira_score,
    fit_noise_augmented_distributions,
)
from channel_lira.core import LatentDistributions, latent_lira_score


class ContinuousChannelLiRATests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LatentDistributions(
            mean_in=np.array([1.0, 0.4]),
            std_in=np.array([0.5, 0.3]),
            mean_out=np.array([-0.4, -0.2]),
            std_out=np.array([0.6, 0.4]),
        )
        self.channel = AffineGaussianChannel(intercept=-0.2, slope=0.75, scale=0.35)

    def test_affine_fit_recovers_known_relationship(self) -> None:
        exact = np.linspace(-3.0, 3.0, 101)
        observed = -0.3 + 0.8 * exact
        fitted = AffineGaussianChannel.fit(exact, observed, min_scale=0.02)
        self.assertAlmostEqual(fitted.intercept, -0.3, places=12)
        self.assertAlmostEqual(fitted.slope, 0.8, places=12)
        self.assertAlmostEqual(fitted.scale, 0.02, places=12)

    def test_affine_marginal_matches_normal_convolution(self) -> None:
        observed = np.array([0.7, -0.1])
        score = affine_channel_lira_score(observed, self.model, self.channel)

        def logpdf(value: float, mean: float, std: float) -> float:
            return -0.5 * ((value - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)

        expected = []
        for index, value in enumerate(observed):
            mean_in = self.channel.intercept + self.channel.slope * self.model.mean_in[index]
            mean_out = self.channel.intercept + self.channel.slope * self.model.mean_out[index]
            std_in = np.sqrt((self.channel.slope * self.model.std_in[index]) ** 2 + self.channel.scale**2)
            std_out = np.sqrt((self.channel.slope * self.model.std_out[index]) ** 2 + self.channel.scale**2)
            expected.append(logpdf(value, mean_in, std_in) - logpdf(value, mean_out, std_out))
        self.assertTrue(np.allclose(score, expected))

    def test_deconvolution_reduces_to_latent_lira_without_residual(self) -> None:
        latent = np.array([0.8, 0.0])
        observed = self.channel.mean(latent)
        self.assertTrue(
            np.allclose(
                deconvolved_continuous_lira_score(observed, self.model, self.channel),
                latent_lira_score(latent, self.model),
            )
        )

    def test_empirical_mixture_matches_manual_likelihood(self) -> None:
        references = np.array([
            [1.0, -0.4], [1.4, -0.1], [-0.5, 0.5], [-0.2, 0.8]
        ])
        inclusion = np.array([
            [1, 0], [1, 0], [0, 1], [0, 1]
        ], dtype=bool)
        observed = np.array([0.4, 0.2])
        score = empirical_channel_lira_score(observed, references, inclusion, self.channel)
        self.assertEqual(score.shape, (2,))
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(score[0], 0.0)
        self.assertGreater(score[1], 0.0)

    def test_noise_augmentation_returns_finite_model(self) -> None:
        references = np.array([
            [1.0, -0.4], [1.4, -0.1], [-0.5, 0.5], [-0.2, 0.8]
        ])
        inclusion = np.array([
            [1, 0], [1, 0], [0, 1], [0, 1]
        ], dtype=bool)
        fitted = fit_noise_augmented_distributions(
            references,
            inclusion,
            self.channel,
            draws=16,
            rng=np.random.default_rng(7),
            variance_shrinkage=0.0,
        )
        self.assertTrue(np.isfinite(fitted.mean_in).all())
        self.assertTrue(np.all(fitted.std_out > 0.0))

    def test_balanced_subset_is_exactly_balanced(self) -> None:
        first_half = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=bool)
        inclusion = np.vstack([first_half, ~first_half])
        selected = balanced_reference_subset(inclusion, 4)
        self.assertTrue(np.array_equal(inclusion[selected].sum(axis=0), [2, 2]))


if __name__ == "__main__":
    unittest.main()
