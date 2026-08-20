from __future__ import annotations

import unittest

import numpy as np

from channel_lira.core import (
    BinaryChannel,
    LatentDistributions,
    attack_metrics,
    channel_lira_score,
    fit_latent_distributions,
    latent_lira_score,
    sigmoid,
)


class ChannelLiRACoreTests(unittest.TestCase):
    def test_binary_channel_round_trip(self) -> None:
        channel = BinaryChannel(false_positive=0.08, false_negative=0.13)
        latent = np.array([0.1, 0.4, 0.9])
        self.assertTrue(np.allclose(channel.invert(channel.apply(latent)), latent))

    def test_balanced_references_fit_candidate_distributions(self) -> None:
        scores = np.array([
            [2.0, -1.2], [1.8, -0.8], [-0.2, 0.9], [0.1, 1.1],
        ])
        inclusion = np.array([
            [1, 0], [1, 0], [0, 1], [0, 1],
        ], dtype=bool)
        model = fit_latent_distributions(scores, inclusion, variance_shrinkage=0.0)
        self.assertTrue(np.allclose(model.mean_in, [1.9, 1.0]))
        self.assertTrue(np.allclose(model.mean_out, [-0.05, -1.0]))

    def test_aggregated_counts_are_a_sufficient_statistic(self) -> None:
        model = LatentDistributions(
            mean_in=np.array([1.2, 0.8]), std_in=np.array([0.4, 0.5]),
            mean_out=np.array([0.1, -0.2]), std_out=np.array([0.5, 0.4]),
        )
        channel = BinaryChannel.symmetric(0.07)
        query_counts = np.array([[9, 7], [8, 6], [7, 8], [9, 5]])
        aggregated = query_counts.sum(axis=0)
        latent_grid = np.linspace(-4.0, 4.0, 257)
        probability = channel.apply(sigmoid(latent_grid))
        for candidate in range(query_counts.shape[1]):
            split_kernel = sum(
                count * np.log(probability)
                + (10 - count) * np.log1p(-probability)
                for count in query_counts[:, candidate]
            )
            aggregate_kernel = (
                aggregated[candidate] * np.log(probability)
                + (40 - aggregated[candidate]) * np.log1p(-probability)
            )
            self.assertTrue(np.allclose(split_kernel, aggregate_kernel))
        split_score = channel_lira_score(aggregated, 40, model, channel)
        single_score = channel_lira_score(aggregated, 40, model, channel)
        self.assertTrue(np.array_equal(split_score, single_score))

    def test_more_shots_approach_the_latent_decision(self) -> None:
        model = LatentDistributions(
            mean_in=np.array([1.5]), std_in=np.array([0.25]),
            mean_out=np.array([-0.5]), std_out=np.array([0.25]),
        )
        channel = BinaryChannel.symmetric(0.1)
        probability = channel.apply(sigmoid(np.array([1.5])))
        low = channel_lira_score(np.rint(16 * probability), 16, model, channel)[0]
        high = channel_lira_score(np.rint(4096 * probability), 4096, model, channel)[0]
        latent = latent_lira_score(np.array([1.5]), model)[0]
        self.assertGreater(high, low)
        self.assertGreater(high, 0.0)
        self.assertGreater(high / latent, 0.9)

    def test_adaptive_quadrature_matches_dense_integration(self) -> None:
        model = LatentDistributions(
            mean_in=np.array([0.9]), std_in=np.array([0.55]),
            mean_out=np.array([-0.25]), std_out=np.array([0.7]),
        )
        channel = BinaryChannel(false_positive=0.08, false_negative=0.12)
        observed = channel_lira_score(
            np.array([73]), 128, model, channel, quadrature_order=48
        )[0]
        grid = np.linspace(-8.0, 8.0, 200_001)

        def dense_log_likelihood(mean: float, std: float) -> float:
            probability = channel.apply(sigmoid(grid))
            prior = np.exp(-0.5 * ((grid - mean) / std) ** 2) / (
                std * np.sqrt(2.0 * np.pi)
            )
            kernel = probability**73 * (1.0 - probability) ** 55
            integrate = getattr(np, "trapezoid", np.trapz)
            return float(np.log(integrate(prior * kernel, grid)))

        expected = dense_log_likelihood(0.9, 0.55) - dense_log_likelihood(-0.25, 0.7)
        self.assertAlmostEqual(observed, expected, places=6)

    def test_metrics_are_tie_safe(self) -> None:
        labels = np.array([0, 0, 1, 1])
        perfect = attack_metrics(labels, np.array([0.0, 0.1, 0.9, 1.0]))
        tied = attack_metrics(labels, np.ones(4))
        self.assertAlmostEqual(perfect["auc"], 1.0)
        self.assertAlmostEqual(tied["auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
