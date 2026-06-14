#!/usr/bin/env python
"""Focused tests for the mismatch and commitment-cost proxy loss layer."""

import unittest

import tensorflow as tf

from legacy.rnncell_model import OnlyMismatchLoss


class OnlyMismatchLossTest(unittest.TestCase):
    def test_adds_commitment_cost_proxy_when_enabled(self):
        layer = OnlyMismatchLoss(
            num_gens=2,
            demand_normalizer_mw=100.0,
            balance_loss_weight=0.0,
            noload_cost_vals=[10.0, 20.0],
            startup_cost_vals=[100.0, 200.0],
            initial_status_vals=[0.0, 1.0],
            cost_normalizer=1000.0,
            cost_loss_weight=0.5,
        )
        demand = tf.constant([[[50.0], [70.0]]], dtype=tf.float32)
        outputs = tf.constant(
            [[[1.0, 1.0, 30.0, 20.0], [1.0, 0.0, 70.0, 0.0]]],
            dtype=tf.float32,
        )
        layer([demand, outputs])

        self.assertEqual(len(layer.losses), 2)
        self.assertAlmostEqual(float(layer.losses[0].numpy()), 0.0, places=6)
        self.assertAlmostEqual(float(layer.losses[1].numpy()), 0.07, places=6)
        metrics = {metric.name: float(metric.result().numpy()) for metric in layer.metrics}
        self.assertAlmostEqual(
            metrics["normalized_commitment_cost_proxy"], 0.14, places=6
        )
        self.assertAlmostEqual(
            metrics["weighted_commitment_cost_proxy"], 0.07, places=6
        )

    def test_records_zero_weight_proxy_without_adding_cost_loss(self):
        layer = OnlyMismatchLoss(
            num_gens=2,
            demand_normalizer_mw=100.0,
            balance_loss_weight=0.0,
            noload_cost_vals=[10.0, 20.0],
            startup_cost_vals=[100.0, 200.0],
            initial_status_vals=[0.0, 1.0],
            cost_normalizer=1000.0,
            cost_loss_weight=0.0,
        )
        demand = tf.constant([[[50.0], [70.0]]], dtype=tf.float32)
        outputs = tf.constant(
            [[[1.0, 1.0, 30.0, 20.0], [1.0, 0.0, 70.0, 0.0]]],
            dtype=tf.float32,
        )
        layer([demand, outputs])

        self.assertEqual(len(layer.losses), 1)
        metrics = {metric.name: float(metric.result().numpy()) for metric in layer.metrics}
        self.assertAlmostEqual(
            metrics["normalized_commitment_cost_proxy"], 0.14, places=6
        )
        self.assertAlmostEqual(
            metrics["weighted_commitment_cost_proxy"], 0.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
