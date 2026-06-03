#!/usr/bin/env python
"""Focused synthetic tests for cost-aware allocation."""

import unittest

import tensorflow as tf

from rnncell_strict_allocation_cost_aware_model import CostAwareLookAheadRepairUCCell


class CostAwareLookAheadRepairUCCellTest(unittest.TestCase):
    def make_cell(self):
        cell = CostAwareLookAheadRepairUCCell(
            hidden_dim=2,
            num_gens=2,
            mut_vals=[2, 2],
            mdt_vals=[4, 4],
            p_min_vals=[0.0, 0.0],
            p_max_vals=[100.0, 100.0],
            ramp_up_vals=[100.0, 100.0],
            ramp_down_vals=[100.0, 100.0],
            startup_cap_vals=[100.0, 100.0],
            shutdown_cap_vals=[100.0, 100.0],
            lookahead_hours=4,
            linear_cost_vals=[10.0, 100.0],
        )
        cell.build((1, 6))
        cell.dense_u.kernel.assign(tf.zeros_like(cell.dense_u.kernel))
        cell.dense_u.bias.assign([20.0, 20.0])
        cell.dense_p.kernel.assign(tf.zeros_like(cell.dense_p.kernel))
        cell.dense_p.bias.assign(tf.zeros_like(cell.dense_p.bias))
        return cell

    def call_cell(self, demand):
        cell = self.make_cell()
        zero_hidden = tf.zeros((1, 2))
        output, _ = cell(
            tf.constant([[demand, 24.0, demand, demand, demand, demand]], dtype=tf.float32),
            (
                zero_hidden,
                zero_hidden,
                tf.constant([[0.0, 0.0]], dtype=tf.float32),
                tf.constant([[1.0, 1.0]], dtype=tf.float32),
                tf.constant([[5.0, 5.0]], dtype=tf.float32),
                tf.constant([[0.0, 0.0]], dtype=tf.float32),
            ),
        )
        return output.numpy()[0]

    def test_dispatches_cheap_generator_before_expensive_generator(self):
        output = self.call_cell(100.0)
        self.assertEqual(output[:2].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(float(output[2]), 100.0, places=5)
        self.assertAlmostEqual(float(output[3]), 0.0, places=5)

    def test_uses_expensive_generator_after_cheap_headroom_is_full(self):
        output = self.call_cell(150.0)
        self.assertEqual(output[:2].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(float(output[2]), 100.0, places=5)
        self.assertAlmostEqual(float(output[3]), 50.0, places=5)


if __name__ == "__main__":
    unittest.main()
