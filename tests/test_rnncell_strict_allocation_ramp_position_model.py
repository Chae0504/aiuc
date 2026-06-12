#!/usr/bin/env python
"""Focused synthetic tests for ramp-position-aware allocation."""

import unittest

import tensorflow as tf

from rnncell_strict_allocation_ramp_position_model import (
    RampPositionAwareStartupRepairUCCell,
)


class RampPositionAwareStartupRepairUCCellTest(unittest.TestCase):
    def make_cell(self):
        cell = RampPositionAwareStartupRepairUCCell(
            hidden_dim=2,
            num_gens=2,
            mut_vals=[1, 1],
            mdt_vals=[1, 1],
            p_min_vals=[0.0, 0.0],
            p_max_vals=[100.0, 100.0],
            ramp_up_vals=[100.0, 10.0],
            ramp_down_vals=[100.0, 100.0],
            startup_cap_vals=[100.0, 100.0],
            shutdown_cap_vals=[100.0, 100.0],
            lookahead_hours=1,
            linear_cost_vals=[10.0, 100.0],
        )
        cell.build((1, 3))
        return cell

    def test_repositions_dispatch_to_make_next_hour_reachable(self):
        cell = self.make_cell()
        p_lower = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        p_upper = tf.constant([[100.0, 100.0]], dtype=tf.float32)
        u_final = tf.constant([[1.0, 1.0]], dtype=tf.float32)
        p_clip = tf.constant([[50.0, 50.0]], dtype=tf.float32)
        output = cell._finalize_power(
            tf.constant([100.0], dtype=tf.float32),
            p_clip,
            p_lower,
            p_upper,
            u_final,
            cell_inputs=tf.constant([[100.0, 24.0, 150.0]], dtype=tf.float32),
        )

        self.assertAlmostEqual(float(tf.reduce_sum(output).numpy()), 100.0, places=5)
        self.assertAlmostEqual(float(output[0, 0].numpy()), 60.0, places=5)
        self.assertAlmostEqual(float(output[0, 1].numpy()), 40.0, places=5)
        next_upper = tf.reduce_sum(tf.minimum(cell.p_max, output + cell.ru) * u_final)
        self.assertAlmostEqual(float(next_upper.numpy()), 150.0, places=5)

    def test_keeps_cost_aware_dispatch_when_next_hour_is_reachable(self):
        cell = self.make_cell()
        p_lower = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        p_upper = tf.constant([[100.0, 100.0]], dtype=tf.float32)
        u_final = tf.constant([[1.0, 1.0]], dtype=tf.float32)
        p_clip = tf.constant([[50.0, 50.0]], dtype=tf.float32)
        output = cell._finalize_power(
            tf.constant([100.0], dtype=tf.float32),
            p_clip,
            p_lower,
            p_upper,
            u_final,
            cell_inputs=tf.constant([[100.0, 24.0, 110.0]], dtype=tf.float32),
        )

        self.assertAlmostEqual(float(output[0, 0].numpy()), 100.0, places=5)
        self.assertAlmostEqual(float(output[0, 1].numpy()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
