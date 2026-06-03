#!/usr/bin/env python
"""Focused synthetic tests for look-ahead startup repair."""

import unittest

import tensorflow as tf

from rnncell_strict_allocation_startup_repair_model import (
    LookAheadStartupRepairUCCell,
)


class LookAheadStartupRepairUCCellTest(unittest.TestCase):
    def make_cell(self):
        cell = LookAheadStartupRepairUCCell(
            hidden_dim=2,
            num_gens=2,
            mut_vals=[1, 1],
            mdt_vals=[4, 4],
            p_min_vals=[0.0, 0.0],
            p_max_vals=[100.0, 100.0],
            ramp_up_vals=[10.0, 10.0],
            ramp_down_vals=[100.0, 100.0],
            startup_cap_vals=[10.0, 10.0],
            shutdown_cap_vals=[100.0, 100.0],
            lookahead_hours=4,
            linear_cost_vals=[10.0, 20.0],
        )
        cell.build((1, 6))
        cell.dense_u.kernel.assign(tf.zeros_like(cell.dense_u.kernel))
        cell.dense_u.bias.assign([20.0, -20.0])
        cell.dense_p.kernel.assign(tf.zeros_like(cell.dense_p.kernel))
        cell.dense_p.bias.assign(tf.zeros_like(cell.dense_p.bias))
        return cell

    def call_cell(self, future_demand):
        cell = self.make_cell()
        zero_hidden = tf.zeros((1, 2))
        output, _ = cell(
            tf.constant([[50.0, 24.0, *future_demand]], dtype=tf.float32),
            (
                zero_hidden,
                zero_hidden,
                tf.constant([[50.0, 0.0]], dtype=tf.float32),
                tf.constant([[1.0, 0.0]], dtype=tf.float32),
                tf.constant([[5.0, 0.0]], dtype=tf.float32),
                tf.constant([[0.0, 4.0]], dtype=tf.float32),
            ),
        )
        return output.numpy()[0]

    def test_starts_eligible_unit_before_future_ramp_shortfall(self):
        output = self.call_cell([85.0, 85.0, 85.0, 85.0])
        self.assertGreater(float(output[0]), 0.5)
        self.assertGreater(float(output[1]), 0.5)

    def test_does_not_start_extra_unit_when_future_capacity_is_sufficient(self):
        output = self.call_cell([75.0, 75.0, 75.0, 75.0])
        self.assertGreater(float(output[0]), 0.5)
        self.assertLess(float(output[1]), 0.5)


if __name__ == "__main__":
    unittest.main()
