#!/usr/bin/env python
"""Focused synthetic tests for the strict-clipping baseline UC cell."""

import unittest

import tensorflow as tf

from baselines.strict_clipping.rnncell_strict_model import StrictClippingUCCell


class StrictClippingUCCellTest(unittest.TestCase):
    def make_cell(self):
        cell = StrictClippingUCCell(
            hidden_dim=2,
            num_gens=1,
            mut_vals=[5],
            mdt_vals=[4],
            p_min_vals=[10.0],
            p_max_vals=[100.0],
            ramp_up_vals=[10.0],
            ramp_down_vals=[10.0],
            startup_cap_vals=[10.0],
            shutdown_cap_vals=[10.0],
        )
        cell.build((1, 2))
        cell.dense_u.kernel.assign(tf.zeros_like(cell.dense_u.kernel))
        cell.dense_p.kernel.assign(tf.zeros_like(cell.dense_p.kernel))
        cell.dense_p.bias.assign(tf.zeros_like(cell.dense_p.bias))
        return cell

    def call_cell(
        self,
        *,
        status_bias,
        demand,
        remaining,
        previous_power,
        previous_status,
        previous_up,
        previous_down,
    ):
        cell = self.make_cell()
        cell.dense_u.bias.assign([status_bias])
        zero_hidden = tf.zeros((1, 2))
        output, _ = cell(
            tf.constant([[demand, remaining]], dtype=tf.float32),
            (
                zero_hidden,
                zero_hidden,
                tf.constant([[previous_power]], dtype=tf.float32),
                tf.constant([[previous_status]], dtype=tf.float32),
                tf.constant([[previous_up]], dtype=tf.float32),
                tf.constant([[previous_down]], dtype=tf.float32),
            ),
        )
        return output.numpy()[0]

    def test_startup_is_limited_to_startup_cap(self):
        output = self.call_cell(
            status_bias=20.0,
            demand=100.0,
            remaining=24.0,
            previous_power=0.0,
            previous_status=0.0,
            previous_up=0.0,
            previous_down=5.0,
        )
        self.assertAlmostEqual(output[0], 1.0, places=5)
        self.assertAlmostEqual(output[1], 10.0, places=5)

    def test_stay_online_output_is_clipped_without_allocation(self):
        output = self.call_cell(
            status_bias=20.0,
            demand=100.0,
            remaining=24.0,
            previous_power=50.0,
            previous_status=1.0,
            previous_up=5.0,
            previous_down=0.0,
        )
        self.assertAlmostEqual(output[0], 1.0, places=5)
        self.assertAlmostEqual(output[1], 55.0, places=5)

    def test_shutdown_is_blocked_above_shutdown_cap(self):
        output = self.call_cell(
            status_bias=-20.0,
            demand=0.0,
            remaining=24.0,
            previous_power=50.0,
            previous_status=1.0,
            previous_up=5.0,
            previous_down=0.0,
        )
        self.assertAlmostEqual(output[0], 1.0, places=5)
        self.assertAlmostEqual(output[1], 55.0, places=5)


if __name__ == "__main__":
    unittest.main()
