#!/usr/bin/env python
"""Focused synthetic tests for strict UC transition handling."""

import unittest

import tensorflow as tf

from rnncell_strict_allocation_model import StrictRampAwareAllocationUCCell


class StrictRampAwareAllocationUCCellTest(unittest.TestCase):
    def make_cell(self, mut=5, mdt=4):
        cell = StrictRampAwareAllocationUCCell(
            hidden_dim=2,
            num_gens=1,
            mut_vals=[mut],
            mdt_vals=[mdt],
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
        cell,
        *,
        status_bias,
        demand,
        remaining,
        previous_power,
        previous_status,
        previous_up,
        previous_down,
    ):
        cell.dense_u.bias.assign([status_bias])
        zero_hidden = tf.zeros((1, 2))
        output, states = cell(
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
        return output.numpy()[0], tuple(state.numpy()[0] for state in states)

    def test_startup_is_limited_to_startup_cap(self):
        output, _ = self.call_cell(
            self.make_cell(),
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

    def test_shutdown_is_blocked_above_shutdown_cap(self):
        output, _ = self.call_cell(
            self.make_cell(),
            status_bias=-20.0,
            demand=0.0,
            remaining=24.0,
            previous_power=50.0,
            previous_status=1.0,
            previous_up=5.0,
            previous_down=0.0,
        )
        self.assertAlmostEqual(output[0], 1.0, places=5)
        self.assertAlmostEqual(output[1], 40.0, places=5)

    def test_shutdown_is_allowed_at_shutdown_cap(self):
        output, _ = self.call_cell(
            self.make_cell(),
            status_bias=-20.0,
            demand=0.0,
            remaining=24.0,
            previous_power=10.0,
            previous_status=1.0,
            previous_up=5.0,
            previous_down=0.0,
        )
        self.assertAlmostEqual(output[0], 0.0, places=5)
        self.assertAlmostEqual(output[1], 0.0, places=5)

    def test_terminal_startup_is_blocked(self):
        output, _ = self.call_cell(
            self.make_cell(),
            status_bias=20.0,
            demand=100.0,
            remaining=4.0,
            previous_power=0.0,
            previous_status=0.0,
            previous_up=0.0,
            previous_down=5.0,
        )
        self.assertAlmostEqual(output[0], 0.0, places=5)
        self.assertAlmostEqual(output[1], 0.0, places=5)

    def test_minimum_down_time_handles_zero_elapsed_hours(self):
        output, _ = self.call_cell(
            self.make_cell(),
            status_bias=20.0,
            demand=100.0,
            remaining=24.0,
            previous_power=0.0,
            previous_status=0.0,
            previous_up=0.0,
            previous_down=0.0,
        )
        self.assertAlmostEqual(output[0], 0.0, places=5)
        self.assertAlmostEqual(output[1], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
