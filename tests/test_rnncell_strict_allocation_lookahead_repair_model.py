#!/usr/bin/env python
"""Focused synthetic tests for look-ahead commitment repair."""

import unittest

import tensorflow as tf

from rnncell_strict_allocation_lookahead_repair_model import (
    FutureDemandWindow,
    LookAheadCommitmentRepairUCCell,
)


class FutureDemandWindowTest(unittest.TestCase):
    def test_exposes_future_demand_and_zero_pads_terminal_hours(self):
        layer = FutureDemandWindow(lookahead_hours=3)
        output = layer(tf.constant([[[10.0], [20.0], [30.0], [40.0]]]))
        self.assertEqual(
            output.numpy().tolist(),
            [[[20.0, 30.0, 40.0], [30.0, 40.0, 0.0], [40.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        )


class LookAheadCommitmentRepairUCCellTest(unittest.TestCase):
    def make_cell(self, lookahead_safety_margin_mw=0.0):
        cell = LookAheadCommitmentRepairUCCell(
            hidden_dim=2,
            num_gens=2,
            mut_vals=[2, 2],
            mdt_vals=[4, 4],
            p_min_vals=[10.0, 10.0],
            p_max_vals=[100.0, 100.0],
            ramp_up_vals=[10.0, 10.0],
            ramp_down_vals=[10.0, 10.0],
            startup_cap_vals=[10.0, 10.0],
            shutdown_cap_vals=[10.0, 10.0],
            lookahead_hours=4,
            lookahead_safety_margin_mw=lookahead_safety_margin_mw,
        )
        cell.build((1, 6))
        cell.dense_u.kernel.assign(tf.zeros_like(cell.dense_u.kernel))
        cell.dense_p.kernel.assign(tf.zeros_like(cell.dense_p.kernel))
        cell.dense_p.bias.assign(tf.zeros_like(cell.dense_p.bias))
        return cell

    def call_cell(
        self,
        future_demand,
        status_bias=(20.0, 10.0),
        lookahead_safety_margin_mw=0.0,
    ):
        cell = self.make_cell(lookahead_safety_margin_mw)
        cell.dense_u.bias.assign(status_bias)
        zero_hidden = tf.zeros((1, 2))
        output, _ = cell(
            tf.constant([[10.0, 24.0, *future_demand]], dtype=tf.float32),
            (
                zero_hidden,
                zero_hidden,
                tf.constant([[10.0, 10.0]], dtype=tf.float32),
                tf.constant([[1.0, 1.0]], dtype=tf.float32),
                tf.constant([[5.0, 5.0]], dtype=tf.float32),
                tf.constant([[0.0, 0.0]], dtype=tf.float32),
            ),
        )
        return output.numpy()[0]

    def test_keeps_low_score_generator_for_near_term_demand_increase(self):
        output = self.call_cell([50.0, 50.0, 50.0, 50.0])
        self.assertGreater(float(output[0]), 0.5)
        self.assertGreater(float(output[1]), 0.5)

    def test_removes_low_score_generator_when_future_capacity_is_sufficient(self):
        output = self.call_cell([20.0, 20.0, 20.0, 20.0])
        self.assertEqual(output[:2].tolist(), [1.0, 0.0])

    def test_counts_restart_capacity_after_mdt_is_satisfied(self):
        output = self.call_cell([20.0, 30.0, 40.0, 70.0])
        self.assertEqual(output[:2].tolist(), [1.0, 0.0])

    def test_overrides_ai_shutdown_that_would_create_future_shortage(self):
        output = self.call_cell(
            [50.0, 50.0, 50.0, 50.0],
            status_bias=(20.0, -20.0),
        )
        self.assertEqual(output[:2].tolist(), [1.0, 1.0])

    def test_accepts_ai_shutdown_when_future_capacity_is_sufficient(self):
        output = self.call_cell(
            [20.0, 20.0, 20.0, 20.0],
            status_bias=(20.0, -20.0),
        )
        self.assertGreater(float(output[0]), 0.5)
        self.assertLess(float(output[1]), 0.5)

    def test_safety_margin_blocks_shutdown_near_future_capacity_limit(self):
        output = self.call_cell(
            [20.0, 20.0, 20.0, 20.0],
            lookahead_safety_margin_mw=25.0,
        )
        self.assertGreater(float(output[0]), 0.5)
        self.assertGreater(float(output[1]), 0.5)


if __name__ == "__main__":
    unittest.main()
