#!/usr/bin/env python
"""Focused synthetic tests for MUT/MDT-aware commitment repair."""

import unittest

import tensorflow as tf

from rnncell_strict_allocation_repair_model import (
    MutMdtAwareCommitmentRepairUCCell,
)


class MutMdtAwareCommitmentRepairUCCellTest(unittest.TestCase):
    def make_cell(self, mut=5, mdt=4):
        cell = MutMdtAwareCommitmentRepairUCCell(
            hidden_dim=2,
            num_gens=2,
            mut_vals=[mut, mut],
            mdt_vals=[mdt, mdt],
            p_min_vals=[10.0, 10.0],
            p_max_vals=[100.0, 100.0],
            ramp_up_vals=[10.0, 10.0],
            ramp_down_vals=[10.0, 10.0],
            startup_cap_vals=[10.0, 10.0],
            shutdown_cap_vals=[10.0, 10.0],
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
        cell.dense_u.bias.assign(status_bias)
        zero_hidden = tf.zeros((1, 2))
        output, _ = cell(
            tf.constant([[demand, remaining]], dtype=tf.float32),
            (
                zero_hidden,
                zero_hidden,
                tf.constant([previous_power], dtype=tf.float32),
                tf.constant([previous_status], dtype=tf.float32),
                tf.constant([previous_up], dtype=tf.float32),
                tf.constant([previous_down], dtype=tf.float32),
            ),
        )
        return output.numpy()[0]

    def test_starts_mdt_eligible_generator_to_cover_shortfall(self):
        output = self.call_cell(
            status_bias=[20.0, -20.0],
            demand=30.0,
            remaining=24.0,
            previous_power=[10.0, 0.0],
            previous_status=[1.0, 0.0],
            previous_up=[5.0, 0.0],
            previous_down=[0.0, 5.0],
        )
        self.assertEqual(output[:2].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(sum(output[2:]), 30.0, places=5)

    def test_does_not_start_generator_before_mdt_is_satisfied(self):
        output = self.call_cell(
            status_bias=[20.0, -20.0],
            demand=30.0,
            remaining=24.0,
            previous_power=[10.0, 0.0],
            previous_status=[1.0, 0.0],
            previous_up=[5.0, 0.0],
            previous_down=[0.0, 2.0],
        )
        self.assertAlmostEqual(float(output[0]), 1.0)
        self.assertAlmostEqual(float(output[1]), 0.0)
        self.assertAlmostEqual(sum(output[2:]), 20.0, places=5)

    def test_does_not_start_generator_too_late_to_honor_mut(self):
        output = self.call_cell(
            status_bias=[20.0, -20.0],
            demand=30.0,
            remaining=4.0,
            previous_power=[10.0, 0.0],
            previous_status=[1.0, 0.0],
            previous_up=[5.0, 0.0],
            previous_down=[0.0, 5.0],
        )
        self.assertAlmostEqual(float(output[0]), 1.0)
        self.assertAlmostEqual(float(output[1]), 0.0)
        self.assertAlmostEqual(sum(output[2:]), 20.0, places=5)

    def test_keeps_online_generator_when_its_capacity_is_needed(self):
        output = self.call_cell(
            status_bias=[-20.0, -20.0],
            demand=20.0,
            remaining=24.0,
            previous_power=[10.0, 0.0],
            previous_status=[1.0, 0.0],
            previous_up=[5.0, 0.0],
            previous_down=[0.0, 5.0],
        )
        self.assertAlmostEqual(float(output[0]), 1.0)
        self.assertAlmostEqual(float(output[1]), 0.0)
        self.assertAlmostEqual(sum(output[2:]), 20.0, places=5)

    def test_removes_low_score_generator_when_shutdown_is_safe(self):
        output = self.call_cell(
            status_bias=[20.0, 10.0],
            demand=10.0,
            remaining=24.0,
            previous_power=[10.0, 10.0],
            previous_status=[1.0, 1.0],
            previous_up=[5.0, 5.0],
            previous_down=[0.0, 0.0],
        )
        self.assertEqual(output[:2].tolist(), [1.0, 0.0])
        self.assertAlmostEqual(sum(output[2:]), 10.0, places=5)

    def test_does_not_shutdown_generator_before_mut_is_satisfied(self):
        output = self.call_cell(
            status_bias=[20.0, 10.0],
            demand=10.0,
            remaining=24.0,
            previous_power=[10.0, 10.0],
            previous_status=[1.0, 1.0],
            previous_up=[2.0, 2.0],
            previous_down=[0.0, 0.0],
        )
        self.assertEqual(output[:2].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(sum(output[2:]), 20.0, places=5)


if __name__ == "__main__":
    unittest.main()
