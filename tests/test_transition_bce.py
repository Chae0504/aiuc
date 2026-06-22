#!/usr/bin/env python
import unittest

import tensorflow as tf

from legacy.rnncell_model import TransitionBinaryCrossentropy


class TransitionBinaryCrossentropyTest(unittest.TestCase):
    def test_adds_startup_shutdown_transition_loss(self):
        loss = TransitionBinaryCrossentropy(
            transition_weight=2.0,
            initial_status_vals=[0.0],
        )
        y_true = tf.constant([[[0.0], [1.0], [0.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[0.0], [0.0], [0.0]]], dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        expected_bce = -tf.math.log(1.0 - epsilon) * 2.0 / 3.0 - tf.math.log(
            epsilon
        ) / 3.0
        # Missed one startup and one shutdown over three transition slots.
        expected_transition = 2.0 / 3.0
        expected = expected_bce + 2.0 * expected_transition

        self.assertAlmostEqual(
            float(loss(y_true, y_pred).numpy()), float(expected), places=5
        )


if __name__ == "__main__":
    unittest.main()
