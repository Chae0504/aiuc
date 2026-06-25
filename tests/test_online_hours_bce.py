#!/usr/bin/env python
import unittest

import tensorflow as tf

from legacy.rnncell_model import OnlineHoursBinaryCrossentropy


class OnlineHoursBinaryCrossentropyTest(unittest.TestCase):
    def test_adds_normalized_online_hours_loss(self):
        loss = OnlineHoursBinaryCrossentropy(online_hours_weight=3.0)
        y_true = tf.constant([[[1.0, 0.0], [1.0, 0.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[1.0, 1.0], [1.0, 1.0]]], dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        expected_bce = tf.reduce_mean(
            -(
                y_true * tf.math.log(y_pred_clipped)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred_clipped)
            )
        )
        expected_online = 2.0 / 4.0
        expected = expected_bce + 3.0 * expected_online

        self.assertAlmostEqual(
            float(loss(y_true, y_pred).numpy()), float(expected), places=5
        )


if __name__ == "__main__":
    unittest.main()
