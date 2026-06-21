#!/usr/bin/env python
import unittest

import tensorflow as tf

from legacy.rnncell_model import CostWeightedBinaryCrossentropy


class CostWeightedBinaryCrossentropyTest(unittest.TestCase):
    def test_false_on_weight_is_generator_specific(self):
        loss = CostWeightedBinaryCrossentropy([1.0, 2.0])
        y_true = tf.constant([[[0.0, 0.0], [1.0, 1.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[0.25, 0.25], [0.75, 0.75]]], dtype=tf.float32)
        bce = -tf.math.log(0.75)
        expected = (bce * 1.0 + bce * 2.0 + bce + bce) / 4.0
        self.assertAlmostEqual(
            float(loss(y_true, y_pred).numpy()), float(expected), places=6
        )


if __name__ == "__main__":
    unittest.main()
