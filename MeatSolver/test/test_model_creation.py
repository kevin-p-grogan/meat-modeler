import unittest
import os

import tensorflow as tf

from src.pork_tenderloin_solver import save_tflite_model

class TestModelCreation(unittest.TestCase):
    _test_model_filepath = "model/test_model.tflite"

    def test_save_model(self):
        model = self._make_test_model()
        save_tflite_model(model, self._test_model_filepath)
        self.assertTrue(os.path.exists(self._test_model_filepath))

    @staticmethod
    def _make_test_model() -> tf.keras.Model:
        input = tf.keras.Input(3, dtype=tf.float32)
        a = input[:, 0]
        b = input[:, 1]
        c = input[:, 2]
        x = tf.cast(tf.linspace(0, 1, 3), tf.float32)
        y = tf.cast(tf.linspace(0, 1, 3), tf.float32)
        x, y = tf.meshgrid(x, y)
        output = a * x + b * y + c
        model = tf.keras.Model(inputs=input, outputs=output)
        return model


if __name__ == '__main__':
    unittest.main()
