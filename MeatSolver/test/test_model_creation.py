import unittest
import os
from tempfile import NamedTemporaryFile

import tensorflow as tf

from src.pork_tenderloin_solver import PorkTenderloinModel


class TestModelCreation(unittest.TestCase):
    _test_model_filepath = "model/test_model.tflite"

    def test_save_model(self):
        pork_tenderloin_model = PorkTenderloinModel()
        x = tf.cast(tf.linspace(0, 1, 3), tf.float32)
        y = tf.cast(tf.linspace(0, 1, 3), tf.float32)
        pork_tenderloin_model._model = self._make_test_model(x, y)
        pork_tenderloin_model.save(self._test_model_filepath)
        self.assertTrue(os.path.exists(self._test_model_filepath))

    @staticmethod
    def _make_test_model(x: tf.Tensor, y: tf.Tensor) -> tf.keras.Model:
        input = tf.keras.Input(3, dtype=tf.float32)
        a = input[:, 0]
        b = input[:, 1]
        c = input[:, 2]
        x, y = tf.meshgrid(x, y)
        output = a * x + b * y + c
        model = tf.keras.Model(inputs=input, outputs=output)
        return model

    def test_get_zero_contours(self):
        num_beta = 10
        pork_tenderloin_model = PorkTenderloinModel(num_beta=num_beta)
        nusselt_numbers, beta_contours = pork_tenderloin_model._compute_zero_contours()
        self.assertEqual(num_beta, beta_contours.shape[0])
        self.assertEqual(nusselt_numbers.shape[0], beta_contours.shape[1])

    def test_compute_betas(self):
        num_beta = 3
        pork_tenderloin_model = PorkTenderloinModel(num_beta=num_beta)
        betas = pork_tenderloin_model._compute_betas(nusselt_number=10 * tf.ones(1))
        self.assertEqual(betas.shape, (num_beta,))

    def test_saving_model_with_betas(self):
        num_beta = 3
        pork_tenderloin_model = PorkTenderloinModel(num_beta=num_beta)
        x = pork_tenderloin_model._compute_betas(nusselt_number=10 * tf.ones(1))
        y = pork_tenderloin_model._compute_betas(nusselt_number=50 * tf.ones(1))
        pork_tenderloin_model._model = self._make_test_model(x, y)
        with NamedTemporaryFile(dir="model") as tmp_file:
            pork_tenderloin_model.save(tmp_file.name)
            self.assertTrue(os.path.exists(tmp_file.name))


if __name__ == '__main__':
    unittest.main()
