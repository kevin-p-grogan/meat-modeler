import unittest
import os
from tempfile import NamedTemporaryFile

import tensorflow as tf

from src.pork_tenderloin_solver import PorkTenderloinModel
from src.pork_parameters import MIN_KAPPA, MAX_KAPPA


class TestModelCreation(unittest.TestCase):
    _test_model_filepath = "model/test_model.tflite"

    def test_save_model(self):
        pork_tenderloin_model = PorkTenderloinModel()
        x = tf.cast(tf.linspace(0.0, 0.5, 3), tf.float32)
        y = tf.cast(tf.linspace(0, 1, 3), tf.float32)
        pork_tenderloin_model._model = self._make_test_model(x, y)
        pork_tenderloin_model.save(self._test_model_filepath)
        self.assertTrue(os.path.exists(self._test_model_filepath))

    @staticmethod
    def _make_test_model(x: tf.Tensor, y: tf.Tensor) -> tf.keras.Model:
        input = tf.keras.Input(2, dtype=tf.float32)
        a = input[:, 0]
        b = input[:, 1]
        x, y = tf.meshgrid(x, y)
        output = a * x + b * y
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
        pork_tenderloin_model = PorkTenderloinModel(num_beta=num_beta, nusselt_number=10)
        betas = pork_tenderloin_model._compute_betas()
        self.assertEqual(betas.shape, (num_beta,))

    def test_saving_model_with_betas(self):
        num_beta = 3
        pork_tenderloin_model = PorkTenderloinModel(num_beta=num_beta, nusselt_number=10)
        x = pork_tenderloin_model._compute_betas()
        pork_tenderloin_model._nusselt_number = 50.0
        y = pork_tenderloin_model._compute_betas()
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        pork_tenderloin_model._model = self._make_test_model(x, y)
        self._test_save(pork_tenderloin_model)

    def _test_save(self, pork_tenderloin_model: PorkTenderloinModel):
        with NamedTemporaryFile(dir="model") as tmp_file:
            pork_tenderloin_model.save(tmp_file.name)
            self.assertTrue(os.path.exists(tmp_file.name))

    def test_make_variables(self):
        num_rho = 10
        num_tau = 4
        pork_tenderloin_model = PorkTenderloinModel(num_rho=num_rho, num_tau=num_tau)
        rhos, taus = pork_tenderloin_model._make_variables()
        self.assertEqual(len(rhos), num_rho)
        self.assertEqual(len(taus), num_tau)
        self.assertEqual(rhos[0], pork_tenderloin_model.MIN_RHO)
        self.assertAlmostEqual(rhos[-1], pork_tenderloin_model.MAX_RHO, places=3)
        self.assertEqual(taus[0], pork_tenderloin_model.MIN_TAU)
        self.assertAlmostEqual(taus[-1], pork_tenderloin_model.MAX_TAU, places=3)

    def test_create(self):
        num_rho = 10
        num_tau = 10
        num_beta = 50
        pork_tenderloin_model = PorkTenderloinModel(num_rho=num_rho, num_tau=num_tau, num_beta=num_beta)
        pork_tenderloin_model.create()
        pork_tenderloin_model.plot(kappa=MIN_KAPPA, theta0=-1.0, output_directory='.')
        pork_tenderloin_model.plot(kappa=MAX_KAPPA, theta0=-1.0, output_directory='.')
        self.assertIsInstance(pork_tenderloin_model._model, tf.keras.Model)
        self._test_save(pork_tenderloin_model)


if __name__ == '__main__':
    unittest.main()
