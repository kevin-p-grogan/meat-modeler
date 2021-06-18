import unittest

import numpy as np
import tensorflow as tf

from src.flat_frustum_model_maker import FlatFrustumModelMaker


class TestFlatFrustumModelMaker(unittest.TestCase):
    _test_model_filepath = "model/test_flat_frustum_model.tflite"

    def test_make(self):
        flat_frustum_model_maker = FlatFrustumModelMaker()
        flat_frustum_model_maker.make()
        self.assertIsInstance(flat_frustum_model_maker._model, tf.keras.Model)
        flat_frustum_model_maker.save(self._test_model_filepath)

    def test_create_domain(self):
        model_maker = FlatFrustumModelMaker()
        lower = tf.constant(0.0)
        upper = tf.constant(1.0)
        num_cells = 10
        for ax in range(model_maker.NUM_DIMS):
            domain = model_maker._create_domain(lower, upper, num_cells, ax)
            rank = len(domain.shape)
            self.assertEqual(rank, model_maker.NUM_DIMS)
            self.assertLessEqual(tf.reduce_min(domain), lower)
            self.assertGreaterEqual(tf.reduce_max(domain), upper)
            self.assertEqual(np.argmax(domain.shape), ax)
            non_axis_dimensions_are_singletons = all(d == 1 for i, d in enumerate(domain.shape) if i != ax)
            self.assertTrue(non_axis_dimensions_are_singletons)

        # Check to see that this works for a KerasTensor input
        lower = tf.keras.Input(3, batch_size=1)[0, 0]
        domain = model_maker._create_domain(lower, upper, num_cells, 0)
        self.assertEqual(np.argmax(domain.shape), 0)

    def test_get_radial_boundary_coefficients(self):
        model_maker = FlatFrustumModelMaker()
        one = tf.constant(1.0, dtype=tf.float32)
        zero = tf.constant(0, dtype=tf.float32)
        a = one
        r = 0.5
        rho = model_maker._create_domain(zero, r * one, 10, axis=0)
        phi = model_maker._create_domain(zero, 2*np.pi * one, 10, axis=1)
        xi = model_maker._create_domain(zero, a, 10, axis=2)
        # use asymptotic coefficients for verification
        p = tf.constant(0.0, dtype=tf.float32)
        f = tf.constant(0.0, dtype=tf.float32)
        predicted_coefficients = model_maker._get_radial_boundary_constants(a, f, p, phi, rho, xi)
        shape = max(phi.shape), max(xi.shape)
        self.assertEqual(shape + (3, 3), predicted_coefficients.shape)

        phi_matrix = tf.repeat(tf.squeeze(phi)[..., tf.newaxis], max(xi.shape), axis=-1)
        actual_coefficeints = tf.stack([
            tf.stack([tf.cos(phi_matrix), tf.sin(phi_matrix), tf.zeros(shape)], axis=-1),
            tf.stack([-tf.sin(phi_matrix) / r, tf.cos(phi_matrix) / r, tf.zeros(shape)], axis=-1),
            tf.stack([tf.zeros(shape), tf.zeros(shape), tf.ones(shape)], axis=-1),
        ], axis=-1)
        self.assertTrue(np.allclose(actual_coefficeints, predicted_coefficients))

        # test that the routine work with Keras Tensors
        kt = tf.keras.Input(3, batch_size=1)[0, 0]
        predicted_coefficients = model_maker._get_radial_boundary_constants(a * kt, f * kt, p * kt, phi * kt, rho * kt,
                                                                            xi * kt)
        self.assertEqual(shape + (3, 3), predicted_coefficients.shape)


if __name__ == '__main__':
    unittest.main()
