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


if __name__ == '__main__':
    unittest.main()
