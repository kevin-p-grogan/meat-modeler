import unittest

import tensorflow as tf

from src.flat_frustum_model_maker import FlatFrustumModelMaker


class TestFlatFrustumModelMaker(unittest.TestCase):
    _test_model_filepath = "model/test_flat_frustum_model.tflite"

    def test_make(self):
        flat_frustum_model_maker = FlatFrustumModelMaker()
        flat_frustum_model_maker.make()
        self.assertIsInstance(flat_frustum_model_maker._model, tf.keras.Model)
        flat_frustum_model_maker.save(self._test_model_filepath)


if __name__ == '__main__':
    unittest.main()
