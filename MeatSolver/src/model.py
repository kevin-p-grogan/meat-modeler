from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Optional

import tensorflow as tf


class ModelMaker(ABC):
    _model: Optional[tf.keras.Model]

    # Base datatype for tensorflow
    DTYPE = tf.float32

    @abstractmethod
    def make(self):
        pass

    def save(self, filepath: str):
        """Saves the model as a tflite model to a filepath."""
        assert self._model
        with TemporaryDirectory(dir='.') as tmp_dir:
            self._model.save(tmp_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
            tflite_model = converter.convert()

            with open(filepath, 'wb') as f:
                f.write(tflite_model)