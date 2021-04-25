from tempfile import TemporaryDirectory

import keras
import tensorflow as tf


def save_tflite_model(model: keras.Model, filepath: str):
    """Saves the model as a tflite model to a filepath."""
    with TemporaryDirectory(dir='.') as tmp_dir:
        model.save(tmp_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        tflite_model = converter.convert()

        with open(filepath, 'wb') as f:
            f.write(tflite_model)