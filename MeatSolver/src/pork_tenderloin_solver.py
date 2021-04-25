from tempfile import TemporaryDirectory
from typing import Optional, Dict

import keras
import tensorflow as tf
import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt

# Output parameters.
NUM_BETA = 10
NUM_RHO = 10
NUM_TAU = 10
OUTPUT_FILEPATH = "model/pork_tenderloin.tflite"

# Parameters for determining the transcendental equation. Parameters investigated in notebooks/transcendental_equation
MIN_NUSSELT_NUMBER = 5.0
MAX_NUSSELT_NUMBER = 500.0
MIN_BETA = 0.0
MAX_BETA = 100
NUM_SAMPLES = 500

# Base datatype for tensorflow
DTYPE = tf.float32


class PorkTenderloinModel:
    _num_beta: int
    _num_rho: int
    _num_tau: int
    _model: Optional[keras.Model]

    def __init__(self, num_beta: int = 1, num_rho: int = 2, num_tau: int = 2):
        self._num_beta = num_beta
        self._num_rho = num_rho
        self._num_tau = num_tau
        self._model = None

    def create(self):
        input = tf.keras.Input(3, dtype=tf.float32)
        kappa, theta, nusselt_number = (input[:, i] for i in range(3))
        betas = self._compute_betas(nusselt_number)
        pass  # TODO: Compute Theta

    def _compute_betas(self, nusselt_number: tf.Tensor) -> tf.Tensor:
        nusselt_numbers, betas = self._compute_zero_contours()
        index = tf.argmin(tf.abs(nusselt_numbers - nusselt_number))  # Simple nearest neighbor.
        return betas[:, index]

    def _compute_zero_contours(self) -> (tf.Tensor, tf.Tensor):
        nusselt_numbers = np.logspace(np.log10(MIN_NUSSELT_NUMBER), np.log10(MAX_NUSSELT_NUMBER), NUM_SAMPLES)
        betas = np.linspace(MIN_BETA, MAX_BETA, NUM_SAMPLES)
        nusselt_number_grid, beta_grid = np.meshgrid(nusselt_numbers, betas)
        residuals = PorkTenderloinModel._transcendental_equation(beta_grid, nusselt_number_grid)
        paths = plt.contour(nusselt_number_grid, beta_grid, residuals, levels=[0]).collections[0].get_paths()
        paths = sorted(paths, key=lambda p: p.vertices[:, 1].min())
        beta_contours = []
        assert len(paths) >= self._num_beta
        for path in paths[:self._num_beta]:
            nusselt_number_contour = path.vertices[:, 0]
            beta_contour = path.vertices[:, 1]
            sort_indices = np.argsort(nusselt_number_contour)
            nusselt_number_contour = nusselt_number_contour[sort_indices]
            beta_contour = beta_contour[sort_indices]
            beta_contour = np.interp(nusselt_numbers, nusselt_number_contour, beta_contour)
            beta_contours.append(beta_contour)

        beta_contours = tf.convert_to_tensor(beta_contours, dtype=DTYPE)
        nusselt_numbers = tf.convert_to_tensor(nusselt_numbers, dtype=DTYPE)
        return nusselt_numbers, beta_contours

    @staticmethod
    def _transcendental_equation(beta: np.ndarray, nusselt_number: np.ndarray) -> np.ndarray:
        """Yields the residual of the transcendental equation"""
        residual = beta * jv(1, beta / 2.0) - nusselt_number * jv(0, beta / 2.0)
        return residual

    def save(self, filepath: str):
        """Saves the model as a tflite model to a filepath."""
        assert self._model
        with TemporaryDirectory(dir='.') as tmp_dir:
            self._model.save(tmp_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
            tflite_model = converter.convert()

            with open(filepath, 'wb') as f:
                f.write(tflite_model)


if __name__ == "__main__":
    pork_tenderloin_model = PorkTenderloinModel(NUM_BETA, NUM_RHO, NUM_TAU)
    pork_tenderloin_model.create()
    pork_tenderloin_model.save("model/pork_tenderloin.tflite")