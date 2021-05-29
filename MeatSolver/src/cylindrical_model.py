from tempfile import TemporaryDirectory
from typing import Optional
import os

import tensorflow as tf
import numpy as np
from scipy.special import j0, j1
import matplotlib.pyplot as plt

# Output parameters.
NUM_BETA = 50
NUM_RHO = 100
NUM_TAU = 100
NUSSELT_NUMBER = 10
OUTPUT_FILEPATH = "model/cylindrical_model.tflite"


class CylindricalModel:
    _num_beta: int
    _num_rho: int
    _num_tau: int
    _nusselt_number: float
    _model: Optional[tf.keras.Model]

    # Parameters for determining the transcendental equation. Parameters investigated in notebooks/transcendental_equation
    MIN_NUSSELT_NUMBER = 5.0
    MAX_NUSSELT_NUMBER = 500.0
    MIN_BETA = 0.0
    MAX_BETA = 1000
    NUM_SAMPLES = 500

    # Parameters for the variables, tau and rho
    MIN_TAU = 0.0
    MAX_TAU = 1.0
    MIN_RHO = 0.0
    MAX_RHO = 0.5

    # Base datatype for tensorflow
    DTYPE = tf.float32

    def __init__(self, num_beta: int = 1, num_rho: int = 2, num_tau: int = 2, nusselt_number: float = 50):
        self._num_beta = num_beta
        self._num_rho = num_rho
        self._num_tau = num_tau
        self._nusselt_number = nusselt_number
        self._model = None

    def create(self):
        input = tf.keras.Input(2, dtype=self.DTYPE)
        kappa = input[:, 0]
        theta0 = input[:, 1]
        betas = self._compute_betas()
        rhos, taus = self._make_variables()
        theta = self._compute_theta(rhos, taus, betas, kappa, theta0)
        self._model = tf.keras.Model(inputs=input, outputs=theta)

    def _compute_betas(self) -> np.ndarray:
        nusselt_numbers, betas = self._compute_zero_contours()
        index = np.argmin(np.abs(nusselt_numbers - self._nusselt_number))  # Simple nearest neighbor.
        return betas[:, index]

    def _compute_zero_contours(self) -> (np.ndarray, np.ndarray):
        nusselt_numbers = np.logspace(
            np.log10(self.MIN_NUSSELT_NUMBER),
            np.log10(self.MAX_NUSSELT_NUMBER),
            self.NUM_SAMPLES
        )
        betas = np.linspace(self.MIN_BETA, self.MAX_BETA, self.NUM_SAMPLES)
        nusselt_number_grid, beta_grid = np.meshgrid(nusselt_numbers, betas)
        residuals = CylindricalModel._transcendental_equation(beta_grid, nusselt_number_grid)
        paths = plt.contour(nusselt_number_grid, beta_grid, residuals, levels=[0]).collections[0].get_paths()
        plt.close("all")
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

        return nusselt_numbers, np.array(beta_contours)

    @staticmethod
    def _transcendental_equation(beta: np.ndarray, nusselt_number: np.ndarray) -> np.ndarray:
        """Yields the residual of the transcendental equation"""
        residual = beta * j1(beta / 2.0) - nusselt_number * j0(beta / 2.0)
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

    def _make_variables(self) -> (np.ndarray, np.ndarray):
        """Create 1D tensors for rho and tau"""
        taus = np.linspace(self.MIN_TAU, self.MAX_TAU, self._num_tau)
        # sample the radius such that the volume is uniform
        rhos = [self.MIN_RHO]
        delta = (self.MAX_RHO**2.0-self.MIN_RHO**2.0) / (self._num_rho-1.0)
        for index in range(self._num_rho-1):
            rho = (rhos[index]**2.0 + delta)**0.5
            rhos.append(rho)
        rhos = np.array(rhos)
        return rhos, taus

    @staticmethod
    def _compute_theta(
            rhos: np.ndarray,
            taus: np.ndarray,
            betas: np.ndarray,
            kappa: tf.Tensor,
            theta0: tf.Tensor) -> tf.Tensor:

        taus = np.reshape(taus, (-1, 1, 1))
        rhos = np.reshape(rhos, (1, -1, 1))
        betas = np.reshape(betas, (1, 1, -1))
        temporal_decay = tf.exp(-betas ** 2.0 * kappa * taus)
        radial_component = j0(betas * rhos)
        amplitudes_numerator = 4.0 * theta0 * j1(betas / 2)
        amplitudes_denominator = betas * (j1(betas / 2) ** 2.0 + j0(betas / 2) ** 2.0)
        amplitudes = amplitudes_numerator / amplitudes_denominator
        theta = amplitudes * radial_component * temporal_decay
        theta = tf.reduce_sum(theta, axis=-1)
        return theta

    def plot(self, kappa: float, theta0: float, output_directory: Optional[str] = None):
        assert self._model
        inputs = tf.convert_to_tensor([kappa, theta0], dtype=tf.float32)[tf.newaxis, ...]
        thetas = self._model.predict(inputs)
        rhos, taus = self._make_variables()
        num_tau, num_rho = thetas.shape
        plt.close("all")
        plt.figure(1)
        for i in range(num_tau):
            plt.plot(rhos, thetas[i, :], label=f"{taus[i]}")
        title = f"kappa={kappa}, theta0={theta0}"
        plt.title(title)
        plt.xlabel("rho")
        plt.ylabel("theta")
        plt.legend(loc="best")
        if output_directory:
            plt.savefig(os.path.join(output_directory, title + ".png"))


        plt.figure(2)
        for j in range(num_rho):
            plt.plot(taus, thetas[:, j], label=f"{rhos[j]}")
        plt.title(f"kappa={kappa}, theta0={theta0}")
        plt.xlabel("tau")
        plt.ylabel("theta")
        plt.legend(loc="best")
        if output_directory:
            plt.savefig(os.path.join(output_directory, title + ".png"))


if __name__ == "__main__":
    cylindrical_model = CylindricalModel(NUM_BETA, NUM_RHO, NUM_TAU, NUSSELT_NUMBER)
    cylindrical_model.create()
    cylindrical_model.save("../model/cylindrical_model.tflite")