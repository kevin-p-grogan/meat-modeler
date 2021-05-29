from typing import Dict

import tensorflow as tf

from src.model import ModelMaker


class FlatFrustumModelMaker(ModelMaker):
    _kappa: tf.Tensor
    _tau: tf.Variable
    _dtau: tf.constant
    _num_rho: int
    _num_theta: int
    _num_xi: int

    # simulation parameters
    NUM_RUNTIME_PARAMETERS = 2
    NUM_TAU = 100
    MIN_TAU = 0.0
    MAX_TAU = 1.0

    def __init__(self, num_rho: int = 2, num_theta: int = 2, num_xi: int = 2):
        self._num_rho = num_rho
        self._num_theta = num_theta
        self._num_xi = num_xi

    def make(self):
        parameters = tf.keras.Input(self.NUM_RUNTIME_PARAMETERS, dtype=self.DTYPE, batch_size=1)
        theta = self._initialize(parameters)
        for _ in range(self.NUM_TAU):
            theta = self._advance(theta)

        self._model = tf.keras.Model(inputs=parameters, outputs=theta)

    def _initialize(self, parameters: tf.Tensor) -> (tf.Tensor, tf.Variable):
        self._kappa = parameters[0, 0]
        self._tau = tf.Variable(self.MIN_TAU)
        self._dtau = tf.constant((self.MAX_TAU - self.MIN_TAU) / self.NUM_TAU, dtype=self.DTYPE)
        theta = parameters[0, 1] * tf.ones((self._num_rho, self._num_theta, self._num_xi), dtype=self.DTYPE)
        return theta

    def _advance(self, theta: tf.Tensor):
        theta = self._set_boundary(theta)
        gradients = self._compute_gradients(theta)
        theta = self._update(theta, gradients)
        return theta

    def _set_boundary(self, theta: tf.Tensor) -> tf.Tensor:
        # TODO: set boundary conditions
        return theta

    def _compute_gradients(self, theta: tf.Tensor) -> Dict[str, tf.Tensor]:
        # TODO: compute gradiebnts
        return {}

    def _update(self, theta: tf.Tensor, gradients: Dict[str, tf.Tensor]) -> tf.Tensor:
        theta = theta + self._dtau * self._kappa * theta
        self._tau = self._tau + self._dtau
        return theta