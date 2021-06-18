from typing import Dict
from math import pi

import tensorflow as tf
import numpy as np

from src.model import ModelMaker


class FlatFrustumModelMaker(ModelMaker):
    # Runtime Parameters
    _diffusivity: tf.Tensor
    _nusselt_number: tf.Tensor
    _theta0: tf.Tensor
    _aspect_ratio: tf.Tensor
    _pointiness: tf.Tensor
    _flatness: tf.Tensor
    _radial_bc_constants: tf.Tensor

    # Model definition-time variables
    _rho: tf.Tensor
    _phi: tf.Tensor
    _xi: tf.Tensor
    _num_rho: int
    _num_phi: int
    _num_xi: int

    # Runtime variables
    _tau: tf.Variable
    _dtau: tf.constant

    # simulation constants
    NUM_RUNTIME_PARAMETERS = 6
    NUM_TAU = 100
    MIN_TAU = 0.0
    MAX_TAU = 1.0
    MIN_RHO = 0.0
    MAX_RHO = 0.5
    MIN_PHI = 0.0
    MAX_PHI = 2.0 * pi
    MIN_XI = 0.0
    DTYPE = tf.float32
    NUM_GHOST = 1
    NUM_DIMS = 3

    def __init__(self, num_rho: int = 2, num_phi: int = 2, num_xi: int = 2):
        self._num_rho = num_rho
        self._num_phi = num_phi
        self._num_xi = num_xi
        self._tau = tf.Variable(self.MIN_TAU)
        self._dtau = tf.constant((self.MAX_TAU - self.MIN_TAU) / self.NUM_TAU, dtype=self.DTYPE)

    def make(self):
        parameters = tf.keras.Input(self.NUM_RUNTIME_PARAMETERS, dtype=self.DTYPE, batch_size=1)
        theta = self._initialize(parameters)
        for _ in range(self.NUM_TAU):
            theta = self._advance(theta)

        theta = self._finalize(theta)
        self._model = tf.keras.Model(inputs=parameters, outputs=theta)

    def _initialize(self, parameters: tf.Tensor) -> (tf.Tensor, tf.Variable):
        self._set_runtime_parameters(parameters)
        self._set_domain_variables()
        self._set_boundary_constants()
        theta = self._create_theta()
        return theta

    def _set_runtime_parameters(self, parameters: tf.Tensor):
        self._diffusivity = parameters[0, 0]
        self._nusselt_number = parameters[0, 1]
        self._theta0 = parameters[0, 2]
        self._aspect_ratio = parameters[0, 3]
        self._pointiness = parameters[0, 4]
        self._flatness = parameters[0, 5]

    def _create_theta(self) -> tf.Tensor:
        shape = self._num_rho, self._num_phi, self._num_xi
        theta = self._theta0 * tf.ones(shape, dtype=self.DTYPE)
        paddings = [[self.NUM_GHOST, self.NUM_GHOST]] * len(shape)
        theta = tf.pad(theta, paddings)  # zero pad the boundaries
        return theta

    def _set_domain_variables(self):
        """Defined cell-centered coordinates for the domain variables"""
        min_rho, max_rho = tf.constant(self.MIN_RHO), tf.constant(self.MAX_RHO)
        self._rho = self._create_domain(lower=min_rho, upper=max_rho, num_cells=self._num_rho, axis=0)
        min_phi, max_phi = tf.constant(self.MIN_PHI), tf.constant(self.MAX_PHI)
        self._phi = self._create_domain(lower=min_phi, upper=max_phi, num_cells=self._num_phi, axis=1)
        min_xi, max_xi = tf.constant(self.MIN_XI), self._aspect_ratio
        self._xi = self._create_domain(lower=min_xi, upper=max_xi, num_cells=self._num_xi, axis=2)

    def _set_boundary_constants(self):
        pass

    def _create_domain(self,
                       lower: tf.Tensor,
                       upper: tf.Tensor,
                       num_cells: int,
                       axis: int) -> tf.Tensor:
        """Creates a cell-centered grid with ghost cells."""
        delta = (upper - lower) / float(num_cells)
        num_total_cells = num_cells + 2*self.NUM_GHOST
        start = lower - delta/2.0
        end = upper + delta/2.0
        domain = start + (end-start)*np.linspace(0.0, 1.0, num_total_cells)
        shape = [num_total_cells if ax == axis else 1 for ax in range(self.NUM_DIMS)]
        domain = tf.reshape(domain, shape=shape)
        return domain

    def _advance(self, theta: tf.Tensor):
        theta = self._set_boundary(theta)
        gradients = self._compute_gradients(theta)
        theta = self._update(theta, gradients)
        return theta

    def _set_boundary(self, theta: tf.Tensor) -> tf.Tensor:
        # TODO: set boundary conditions
        return theta

    def _compute_gradients(self, theta: tf.Tensor) -> Dict[str, tf.Tensor]:
        # TODO: compute gradients
        return {}

    def _update(self, theta: tf.Tensor, gradients: Dict[str, tf.Tensor]) -> tf.Tensor:
        theta = theta + self._dtau * self._diffusivity * theta
        self._tau = self._tau + self._dtau
        return theta

    @staticmethod
    def _finalize(theta: tf.Tensor) -> tf.Tensor:
        return theta[1:-1, 1:-1, 1:-1]
