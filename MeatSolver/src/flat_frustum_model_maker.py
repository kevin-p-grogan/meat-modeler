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
    _boundary_transformation_vector: tf.Tensor

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
        self._set_boundary_transformation_vector()
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

    def _set_boundary_transformation_vector(self):
        transformation_matrix = self._get_boundary_transformation_matrix(a=self._aspect_ratio,
                                                                         f=self._flatness,
                                                                         p=self._pointiness,
                                                                         rho=self._rho,
                                                                         phi=self._phi,
                                                                         xi=self._xi)
        normal_vector = self._get_boundary_normal_vector(a=self._aspect_ratio,
                                                         f=self._flatness,
                                                         p=self._pointiness,
                                                         rho=self._rho,
                                                         phi=self._phi,
                                                         xi=self._xi)
        self._boundary_transformation_vector = normal_vector @ transformation_matrix

    @staticmethod
    def _get_boundary_transformation_matrix(a: tf.Tensor, f: tf.Tensor, p: tf.Tensor, phi: tf.Tensor,
                                            rho: tf.Tensor, xi: tf.Tensor) -> tf.Tensor:
        """Transformation constants for the gradient at the boundary. Found in coordinate_transformation.ipynb."""
        r = (rho[-1] + rho[-2]) / 2.0  # coordinate at the boundary
        shape = phi.shape[1], xi.shape[2]
        A_00 = a * (-2 * r * a * f ** 2 * tf.sin(phi) ** 2 + 2 * r * a * f * tf.sin(
            phi) + 2 * r * f ** 2 * p * xi * tf.sin(phi) ** 2 - 2 * r * f * p * xi * tf.sin(phi) - a * tf.sqrt(
            r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2)) * tf.cos(phi) / (
                       r * a ** 2 * f ** 3 * tf.sin(phi) ** 3 - r * a ** 2 * f ** 2 * tf.sin(
                   phi) ** 2 + r * a ** 2 * f * tf.sin(
                   phi) - r * a ** 2 - 2 * r * a * f ** 3 * p * xi * tf.sin(
                   phi) ** 3 + 2 * r * a * f ** 2 * p * xi * tf.sin(phi) ** 2 - 2 * r * a * f * p * xi * tf.sin(
                   phi) + 2 * r * a * p * xi + r * f ** 3 * p ** 2 * xi ** 2 * tf.sin(
                   phi) ** 3 - r * f ** 2 * p ** 2 * xi ** 2 * tf.sin(
                   phi) ** 2 + r * f * p ** 2 * xi ** 2 * tf.sin(
                   phi) - r * p ** 2 * xi ** 2 + 2 * a ** 2 * f * tf.sqrt(
                   r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2) * tf.sin(
                   phi) - 2 * a * f * p * xi * tf.sqrt(
                   r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2) * tf.sin(phi))
        A_00 = tf.reshape(A_00, shape)
        A_01 = a * tf.sin(phi) / (r * (a * f * tf.sin(phi) - a - f * p * xi * tf.sin(phi) + p * xi))
        A_01 = tf.reshape(A_01, shape)
        A_02 = tf.zeros(shape)
        A_10 = a * (-r * a * f * tf.sin(phi) ** 2 + r * a * tf.sin(phi) + r * f * p * xi * tf.sin(
            phi) ** 2 - r * p * xi * tf.sin(phi) - 2 * a * f * tf.sqrt(
            r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2) * tf.sin(phi) ** 2 + a * f * tf.sqrt(
            r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2)) / (
                       2 * r * a ** 2 * f ** 2 * tf.sin(phi) ** 2 - 2 * r * a ** 2 * f * tf.sin(
                   phi) - 4 * r * a * f ** 2 * p * xi * tf.sin(phi) ** 2 + 4 * r * a * f * p * xi * tf.sin(
                   phi) + 2 * r * f ** 2 * p ** 2 * xi ** 2 * tf.sin(
                   phi) ** 2 - 2 * r * f * p ** 2 * xi ** 2 * tf.sin(phi) + a ** 2 * f ** 2 * tf.sqrt(
                   r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2) * tf.sin(
                   phi) ** 2 + a ** 2 * tf.sqrt(r ** 2 * (-a + p * xi) ** 2 * (
                       f * tf.sin(phi) - 1) ** 2 / a ** 2) - a * f ** 2 * p * xi * tf.sqrt(
                   r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2) * tf.sin(
                   phi) ** 2 - a * p * xi * tf.sqrt(
                   r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2))
        A_10 = tf.reshape(A_10, shape)
        A_11 = a * tf.cos(phi) / (r * (-a * f * tf.sin(phi) + a + f * p * xi * tf.sin(phi) - p * xi))
        A_11 = tf.reshape(A_11, shape)
        A_12 = tf.zeros(shape)
        A_20 = r * p * (r * a * f ** 3 * tf.sin(phi) ** 3 - 2 * r * a * f ** 2 * tf.sin(
            phi) ** 2 + r * a * f * tf.sin(phi) - r * f ** 3 * p * xi * tf.sin(
            phi) ** 3 + 2 * r * f ** 2 * p * xi * tf.sin(phi) ** 2 - r * f * p * xi * tf.sin(phi) + a * f * tf.sqrt(
            r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2) * tf.sin(phi) - a * tf.sqrt(
            r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2)) / (
                       r * a ** 2 * f ** 3 * tf.sin(phi) ** 3 - r * a ** 2 * f ** 2 * tf.sin(
                   phi) ** 2 + r * a ** 2 * f * tf.sin(
                   phi) - r * a ** 2 - 2 * r * a * f ** 3 * p * xi * tf.sin(
                   phi) ** 3 + 2 * r * a * f ** 2 * p * xi * tf.sin(phi) ** 2 - 2 * r * a * f * p * xi * tf.sin(
                   phi) + 2 * r * a * p * xi + r * f ** 3 * p ** 2 * xi ** 2 * tf.sin(
                   phi) ** 3 - r * f ** 2 * p ** 2 * xi ** 2 * tf.sin(
                   phi) ** 2 + r * f * p ** 2 * xi ** 2 * tf.sin(
                   phi) - r * p ** 2 * xi ** 2 + 2 * a ** 2 * f * tf.sqrt(
                   r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2) * tf.sin(
                   phi) - 2 * a * f * p * xi * tf.sqrt(
                   r ** 2 * (-a + p * xi) ** 2 * (f * tf.sin(phi) - 1) ** 2 / a ** 2) * tf.sin(phi))
        A_20 = tf.reshape(A_20, shape)
        A_21 = tf.zeros(shape)
        A_22 = tf.ones(shape)
        A = tf.stack([  # stack the rows and then the columns
            tf.stack([A_00, A_10, A_20], axis=-1),
            tf.stack([A_01, A_11, A_21], axis=-1),
            tf.stack([A_02, A_12, A_22], axis=-1),
        ], axis=-1)
        return A

    @staticmethod
    def _get_boundary_normal_vector(a: tf.Tensor, f: tf.Tensor, p: tf.Tensor, phi: tf.Tensor,
                                    rho: tf.Tensor, xi: tf.Tensor) -> tf.Tensor:
        """Normal unit vector at the radial boundary. Found in coordinate_transformation.ipynb."""
        r = (rho[-1] + rho[-2]) / 2.0  # coordinate at the boundary
        shape = phi.shape[1], xi.shape[2], 1
        n_0 = r * (a - p * xi) * (-f * tf.sin(2 * phi) + tf.cos(phi)) / a
        n_0 = tf.reshape(n_0, shape)
        n_1 = r * (a - p * xi) * (-2 * f * tf.sin(phi) ** 2 + f + tf.sin(phi)) / a
        n_1 = tf.reshape(n_1, shape)
        n_2 = r ** 2 * p * (a - p * xi) * (f * tf.sin(phi) - 1) ** 2 / a ** 2
        n_2 = tf.reshape(n_2, shape)
        n = tf.stack([n_0, n_1, n_2], axis=-1)
        magnitude = tf.sqrt(tf.reduce_sum(n * n, axis=-1))[..., tf.newaxis]
        n_hat = n / magnitude
        return n_hat

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
