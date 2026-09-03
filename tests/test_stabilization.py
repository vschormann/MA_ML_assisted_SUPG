import numpy as np
import scipy.sparse

from supgml.stabilization import BJKAFC, KuzminAFC, standard_tau


def test_standard_tau_matches_formula():
    diameter = np.array([0.5, 1.0])
    diffusion = 0.25
    speed = 2.0
    peclet = diameter * speed / (2 * diffusion)
    expected = diameter * (1 / np.tanh(peclet) - 1 / peclet) / (2 * speed)
    np.testing.assert_allclose(standard_tau(diameter, diffusion, speed), expected)


def test_standard_tau_has_zero_velocity_limit():
    np.testing.assert_allclose(standard_tau(np.array([0.5, 1.0]), 0.25, 0.0), 0.0)


def test_afc_residual_shapes():
    matrix = scipy.sparse.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
    rhs = np.array([0.0, 0.0])
    values = np.array([0.25, 0.75])
    assert KuzminAFC(matrix, rhs, []).__call__(values).shape == values.shape
    assert BJKAFC(matrix, rhs, []).__call__(values).shape == values.shape
