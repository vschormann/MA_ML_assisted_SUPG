import numpy as np

from supgml.graph import relative_position


def test_relative_position_contains_distance_and_direction():
    result = relative_position([3.0, 4.0], [0.0, 0.0])
    np.testing.assert_allclose(result, [5.0, 0.6, 0.8])


def test_relative_position_handles_self_edge():
    np.testing.assert_allclose(relative_position([1.0, 2.0], [1.0, 2.0]), 0.0)
