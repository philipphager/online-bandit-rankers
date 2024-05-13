import numpy as np
import pytest

from src.simulation.pbm import PBMSimulator, CascadeSimulator


@pytest.mark.parametrize(
    "position_bias,relevance,expected_ctr",
    [
        # No bias, high relevance
        (0.0, np.array([1.0, 1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
        # No bias, mixed relevance
        (0.0, np.array([1.0, 0.8, 0.6, 0.4, 0.2]), np.array([1.0, 0.8, 0.6, 0.4, 0.2])),
        # No bias, low relevance
        (0.0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0])),
        # Bias, high relevance
        (
            1.0,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5]),
        ),
        # Bias, mixed relevance
        (1.0, np.array([0.2, 0.4, 0.6, 0.8, 1.0]), np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
        # Bias, low relevance
        (1.0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0])),
    ],
)
def test_pbm_simulator(
    position_bias,
    relevance,
    expected_ctr,
    n_clicks: int = 50_000,
    absolute_tolerance=0.01,
):
    simulator = PBMSimulator(position_bias=position_bias)
    actual_clicks = []

    for i in range(n_clicks):
        clicks = simulator(relevance)
        actual_clicks.append(clicks)

    actual_clicks = np.array(actual_clicks)
    actual_ctr = actual_clicks.mean(axis=0)

    assert np.allclose(actual_ctr, expected_ctr, atol=absolute_tolerance)


@pytest.mark.parametrize(
    "relevance,expected_ctr",
    [
        # All relevant items, click on first
        (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0, 0.0])),
        # First relevant item is only relevant item
        (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0, 0.0])),
        # Last relevant item is only relevant item
        (np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 0.0, 1.0])),
        # All items are equally somewhat relevant
        (np.array([0.5, 0.5, 0.5, 0.5, 0.5]), np.array([0.5 ** 1, 0.5 ** 2, 0.5 ** 3, 0.5 ** 4, 0.5 ** 5])),
        # Second item is most relevant
        (np.array([0.5, 1.0, 0.5, 1.0, 0.5]), np.array([0.5, 0.5, 0.0, 0.0, 0.0])),
        # Increasing relevance
        (np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([0.1, 0.9 * 0.2, 0.9 * 0.8 * 0.3, 0.9 * 0.8 * 0.7 * 0.4, 0.9 * 0.8 * 0.7 * 0.6 * 0.5])),
    ],
)
def test_cascade_simulator(
    relevance,
    expected_ctr,
    n_clicks: int = 50_000,
    absolute_tolerance=0.01,
):
    simulator = CascadeSimulator()
    actual_clicks = []

    for i in range(n_clicks):
        clicks = simulator(relevance)
        actual_clicks.append(clicks)

    actual_clicks = np.array(actual_clicks)
    actual_ctr = actual_clicks.mean(axis=0)

    assert np.allclose(actual_ctr, expected_ctr, atol=absolute_tolerance)
