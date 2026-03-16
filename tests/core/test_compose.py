import numpy as np
import pytest

import ptspy.np_wrapper as npw
from ptspy.core import compose

try:
    import cupy  # type: ignore[import-untyped]
    HAS_CUPY = True
    HAS_CUDA = cupy.cuda.is_available()
except ImportError:
    HAS_CUPY = False
    HAS_CUDA = False


def test_populate_sequential_and_constant():
    seq = compose.populate((2, 3), start=1, increment=2)
    assert np.array_equal(seq, np.array([[1, 3, 5], [7, 9, 11]], dtype=float))

    const = compose.populate((2, 2), start=5, increment=0)
    assert np.array_equal(const, np.full((2, 2), 5.0))


def test_steps_inclusive_and_exclusive():
    arr = compose.steps(0, 10, 5)
    assert np.allclose(arr, np.array([0, 2.5, 5, 7.5, 10], dtype=float))

    arr_exclusive = compose.steps(0, 3, 3, inclusive=False)
    assert np.allclose(arr_exclusive, np.array([0.75, 1.5, 2.25], dtype=float))

    with pytest.raises(ValueError):
        compose.steps(0, 1, -1)

    shaped = compose.steps([0, 0], [10, 5], 3, shaping="quadraticIn")
    assert shaped.shape == (3, 2)
    assert np.allclose(shaped, np.array([[0.0, 0.0], [2.5, 1.25], [10.0, 5.0]]))
    assert shaped[1, 0] < 5  # quadratic in eases in toward the start


def test_gridpoints_spacing_broadcast_and_validation():
    grid = compose.gridPoints([2, 3], 1)
    assert grid.shape == (6, 2)
    assert np.array_equal(grid[0], np.array([0.0, 0.0]))
    assert np.array_equal(grid[-1], np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        compose.gridPoints([2, 2], [1, 2, 3])

    grid_broadcast = compose.gridPoints(3, [2])
    assert np.array_equal(grid_broadcast[:, 0], np.array([0.0, 2.0, 4.0]))


def test_gridpoints_rejects_non_positive_repeats():
    with pytest.raises(ValueError):
        compose.gridPoints(0, 1)
    with pytest.raises(ValueError):
        compose.gridPoints([2, 0], [1, 1])


def test_random_seed_and_range():
    expected = np.random.default_rng(42).uniform(-1, 1, (2, 2)).astype(float)
    actual = compose.random((2, 2), min=-1, max=1, seed=42)
    assert np.allclose(actual, expected)

    scalar = compose.random(min=0.25, max=0.75, seed=1)
    assert 0.25 <= scalar < 0.75

    rng_seed = np.random.default_rng(5)
    expected_from_rng = rng_seed.uniform(0, 1, (2,)).astype(float)
    rng = np.random.default_rng(5)
    with_rng = compose.random((2,), min=0, max=1, rng=rng)
    assert np.allclose(with_rng, expected_from_rng)


@pytest.mark.skipif(not HAS_CUDA, reason="CuPy/CUDA not available")
def test_random_with_numpy_rng_stays_on_active_cupy_backend():
    npw.reset('cupy')
    try:
        rng = np.random.default_rng(5)
        result = compose.random((2,), min=0, max=1, rng=rng)
        assert isinstance(result, cupy.ndarray)
        assert result.dtype == cupy.float32
        np.testing.assert_allclose(cupy.asnumpy(result), np.random.default_rng(5).uniform(0, 1, (2,)).astype(np.float32))
    finally:
        npw.reset('numpy')


def test_irregular_bounds_and_sorting():
    rng = np.random.default_rng(0)
    vals = compose.irregular(maxValue=10, steps=5, rng=rng)
    assert vals.shape == (5,)
    assert np.all(vals >= 0)
    assert np.all(vals < 10)
    assert np.all(np.diff(vals) >= 0)  # monotonic increasing bins

    empty = compose.irregular(maxValue=1, steps=0)
    assert empty.shape == (0,)

    with pytest.raises(ValueError):
        compose.irregular(maxValue=-1, steps=1)
    with pytest.raises(ValueError):
        compose.irregular(maxValue=1, steps=-1)
