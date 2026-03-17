import numpy as np
import pytest
import skia

from ptspy.core import colors
from ptspy.core.colors import _gamutMapSoft


def test_cube_hcl_validations_and_shape():
    with pytest.raises(ValueError):
        colors.cubeHCL(0)
    with pytest.raises(ValueError):
        colors.cubeHCL(1)

    grid = colors.cubeHCL(steps=3)
    assert grid.shape == (3, 3, 3, 3)
    # Axes are now (H, C, L) - test with custom ranges
    ranged = colors.cubeHCL(steps=4, h_range=(0.0, np.pi), c_range=(5, 80), l_range=(10, 90), includeHueMax=True)
    # grid[0, 0, 0, 0] is H value at first H index = 0.0
    assert np.isclose(ranged[0, 0, 0, 0], 0.0)
    # grid[-1, 0, 0, 0] is H value at last H index = pi (with includeHueMax=True)
    assert np.isclose(ranged[-1, 0, 0, 0], np.pi)
    # grid[0, 0, 0, 2] is L value at first L index = 10
    assert np.isclose(ranged[0, 0, 0, 2], 10)
    # grid[0, 0, -1, 2] is L value at last L index = 90
    assert np.isclose(ranged[0, 0, -1, 2], 90)

    reversed_axes = colors.cubeHCL(steps=4, h_range=(2 * np.pi, np.pi), c_range=(50, 10), l_range=(100, 0))
    # Axis 0 is H, axis 1 is C, axis 2 is L
    assert np.all(np.diff(reversed_axes[:, 0, 0, 0]) < 0)  # H decreasing along axis 0
    assert np.all(np.diff(reversed_axes[0, :, 0, 1]) < 0)  # C decreasing along axis 1
    assert np.all(np.diff(reversed_axes[0, 0, :, 2]) < 0)  # L decreasing along axis 2

    with pytest.warns(ResourceWarning):
        colors.cubeHCL(steps=230)  # float32: 230³ × 3 × 4 ≈ 146 MB > 128 MB threshold


def test_cube_hcl_hue_endpoint_exclusive_when_disabled():
    grid = colors.cubeHCL(steps=6, includeHueMax=False)
    # H is on axis 0: grid[0, 0, 0, 0] is hue at first H index
    assert np.isclose(grid[0, 0, 0, 0], 0.0)
    # grid[-1, 0, 0, 0] is hue at last H index, should be less than 2*pi
    assert grid[-1, 0, 0, 0] < 2 * np.pi


def test_color_cube_small_grid_dtype_and_range():
    cube = colors.colorCube(steps=2)
    assert cube.shape == (2, 2, 2)
    assert cube.dtype == np.uint32
    assert np.all(cube <= 0xFFFFFFFF)


def test_color_interpolation_path_and_subcube():
    cube = np.arange(27, dtype=np.uint32).reshape(3, 3, 3)

    path = colors.colorInterpolation(cube, [0, 0, 0], [2, 2, 2], steps=2, asPath=True)
    assert np.array_equal(path, np.array([0, 13, 26], dtype=np.uint32))
    path_no_end = colors.colorInterpolation(cube, [0, 0, 0], [2, 2, 2], steps=2, asPath=True, includeEnd=False)
    assert path_no_end.shape == (2,)
    assert path_no_end[-1] == cube[1, 1, 1]

    sub = colors.colorInterpolation(cube, [0, 0, 0], [1, 1, 1], steps=2, asPath=False)
    assert sub.shape[-3:] == (3, 3, 3)
    squeezed = np.squeeze(sub)
    assert squeezed.shape[-3:] == (3, 3, 3)
    assert squeezed[..., 0, 0, 0] == 0 and squeezed[..., -1, -1, -1] == cube[1, 1, 1]

    with pytest.raises(ValueError):
        colors.colorInterpolation(cube, [0, 0], [1, 1, 1], steps=2)
    with pytest.raises(ValueError):
        colors.colorInterpolation(cube, [0, 0, 0], [1, 1, 1], steps=0)
    with pytest.raises(IndexError):
        colors.colorInterpolation(cube, [0, 0, 0], [3, 0, 0], steps=2)


def test_color_picker_returns_indices_and_value():
    cube = np.arange(8, dtype=np.uint32).reshape(2, 2, 2)
    idx = colors.colorPicker(cube, 0.6, 0.6, 0.6, asArray=True)
    assert np.array_equal(idx, np.array([1, 1, 1]))
    assert colors.colorPicker(cube, 0.6, 0.6, 0.6) == cube[1, 1, 1]
    # Endpoints and out-of-range values are clamped to cube bounds.
    assert colors.colorPicker(cube, 1.0, 0.0, 0.0) == cube[1, 0, 0]
    clampedLow = colors.colorPicker(cube, -1e-12, -1e-12, -1e-12)
    assert clampedLow == cube[0, 0, 0]


def test_color_scheme_respects_bounds_and_shape():
    cube = np.arange(64, dtype=np.uint32).reshape(4, 4, 4)
    rng = np.random.default_rng(0)
    scheme = colors.colorScheme(cube, hues=2, swatches=2, intensity=0.5, variant=rng)
    assert scheme.shape == (2, 2, 4)
    for entry in scheme.reshape(-1, 4):
        # Return order is [RGB, h-index, c-index, l-index]
        rgb, h, c, l = entry
        assert 0 <= h < cube.shape[0]  # H is axis 0
        assert 0 <= c < cube.shape[1]  # C is axis 1
        assert 0 <= l < cube.shape[2]  # L is axis 2
        assert rgb == cube[h, c, l]


def test_gradient_variants_and_errors():
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    shader = colors.gradient(pts, [0xFF0000FF, 0x00FF00FF], kind="linear")
    assert isinstance(shader, skia.Shader)

    radial_pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    radial = colors.gradient(radial_pts, [0xFF0000FF, 0x00FF00FF], kind="radial")
    assert isinstance(radial, skia.Shader)

    sweep = colors.gradient(np.array([[5.0, 5.0]]), [0xFF0000FF, 0x00FF00FF], kind="sweep")
    assert isinstance(sweep, skia.Shader)

    with pytest.raises(ValueError):
        colors.gradient(np.array([[0.0, 0.0]]), [0xFF0000], kind="linear")
    with pytest.raises(ValueError):
        colors.gradient(np.array([[0.0, 0.0]]), [0xFF0000], kind="conical")
    with pytest.raises(ValueError):
        colors.gradient(pts, [0xFF0000], kind="unknown")


def test_color_cube_observer_illuminant_and_determinism():
    arr2 = colors.colorCube(steps=3, observer="2")
    arr10 = colors.colorCube(steps=3, observer="10")
    arrD50 = colors.colorCube(steps=3, illuminant="D50")
    assert not np.array_equal(arr2, arr10) or not np.array_equal(arr10, arrD50)
    a = colors.colorCube(steps=2, observer="10", illuminant="D65")
    b = colors.colorCube(steps=2, observer="10", illuminant="D65")
    assert np.array_equal(a, b)
    with pytest.raises(Exception):
        colors.colorCube(steps=2, observer="X")  # type: ignore


def test_color_interpolation_batching_and_resource_warning():
    cube = np.arange(64, dtype=np.uint32).reshape(4, 4, 4)
    start = np.array([[0, 0, 0], [1, 1, 1]])
    end = np.array([[3, 3, 3], [2, 2, 2]])
    grid = colors.colorInterpolation(cube, start, end, steps=3, asPath=False)
    assert grid.shape[0] == 2
    assert grid.shape[-3:] == (4, 4, 4)
    squeezed = np.squeeze(grid)
    assert squeezed[0, 0, 0, 0] == cube[0, 0, 0]
    assert squeezed[1, -1, -1, -1] == cube[2, 2, 2]

    with pytest.warns(ResourceWarning):
        colors.colorInterpolation(cube, [0, 0, 0], [3, 3, 3], steps=256, asPath=False)


def test_gamut_map_soft_clamps_to_unit_interval():
    """Soft mapping should always produce values in [0, 1]."""
    # Out-of-gamut linear RGB values
    rgb = np.array([[-0.5, 1.5, 0.5], [0.0, 2.0, -1.0], [0.3, 0.7, 0.5]])
    mapped = _gamutMapSoft(rgb)
    assert mapped.min() >= 0.0
    assert mapped.max() <= 1.0


def test_gamut_map_soft_preserves_in_gamut_values():
    """Values well within gamut should remain mostly unchanged."""
    rgb = np.array([[0.3, 0.4, 0.5], [0.1, 0.2, 0.3]])
    mapped = _gamutMapSoft(rgb, threshold=0.85, strength=1.5)
    # Values below threshold should be nearly identical
    assert np.allclose(rgb, mapped, atol=0.05)


def test_gamut_map_soft_smooth_gradient():
    """Soft mapping should produce smooth gradients (no hard discontinuities)."""
    # Create a gradient from in-gamut to out-of-gamut
    t = np.linspace(0.5, 1.5, 100)
    rgb = np.stack([t, t * 0.5, np.zeros_like(t)], axis=-1)
    mapped = _gamutMapSoft(rgb)

    # Check that output changes smoothly (no jumps > threshold)
    diffs = np.abs(np.diff(mapped, axis=0))
    maxJump = diffs.max()
    assert maxJump < 0.1, f"Discontinuity detected: max jump = {maxJump}"


def test_gamut_map_soft_threshold_and_strength():
    """Different threshold/strength values should produce different results."""
    rgb = np.array([[0.0, 1.2, 0.6]])

    # Lower threshold = earlier compression
    low_t = _gamutMapSoft(rgb, threshold=0.5, strength=1.5)
    high_t = _gamutMapSoft(rgb, threshold=0.9, strength=1.5)
    assert not np.allclose(low_t, high_t)

    # Higher strength = more aggressive compression
    low_s = _gamutMapSoft(rgb, threshold=0.85, strength=0.5)
    high_s = _gamutMapSoft(rgb, threshold=0.85, strength=3.0)
    assert not np.allclose(low_s, high_s)


def test_gamut_map_soft_edge_cases():
    """Black, white, and grayscale should be handled correctly."""
    edges = np.array(
        [
            [0.0, 0.0, 0.0],  # Black
            [1.0, 1.0, 1.0],  # White
            [0.5, 0.5, 0.5],  # Gray
        ]
    )
    mapped = _gamutMapSoft(edges)

    # Black should remain black
    assert np.allclose(mapped[0], [0.0, 0.0, 0.0], atol=0.01)
    # White should remain very close to white
    assert np.allclose(mapped[1], [1.0, 1.0, 1.0], atol=0.05)
    # Gray should remain neutral
    assert np.allclose(mapped[2], [0.5, 0.5, 0.5], atol=0.05)


def test_color_cube_soft_vs_clip_differ_at_high_chroma():
    """Soft and clip gamut mapping should produce different results at high chroma."""
    soft = colors.colorCube(steps=10, gamutMap="soft")
    clip = colors.colorCube(steps=10, gamutMap="clip")

    # At high chroma (axis 1), results should differ
    highChromaSoft = soft[:, -1, :]  # Last chroma slice
    highChromaClip = clip[:, -1, :]

    # They should not be identical
    assert not np.array_equal(highChromaSoft, highChromaClip)


def test_color_cube_gamut_parameters_affect_output():
    """Gamut threshold and strength parameters should affect the result."""
    default = colors.colorCube(steps=5)
    lowThreshold = colors.colorCube(steps=5, gamutThreshold=0.5)
    highStrength = colors.colorCube(steps=5, gamutStrength=5.0)

    # Different parameters should produce different cubes
    assert not np.array_equal(default, lowThreshold)
    assert not np.array_equal(default, highStrength)
