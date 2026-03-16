import numpy as np
import pytest
import skia

import ptspy.np_wrapper as npw
from ptspy.core import util
from ptspy.core.form import Form

try:
    import cupy  # type: ignore[import-untyped]
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def test_pretty_formats_empty_and_nested():
    assert "No data" in util.pretty([])
    text = util.pretty([[1, 2], [3, 4]], header="Matrix")
    assert "Matrix" in text
    assert "[1.0, 2.0]" in text
    deeper = util.pretty([[1, 2, 3]])
    assert "[" in deeper


def test_pretty_handles_scalar():
    result = util.pretty(5.0, header="Scalar")
    assert "5.0" in result
    assert "Scalar" in result


def test_get_skiaprop_and_invalid_key():
    assert util.getSkiaProp("stroke-cap-round") == skia.Paint.kRound_Cap
    with pytest.raises(KeyError):
        util.getSkiaProp("missing")


def test_color_conversions_round_trip_and_formats():
    assert util.toRGBA(0xFF112233) == 0x112233FF
    assert util.toARGB(0x112233FF) == 0xFF112233
    assert util.toARGB(0x112233, hasAlpha=False) == 0xFF112233

    assert util.toARGBInt("#abc") == 0xFFaabbcc
    assert util.toARGBInt("#112233FF") == 0xFF112233
    assert util.toARGBInt(0x112233, hasAlpha=False) == 0xFF112233
    assert util.toARGBInt([0x112233]) == 0xFF112233

    with pytest.raises(ValueError):
        util.toARGBInt([1, 2])  # invalid sequence length
    with pytest.raises(TypeError):
        util.toARGBInt(object())
    with pytest.raises(ValueError):
        util.toARGBInt([1, 2, 3, 4, 5])


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_color_scalar_numpy_integer_still_works_under_cupy_backend():
    npw.reset('cupy')
    try:
        color = np.uint32(0x112233)
        assert util.validateColorLike(color) is color
        assert util.toARGBInt(color, hasAlpha=False) == 0xFF112233
        assert util.toARGBInt([color]) == 0xFF112233
    finally:
        npw.reset('numpy')


def test_get_color_clamps_and_alpha():
    cube = np.arange(8, dtype=np.uint32).reshape(2, 2, 2)
    color = util.getColor(cube, np.array([3, 3, 3]))
    assert color == cube[1, 1, 1]

    with_alpha = util.getColor(cube, np.array([0, 0, 0]), alpha=128)
    assert with_alpha >> 24 == 128
    assert with_alpha & 0x00FFFFFF == cube[0, 0, 0]

    empty = util.getColor(cube, np.empty((0, 3), dtype=int))
    assert isinstance(empty, np.ndarray) and empty.shape == (0,)

    arr = util.getColor(cube, np.array([[0, 0, 1], [1, 1, 1]]), alpha=0)
    assert (arr >> 24 == 0).all()


def test_image_array_conversions_and_validation():
    form = Form(width=2, height=2)
    form.clear(0x000000)  # black
    img = form.toImage()
    arr = util.imageToArray(img)
    assert arr.shape == (2, 2, 4)
    assert np.all((0.0 <= arr) & (arr <= 1.0))

    gray = util.imageToArray(img, asGrayscale=True)
    assert gray.shape == (2, 2)

    gray_pixels = np.array([[0.0, 1.0]], dtype=float)
    packed_gray = util.arrayToImage(gray_pixels, alwaysRGBA=True)
    assert packed_gray.shape == (1, 2)
    assert (packed_gray >> 24 == 255).all()

    rgb_pixels = np.array([[[0.0, 0.5, 1.0]]], dtype=float)
    packed_rgb = util.arrayToImage(rgb_pixels, alwaysRGBA=False)
    assert packed_rgb.shape == (1, 1)

    rgba_pixels = np.array([[[0.0, 0.0, 0.0, 0.5]]], dtype=float)
    packed_rgba = util.arrayToImage(rgba_pixels)
    assert packed_rgba.shape == (1, 1)
    assert packed_rgba.dtype == np.uint32

    with pytest.raises(ValueError):
        util.arrayToImage(np.ones((2, 2, 5)))

    # custom min/max scaling
    scaled = util.arrayToImage(np.array([[0.5, 1.0]]), min=0.0, max=1.0)
    assert scaled.dtype == np.uint32
