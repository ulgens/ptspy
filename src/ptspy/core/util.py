"""Utility helpers for colors, arrays, and Skia integration."""

import numpy as _cpu_numpy  # Permanent reference to CPU numpy for Skia boundary operations
from .. import np_wrapper as np

from typing import Any, Optional, Union, Sequence
from numpy.typing import ArrayLike, NDArray
from ptspy.typing import (
    NumberSequenceND,
    PArray,
    ColorLike,
    IntArrayOrScalar,
    IntArray,
    NumberSequence1D,
    NumberSequence2D,
)
from .elements import toPArray, partial
import skia


def _validateNDim(name: str, arr: np.ndarray, ndim: int) -> None:
    """
    Ensure an array has an exact number of dimensions.

    Data:
        internal: true
    """
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {arr.shape}")


def _validateLastDim(name: str, arr: np.ndarray, lastDim: int) -> None:
    """
    Ensure an array has the expected last-dimension width.

    Data:
        internal: true
    """
    if arr.ndim == 0 or arr.shape[-1] != lastDim:
        raise ValueError(f"{name} must have last dimension == {lastDim}, got shape {arr.shape}")


def _validateFinite(name: str, arr: np.ndarray) -> None:
    """
    Ensure all values in an array are finite.

    Data:
        internal: true
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite values only")


def validateNumberSequence1D(
    value: NumberSequence1D,
    *,
    name: str = "value",
    length: int | None = None,
    minLength: int | None = None,
    finite: bool = False,
) -> PArray:
    """
    Validate and coerce an input as a 1D float32 vector.

    Args:
        value (NumberSequence1D): Input values to validate.
        name (str, optional): Field name used in error messages. Defaults to `"value"`.
        length (int, optional): If provided, enforce exact vector length.
        minLength (int, optional): If provided, enforce minimum vector length.
        finite (bool, optional): If True, reject NaN and infinity values.

    Returns:
        PArray: A validated float32 1D array.

    Raises:
        ValueError: If dimensionality, length, or finiteness checks fail.

    Examples:
        >>> validateNumberSequence1D([1, 2, 3], length=3)
        array([1., 2., 3.], dtype=float32)
    """
    arr = np.asarray(value, dtype=np.float32)
    _validateNDim(name, arr, 1)

    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}")
    if minLength is not None and arr.shape[0] < minLength:
        raise ValueError(f"{name} must have length >= {minLength}, got {arr.shape[0]}")
    if finite:
        _validateFinite(name, arr)

    return arr


def validateNumberSequence2D(
    value: NumberSequence2D,
    *,
    name: str = "value",
    cols: int | None = None,
    minRows: int | None = None,
    finite: bool = False,
) -> PArray:
    """
    Validate and coerce an input as a 2D float32 table.

    Args:
        value (NumberSequence2D): Input values to validate.
        name (str, optional): Field name used in error messages. Defaults to `"value"`.
        cols (int, optional): If provided, enforce exact number of columns.
        minRows (int, optional): If provided, enforce minimum number of rows.
        finite (bool, optional): If True, reject NaN and infinity values.

    Returns:
        PArray: A validated float32 2D array.

    Raises:
        ValueError: If dimensionality, row/column constraints, or finiteness checks fail.

    Examples:
        >>> validateNumberSequence2D([[0, 1], [2, 3]], cols=2)
        array([[0., 1.],
               [2., 3.]], dtype=float32)
    """
    arr = np.asarray(value, dtype=np.float32)
    _validateNDim(name, arr, 2)

    if cols is not None and arr.shape[1] != cols:
        raise ValueError(f"{name} must have {cols} columns, got shape {arr.shape}")
    if minRows is not None and arr.shape[0] < minRows:
        raise ValueError(f"{name} must have at least {minRows} rows, got {arr.shape[0]}")
    if finite:
        _validateFinite(name, arr)

    return arr


def validateShapeND(
    value: ArrayLike,
    *,
    name: str = "value",
    ndim: int | None = None,
    lastDim: int | None = None,
    dtype: Any = np.float32,
    finite: bool = False,
) -> NDArray[Any]:
    """
    Generic shape validator for public APIs.

    Args:
        value (ArrayLike): Input value to validate and coerce.
        name (str, optional): Field name used in error messages. Defaults to `"value"`.
        ndim (int, optional): If provided, enforce exact number of dimensions.
        lastDim (int, optional): If provided, enforce exact size of the last dimension.
        dtype (Any, optional): Target dtype used by `np.asarray`. Defaults to `np.float32`.
        finite (bool, optional): If True, reject NaN and infinity values.

    Returns:
        NDArray[Any]: A validated NumPy-like array.

    Raises:
        ValueError: If dimensionality, last-dimension, or finiteness checks fail.

    Examples:
        >>> validateShapeND([[1, 2], [3, 4]], ndim=2, lastDim=2)
        array([[1., 2.],
               [3., 4.]], dtype=float32)
    """
    arr = np.asarray(value, dtype=dtype)
    if ndim is not None:
        _validateNDim(name, arr, ndim)
    if lastDim is not None:
        _validateLastDim(name, arr, lastDim)
    if finite:
        _validateFinite(name, arr)
    return arr


def validateColorLike(value: ColorLike, *, name: str = "color") -> ColorLike:
    """
    Validate common color inputs used by drawing APIs.

    Accepted formats: int / numpy integer, hex-like string, 1D sequence or array with 3 (RGB) or 4 (RGBA) numeric channels

    Args:
        value (ColorLike): Color value to validate.
        name (str, optional): Field name used in error messages. Defaults to `"color"`.

    Returns:
        ColorLike: The original color value if validation succeeds.

    Raises:
        TypeError: If the input type is unsupported or sequence values are not numeric.
        ValueError: If sequence inputs do not contain 3 or 4 channels, or contain non-finite values.

    Examples:
        >>> validateColorLike("#ffaa11")
        '#ffaa11'
        >>> validateColorLike([255, 170, 17, 128])
        [255, 170, 17, 128]
    """
    if isinstance(value, (int, _cpu_numpy.integer, str)):
        return value

    if not np.isarray(value) and not isinstance(value, (str, bytes, bytearray)):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                scalar = item()
            except (TypeError, ValueError):
                pass
            else:
                if isinstance(scalar, (int, _cpu_numpy.integer, str)):
                    return value

    if np.isarray(value):
        arr = np.asarray(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        arr = np.asarray(value)
    else:
        raise TypeError(f"{name} must be an int, string, or numeric sequence")

    flat = arr.ravel()
    if flat.size not in (3, 4):
        raise ValueError(f"{name} sequence must have 3 (RGB) or 4 (RGBA) components, got {flat.size}")

    try:
        numeric = flat.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} sequence must be numeric") from exc

    _validateFinite(name, numeric)
    return value


def pretty(arr: NumberSequenceND, header: Optional[str] = None, indent=0):
    """
    Format a multi-dimensional array into a readable string.

    Returns a clearly indented string with an optional header, useful for inspecting nested arrays.

    Args:
        arr (NumberSequenceND): Array-like data to format.
        header (str, optional): A title displayed above the formatted array.
        indent (int, optional): Spaces used for indentation. Defaults to 0.

    Returns:
        str: A neatly formatted string representation of the array.

    Examples:
        >>> pretty([[1, 2], [3, 4]], header="2D Array")
        '\n\n2D Array\n[\n  [1, 2],\n  [3, 4]\n]'
    """

    arr = toPArray(arr)

    if arr.ndim == 0:
        h = f"\n\n{header}\n" if header else ""
        return h + str(arr)

    if len(arr) == 0:
        return f"\n\n{header or ''}\nNo data"

    pad = " " * indent

    if not np.isarray(arr):  # base case
        return f"\n\n{header}\n{pad + str(arr)}"
    elif len(arr.shape) == 1:  # 1D arrays
        h = f"\n\n{header}\n" if header and indent == 0 else ""
        return h + pad + "[" + ", ".join(map(str, arr)) + "]"
    else:
        parts = [pretty(subarr, None, indent + 2) for subarr in arr]
        if indent == 0 and header:  # Add header at top level
            return f"\n\n{header}\n" + pad + "[\n" + ",\n".join(parts) + "\n" + pad + "]"
        else:
            return pad + "[\n" + ",\n".join(parts) + "\n" + pad + "]"


def getSkiaProp(id: str) -> Any:
    """
    Retrieves the corresponding Skia property based on a given identifier.

    This function maps a string identifier to a Skia `Paint` property, such as stroke cap or stroke join styles.

    Args:
        id (str): The identifier for the Skia property. Expected to be one of: "stroke-cap-bevel", "stroke-cap-last", "stroke-cap-miter", "stroke-cap-round", "stroke-join-bevel", "stroke-join-last", "stroke-join-miter", "stroke-join-round"

    Returns:
        Any: The corresponding Skia `Paint` property.

    Raises:
        KeyError: If the identifier is not found in the mapping.

    Data:
        internal: true
    """
    mapping = {
        # Caps (Skia names: Butt, Round, Square). Keep legacy ids for compatibility.
        "stroke-cap-bevel": skia.Paint.kButt_Cap,
        "stroke-cap-last": skia.Paint.kSquare_Cap,
        "stroke-cap-miter": skia.Paint.kSquare_Cap,
        "stroke-cap-round": skia.Paint.kRound_Cap,
        # Joins
        "stroke-join-bevel": skia.Paint.kBevel_Join,
        "stroke-join-last": skia.Paint.kLast_Join,
        "stroke-join-miter": skia.Paint.kMiter_Join,
        "stroke-join-round": skia.Paint.kRound_Join,
    }
    try:
        return mapping[id]
    except KeyError as exc:
        raise KeyError(f"Unknown skia prop id '{id}'") from exc


def toRGBA(argb: IntArrayOrScalar):
    """
    Converts an ARGB color value to RGBA format.

    This function reorders the bytes of a 32-bit color value from ARGB (Alpha, Red, Green, Blue)
    to RGBA (Red, Green, Blue, Alpha).

    Args:
        argb (IntArrayOrScalar): A single ARGB color value or an array of ARGB values.

    Returns:
        Union[int, np.ndarray]: The corresponding RGBA value(s). If the input is an integer, returns an integer.
        If the input is an array, returns an array of the same shape as the input.

    Examples:
        >>> toRGBA(0xFF112233)
        0x112233FF

        >>> toRGBA(np.array([0xFF112233, 0xAA445566]))
        array([0x112233FF, 0x445566AA])
    """
    a = (argb & 0xFF000000) >> 24
    a &= 0xFF
    rgb = argb & 0x00FFFFFF
    rgba = (rgb << 8) | a
    return rgba


def toARGB(rgba: IntArrayOrScalar, hasAlpha: bool = True) -> Union[int, np.ndarray]:
    """
    Converts an RGBA or RGB color value to ARGB format.

    This function reorders the bytes of a 32-bit color value from RGBA (Red, Green, Blue, Alpha)
    to ARGB (Alpha, Red, Green, Blue). If the input is in RGB format (i.e., the alpha channel is not present),
    set `hasAlpha` to False.

    Args:
        rgba (IntArrayOrScalar): A single RGBA or RGB color value or an array of such values.
        hasAlpha (bool): A flag indicating whether the input is in RGB format. Default is True (indicating RGBA).

    Returns:
        Union[int, np.ndarray]: The corresponding ARGB value(s). If the input is an integer, returns an integer.
        If the input is an array, returns an array of the same shape as the input.

    Examples:
        >>> toARGB(0x112233FF)
        0xFF112233

        >>> toARGB(np.array([0x112233FF, 0x445566AA]))
        array([0xFF112233, 0xAA445566])
    """
    if hasAlpha:
        rgb = (rgba >> 8) & 0x00FFFFFF
        a = rgba & 0x000000FF
    else:
        rgb = rgba & 0x00FFFFFF
        a = 0xFF

    return (a << 24) | rgb


def toARGBInt(*args: Union[int, ColorLike], hasAlpha: Optional[bool] = None) -> int:
    """
    Converts a color represented as an RGBA integer, a string, or an array-like object into an ARGB integer
    suitable for use in `skia-python` functions.

    Accepts int (RGB or RGBA), hex string (``"#RGB"``, ``"#RGBA"``, ``"#RRGGBB"``, ``"#RRGGBBAA"``),
    or a 3/4-element sequence. Short hex like ``"#abc"`` is expanded to ``"#aabbcc"``.

    For int inputs < 0xFFFFFF, the value is assumed to be RGB with alpha=255 (use ``hasAlpha=True``
    to override). A single-element list like ``[0x112233]`` is treated as a scalar int.

    Args:
        *args (Union[int, ColorLike]): The color input. Can be: a single integer, a hex string or a list/tuple of RGBA components.
        hasAlpha (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

    Returns:
        int: The corresponding ARGB integer.

    Examples:
        >>> toARGBInt(0x112233)
        0xFF112233

        >>> toARGBInt("#112233FF")
        0xFF112233

        >>> toARGBInt([17, 34, 51])
        0xFF112233

        >>> toARGBInt(17, 34, 51, 128)
        0x80112233
    """

    numArgs = len(args)
    color: ColorLike = args[0] if numArgs == 1 else args

    # single integer parameter
    if numArgs == 1 and isinstance(color, (int, _cpu_numpy.integer)):
        c = int(color)  # type: ignore
        return toARGB(c, c >= 0xFFFFFF if hasAlpha is None else hasAlpha)

    # single string parameter
    if isinstance(color, str):
        c = color.lstrip("#")
        if len(c) in (3, 4):  # support short format like "#FFF" or "#FFFA"
            c = "".join(ch * 2 for ch in c)
        value = int(c, 16)
        return (value | 0xFF000000) if len(c) <= 6 else toARGB(value)

    if numArgs == 1 and not np.isarray(color) and not isinstance(color, (str, bytes, bytearray)):
        item = getattr(color, "item", None)
        if callable(item):
            try:
                scalar = item()
            except (TypeError, ValueError):
                pass
            else:
                if isinstance(scalar, (int, _cpu_numpy.integer)):
                    c = int(scalar)
                    return toARGB(c, c >= 0xFFFFFF if hasAlpha is None else hasAlpha)
                if isinstance(scalar, str):
                    return toARGBInt(scalar, hasAlpha=hasAlpha)

    # array parameter
    if isinstance(color, Sequence) or np.isarray(color):
        # Catch the single element array case eg, (0x112233,) and treat it as a scalar
        if len(color) == 1:  # type: ignore
            return toARGBInt(color[0], hasAlpha=hasAlpha)  # type: ignore

        comps = color.ravel() if np.isarray(color) else color  # type: ignore
        if not 3 <= len(comps) <= 4:  # type: ignore
            raise ValueError("Sequence must have 3 (RGB) or 4 (RGBA) components")

        r, g, b = (int(round(x)) for x in comps[:3])  # type: ignore
        a = int(round(comps[3])) if len(comps) == 4 else 255  # type: ignore
        return (a << 24) | (r << 16) | (g << 8) | b

    raise TypeError("Unsupported color format")


@partial(optional=True)
def getColor(cube: IntArray, indices: IntArray, alpha: Optional[int] = None) -> Union[int, IntArray]:
    """
    Retrieves a color from a 3D array of colors in HCL space at specified indices.

    This function allows you to extract a color from a 3D array (such as one generated by `colorCube()`)
    using a set of indices. Optionally, an alpha value can be provided to include transparency in the resulting color.

    Args:
        cube (IntArray): A 3D array of colors, such as from `colorCube()`. Axes are (H, C, L).
        indices (IntArray): A set of indices specifying which color to retrieve (hue, chroma, lightness). Can be a single set of indices or an array of indices.
        alpha (int, optional): An optional alpha value to apply to the color. If provided, the returned color will be in ARGB format with the specified alpha.

    Returns:
        Union[int, IntArray]: The color at the specified indices. If `alpha` is provided, returns an integer in ARGB format; otherwise, returns the raw color value(s).

    Examples:
        >>> hcl = np.array([[[255, 128, 64], [128, 64, 32]], [[0, 0, 0], [255, 255, 255]]])
        >>> indices = np.array([0, 1, 2])
        >>> getColor(hcl, indices)
        32

        >>> getColor(hcl, indices, alpha=128)
        2147483680
    """
    maxidx = np.array(cube.shape, dtype=int) - 1
    k = np.clip(indices, 0, maxidx).astype(int)

    # build tuple of arrays for advanced indexing
    if k.ndim == 1:
        idx = tuple(k.tolist())
    else:
        idx = tuple(k[..., i] for i in range(k.shape[-1]))
    color = cube[idx]

    if alpha is None:
        return int(color) if np.isscalar(color) else color

    a = (int(alpha) & 0xFF) << 24
    if np.isscalar(color):
        return a | (int(color) & 0x00FFFFFF)
    return (color & 0x00FFFFFF) | a


def imageToArray(image: skia.Image, asGrayscale: bool = False) -> PArray:
    """
    Convert a skia.Image to a float32 array normalized to [0.0, 1.0].

    Args:
        image (skia.Image): The skia.Image to convert.
        asGrayscale (bool, optional): If True, returns a luminance array using ITU-R BT.601 weights
            (0.299 R + 0.587 G + 0.114 B).

    Returns:
        PArray: Array of shape (height, width, 4) in RGBA order, or (height, width) if ``asGrayscale``.
    """

    array = np.fromcpu(_cpu_numpy.array(image))
    if asGrayscale:
        array = np.dot(array[..., :3], [0.299, 0.587, 0.114])
    return array.astype(np.float32) / 255.0


def arrayToImage(imageData: PArray, min=None, max=None, alwaysRGBA: bool = False) -> IntArray:
    """
    Convert a numeric array to packed 32-bit pixels.

    Normalizes values to [0, 255] (optionally using `min`/`max`), then packs channels into
    32-bit integers.

    Args:
        imageData (PArray): Grayscale (H, W), RGB (H, W, 3), or RGBA (H, W, 4) array.
        min (float, optional): Lower bound for normalization. Defaults to `imageData.min()`.
        max (float, optional): Upper bound for normalization. Defaults to `imageData.max()`.
        alwaysRGBA (bool, optional): If True, force ARGB output with alpha=255 for gray/RGB.

    Returns:
        IntArray: Packed pixels as RGB or ARGB integers.

    Examples:
        >>> gray = np.array([[0.0, 1.0]])
        >>> arrayToImage(gray)
        array([[       0, 16777215]], dtype=uint32)

        >>> rgb = np.array([[[0.0, 0.5, 1.0]]])
        >>> arrayToImage(rgb, alwaysRGBA=True)
        array([[4278222847]], dtype=uint32)
    """

    minVal = np.min(imageData) if min is None else min
    maxVal = np.max(imageData) if max is None else max
    pixel = (imageData - minVal) / (maxVal - minVal + 1e-9)

    if imageData.ndim == 2:  # Grayscale
        u8 = np.rint(pixel * 255).astype(np.uint32)
        if alwaysRGBA:
            return (255 << 24) | (u8 << 16) | (u8 << 8) | u8
        else:
            return (u8 << 16) | (u8 << 8) | u8

    elif imageData.ndim == 3 and imageData.shape[-1] == 3:  # RGB
        rgb = np.rint(pixel * 255).astype(np.uint32)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        if alwaysRGBA:
            return (255 << 24) | (r << 16) | (g << 8) | b
        else:
            return (r << 16) | (g << 8) | b

    elif imageData.ndim == 3 and imageData.shape[-1] == 4:  # RGBA
        rgba = np.rint(pixel * 255).astype(np.uint32)
        r, g, b, a = rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]
        return (a << 24) | (r << 16) | (g << 8) | b

    else:
        raise ValueError(
            f"Unsupported array shape: {imageData.shape}. Expected grayscale (height, width), RGB (height, width, 3), or RGBA (height, width, 4)"
        )
