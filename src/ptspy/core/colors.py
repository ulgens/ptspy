"""Color helpers and palettes built on HCL color space"""
from .elements import P, P3, partial, toPArray
from .geom import TWO_PI
from .util import toARGBInt
from ptspy.typing import NumberSequence1D, PArray, IntArray, ColorLike, NumberSequenceND, GradientStyle
from .. import np_wrapper as np
from typing import List, Literal, Optional

import warnings
import skia
from .compose import irregular
import math


# Reference white points (X, Y, Z) for different illuminants and observers
# Values from CIE standards, normalized to Y=100
_ILLUMINANTS = {
    "A": {"2": (109.850, 100.0, 35.585), "10": (111.144, 100.0, 35.200), "R": (109.850, 100.0, 35.585)},
    "B": {"2": (99.0927, 100.0, 85.313), "10": (99.178, 100.0, 84.3493), "R": (99.0927, 100.0, 85.313)},
    "C": {"2": (98.074, 100.0, 118.232), "10": (97.285, 100.0, 116.145), "R": (98.074, 100.0, 118.232)},
    "D50": {"2": (96.422, 100.0, 82.521), "10": (96.720, 100.0, 81.427), "R": (96.422, 100.0, 82.521)},
    "D55": {"2": (95.682, 100.0, 92.149), "10": (95.799, 100.0, 90.926), "R": (95.682, 100.0, 92.149)},
    "D65": {"2": (95.047, 100.0, 108.883), "10": (94.811, 100.0, 107.304), "R": (95.047, 100.0, 108.883)},
    "D75": {"2": (94.972, 100.0, 122.638), "10": (94.416, 100.0, 120.641), "R": (94.972, 100.0, 122.638)},
    "E": {"2": (100.0, 100.0, 100.0), "10": (100.0, 100.0, 100.0), "R": (100.0, 100.0, 100.0)},
}


def _hcl2lab(hcl):
    """HCL → LAB (polar to cartesian)"""
    H, C, L = hcl[..., 0], hcl[..., 1], hcl[..., 2]
    return np.stack([L, C * np.cos(H), C * np.sin(H)], axis=-1)


def _gamutMapSoft(rgb, threshold=0.85, strength=1.5, chromaFallback=True):
    """
    Soft gamut mapping with smooth shoulder compression.

    Applies a Reinhard-style shoulder curve that smoothly compresses out-of-gamut
    values toward the boundary, avoiding harsh clipping artifacts while preserving
    hue relationships. In-gamut values pass through unchanged.

    Args:
        rgb: Linear RGB array (..., 3), may contain values outside [0, 1]
        threshold: Blend zone width for smooth transition (0.85 = 15% blend zone).
            Values in (threshold, 1] are gently compressed; values > 1 are strongly compressed.
        strength: Compression intensity (higher = more aggressive compression)
        chromaFallback: If True, iteratively reduce saturation for extreme cases

    Returns:
        RGB array with values in [0, 1]
    """
    def _softShoulder(x, t, s):
        # Separate handling for different regions
        above1 = x > 1  # Out of gamut (overflow)
        inBlend = (x > t) & (x <= 1)  # In blend zone (in-gamut but near boundary)
        below0 = x < 0  # Out of gamut (underflow)

        # Overflow compression: maps (1, ∞) → (1, 1+1/s)
        # Reinhard-style: y = 1 + (x-1)/(1 + s*(x-1)) ensures C0 continuity at x=1
        overflowCompressed = 1 + (x - 1) / (1 + s * (x - 1))

        # Blend zone: gentle S-curve for values in (threshold, 1]
        # This smooths the transition but keeps 1.0 → 1.0
        blendT = (x - t) / (1 - t)  # Normalized: 0 at threshold, 1 at 1.0
        blendCompressed = t + (1 - t) * blendT / (1 + s * blendT * (1 - blendT))

        # Underflow compression: maps (-∞, 0) → approaches 0 asymptotically
        underflowCompressed = x / (1 - s * x)

        result = np.where(above1, overflowCompressed, x)
        result = np.where(inBlend, blendCompressed, result)
        result = np.where(below0, underflowCompressed, result)
        return result

    mapped = _softShoulder(rgb, threshold, strength)

    # Chroma reduction fallback for extreme out-of-gamut colors
    if chromaFallback:
        for _ in range(3):
            outOfGamut = (mapped < 0) | (mapped > 1)
            if not np.any(outOfGamut):
                break
            # Reduce saturation towards luminance (Rec.709 weights)
            lum = 0.2126 * mapped[..., 0] + 0.7152 * mapped[..., 1] + 0.0722 * mapped[..., 2]
            mapped = lum[..., None] + 0.9 * (mapped - lum[..., None])

    return np.clip(mapped, 0, 1)


def _lab2rgb(lab, illuminant: str = "D65", observer: Literal["2", "10", "R"] = "10",
             gamutMap: Literal["soft", "clip"] = "soft", gamutThreshold: float = 0.85,
             gamutStrength: float = 1.5):
    """LAB → sRGB conversion with configurable illuminant, observer, and gamut mapping."""
    if illuminant not in _ILLUMINANTS:
        raise ValueError(f"Unknown illuminant '{illuminant}'. Valid: {list(_ILLUMINANTS.keys())}")
    if observer not in _ILLUMINANTS[illuminant]:
        raise ValueError(f"Unknown observer '{observer}'. Valid: 2, 10, R")

    Xn, Yn, Zn = _ILLUMINANTS[illuminant][observer]

    # LAB → XYZ
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    delta = 6 / 29

    def finv(t):
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))

    X = Xn * finv(fx) / 100
    Y = Yn * finv(fy) / 100
    Z = Zn * finv(fz) / 100

    # XYZ → linear RGB (sRGB matrix)
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    bl = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    linearRgb = np.stack([r, g, bl], axis=-1)

    # Gamut mapping in linear space (before gamma)
    if gamutMap == "soft":
        linearRgb = _gamutMapSoft(linearRgb, gamutThreshold, gamutStrength)
    elif gamutMap == "clip":
        linearRgb = np.clip(linearRgb, 0, 1)

    # Gamma correction (no clipping needed - already in gamut)
    def gamma(u):
        return np.where(u <= 0.0031308, 12.92 * u, 1.055 * u ** (1 / 2.4) - 0.055)

    return np.stack([gamma(linearRgb[..., i]) for i in range(3)], axis=-1)


def cubeHCL(steps: int = 36, h_range=(0, TWO_PI), c_range=(0, 100), l_range=(0, 100), includeHueMax=False) -> PArray:
    """
    Generate a 3D grid of HCL color values.

    This function creates a 3D grid of HCL (Hue, Chroma, Lightness) values by interpolating
    between the provided ranges.

    Args:
        steps (int, optional): Number of points along each H, C, and L axis (must be >= 2). Defaults to 36.
        h_range (tuple, optional): Min and max values for hue (in radians). Defaults to (0, TWO_PI).
        c_range (tuple, optional): Min and max values for chroma. Defaults to (0, 100).
        l_range (tuple, optional): Min and max values for lightness. Defaults to (0, 100).
        includeHueMax (bool, optional): If True, includes hue endpoint (e.g., to duplicate 0 and 2π). Default is False.

    Returns:
        PArray: An array shaped `(steps, steps, steps, 3)`, containing HCL coordinates with axes (H, C, L).

    Examples:
        >>> hcl_cube = cubeHCL(10)
        >>> hcl_cube.shape
        (10, 10, 10, 3)

        >>> hcl_cube[0, 0, 0]
        array([0., 0., 0.])
    """
    if steps < 2:
        raise ValueError("steps must be ≥ 2")

    est_bytes = steps**3 * 3 * 4  # float32 grid before packing
    if est_bytes > 1 << 27:  # if more than 128 MiB
        warnings.warn(
            f"colorCube: estimated temporary memory {est_bytes / (1 << 30):.1f} GiB",
            ResourceWarning,
            stacklevel=2,
        )

    h_vals = np.linspace(h_range[0], h_range[1], steps, endpoint=includeHueMax)
    c_vals = np.linspace(*c_range, steps)
    l_vals = np.linspace(*l_range, steps)
    return P3(h_vals, c_vals, l_vals)


def colorCube(steps=36, observer: Literal["2", "10", "R"] = "10", illuminant: str = "D65",
              gamutMap: Literal["soft", "clip"] = "soft", gamutThreshold: float = 0.85,
              gamutStrength: float = 1.5) -> IntArray:
    """
    Generates a 3D cube of RGB integers by sampling the HCL color space (hue, chroma, and lightness).

    This function creates a 3D array of shape `(steps, steps, steps)` where each element is an RGB integer.
    The RGB values are generated by interpolating through the HCL color space and converting the results to a structured cube of RGB.
    The cube axes are ordered as (H, C, L) - hue on axis 0, chroma on axis 1, lightness on axis 2.

    Args:
        steps (int, optional): Number of samples per H, C, and L axis. Defaults to 36. Large values may require significant memory.
        observer (Literal["2", "10", "R"], optional): Standard observer angle (2, 10, or R). Defaults to "10".
        illuminant (str, optional): The illuminant used for color conversion. Defaults to "D65".
        gamutMap (Literal["soft", "clip"], optional): Gamut mapping strategy. "soft" uses smooth shoulder
            compression for gradual transitions; "clip" uses hard clipping. Defaults to "soft".
        gamutThreshold (float, optional): For soft mapping, start compression at this fraction of gamut
            (0.85 = 85%). Lower values compress earlier for smoother results. Defaults to 0.85.
        gamutStrength (float, optional): For soft mapping, compression intensity. Higher values produce
            more aggressive compression. Defaults to 1.5.

    Returns:
        IntArray: An integer array of shape `(steps, steps, steps)`, with RGB values packed as 0xRRGGBB.
            Axes are (hue, chroma, lightness).

    Examples:
        >>> cube = colorCube(10)
        >>> cube.shape
        (10, 10, 10)

        >>> cube[0, 0, 0]
        0x123456  # Example output representing an RGB color

        >>> # Hard clipping (legacy behavior)
        >>> cube = colorCube(10, gamutMap="clip")

        >>> # Aggressive soft compression
        >>> cube = colorCube(10, gamutThreshold=0.7, gamutStrength=3.0)
    """

    hclGrid = cubeHCL(steps)  # (n, n, n, 3) with axes (H, C, L)
    rgbFloat = _lab2rgb(_hcl2lab(hclGrid), illuminant=illuminant, observer=observer,
                        gamutMap=gamutMap, gamutThreshold=gamutThreshold, gamutStrength=gamutStrength)

    rgbU8 = np.clip(np.rint(rgbFloat * 255.0), 0, 255).astype(np.uint8)
    r, g, b = (rgbU8[..., 0].astype(np.uint32), rgbU8[..., 1].astype(np.uint32), rgbU8[..., 2].astype(np.uint32))

    packed = (r << 16) | (g << 8) | b
    return packed.astype(np.uint32)


@partial(optional=True)
def colorInterpolation(cube: IntArray, start: NumberSequenceND, end: NumberSequenceND, steps: int = 10, asPath: bool = True, includeEnd: bool = True) -> IntArray:
    """
    Interpolates between two colors in a color cube.

    This function interpolates colors within a 3D array of colors (a color cube) between a specified start and end color.
    The interpolation can be performed along a straight path between the two colors, or within a sub-cube of the color cube,
    depending on the `asPath` argument.


    Args:
        cube (IntArray): A 3D array of colors, typically generated by the `colorCube()` function.
        start (PArray): The starting color as (hue, chroma, lightness) indices. It can be a batch of colors or a single color.
        end (PArray): The ending color as (hue, chroma, lightness) indices. It can be a batch of colors or a single color.
        steps (int, optional): Number of interpolation steps (integer >= 1). Effective per-axis/path
            sample count is `N = steps + (1 if includeEnd else 0)`. Default is 10.
        asPath (bool, optional): If True, interpolates colors along a straight line path between `start` and `end`
            with O(N) storage. If False, samples a dense sub-cube with O(N^3) storage. Default is True.
        includeEnd (bool, optional): If True, include endpoint sample(s) from `end`. If False, endpoint is excluded:
            path mode returns exactly `steps` samples and sub-cube mode returns `steps` samples per axis. Default is True.

    Returns:
        IntArray: Interpolated colors with shape determined by `asPath` (see table above).

    Examples:
        >>> cube = colorCube(36)
        >>> start = P(10, 20, 30)  # (hue, chroma, lightness) indices
        >>> end = P(25, 35, 5)
        >>> colors = colorInterpolation(cube, start, end, steps=5)
        >>> colors.shape
        (6,)

        >>> colors_as_subcube = colorInterpolation(cube, start, end, steps=5, asPath=False)
        >>> colors_as_subcube.shape
        (6, 6, 6)
    """

    if not (isinstance(steps, int) and steps >= 1):
        raise ValueError("steps must be a positive integer")

    startArr = np.asarray(start, dtype=np.intp)
    endArr   = np.asarray(end,   dtype=np.intp)

    if startArr.shape[-1] != 3 or endArr.shape[-1] != 3:
        raise ValueError("start/end must have last dimension == 3")

    try:
        startB, endB = np.broadcast_arrays(startArr, endArr)
    except ValueError as e:
        raise ValueError("start and end are not broadcast-compatible") from e

    for axis, dim in enumerate(cube.shape):
        if (
            (startB[..., axis] < 0).any()
            or (startB[..., axis] >= dim).any()
            or (endB[..., axis] < 0).any()
            or (endB[..., axis] >= dim).any()
        ):
            raise IndexError("start/end indices out of cube bounds")

    nPts = steps + (1 if includeEnd else 0)

    # as path
    if asPath:
        t = np.linspace(0.0, 1.0, nPts, endpoint=includeEnd, dtype=float)

        # reshape t → (1,…,1,L,1) so it broadcasts over batch dims and channel dim
        t = t.reshape((1,) * (startB.ndim - 1) + (nPts, 1))

        diff = (endB - startB)[..., None, :]          # … × 1 × 3
        idx  = np.round(startB[..., None, :] + t * diff).astype(np.intp)

        # idx order is (hue, chroma, lightness) matching cube axes
        return cube[idx[..., 0], idx[..., 1], idx[..., 2]]

    # as sub-cube
    estBytes = np.prod(startB.shape[:-1], dtype=int) * (nPts ** 3) * cube.itemsize
    if estBytes > 1 << 24:  # if more than 16 MiB
        warnings.warn(
            "colorInterpolation(asPath=False): dense allocation "
            f"~{estBytes / (1 << 20):.1f} MiB with O(N^3) growth. "
            "Use asPath=True for O(N) memory when possible.",
            ResourceWarning,
            stacklevel=2,
        )

    # build integer linspaces directly, one per axis (H, C, L order)
    hue = np.round(
        np.linspace(startB[..., 0][..., None], endB[..., 0][..., None],
                    nPts, endpoint=includeEnd, axis=-1)
    ).astype(np.intp)
    chroma = np.round(
        np.linspace(startB[..., 1][..., None], endB[..., 1][..., None],
                    nPts, endpoint=includeEnd, axis=-1)
    ).astype(np.intp)
    lightness = np.round(
        np.linspace(startB[..., 2][..., None], endB[..., 2][..., None],
                    nPts, endpoint=includeEnd, axis=-1)
    ).astype(np.intp)

    # broadcast to full grid with axes (hue, chroma, lightness)
    return cube[hue[..., :, None, None], chroma[..., None, :, None], lightness[..., None, None, :]].astype(np.uint32)


@partial(optional=True)
def colorPicker(cube: IntArray,  hue: float, chroma: float, lightness: float, asArray: bool = False):
    """
    Pick a color from a color cube using normalized hue, chroma, and lightness.

    Retrieves a color from a 3D HCL color cube. Inputs are treated as normalized values and
    mapped to indices in the cube. Values outside the expected range are clamped to valid
    cube bounds. Returns either the RGB integer or the cube indices.

    Args:
        cube (IntArray): A 3D color array, typically from `colorCube()`. Axes are (H, C, L).
        hue (float): Normalized hue (typically in [0, 1], inclusive; clamped if outside).
        chroma (float): Normalized chroma (typically in [0, 1], inclusive; clamped if outside).
        lightness (float): Normalized lightness (typically in [0, 1], inclusive; clamped if outside).
        asArray (bool, optional): If True, returns indices array; if False, returns RGB integer. Default is False.

    Returns:
        Union[int, PArray]: RGB integer (when asArray=False) or array of [h, c, l] indices (when asArray=True).

    Examples:
        >>> cube = colorCube(36)
        >>> color = colorPicker(cube, 0.5, 0.5, 0.5)
        >>> color
        0x123456  # RGB color value

        >>> indices = colorPicker(cube, 0.5, 0.5, 0.5, asArray=True)
        >>> indices
        array([18, 18, 18])
    """
    indices = np.floor(P(hue, chroma, lightness) * cube.shape).astype(int)
    indices = np.clip(indices, 0, np.asarray(cube.shape, dtype=int) - 1)
    return P(indices[0], indices[1], indices[2]) if asArray else cube[indices[0], indices[1], indices[2]]


@partial(optional=True)
def colorScheme(
    cube: IntArray, hues: int = 8, swatches: int = 5, intensity: float = 0.4, variant: np.random.Generator = None
) -> PArray:
    """
    Generates a color scheme from a 3D array of colors in HCL space. Experimental.

    This function creates a color scheme by sampling from a 3D HCL color cube. The scheme is designed to vary in lightness and chroma,
    with the top part of the scheme getting darker and less saturated, while the bottom part gets brighter. The function returns an
    array of shape `(hues, swatches, 4)`, where each entry contains the RGB integer, and the indices from the HCL cube corresponding
    to hue, chroma, and lightness.

    Args:
        cube (IntArray): A 3D array of colors, typically generated from the `colorCube()` function. Axes are (H, C, L).
        hues (int, optional): The number of different hues in the scheme. Default is 8.
        swatches (int, optional): The number of swatches per hue. Default is 5.
        intensity (float, optional): Controls the variation in lightness and chroma. Lower values create more contrast. Default is 0.4.
        variant (np.random.Generator, optional): If an rng is provided, uses irregular spacing for the hue samples. Default is None.

    Returns:
        PArray: An array of shape `(hues, swatches, 4)` where the last dimension contains:
                - RGB integer color
                - Hue index
                - Chroma index
                - Lightness index

    Examples:
        >>> cube = colorCube(36)
        >>> scheme = colorScheme(cube, hues=6, swatches=4)
        >>> scheme.shape
        (6, 4, 4)

        >>> scheme[0, 0]  # Example of the first color's data
        array([0x123456, 0, 12, 18])  # [RGB integer, h-index, c-index, l-index]
    """
    hues += 1
    # Hue is now on axis 0 of the cube
    if variant is not None:
        _hs = irregular(maxValue=cube.shape[0] - 1, steps=hues, rng=variant)
    else:
        _hs = np.linspace(0, cube.shape[0] - 1, hues)
    hs = np.floor(_hs).astype(int)
    all = cube[hs]  # shape: (num_hues, C, L)

    def getSwatches(hueIndex: int):
        # Transpose to get (L, C) shape for consistent algorithm behavior
        colors = all[hueIndex].T  # shape: (L, C)

        x = np.linspace(0, colors.shape[0] - 1, swatches, dtype=int) - colors.shape[0] // 2
        dx = np.clip(x + colors.shape[0] // 2, 0, colors.shape[0] - 1).astype(int)

        y = colors.shape[0] - x**2 / (intensity * colors.shape[0])
        offy = colors.shape[1] / ((1.01 - intensity) * colors.shape[1] / 2)
        dy = np.clip(np.floor(y / 2 + offy), 0, colors.shape[1] - 1).astype(int)

        hueIndices = np.full(swatches, hueIndex, dtype=int)
        # Return [RGB, h-index, c-index, l-index] - note dy is chroma, dx is lightness after transpose
        return np.stack([colors[dx, dy], hueIndices, dy, dx], axis=-1)

    return np.array([getSwatches(h) for h in range(hues)])[:, -swatches:][:-1]


@partial(optional=True)
def gradient(pts: PArray, colors: List[ColorLike], kind: GradientStyle = "linear", stops: Optional[NumberSequence1D] = None):
    """
    Creates a gradient shader based on the specified kind and color stops. You can then use [Form.fill](Form.fill) to fill the gradient.

    This function generates a gradient shader, either linear, radial, sweep, or conical, depending on the `kind` argument.

    The gradient is defined by the `pts` for the start and end points:

    - linear: 2 points [start, end]

    - radial: 2 points [center, radius]

    - sweep: 1 point [center]

    - conical: 2 points [start_center, end_center]

    Args:
        pts (PArray): An array of points defining the gradient's start and end positions.
        colors (List[ColorLike]): A list of colors to be used in the gradient. These can be RGBA integers or other color formats.
        kind (str, optional): The type of gradient. Can be "linear", "radial", "sweep", or "conical". Default is "linear".
        stops (NumberSequence1D, optional): A sequence of stops for the gradient colors. If None, the colors are evenly spaced.

    Returns:
        skia.Shader: A Skia shader object that can be used for drawing the gradient.

    Examples:
        >>> pts = P([0, 0], [100, 100])
        >>> colors = [0xFF0000FF, 0xFFFF00FF, 0xFF00FFFF]
        >>> shader = gradient(pts, colors, kind="linear")
        >>> form.fill(shader)
    """

    argbColors = [toARGBInt(c) for c in colors]
    cpuPts = toPArray(pts)

    if kind == "linear":
        if cpuPts.shape[0] != 2:
            raise ValueError("linear gradient needs exactly 2 points")
        p0, p1 = [skia.Point(p[0], p[1]) for p in cpuPts]
        return skia.GradientShader.MakeLinear([p0, p1], argbColors, stops)

    if kind == "radial":
        if cpuPts.shape[0] != 2:
            raise ValueError("radial gradient needs centre and radius point")
        centre, edge = cpuPts
        radius = math.hypot(*(edge - centre))
        return skia.GradientShader.MakeRadial(
            skia.Point(*centre), radius, argbColors, stops
        )

    if kind == "sweep":
        cx, cy = cpuPts[0]
        return skia.GradientShader.MakeSweep(cx, cy, argbColors, stops)

    if kind == "conical":
        if cpuPts.shape[0] != 2:
            raise ValueError("conical gradient needs two centre points")
        start, end = map(lambda p: skia.Point(*p), cpuPts)
        r1 = math.hypot(*(cpuPts[1] - cpuPts[0]))
        return skia.GradientShader.MakeTwoPointConical(
            start, 0.0, end, r1, argbColors, stops
        )

    raise ValueError(f"Unknown gradient type '{kind}'")
