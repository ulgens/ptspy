"""Type aliases shared across ptspy."""

from __future__ import annotations

from typing import Literal, TypeAlias
from collections.abc import Callable, Sequence

import skia
from numpy.typing import ArrayLike, NDArray

from . import np_wrapper as np


# Either an int value or a float value
Number: TypeAlias = int | float | np.intp | np.float32 | np.float64

# Names for stroke joins and caps
StrokeJoinCap: TypeAlias = Literal["bevel", "round", "miter", "last"]

# Visual styles for rendering points
PointStyle: TypeAlias = Literal["circle", "rect"]

# A numpy-like array of floats (float32 for GPU compatibility and Skia alignment)
PArray: TypeAlias = NDArray[np.float32]

# Either a float value or a numpy-like array of floats
PArrayOrScalar: TypeAlias = float | np.float32 | PArray

# A numpy-like array of ints
IntArray: TypeAlias = NDArray[np.intp]

# Either an int value or a numpy-like array of ints
IntArrayOrScalar: TypeAlias = int | np.intp | IntArray

# Either a numpy-like array or a python sequence of ints
IntArrayOrList: TypeAlias = IntArray | Sequence[int]

# A numpy-like 1D array, list or tuple of floats or ints
NumberSequence1D: TypeAlias = PArray | Sequence[Number]

# A numpy-like 2D array, list or tuple of floats or ints
NumberSequence2D: TypeAlias = PArray | Sequence[Sequence[Number]]

# A numpy-like ND array, list or tuple of floats or ints
NumberSequenceND: TypeAlias = "PArray | Sequence[Number | NumberSequenceND]"

# Same as ArrayLike from numpy.typing
PArrayLike: TypeAlias = ArrayLike

# A flexible color type: Either an RGB/RGBA array, an RGB/RGBA int, or a string
ColorLike: TypeAlias = int | NumberSequence1D | str

# A color value or a shader
ColorOrShader: TypeAlias = ColorLike | skia.Shader

# A shaping function for interpolation
ShapingFunction: TypeAlias = Callable[[PArrayOrScalar, PArrayOrScalar, PArrayOrScalar], PArrayOrScalar]

# Shaping method identifiers for interpolation
ShapingMethod: TypeAlias = Literal[
    "linear",
    "quadraticIn",
    "quadraticOut",
    "quadraticInOut",
    "cubicIn",
    "cubicOut",
    "cubicInOut",
    "exponentialIn",
    "exponentialOut",
    "cosineInOut",
    "sigmoid",
    "seat",
    "step",
]

# Gradient styles for color gradients
GradientStyle: TypeAlias = Literal["linear", "radial", "sweep", "conical"]

# Padding modes for various operations
PadMode: TypeAlias = Literal["last", "value", "wrap", "mirror", "truncate"]

# Rounding modes for various operations
RoundingMode: TypeAlias = Literal["floor", "ceil", "round"]


__all__ = (
    "Number",
    "StrokeJoinCap",
    "PointStyle",
    "PArray",
    "PArrayOrScalar",
    "IntArray",
    "IntArrayOrScalar",
    "IntArrayOrList",
    "NumberSequence1D",
    "NumberSequence2D",
    "NumberSequenceND",
    "PArrayLike",
    "ColorLike",
    "ColorOrShader",
    "ShapingFunction",
    "ShapingMethod",
    "GradientStyle",
    "PadMode",
    "RoundingMode",
)
