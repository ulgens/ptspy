"""Geometry constants and helpers. Work-in-progress."""

from ptspy.typing import NumberSequence1D, NumberSequence2D, PArray, NumberSequenceND, PArrayOrScalar
from .elements import P, partial, toPArray, TBD
from .. import np_wrapper as np
from types import SimpleNamespace
from typing import Optional, cast

from math import sqrt

# Represents an arbitrary very small number. It is set as 0.00001 here.
SMALL_NUMBER = 1e-5

# π radian (180 deg)
PI = 3.141592653589793

# Two π radian (360deg)
TWO_PI = PI * 2

# Half π radian (90deg)
HALF_PI = PI / 2

# π/4 radian (45deg)
QUARTER_PI = PI / 4

# π/180 or 1 degree in radian
ONE_DEGREE = PI / 180

# Multiply this constant with a radian to get a degree
RADIAN_TO_DEGREE = 180 / PI

# Multiply this constant with a degree to get a radian
DEGREE_TO_RADIAN = ONE_DEGREE

# Gaussian constant for normalization
GAUSSIAN = 1.0 / sqrt(2 * PI)


def toRadian(degrees: PArrayOrScalar) -> PArrayOrScalar:
    """
    Convert degrees to radians.

    Args:
        degrees (PArrayOrScalar): The angle in degrees. Can be a single scalar value or an array of values.

    Returns:
        PArrayOrScalar: The angle in radians, with the same shape as the input.

    Examples:
        >>> toRadian(180)
        3.141592653589793

        >>> toRadian(P(0, 90, 180, 270, 360))
        array([0.        , 1.57079633, 3.14159265, 4.71238898, 6.28318531])
    """
    return degrees * DEGREE_TO_RADIAN


def toDegree(radians: PArrayOrScalar) -> PArrayOrScalar:
    """
    Convert radians to degrees.

    Args:
        radians (PArrayOrScalar): The angle in radians. Can be a single scalar value or an array of values.

    Returns:
        PArrayOrScalar: The angle in degrees, with the same shape as the input.

    Examples:
        >>> toDegree(3.141592653589793)
        180.0

        >>> toDegree(P(0, 1.57079633, 3.14159265, 4.71238898, 6.28318531))
        array([  0.,  90., 180., 270., 360.])
    """
    return radians * RADIAN_TO_DEGREE


@partial(optional=True, placeholder=TBD)
def regularPolygon(center: NumberSequence1D, sides: int, radius: int = 1, closed: bool = False) -> PArray:
    """
    Generate the vertices of a regular polygon.

    Computes evenly spaced points around a circle centered at `center` with the given
    `radius` and number of `sides`. The first vertex is at angle 0 (positive x-axis) and
    vertices proceed counter-clockwise. When radius is 1 and the center is [0, 0], the polygon
    is inscribed in a unit circle, which is useful for normalization and scaling.

    Args:
        center (NumberSequence1D): Polygon center coordinates `[x, y]`.
        sides (int): Number of sides (and vertices) of the polygon. Must be at least 3.
        radius (int): Distance from center to each vertex.
        closed (bool, optional): If `True`, the polygon is closed by repeating the first vertex.

    Returns:
        PArray: Array of shape `(sides, 2)` containing the polygon's vertex coordinates.

    Examples:
        >>> regularPolygon([0, 0], sides=4, radius=1)
        array([[ 1. ,  0. ],
               [ 0. ,  1. ],
               [-1. ,  0. ],
               [ 0. , -1. ]])
    """
    if sides < 3:
        raise ValueError("sides must be at least 3")
    z = radius * np.exp(1j * np.linspace(0, 2 * np.pi, sides, False))
    vertices = np.column_stack([z.real, z.imag]) + center

    if closed:
        vertices = np.vstack([vertices, vertices[0]])

    return vertices


@partial(optional=True, placeholder=TBD)
def translate2D(pts: NumberSequenceND, offset: NumberSequenceND) -> PArray:
    """
    Translate (shift) 2D points by an offset vector.

    Args:
        pts (NumberSequenceND): Points with shape (..., 2).
        offset (NumberSequenceND): Translation vector [dx, dy].

    Returns:
        PArray: Translated points with the same shape as input.

    Examples:
        >>> translate2D(P(1, 2), P(10, 20))
        array([11., 22.])
    """
    return toPArray(pts, gpu=True) + toPArray(offset, gpu=True)


@partial(optional=True, placeholder=TBD)
def rotate2D(pts: NumberSequenceND, angle: float, origin: Optional[NumberSequenceND] = None) -> PArray:
    """
    Rotate 2D points clockwise by an angle in radians (screen coordinates, +y down).

    Args:
        pts (NumberSequenceND): Points with shape (..., 2).
        angle (float): Rotation angle in radians (clockwise in screen space).
        origin (Optional[NumberSequenceND]): Center of rotation [x, y]. None means [0, 0].

    Returns:
        PArray: Rotated points with the same shape as input.

    Examples:
        >>> rotate2D(P(1, 0), PI / 2)
        array([0., 1.])
    """
    pts = toPArray(pts, gpu=True)
    cosA = np.cos(angle)
    sinA = np.sin(angle)
    if origin is not None:
        origin = toPArray(origin, gpu=True)
        pts = pts - origin
    x = pts[..., 0]
    y = pts[..., 1]
    result = np.empty_like(pts)
    result[..., 0] = x * cosA - y * sinA
    result[..., 1] = x * sinA + y * cosA
    if origin is not None:
        result = result + origin
    return result


@partial(optional=True, placeholder=TBD)
def scale2D(pts: NumberSequenceND, factor: NumberSequenceND, origin: Optional[NumberSequenceND] = None) -> PArray:
    """
    Scale 2D points by a factor from an origin.

    Args:
        pts (NumberSequenceND): Points with shape (..., 2).
        factor (NumberSequenceND): Scalar for uniform scale, or [sx, sy] for non-uniform.
        origin (Optional[NumberSequenceND]): Center of scaling [x, y]. None means [0, 0].

    Returns:
        PArray: Scaled points with the same shape as input.

    Examples:
        >>> scale2D(P(1, 2), 3)
        array([3., 6.])
    """
    pts = toPArray(pts, gpu=True)
    factor = np.asarray(factor, dtype=np.float32)
    if origin is not None:
        origin = toPArray(origin, gpu=True)
        return (pts - origin) * factor + origin
    return pts * factor


@partial(optional=True, placeholder=TBD)
def skew2D(pts: NumberSequenceND, shear: NumberSequenceND, origin: Optional[NumberSequenceND] = None) -> PArray:
    """
    Skew (shear) 2D points by factors [sx, sy].

    Args:
        pts (NumberSequenceND): Points with shape (..., 2).
        shear (NumberSequenceND): Shear factors [sx, sy].
        origin (Optional[NumberSequenceND]): Center of shear [x, y]. None means [0, 0].

    Returns:
        PArray: Skewed points with the same shape as input.

    Examples:
        >>> skew2D(P(1, 0), P(0.5, 0))
        array([1., 0.])
    """
    pts = toPArray(pts, gpu=True)
    shear = toPArray(shear, gpu=True)
    if origin is not None:
        origin = toPArray(origin, gpu=True)
        pts = pts - origin
    x = pts[..., 0]
    y = pts[..., 1]
    result = np.empty_like(pts)
    result[..., 0] = x + shear[0] * y
    result[..., 1] = y + shear[1] * x
    if origin is not None:
        result = result + origin
    return result


@partial(optional=True, placeholder=TBD)
def reflect2D(pts: NumberSequenceND, angle: float = 0, origin: Optional[NumberSequenceND] = None) -> PArray:
    """
    Reflect 2D points across a line through the origin (or a given point) at the specified angle.

    The reflection axis is a line at `angle` radians from the positive x-axis.
    `angle=0` reflects across the x-axis (flips y) and `angle=PI/2` reflects across the y-axis (flips x).

    Args:
        pts (NumberSequenceND): Points with shape (..., 2).
        angle (float): Angle of the reflection axis in radians. Defaults to 0 (x-axis).
        origin (Optional[NumberSequenceND]): Center of reflection [x, y]. None means [0, 0].

    Returns:
        PArray: Reflected points with the same shape as input.

    Examples:
        >>> reflect2D(P(1, 1), 0)
        array([ 1., -1.])
    """
    pts = toPArray(pts, gpu=True)
    cos2A = np.cos(2 * angle)
    sin2A = np.sin(2 * angle)
    if origin is not None:
        origin = toPArray(origin, gpu=True)
        pts = pts - origin
    x = pts[..., 0]
    y = pts[..., 1]
    result = np.empty_like(pts)
    result[..., 0] = x * cos2A + y * sin2A
    result[..., 1] = x * sin2A - y * cos2A
    if origin is not None:
        result = result + origin
    return result


@partial(optional=True, placeholder=TBD)
def transform2D(pts: NumberSequenceND, matrix: NumberSequenceND) -> PArray:
    """
    Apply a general affine transform to 2D points.

    Args:
        pts (NumberSequenceND): Points with shape (..., 2).
        matrix (NumberSequenceND): Either a (2, 2) linear matrix or a (3, 3) affine matrix.

    Returns:
        PArray: Transformed points with the same shape as input.

    Raises:
        ValueError: If matrix shape is not (2, 2) or (3, 3).

    Examples:
        >>> transform2D(P(1, 0), np.eye(2))
        array([1., 0.])
    """
    pts = toPArray(pts, gpu=True)
    matrix = toPArray(matrix, gpu=True)
    if matrix.shape == (2, 2):
        return pts @ matrix.T
    elif matrix.shape == (3, 3):
        shape = pts.shape
        flat = pts.reshape(-1, 2)
        ones = np.ones((flat.shape[0], 1), dtype=np.float32)
        homogeneous = np.concatenate([flat, ones], axis=1)
        transformed = homogeneous @ matrix.T
        return transformed[..., :2].reshape(shape)
    else:
        raise ValueError(f"Matrix must be (2, 2) or (3, 3), got {matrix.shape}")


# --- Matrix2D helpers (private) ---


def _translateM(offset: PArray) -> PArray:
    """Build a 3x3 translation matrix."""
    m = np.eye(3, dtype=np.float32)
    m[0, 2] = offset[0]
    m[1, 2] = offset[1]
    return m


def _originWrap(m: PArray, origin: Optional[NumberSequenceND]) -> PArray:
    """Wrap a transform matrix with translate-to-origin and back."""
    if origin is None:
        return m
    o = toPArray(origin, gpu=True)
    return _translateM(o) @ m @ _translateM(-o)


class Matrix2D:
    """
    Fluent builder for 2D affine transform matrices.

    Chains transform calls left-to-right to build a 3x3 affine matrix. The result can be used in transform2D function.

    Examples:
        >>> m = Matrix2D().scale(2).rotate(PI / 4)
        >>> transform2D(pts, m)

        >>> mInv = m.inverse()
        >>> transform2D(transformedPts, mInv)  # undo the transform
    """

    __slots__ = ("_m",)

    def __init__(self):
        self._m = np.eye(3, dtype=np.float32)

    def translate(self, offset: NumberSequenceND) -> "Matrix2D":
        """
        Translate by offset.

        Args:
            offset (NumberSequenceND): Translation vector `[dx, dy]`.

        Returns:
            Matrix2D: Self, for chaining.
        """
        self._m = _translateM(toPArray(offset, gpu=True)) @ self._m
        return self

    def rotate(self, angle: float, origin: Optional[NumberSequenceND] = None) -> "Matrix2D":
        """
        Rotate clockwise (in screen space) by angle in radians.

        Args:
            angle (float): Rotation angle in radians.
            origin (NumberSequenceND, optional): Center of rotation. Defaults to `[0, 0]`.

        Returns:
            Matrix2D: Self, for chaining.
        """
        c, s = float(np.cos(angle)), float(np.sin(angle))
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        self._m = _originWrap(r, origin) @ self._m
        return self

    def scale(self, factor: NumberSequenceND, origin: Optional[NumberSequenceND] = None) -> "Matrix2D":
        """
        Scale uniformly by a scalar or non-uniformly by `[sx, sy]`.

        Args:
            factor (NumberSequenceND): Scalar for uniform scale, or `[sx, sy]` for per-axis.
            origin (NumberSequenceND, optional): Center of scaling. Defaults to `[0, 0]`.

        Returns:
            Matrix2D: Self, for chaining.
        """
        f = np.asarray(factor, dtype=np.float32)
        if f.ndim == 0:
            s = np.diag(np.array([f, f, 1], dtype=np.float32))
        else:
            s = np.diag(np.array([f[0], f[1], 1], dtype=np.float32))
        self._m = _originWrap(s, origin) @ self._m
        return self

    def skew(self, shear: NumberSequenceND, origin: Optional[NumberSequenceND] = None) -> "Matrix2D":
        """
        Skew by shear factors.

        Args:
            shear (NumberSequenceND): Shear factors `[sx, sy]`.
            origin (NumberSequenceND, optional): Center of shearing. Defaults to `[0, 0]`.

        Returns:
            Matrix2D: Self, for chaining.
        """
        sh = toPArray(shear, gpu=True)
        k = np.array([[1, sh[0], 0], [sh[1], 1, 0], [0, 0, 1]], dtype=np.float32)
        self._m = _originWrap(k, origin) @ self._m
        return self

    def reflect(self, angle: float = 0, origin: Optional[NumberSequenceND] = None) -> "Matrix2D":
        """
        Reflect across a line at a given angle from the x-axis.

        Args:
            angle (float): Angle of the reflection axis in radians. Defaults to 0 (x-axis).
            origin (NumberSequenceND, optional): Center of reflection. Defaults to `[0, 0]`.

        Returns:
            Matrix2D: Self, for chaining.
        """
        c2, s2 = float(np.cos(2 * angle)), float(np.sin(2 * angle))
        r = np.array([[c2, s2, 0], [s2, -c2, 0], [0, 0, 1]], dtype=np.float32)
        self._m = _originWrap(r, origin) @ self._m
        return self

    def inverse(self) -> "Matrix2D":
        """
        Return a new Matrix2D with the inverted transform.

        Returns:
            Matrix2D: A new instance whose matrix is the inverse of this one.
        """
        result = Matrix2D()
        result._m = np.linalg.inv(self._m).astype(np.float32)
        return result

    @property
    def m(self) -> PArray:
        """
        The 3x3 affine matrix.

        Returns:
            PArray: The accumulated 3x3 affine transform matrix.
        """
        return self._m

    def __array__(self, dtype=None):
        return self._m if dtype is None else self._m.astype(dtype)

    def __repr__(self):
        return f"Matrix2D(\n{self._m}\n)"


class Rectangle:
    "A class for various static helper methods related to rectangles."

    @staticmethod
    def size(rect: PArray) -> PArray:
        """
        Calculate the size of a rectangle.

        Takes a rectangle defined by two corner points and returns its width and height.

        Args:
            rect (PArray): An array of shape (2, 2) where rect[0] is one corner and rect[1] is the opposite corner.

        Returns:
            PArray: An array of shape (2,) containing [width, height].

        Examples:
            >>> rect = P([0, 0], [10, 20])
            >>> size = Rectangle.size(rect)
            >>> size
            array([10, 20])
        """
        return P(rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1])

    @staticmethod
    def center(rect: PArray) -> PArray:
        """
        Calculate the center point of a rectangle.

        Takes a rectangle defined by two corner points and returns its center.

        Args:
            rect (PArray): An array of shape (2, 2) where rect[0] is one corner and rect[1] is the opposite corner.

        Returns:
            PArray: An array of shape (2,) containing [x, y] coordinates of the center.

        Examples:
            >>> rect = P([0, 0], [10, 20])
            >>> center = Rectangle.center(rect)
            >>> center
            array([5.0, 10.0])
        """
        return P((rect[0, 0] + rect[1, 0]) / 2, (rect[0, 1] + rect[1, 1]) / 2)

    @staticmethod
    def fromTopLeft(topLeft: PArray, size: PArrayOrScalar) -> PArray:
        """
        Create a rectangle from the top-left corner and size.

        This method takes the top-left point of a rectangle and its size (width and height),
        and returns the rectangle defined by the top-left and bottom-right points.

        Args:
            topLeft (PArray): A 2D array representing the top-left point of the rectangle.
            size (PArrayOrScalar): A 2D array or scalar representing the size of the rectangle. If a scalar is provided, it is used for both width and height.

        Returns:
            PArray: A 2D array where the first row is the top-left point and
                                     the second row is the bottom-right point.

        Examples:
            >>> topLeft = P(0, 0)
            >>> size = P(10, 20)
            >>> rect = Rectangle.fromTopLeft(topLeft, size)
            >>> rect
            array([[ 0,  0],
                   [10, 20]])
        """
        return P(topLeft, topLeft + size)

    @staticmethod
    def fromCenter(center: PArray, size: PArrayOrScalar) -> PArray:
        """
        Create a bounding box from a center point and size.

        Args:
            center (PArray): The center point of the bounding box.
            size (PArrayOrScalar): The size of the bounding box. Can be a scalar or an array specifying the size along each dimension.

        Returns:
            P: A bounding box with coordinates calculated from the center and size.

        Examples:
            >>> center = P(5, 5)
            >>> size = 4
            >>> bbox = Rectangle.fromCenter(center, size)
            >>> bbox
            P(array([3., 3.]), array([7., 7.]))

            >>> size = P(4, 6)
            >>> bbox = Rectangle.fromCenter(center, size)
            >>> bbox
            P(array([3., 2.]), array([7., 8.]))
        """
        return P(center - size / 2, center + size / 2)

    @staticmethod
    def strips(
        rect: NumberSequenceND,
        ratio: PArrayOrScalar = 0.5,
        axis: int = 0,
    ) -> PArray:
        """
        Split a rectangle into strips along an axis based on the `ratio` parameter.

        Args:
            rect (NumberSequenceND): Either a rectangular size `[w, h]`, or a bounding rectangle `[[x0, y0], [x1, y1]]`.
            ratio (PArrayOrScalar): Fraction(s) of the span. Scalar for equal splits, array for proportional splits.
            axis (int): 0 = vertical strips (along x), 1 = horizontal (along y).

        Returns:
            PArray: an array of (number of strips, 2, 2)

        Examples:
            >>> Rectangle.strips([100, 60], 0.25).shape
            (4, 2, 2)
            >>> Rectangle.strips([100, 60], [0.2, 0.5, 0.1]).shape
            (4, 2, 2)
        """
        arr = toPArray(rect, gpu=True)
        if arr.ndim == 1:
            origin = np.zeros(2, dtype=np.float32)
            end = arr[:2].copy()
        else:
            origin = arr[0, :2].copy()
            end = arr[1, :2].copy()

        lo = float(origin[axis])
        hi = float(end[axis])
        span = hi - lo
        if span <= 0:
            raise ValueError(f"rect must have positive extent along axis {axis}")

        ratios = np.atleast_1d(np.asarray(ratio, dtype=np.float32))

        if ratios.size == 1:
            # Scalar: equal splits
            r = float(ratios.item())
            if r <= 0 or r > 1:
                raise ValueError(f"scalar ratio must be in (0, 1], got {r}")
            n = max(1, round(1.0 / r))
            edges = np.linspace(lo, hi, n + 1, dtype=np.float32)
        else:
            # Array: proportional splits
            cumulative = np.cumsum(ratios)
            # Keep ratios that start before 1.0 (truncate overflow)
            keep = int(np.searchsorted(cumulative, 1.0, side="left")) + 1
            ratios = ratios[:keep]
            cumulative = np.minimum(np.cumsum(ratios), 1.0)
            edges = np.concatenate([[0.0], cumulative.astype(np.float32)]) * span + lo
            # Add remainder strip if we haven't reached the end
            if edges[-1] < hi - 1e-6:
                edges = np.append(edges, np.float32(hi))

        n = len(edges) - 1
        mins = np.broadcast_to(origin, (n, 2)).copy()
        maxs = np.broadcast_to(end, (n, 2)).copy()
        mins[:, axis] = edges[:-1]
        maxs[:, axis] = edges[1:]

        return toPArray(np.stack([mins, maxs], axis=1), gpu=True)

    @staticmethod
    def helper(data: NumberSequenceND) -> SimpleNamespace:
        """
        Determine the bounds of a 2D rectangle and provide semantic access to its corners.
        Assumes screen coordinates (y increases downward).

        Args:
            data (NumberSequenceND): Either [width, height] (origin at [0,0]) or [[x1, y1], [x2, y2]].

        Returns:
            SimpleNamespace: A namespace with attributes 'tl' (top left), 'tr' (top right),
                'bl' (bottom left), 'br' (bottom right), and 'center'. It also contains 'top',
                'left', 'right', and 'bottom' segments, plus 'size'.

        Raises:
            ValueError: If `data` does not match the expected formats.

        Examples:
            >>> data = [10, 5]
            >>> bounds = Rectangle.helper(data)
            >>> bounds.tl
            P([0, 0])
            >>> bounds.br
            P([10, 5])

            >>> data = [[2, 3], [5, 7]]
            >>> bounds = Rectangle.helper(data)
            >>> bounds.tl
            P([2, 3])
            >>> bounds.br
            P([5, 7])
        """
        if not np.isarray(data):
            data = toPArray(data, gpu=True)

        # Tell the type checker it's definitely an ndarray:
        data = cast(PArray, data)

        if data.ndim == 1:
            if data.size < 2:
                raise ValueError("Input data must include width and height.")
            # Assuming the format is [width, height]
            p1 = P(0, 0)
            p2 = data[:2]
        elif data.ndim > 1 and data.shape[1] > 1:
            # Assuming the format is [[x1, y1], [x2, y2]]
            p1 = data[0, :2]
            p2 = data[1, :2]
        else:
            raise ValueError("Input data must either be [width, height] or [[x1, y1], [x2, y2]].")

        tl = p1
        tr = P(p2[0], p1[1])
        bl = P(p1[0], p2[1])
        br = p2

        return SimpleNamespace(
            tl=tl,
            tr=tr,
            bl=bl,
            br=br,
            center=(p1 + p2) / 2,
            top=P(tl, tr),
            left=P(tl, bl),
            right=P(tr, br),
            bottom=P(bl, br),
            size=P(p2 - p1),
        )


class Circle:
    "A class for various static helper methods related to circles."

    @staticmethod
    def fromCenter(center: PArray, radius: PArrayOrScalar) -> PArray:
        """
        Create a bounding box from a center point and radius.

        Args:
            center (PArray): The center point of the bounding box.
            radius (PArrayOrScalar): The radius of the bounding box. Can be a scalar or an array specifying the radius along each dimension.

        Returns:
            P: A bounding box with coordinates calculated from the center and radius.

        Examples:
            >>> center = P(5, 5)
            >>> radius = 2
            >>> bbox = Circle.fromCenter(center, radius)
            >>> bbox
            P([3, 3], [7, 7])

            >>> radius = P(2, 3)
            >>> bbox = Circle.fromCenter(center, radius)
            >>> bbox
            P([3, 2], [7, 8])
        """
        return P(center - radius, center + radius)


def hermiteToBezier(
    pts: NumberSequence2D,
    tangents: NumberSequence2D,
    closed: bool = False,
) -> PArray:
    """
    Convert Hermite spline data (positions + tangent vectors) into cubic Bezier
    control points.

    Args:
        pts (NumberSequence2D): Shape (n, 2) — positions the curve passes through.
        tangents (NumberSequence2D): Shape (n, 2) — tangent vectors at each point.
        closed (bool): If True, adds a final segment back to the first point.

    Returns:
        PArray: array of shape (m, 2) like `[start, cp1, cp2, end, cp1, cp2, end, ...]`
    """
    pts = toPArray(pts, gpu=True)
    tangents = toPArray(tangents, gpu=True)
    n = len(pts)

    numSegments = n if closed else n - 1
    idx0 = np.arange(numSegments)
    idx1 = (idx0 + 1) % n

    cp1 = pts[idx0] + tangents[idx0] / 3.0
    cp2 = pts[idx1] - tangents[idx1] / 3.0
    ends = pts[idx1]

    result = np.empty((1 + 3 * numSegments, 2), dtype=np.float32)
    result[0] = pts[0]
    result[1::3] = cp1
    result[2::3] = cp2
    result[3::3] = ends

    return result


def cardinalToBezier(
    pts: NumberSequence2D,
    tension: PArrayOrScalar = 0.5,
    closed: bool = False,
) -> PArray:
    """
    Convert cardinal spline control points into cubic Bezier control points.

    Args:
        pts (NumberSequence2D): Shape (n, 2) — the points the curve passes through.
        tension (PArrayOrScalar): Scalar (default 0.5 = Catmull-Rom) or array of
            shape (n,) for per-point tension. 1.0 produces straight lines.
        closed (bool): If True, wraps tangents and adds a final segment back to the start.

    Returns:
        PArray: array of shape (m, 2) like `[start, cp1, cp2, end, cp1, cp2, end, ...]`
    """
    pts = toPArray(pts, gpu=True)
    n = len(pts)
    idx = np.arange(n)

    if closed:
        im1, ip1 = (idx - 1) % n, (idx + 1) % n
    else:
        im1, ip1 = np.clip(idx - 1, 0, n - 1), np.clip(idx + 1, 0, n - 1)

    s = np.atleast_1d(1.0 - np.asarray(tension, dtype=np.float32))
    tangents = s[:, None] * (pts[ip1] - pts[im1])  # (n, 2)

    return hermiteToBezier(pts, tangents, closed)


def sampleCardinal(
    pts: NumberSequence2D,
    tension: PArrayOrScalar = 0.5,
    numSamples: int = 20,
    closed: bool = False,
) -> PArray:
    """
    Evaluate points along a cardinal spline curve using the Hermite basis matrix.

    Args:
        pts (NumberSequence2D): Shape (n, 2) — control points.
        tension (PArrayOrScalar): Scalar or per-point array of shape (n,).
        numSamples (int): Samples per segment.
        closed (bool): If True, the curve loops back to the start.

    Returns:
        PArray: Shape (totalSamples, 2) of evaluated positions.
    """
    pts = toPArray(pts, gpu=True)
    n = len(pts)
    numSegments = n if closed else n - 1

    # Index arrays for 4 control points per segment: [P_{i-1}, P_i, P_{i+1}, P_{i+2}]
    seg = np.arange(numSegments)
    if closed:
        im1, i0, i1, i2 = (seg - 1) % n, seg, (seg + 1) % n, (seg + 2) % n
    else:
        im1 = np.clip(seg - 1, 0, n - 1)
        i0, i1 = seg, seg + 1
        i2 = np.clip(seg + 2, 0, n - 1)

    G = np.stack([pts[im1], pts[i0], pts[i1], pts[i2]], axis=1)  # (numSegments, 4, d)

    # T matrix: (numSamples, 4) — shared across all segments
    t = np.linspace(0, 1, numSamples, endpoint=False, dtype=np.float32)
    T = np.column_stack([t**3, t**2, t, np.ones_like(t)])

    if np.isscalar(tension):
        s = 1.0 - float(tension)
        M = np.array(
            [
                [-s, 2.0 - s, s - 2.0, s],
                [2.0 * s, s - 3.0, 3.0 - 2.0 * s, -s],
                [-s, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        TM = T @ M  # (numSamples, 4)
        samples = np.einsum("si,nid->nsd", TM, G)  # (numSegments, numSamples, d)
    else:
        s = 1.0 - np.asarray(tension, dtype=np.float32)[i0]  # (numSegments,)
        z = np.zeros_like(s)
        o = np.ones_like(s)
        M = np.array(
            [
                [-s, 2.0 - s, s - 2.0, s],
                [2.0 * s, s - 3.0, 3.0 - 2.0 * s, -s],
                [-s, z, s, z],
                [z, o, z, z],
            ],
            dtype=np.float32,
        ).transpose(2, 0, 1)  # (numSegments, 4, 4)
        samples = np.einsum("si,nij,njd->nsd", T, M, G)  # (numSegments, numSamples, d)

    samples = samples.reshape(-1, pts.shape[1])
    endpoint = pts[0:1] if closed else pts[-1:]
    return np.concatenate([samples, endpoint], axis=0)


def bsplineToBezier(
    pts: NumberSequence2D,
    closed: bool = False,
) -> PArray:
    """
    Convert uniform cubic B-spline control points into cubic Bezier control
    points.

    Args:
        pts (NumberSequence2D): Shape (n, 2) B-spline control points.
            Minimum 4 for open, 3 for closed.
        closed (bool): If True, wraps control points and adds a final segment
            back to the start.

    Returns:
        PArray: array of shape (m, 2) like `[start, cp1, cp2, end, cp1, cp2, end, ...]`
    """
    pts = toPArray(pts, gpu=True)
    n = len(pts)

    if closed:
        # Wrap: each segment i uses pts[i], pts[(i+1)%n], pts[(i+2)%n], pts[(i+3)%n]
        numSegments = n
        idx0 = np.arange(n)
        idx1 = (idx0 + 1) % n
        idx2 = (idx0 + 2) % n
        idx3 = (idx0 + 3) % n
    else:
        # Open: segment i uses pts[i..i+3], giving n-3 segments
        numSegments = n - 3
        idx0 = np.arange(numSegments)
        idx1 = idx0 + 1
        idx2 = idx0 + 2
        idx3 = idx0 + 3

    Q0 = pts[idx0]
    Q1 = pts[idx1]
    Q2 = pts[idx2]
    Q3 = pts[idx3]

    # Uniform cubic B-spline → Bezier conversion
    starts = (Q0 + 4.0 * Q1 + Q2) / 6.0
    cp1s = (2.0 * Q1 + Q2) / 3.0
    cp2s = (Q1 + 2.0 * Q2) / 3.0
    ends = (Q1 + 4.0 * Q2 + Q3) / 6.0

    result = np.empty((1 + 3 * numSegments, 2), dtype=np.float32)
    result[0] = starts[0]
    result[1::3] = cp1s
    result[2::3] = cp2s
    result[3::3] = ends

    return result


def sampleBspline(
    pts: NumberSequence2D,
    numSamples: int = 20,
    closed: bool = False,
) -> PArray:
    """
    Evaluate points along a uniform cubic B-spline using the basis matrix.

    Args:
        pts (NumberSequence2D): Shape (n, 2) — B-spline control points.
            Minimum 4 for open, 3 for closed.
        numSamples (int): Samples per segment.
        closed (bool): If True, the curve loops back to the start.

    Returns:
        PArray: Shape (totalSamples, 2) of evaluated positions.
    """
    pts = toPArray(pts, gpu=True)
    n = len(pts)
    numSegments = n if closed else n - 3

    # Uniform cubic B-spline basis matrix
    M = (
        np.array(
            [
                [-1, 3, -3, 1],
                [3, -6, 3, 0],
                [-3, 0, 3, 0],
                [1, 4, 1, 0],
            ],
            dtype=np.float32,
        )
        / 6.0
    )

    t = np.linspace(0, 1, numSamples, endpoint=False, dtype=np.float32)
    T = np.column_stack([t**3, t**2, t, np.ones_like(t)])
    TM = T @ M  # (numSamples, 4)

    # Index arrays for 4 control points per segment
    seg = np.arange(numSegments)
    G = np.stack(
        [pts[seg % n], pts[(seg + 1) % n], pts[(seg + 2) % n], pts[(seg + 3) % n]], axis=1
    )  # (numSegments, 4, d)

    samples = np.einsum("si,nid->nsd", TM, G)  # (numSegments, numSamples, d)
    samples = samples.reshape(-1, pts.shape[1])

    # Append final endpoint
    if closed:
        endpoint = samples[0:1]
    else:
        endpoint = ((pts[-3] + 4.0 * pts[-2] + pts[-1]) / 6.0)[np.newaxis].astype(np.float32)

    return np.concatenate([samples, endpoint], axis=0)


def sampleBezier(
    pts: NumberSequence2D,
    numSamples: int = 20,
) -> PArray:
    """
    Sample points along a cubic Bezier curve.

    Args:
        pts (NumberSequence2D): array of shape (m, 2) like `[start, cp1, cp2, end, cp1, cp2, end, ...]`
        numSamples (int): Samples per segment.

    Returns:
        PArray: Shape (totalSamples, 2) of evaluated positions.
    """
    pts = toPArray(pts, gpu=True)
    d = pts.shape[1]

    # Parse segments: first point is start, then groups of 3 (cp1, cp2, end)
    rest = pts[1:]
    keep = len(rest) - (len(rest) % 3)
    if keep == 0:
        return pts[:1].copy()

    segments = rest[:keep].reshape(-1, 3, d)  # (numSegments, 3, d)
    numSegments = len(segments)

    # Build anchors: P0[0] = pts[0], P0[i] = end of previous segment
    P0 = np.empty((numSegments, d), dtype=np.float32)
    P0[0] = pts[0]
    if numSegments > 1:
        P0[1:] = segments[:-1, 2]

    # Bernstein basis polynomials — computed once
    t = np.linspace(0, 1, numSamples, endpoint=False, dtype=np.float32)
    u = 1.0 - t
    B = np.column_stack([u**3, 3.0 * u**2 * t, 3.0 * u * t**2, t**3])  # (numSamples, 4)

    # Geometry: (numSegments, 4, d)
    G = np.stack([P0, segments[:, 0], segments[:, 1], segments[:, 2]], axis=1)

    samples = np.einsum("si,nid->nsd", B, G)  # (numSegments, numSamples, d)
    samples = samples.reshape(-1, d)

    return np.concatenate([samples, segments[-1:, 2]], axis=0)
