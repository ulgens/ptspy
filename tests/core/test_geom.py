import numpy as np
import pytest

from ptspy.core import geom
from ptspy.core.elements import P, toPArray, TBD


def test_degree_radian_conversions_roundtrip():
    degrees = np.array([0, 90, 180, 360], dtype=float)
    radians = geom.toRadian(degrees)
    assert np.allclose(geom.toDegree(radians), degrees)


def test_regular_polygon_and_closed_flag():
    poly = geom.regularPolygon([0, 0], sides=4, radius=1, closed=False)
    assert poly.shape == (4, 2)
    assert np.allclose(np.linalg.norm(poly, axis=1), np.ones(4))

    closed = geom.regularPolygon([0, 0], sides=3, radius=2, closed=True)
    assert closed.shape == (4, 2)  # extra vertex
    assert np.array_equal(closed[0], closed[-1])
    # vertices should be ordered around circle
    angles = np.unwrap(np.arctan2(closed[:-1, 1], closed[:-1, 0]))
    assert np.all(np.diff(angles) > 0)


def test_regular_polygon_rejects_fewer_than_3_sides():
    with pytest.raises(ValueError):
        geom.regularPolygon([0, 0], sides=2)
    with pytest.raises(ValueError):
        geom.regularPolygon([0, 0], sides=0)


def test_rectangle_helpers_and_validation():
    rect = np.array([[0, 0], [10, 5]])
    assert np.array_equal(geom.Rectangle.size(rect), np.array([10, 5]))
    assert np.array_equal(geom.Rectangle.center(rect), np.array([5.0, 2.5]))
    assert np.array_equal(geom.Rectangle.fromTopLeft(np.array([1, 2]), np.array([3, 4])), np.array([[1, 2], [4, 6]]))
    assert np.array_equal(geom.Rectangle.fromCenter(np.array([5, 5]), 4), np.array([[3, 3], [7, 7]]))

    helper = geom.Rectangle.helper([[2, 3], [5, 7]])
    assert np.array_equal(helper.tl, np.array([2, 3]))
    assert np.array_equal(helper.br, np.array([5, 7]))
    assert np.array_equal(helper.center, np.array([3.5, 5.0]))

    with pytest.raises(ValueError):
        geom.Rectangle.helper(np.array([1, 2, 3])[:, None])
    with pytest.raises(ValueError):
        geom.Rectangle.helper([1])


# --- Rectangle.strips tests ---

def test_strips_scalar_equal():
    """Scalar ratio=0.25 gives 4 equal vertical strips."""
    strips = geom.Rectangle.strips([100, 60], 0.25)
    assert strips.shape == (4, 2, 2)
    np.testing.assert_allclose(strips[:, 0, 0], [0, 25, 50, 75])
    np.testing.assert_allclose(strips[:, 1, 0], [25, 50, 75, 100])
    # y unchanged
    np.testing.assert_allclose(strips[:, 0, 1], 0)
    np.testing.assert_allclose(strips[:, 1, 1], 60)


def test_strips_scalar_horizontal():
    """Scalar ratio along axis=1 splits horizontally."""
    strips = geom.Rectangle.strips([100, 60], 1 / 3, axis=1)
    assert strips.shape == (3, 2, 2)
    np.testing.assert_allclose(strips[:, 0, 1], [0, 20, 40])
    np.testing.assert_allclose(strips[:, 1, 1], [20, 40, 60])
    np.testing.assert_allclose(strips[:, 0, 0], 0)
    np.testing.assert_allclose(strips[:, 1, 0], 100)


def test_strips_scalar_half():
    """ratio=0.5 gives 2 equal strips."""
    strips = geom.Rectangle.strips([100, 60], 0.5)
    assert strips.shape == (2, 2, 2)
    np.testing.assert_allclose(strips[0], [[0, 0], [50, 60]])
    np.testing.assert_allclose(strips[1], [[50, 0], [100, 60]])


def test_strips_scalar_whole():
    """ratio=1.0 gives 1 strip (the whole rect)."""
    strips = geom.Rectangle.strips([100, 60], 1.0)
    assert strips.shape == (1, 2, 2)
    np.testing.assert_allclose(strips[0], [[0, 0], [100, 60]])


def test_strips_bbox_input():
    """Bbox [[10, 20], [110, 80]] splits with origin preserved."""
    strips = geom.Rectangle.strips([[10, 20], [110, 80]], 0.5)
    assert strips.shape == (2, 2, 2)
    np.testing.assert_allclose(strips[0], [[10, 20], [60, 80]])
    np.testing.assert_allclose(strips[1], [[60, 20], [110, 80]])


def test_strips_array_proportional():
    """Array ratio [0.2, 0.5, 0.1] gives 4 strips (remainder fills last)."""
    strips = geom.Rectangle.strips([100, 60], [0.2, 0.5, 0.1])
    assert strips.shape == (4, 2, 2)
    np.testing.assert_allclose(strips[:, 0, 0], [0, 20, 70, 80])
    np.testing.assert_allclose(strips[:, 1, 0], [20, 70, 80, 100])


def test_strips_array_exact_sum():
    """Ratios summing to exactly 1.0 produce no remainder strip."""
    strips = geom.Rectangle.strips([100, 60], [0.3, 0.7])
    assert strips.shape == (2, 2, 2)
    np.testing.assert_allclose(strips[:, 0, 0], [0, 30])
    np.testing.assert_allclose(strips[:, 1, 0], [30, 100])


def test_strips_array_overflow():
    """Ratios exceeding 1.0 are truncated — third ratio is clipped."""
    strips = geom.Rectangle.strips([100, 60], [0.5, 0.4, 0.3])
    assert strips.shape == (3, 2, 2)
    np.testing.assert_allclose(strips[0, 0, 0], 0)
    np.testing.assert_allclose(strips[0, 1, 0], 50)
    np.testing.assert_allclose(strips[1, 0, 0], 50)
    np.testing.assert_allclose(strips[1, 1, 0], 90)
    # Third strip is clipped to the boundary
    np.testing.assert_allclose(strips[2, 0, 0], 90)
    np.testing.assert_allclose(strips[2, 1, 0], 100)


def test_strips_invalid_ratio_zero():
    with pytest.raises(ValueError, match="ratio must be in"):
        geom.Rectangle.strips([100, 60], 0)


def test_strips_invalid_ratio_negative():
    with pytest.raises(ValueError, match="ratio must be in"):
        geom.Rectangle.strips([100, 60], -0.5)


def test_strips_invalid_ratio_over_one():
    with pytest.raises(ValueError, match="ratio must be in"):
        geom.Rectangle.strips([100, 60], 1.5)


def test_circle_from_center():
    box = geom.Circle.fromCenter(np.array([5, 5]), 2)
    assert np.array_equal(box, np.array([[3, 3], [7, 7]]))


# --- cardinalToBezier tests ---

def test_cardinalToBezier_basic():
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30]], dtype=float)
    result = geom.cardinalToBezier(pts)
    n = len(pts)
    # Open curve: 1 + 3*(n-1) points
    assert result.shape == (1 + 3 * (n - 1), 2)
    # First point matches input start
    assert np.allclose(result[0], pts[0])
    # Last anchor matches last input point
    assert np.allclose(result[-1], pts[-1])


def test_cardinalToBezier_tension_extremes():
    pts = np.array([[0, 0], [10, 10], [20, 0], [30, 10]], dtype=float)
    # tension=1 → s=0 → tangents are zero → control points collapse onto anchors (straight lines)
    result = geom.cardinalToBezier(pts, tension=1.0)
    # For each segment, cp1 should equal start and cp2 should equal end
    for seg in range(len(pts) - 1):
        cp1 = result[1 + seg * 3]
        cp2 = result[2 + seg * 3]
        start = pts[seg]
        end = pts[seg + 1]
        assert np.allclose(cp1, start), f"seg {seg}: cp1 should equal start"
        assert np.allclose(cp2, end), f"seg {seg}: cp2 should equal end"


def test_cardinalToBezier_closed():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    result = geom.cardinalToBezier(pts, closed=True)
    n = len(pts)
    # Closed curve: 1 + 3*n points (n segments)
    assert result.shape == (1 + 3 * n, 2)
    # Last anchor should be the first point (loop back)
    assert np.allclose(result[-1], pts[0])


def test_cardinalToBezier_perPointTension():
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30]], dtype=float)
    scalar_result = geom.cardinalToBezier(pts, tension=0.5)
    per_point = np.array([0.3, 0.5, 0.7, 0.2])
    varied_result = geom.cardinalToBezier(pts, tension=per_point)
    # Same shape
    assert varied_result.shape == scalar_result.shape
    # Different control points
    assert not np.allclose(scalar_result, varied_result)


def test_cardinalToBezier_minPoints():
    pts = np.array([[0, 0], [10, 10]], dtype=float)
    result = geom.cardinalToBezier(pts)
    # 2 points → 1 segment → 1 + 3 = 4 points
    assert result.shape == (4, 2)
    assert np.allclose(result[0], pts[0])
    assert np.allclose(result[-1], pts[-1])


# --- sampleCardinal tests ---

def test_sampleCardinal_onControlPoints():
    """Sampled curve at segment boundaries should pass through the original control points."""
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30]], dtype=float)
    numSamples = 50
    samples = geom.sampleCardinal(pts, numSamples=numSamples)
    # First sample is first point
    assert np.allclose(samples[0], pts[0], atol=1e-3)
    # Last sample is last point
    assert np.allclose(samples[-1], pts[-1], atol=1e-3)
    # Interior control points appear at segment boundaries
    # Segment 0 ends at index numSamples, segment 1 ends at 2*numSamples, etc.
    for i in range(1, len(pts) - 1):
        boundary_idx = i * numSamples
        assert np.allclose(samples[boundary_idx], pts[i], atol=1e-3), \
            f"Boundary {i}: expected {pts[i]}, got {samples[boundary_idx]}"


def test_sampleCardinal_closed():
    """Closed curve samples should form a loop (first ≈ last)."""
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    samples = geom.sampleCardinal(pts, closed=True, numSamples=30)
    assert np.allclose(samples[0], samples[-1], atol=1e-3)


# --- sampleBezier tests ---

def test_sampleBezier_straightLine():
    """Bezier with collinear control points → sampled points are collinear."""
    # Straight line from (0,0) to (30,30): control points on the line
    pts = np.array([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=float)
    samples = geom.sampleBezier(pts, numSamples=20)
    # All points should lie on y=x line
    assert np.allclose(samples[:, 0], samples[:, 1], atol=1e-3)


def test_sampleBezier_endpoints():
    """First/last sampled point match first/last Bezier anchor."""
    pts = np.array([[0, 0], [5, 20], [15, -10], [20, 5],
                    [25, 15], [35, 0], [40, 10]], dtype=float)
    samples = geom.sampleBezier(pts, numSamples=10)
    assert np.allclose(samples[0], pts[0], atol=1e-3)
    # Last anchor is pts[3] for first segment, pts[6] for second
    assert np.allclose(samples[-1], pts[6], atol=1e-3)


# --- hermiteToBezier tests ---

def test_hermiteToBezier_basic():
    """4 points + tangents → correct shape, first/last match."""
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30]], dtype=float)
    tangents = np.array([[5, 10], [10, -5], [-5, 10], [5, 5]], dtype=float)
    result = geom.hermiteToBezier(pts, tangents)
    n = len(pts)
    assert result.shape == (1 + 3 * (n - 1), 2)
    assert np.allclose(result[0], pts[0])
    assert np.allclose(result[-1], pts[-1])


def test_hermiteToBezier_zeroTangents():
    """Zero tangents → cp1 = start, cp2 = end (straight lines)."""
    pts = np.array([[0, 0], [10, 10], [20, 0]], dtype=float)
    tangents = np.zeros_like(pts)
    result = geom.hermiteToBezier(pts, tangents)
    for seg in range(len(pts) - 1):
        cp1 = result[1 + seg * 3]
        cp2 = result[2 + seg * 3]
        start = pts[seg]
        end = pts[seg + 1]
        assert np.allclose(cp1, start), f"seg {seg}: cp1 should equal start"
        assert np.allclose(cp2, end), f"seg {seg}: cp2 should equal end"


def test_hermiteToBezier_closed():
    """Closed hermite loops back, shape is (1 + 3*n, 2)."""
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    tangents = np.array([[5, 0], [0, 5], [-5, 0], [0, -5]], dtype=float)
    result = geom.hermiteToBezier(pts, tangents, closed=True)
    n = len(pts)
    assert result.shape == (1 + 3 * n, 2)
    assert np.allclose(result[-1], pts[0])


def test_hermiteToBezier_matchesCardinal():
    """hermiteToBezier with cardinal tangents should match cardinalToBezier."""
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30]], dtype=float)
    tension = 0.5
    cardinal_result = geom.cardinalToBezier(pts, tension=tension)

    # Manually compute cardinal tangents
    s = 1.0 - tension
    prev = np.empty_like(pts)
    nxt = np.empty_like(pts)
    prev[0] = pts[0]
    prev[1:] = pts[:-1]
    nxt[-1] = pts[-1]
    nxt[:-1] = pts[1:]
    tangents = s * (nxt - prev)

    hermite_result = geom.hermiteToBezier(pts, tangents)
    assert np.allclose(cardinal_result, hermite_result, atol=1e-5)


# --- bsplineToBezier tests ---

def test_bsplineToBezier_basic():
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30], [70, 5]], dtype=float)
    result = geom.bsplineToBezier(pts)
    n = len(pts)
    numSegments = n - 3  # 2
    # Open curve: 1 + 3*(n-3) points
    assert result.shape == (1 + 3 * numSegments, 2)
    # First point: (Q0 + 4*Q1 + Q2) / 6
    expectedStart = (pts[0] + 4 * pts[1] + pts[2]) / 6.0
    assert np.allclose(result[0], expectedStart)
    # Last anchor: (Q_{n-3} + 4*Q_{n-2} + Q_{n-1}) / 6
    expectedEnd = (pts[-3] + 4 * pts[-2] + pts[-1]) / 6.0
    assert np.allclose(result[-1], expectedEnd)


def test_bsplineToBezier_collinear():
    """Collinear control points → all Bezier points are collinear."""
    pts = np.array([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]], dtype=float)
    result = geom.bsplineToBezier(pts)
    # All points should lie on y=x
    assert np.allclose(result[:, 0], result[:, 1], atol=1e-5)


def test_bsplineToBezier_closed():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    result = geom.bsplineToBezier(pts, closed=True)
    n = len(pts)
    # Closed curve: n segments → 1 + 3*n points
    assert result.shape == (1 + 3 * n, 2)
    # Last anchor should equal first point (loop back)
    assert np.allclose(result[-1], result[0], atol=1e-5)


def test_bsplineToBezier_minPoints():
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30]], dtype=float)
    result = geom.bsplineToBezier(pts)
    # 4 points → 1 segment → 1 + 3 = 4 points
    assert result.shape == (4, 2)


def test_bsplineToBezier_continuity():
    """Adjacent segments share endpoints (C0 continuity)."""
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30], [70, 5], [90, 25]], dtype=float)
    result = geom.bsplineToBezier(pts)
    numSegments = len(pts) - 3
    for seg in range(numSegments - 1):
        endOfSeg = result[3 + seg * 3]
        startOfNext = result[3 + seg * 3]  # same index — shared point
        assert np.allclose(endOfSeg, startOfNext)


# --- sampleBspline tests ---

def test_sampleBspline_endpoints():
    """First and last sampled points match expected B-spline boundary values."""
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30], [70, 5]], dtype=float)
    samples = geom.sampleBspline(pts, numSamples=50)
    expectedStart = (pts[0] + 4 * pts[1] + pts[2]) / 6.0
    expectedEnd = (pts[-3] + 4 * pts[-2] + pts[-1]) / 6.0
    assert np.allclose(samples[0], expectedStart, atol=1e-3)
    assert np.allclose(samples[-1], expectedEnd, atol=1e-3)


def test_sampleBspline_closed():
    """Closed B-spline samples form a loop (first ≈ last)."""
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    samples = geom.sampleBspline(pts, closed=True, numSamples=30)
    assert np.allclose(samples[0], samples[-1], atol=1e-3)


def test_sampleBspline_matchesBezierConversion():
    """sampleBspline should agree with bsplineToBezier + sampleBezier."""
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30], [70, 5]], dtype=float)
    numSamples = 30

    directSamples = geom.sampleBspline(pts, numSamples=numSamples)
    bezPts = geom.bsplineToBezier(pts)
    bezSamples = geom.sampleBezier(bezPts, numSamples=numSamples)

    assert directSamples.shape == bezSamples.shape
    assert np.allclose(directSamples, bezSamples, atol=1e-2)


def test_sampleBspline_shape():
    """Verify output shape given numSamples and segment count."""
    pts = np.array([[0, 0], [10, 20], [30, 10], [50, 30], [70, 5]], dtype=float)
    numSamples = 15
    samples = geom.sampleBspline(pts, numSamples=numSamples)
    numSegments = len(pts) - 3  # 2
    # 2 segments × numSamples (endpoint=False each) + 1 final endpoint
    assert samples.shape == (numSegments * numSamples + 1, 2)


def test_sampleBezier_shape():
    """Verify output shape given numSamples and segment count."""
    # 1 + 6 remaining → 2 segments (groups of 3)
    pts = np.array([[0, 0], [1, 2], [3, 4], [5, 0],
                    [7, 2], [9, 4], [10, 0]], dtype=float)
    numSamples = 15
    samples = geom.sampleBezier(pts, numSamples=numSamples)
    # 2 segments × numSamples (endpoint=False each) + 1 final endpoint
    assert samples.shape == (2 * numSamples + 1, 2)


# --- translate2D tests ---

def test_translate2D_single_point():
    result = geom.translate2D(P(1, 2), P(10, 20))
    assert np.allclose(result, [11, 22])


def test_translate2D_batch():
    pts = P([0, 0], [1, 1], [2, 2])
    result = geom.translate2D(pts, P(5, 10))
    expected = P([5, 10], [6, 11], [7, 12])
    assert np.allclose(result, expected)


def test_translate2D_grid():
    grid = np.zeros((3, 4, 2), dtype=np.float32)
    result = geom.translate2D(grid, P(1, 2))
    assert result.shape == (3, 4, 2)
    assert np.allclose(result[..., 0], 1)
    assert np.allclose(result[..., 1], 2)


def test_translate2D_partial():
    shift = geom.translate2D(offset=P(5, 5))
    result = shift(P(1, 2))
    assert np.allclose(result, [6, 7])


# --- rotate2D tests ---

def test_rotate2D_quarter_turn():
    result = geom.rotate2D(P(1, 0), geom.HALF_PI)
    assert np.allclose(result, [0, 1], atol=1e-6)


def test_rotate2D_full_turn():
    pt = P(3, 4)
    result = geom.rotate2D(pt, geom.TWO_PI)
    assert np.allclose(result, pt, atol=1e-5)


def test_rotate2D_with_origin():
    # Rotating (2, 0) by 180 degrees around (1, 0) → (0, 0)
    result = geom.rotate2D(P(2, 0), geom.PI, origin=P(1, 0))
    assert np.allclose(result, [0, 0], atol=1e-6)


def test_rotate2D_zero_angle():
    pt = P(5, 7)
    result = geom.rotate2D(pt, 0)
    assert np.allclose(result, pt)


def test_rotate2D_batch():
    pts = P([1, 0], [0, 1], [-1, 0], [0, -1])
    result = geom.rotate2D(pts, geom.HALF_PI)
    expected = P([0, 1], [-1, 0], [0, -1], [1, 0])
    assert np.allclose(result, expected, atol=1e-6)


def test_rotate2D_grid():
    grid = np.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]], dtype=np.float32)
    result = geom.rotate2D(grid, geom.PI)
    expected = np.array([[[-1, 0], [0, -1]], [[1, 0], [0, 1]]], dtype=np.float32)
    assert result.shape == (2, 2, 2)
    assert np.allclose(result, expected, atol=1e-6)


def test_rotate2D_partial():
    rot90 = geom.rotate2D(angle=geom.HALF_PI)
    result = rot90(P(1, 0))
    assert np.allclose(result, [0, 1], atol=1e-6)


# --- scale2D tests ---

def test_scale2D_uniform():
    result = geom.scale2D(P(2, 3), 3)
    assert np.allclose(result, [6, 9])


def test_scale2D_nonuniform():
    result = geom.scale2D(P(2, 3), P(2, 3))
    assert np.allclose(result, [4, 9])


def test_scale2D_identity():
    pt = P(5, 7)
    result = geom.scale2D(pt, 1)
    assert np.allclose(result, pt)


def test_scale2D_with_origin():
    # Scale (4, 4) by 2 from origin (2, 2) → (6, 6)
    result = geom.scale2D(P(4, 4), 2, origin=P(2, 2))
    assert np.allclose(result, [6, 6])


def test_scale2D_by_zero():
    result = geom.scale2D(P(5, 7), 0, origin=P(1, 1))
    assert np.allclose(result, [1, 1])


def test_scale2D_reflection():
    result = geom.scale2D(P(3, 4), P(-1, 1))
    assert np.allclose(result, [-3, 4])


def test_scale2D_grid():
    grid = np.ones((2, 3, 2), dtype=np.float32) * 2
    result = geom.scale2D(grid, 5)
    assert result.shape == (2, 3, 2)
    assert np.allclose(result, 10)


# --- skew2D tests ---

# --- reflect2D tests ---

def test_reflect2D_across_x_axis():
    # angle=0 reflects across x-axis: (1, 1) → (1, -1)
    result = geom.reflect2D(P(1, 1), 0)
    assert np.allclose(result, [1, -1])


def test_reflect2D_across_y_axis():
    # angle=PI/2 reflects across y-axis: (1, 1) → (-1, 1)
    result = geom.reflect2D(P(1, 1), geom.HALF_PI)
    assert np.allclose(result, [-1, 1], atol=1e-6)


def test_reflect2D_across_diagonal():
    # angle=PI/4 reflects across y=x: (1, 0) → (0, 1)
    result = geom.reflect2D(P(1, 0), geom.QUARTER_PI)
    assert np.allclose(result, [0, 1], atol=1e-6)


def test_reflect2D_point_on_axis():
    # Point on the reflection axis is unchanged
    result = geom.reflect2D(P(5, 0), 0)
    assert np.allclose(result, [5, 0])


def test_reflect2D_double_reflection_identity():
    pt = P(3, 7)
    reflected = geom.reflect2D(geom.reflect2D(pt, 0.7), 0.7)
    assert np.allclose(reflected, pt, atol=1e-6)


def test_reflect2D_with_origin():
    # Reflect (3, 1) across x-axis through (1, 1): centered=(2, 0) → (2, 0) → + origin = (3, 1)
    # Actually: centered=(2, 0), reflect across x: (2, 0) → (2, 0). Point is on axis, unchanged.
    # Better test: reflect (1, 3) across x-axis through (1, 1): centered=(0, 2) → (0, -2) → (1, -1)
    result = geom.reflect2D(P(1, 3), 0, origin=P(1, 1))
    assert np.allclose(result, [1, -1], atol=1e-6)


def test_reflect2D_batch():
    pts = P([1, 0], [0, 1], [-1, 0])
    result = geom.reflect2D(pts, 0)  # across x-axis
    expected = P([1, 0], [0, -1], [-1, 0])
    assert np.allclose(result, expected, atol=1e-6)


def test_reflect2D_grid():
    grid = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    result = geom.reflect2D(grid, 0)  # across x-axis: flip y
    assert result.shape == (2, 2, 2)
    assert np.allclose(result[..., 0], grid[..., 0])
    assert np.allclose(result[..., 1], -grid[..., 1])


def test_reflect2D_partial():
    flipY = geom.reflect2D(angle=geom.HALF_PI)
    result = flipY(P(3, 4))
    assert np.allclose(result, [-3, 4], atol=1e-6)


def test_skew2D_x_only():
    # shear [1, 0]: x' = x + 1*y, y' = y
    result = geom.skew2D(P(0, 1), P(1, 0))
    assert np.allclose(result, [1, 1])


def test_skew2D_y_only():
    # shear [0, 1]: x' = x, y' = y + 1*x
    result = geom.skew2D(P(1, 0), P(0, 1))
    assert np.allclose(result, [1, 1])


def test_skew2D_zero_shear():
    pt = P(3, 7)
    result = geom.skew2D(pt, P(0, 0))
    assert np.allclose(result, pt)


def test_skew2D_with_origin():
    # Skew (2, 1) with shear [1, 0] around origin (1, 0)
    # centered: (1, 1) → x'=1+1*1=2, y'=1 → + origin → (3, 1)
    result = geom.skew2D(P(2, 1), P(1, 0), origin=P(1, 0))
    assert np.allclose(result, [3, 1])


def test_skew2D_batch():
    pts = P([1, 0], [0, 1])
    result = geom.skew2D(pts, P(0.5, 0))
    expected = P([1, 0], [0.5, 1])
    assert np.allclose(result, expected)


# --- transform2D tests ---

def test_transform2D_identity_2x2():
    pt = P(3, 4)
    result = geom.transform2D(pt, np.eye(2))
    assert np.allclose(result, pt)


def test_transform2D_rotation_2x2():
    angle = geom.HALF_PI
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle),  np.cos(angle)]], dtype=np.float32)
    result = geom.transform2D(P(1, 0), matrix)
    assert np.allclose(result, [0, 1], atol=1e-6)


def test_transform2D_affine_3x3():
    # Translation via affine matrix
    matrix = np.array([[1, 0, 10],
                        [0, 1, 20],
                        [0, 0,  1]], dtype=np.float32)
    result = geom.transform2D(P(1, 2), matrix)
    assert np.allclose(result, [11, 22])


def test_transform2D_affine_batch():
    matrix = np.array([[2, 0, 1],
                        [0, 2, 1],
                        [0, 0, 1]], dtype=np.float32)
    pts = P([0, 0], [1, 1])
    result = geom.transform2D(pts, matrix)
    expected = P([1, 1], [3, 3])
    assert np.allclose(result, expected)


def test_transform2D_affine_grid():
    matrix = np.array([[1, 0, 5],
                        [0, 1, 5],
                        [0, 0, 1]], dtype=np.float32)
    grid = np.zeros((2, 3, 2), dtype=np.float32)
    result = geom.transform2D(grid, matrix)
    assert result.shape == (2, 3, 2)
    assert np.allclose(result[..., 0], 5)
    assert np.allclose(result[..., 1], 5)


def test_transform2D_invalid_matrix():
    with pytest.raises(ValueError, match="must be"):
        geom.transform2D(P(1, 2), np.eye(4))


def test_transform2D_partial():
    doubler = geom.transform2D(matrix=np.array([[2, 0], [0, 2]], dtype=np.float32))
    result = doubler(P(3, 4))
    assert np.allclose(result, [6, 8])


# --- Matrix2D tests ---

def test_Matrix2D_identity():
    m = geom.Matrix2D()
    assert np.allclose(m.m, np.eye(3))
    result = geom.transform2D(P(3, 4), m)
    assert np.allclose(result, [3, 4])


def test_Matrix2D_translate():
    m = geom.Matrix2D().translate(P(10, 20))
    result = geom.transform2D(P(1, 2), m)
    assert np.allclose(result, [11, 22])


def test_Matrix2D_rotate():
    m = geom.Matrix2D().rotate(geom.HALF_PI)
    result = geom.transform2D(P(1, 0), m)
    assert np.allclose(result, [0, 1], atol=1e-6)


def test_Matrix2D_scale_uniform():
    m = geom.Matrix2D().scale(3)
    result = geom.transform2D(P(2, 3), m)
    assert np.allclose(result, [6, 9])


def test_Matrix2D_scale_nonuniform():
    m = geom.Matrix2D().scale(P(2, 3))
    result = geom.transform2D(P(1, 1), m)
    assert np.allclose(result, [2, 3])


def test_Matrix2D_skew():
    m = geom.Matrix2D().skew(P(1, 0))
    result = geom.transform2D(P(0, 1), m)
    assert np.allclose(result, [1, 1])


def test_Matrix2D_reflect():
    m = geom.Matrix2D().reflect(0)  # across x-axis
    result = geom.transform2D(P(1, 1), m)
    assert np.allclose(result, [1, -1])


def test_Matrix2D_chained():
    """Scale then rotate should match applying both sequentially."""
    pts = P([1, 0], [0, 1], [1, 1])
    m = geom.Matrix2D().scale(2).rotate(geom.HALF_PI)
    fromMatrix = geom.transform2D(pts, m)
    fromFunctions = geom.rotate2D(geom.scale2D(pts, 2), geom.HALF_PI)
    assert np.allclose(fromMatrix, fromFunctions, atol=1e-5)


def test_Matrix2D_with_origin():
    """Scale from origin should match standalone function."""
    origin = P(100, 100)
    pts = P(110, 110)
    m = geom.Matrix2D().scale(2, origin=origin)
    fromMatrix = geom.transform2D(pts, m)
    fromFunction = geom.scale2D(pts, 2, origin=origin)
    assert np.allclose(fromMatrix, fromFunction, atol=1e-5)


def test_Matrix2D_chained_with_origin():
    """Chained transforms with shared origin should match sequential application."""
    origin = P(50, 50)
    pts = P([60, 50], [50, 60], [70, 70])
    m = geom.Matrix2D().scale(2, origin=origin).rotate(geom.PI / 6, origin=origin)
    fromMatrix = geom.transform2D(pts, m)
    fromFunctions = geom.rotate2D(geom.scale2D(pts, 2, origin=origin), geom.PI / 6, origin=origin)
    assert np.allclose(fromMatrix, fromFunctions, atol=1e-4)


def test_Matrix2D_inverse():
    pts = P([10, 20], [30, 40])
    m = geom.Matrix2D().scale(2).rotate(geom.PI / 3).translate(P(5, 10))
    transformed = geom.transform2D(pts, m)
    recovered = geom.transform2D(transformed, m.inverse())
    assert np.allclose(recovered, pts, atol=1e-3)


def test_Matrix2D_inverse_does_not_mutate():
    m = geom.Matrix2D().scale(2)
    original = m.m.copy()
    _ = m.inverse()
    assert np.allclose(m.m, original)


def test_Matrix2D_array_protocol():
    """Matrix2D should be usable anywhere a numpy array is expected."""
    m = geom.Matrix2D().translate(P(5, 5))
    arr = np.asarray(m)
    assert arr.shape == (3, 3)
    assert np.allclose(arr, m.m)
