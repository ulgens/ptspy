"""Visual regression tests for 2D transforms."""

import numpy as np
from ptspy import (
    frame,
    P,
    Matrix2D,
    translate2D,
    rotate2D,
    scale2D,
    skew2D,
    reflect2D,
    transform2D,
    PI,
    HALF_PI,
    QUARTER_PI,
)

# Base L-shape at origin — asymmetric so orientation is visually obvious
_L = P([0, 0], [0, 30], [10, 30], [10, 10], [25, 10], [25, 0])


def test_translate2D_visual(visualTest):
    """Green L at top-left, red L shifted to bottom-right."""

    @frame(bg="f0f0f0", size=(200, 200))
    def draw(form):
        shape = translate2D(_L, P(20, 20))
        moved = translate2D(shape, P(100, 80))
        form.fillOnly("2ecc71").polygon(shape)
        form.fillOnly("e74c3c").polygon(moved)

    visualTest(draw, "translate2D")


def test_rotate2D_visual(visualTest):
    """Green L at original position, red L rotated 45 deg around center (dot)."""

    @frame(bg="f0f0f0", size=(200, 200))
    def draw(form):
        origin = P(100, 100)
        shape = translate2D(_L, P(120, 70))
        rotated = rotate2D(shape, QUARTER_PI, origin=origin)
        form.fillOnly("2ecc71").polygon(shape)
        form.fillOnly("e74c3c").polygon(rotated)
        form.fillOnly("333").point(origin, 3)

    visualTest(draw, "rotate2D")


def test_scale2D_visual(visualTest):
    """Green L at original size, red L scaled 1.8x from center (dot)."""

    @frame(bg="f0f0f0", size=(200, 200))
    def draw(form):
        origin = P(100, 100)
        shape = translate2D(_L, P(88, 85))
        scaled = scale2D(shape, 1.8, origin=origin)
        form.fillOnly("2ecc71").polygon(shape)
        form.fillOnly("e74c3c").polygon(scaled)
        form.fillOnly("333").point(origin, 3)

    visualTest(draw, "scale2D")


def test_skew2D_visual(visualTest):
    """Green L original, red L sheared from center (dot)."""

    @frame(bg="f0f0f0", size=(200, 200))
    def draw(form):
        origin = P(100, 100)
        shape = translate2D(_L, P(80, 75))
        skewed = skew2D(shape, P(0.4, 0.15), origin=origin)
        form.fillOnly("2ecc71").polygon(shape)
        form.fillOnly("e74c3c").polygon(skewed)
        form.fillOnly("333").point(origin, 3)

    visualTest(draw, "skew2D")


def test_reflect2D_visual(visualTest):
    """Green L on left, red L mirrored across vertical axis (gray line) through center."""

    @frame(bg="f0f0f0", size=(200, 200))
    def draw(form):
        origin = P(100, 100)
        shape = translate2D(_L, P(30, 75))
        reflected = reflect2D(shape, HALF_PI, origin=origin)
        form.fillOnly("2ecc71").polygon(shape)
        form.fillOnly("e74c3c").polygon(reflected)
        form.strokeOnly("999", 1).line(P([100, 10], [100, 190]))
        form.fillOnly("333").point(origin, 3)

    visualTest(draw, "reflect2D")


def test_transform2D_visual(visualTest):
    """Green L original, red L after combined rotation+scale via 3x3 affine matrix."""

    @frame(bg="f0f0f0", size=(200, 200))
    def draw(form):
        shape = translate2D(_L, P(80, 75))
        # Affine matrix: rotate 30 deg + scale 1.5x, anchored at (100, 100)
        angle = PI / 6
        s, c = float(np.sin(angle)), float(np.cos(angle))
        sx = 1.5
        ox, oy = 100.0, 100.0
        matrix = np.array(
            [
                [sx * c, -sx * s, ox - sx * (c * ox - s * oy)],
                [sx * s, sx * c, oy - sx * (s * ox + c * oy)],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        transformed = transform2D(shape, matrix)
        form.fillOnly("2ecc71").polygon(shape)
        form.fillOnly("e74c3c").polygon(transformed)
        form.fillOnly("333").point(P(100, 100), 3)

    visualTest(draw, "transform2D")


def test_Matrix2D_visual(visualTest):
    """Green L original, red L after chained scale+rotate+translate via Matrix2D,
    blue L recovered via inverse (overlaps green)."""

    @frame(bg="f0f0f0", size=(200, 200))
    def draw(form):
        origin = P(100, 100)
        shape = translate2D(_L, P(85, 80))
        m = Matrix2D().scale(1.5, origin=origin).rotate(PI / 5, origin=origin).translate(P(10, -15))
        transformed = transform2D(shape, m)
        recovered = transform2D(transformed, m.inverse())
        form.fillOnly("2ecc71").polygon(shape)
        form.fillOnly("e74c3c").polygon(transformed)
        form.strokeOnly("3498db", 2).polygon(recovered)
        form.fillOnly("333").point(origin, 3)

    visualTest(draw, "Matrix2D")
