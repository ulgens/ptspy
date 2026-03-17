"""Visual regression tests based on QuickStart and Stamps notebook examples."""

import numpy as np
from ptspy import (
    frame,
    frames,
    P,
    steps,
    expand,
    colorCube,
    colorPicker,
    colorInterpolation,
    colorScheme,
    gradient,
    randomizer,
    regularPolygon,
    populate,
    pipeline,
    F,
    TBD,
    sampleBezier,
    sampleBspline,
    sampleCardinal,
    hermiteToBezier,
)


# =============================================================================
# QuickStart examples
# =============================================================================


def test_quickstart_dot(visualTest):
    """Simple centered dot."""

    @frame(size=(100, 100))
    def draw(form):
        form.fillOnly("000").point(form.size / 2, 10)

    visualTest(draw, "quickstart_dot")


def test_quickstart_dot_dark(visualTest):
    """Dot on dark background."""

    @frame(bg="000", size=(100, 100))
    def draw(form):
        form.fillOnly("f0f3f7").point(form.size / 2, 30)

    visualTest(draw, "quickstart_dot_dark")


def test_quickstart_color_cube_slice(visualTest):
    """Color cube middle slice."""

    @frame(size=(100, 100))
    def draw(form):
        cube = colorCube(25)
        slice = cube[:, 14, :]  # middle chroma slice (original uses index 14)
        form.image(slice, P([0, 0], form.size))

    visualTest(draw, "quickstart_color_cube_slice")


def test_quickstart_color_picker(visualTest):
    """Nested rectangles with picked colors."""

    @frame(size=(100, 100))
    def draw(form):
        cube = colorCube(30)
        form.fillOnly(colorPicker(cube, 0.20, 0.9, 0.7)).rect(P([[0, 0], form.size]))
        form.fillOnly(colorPicker(cube, 0.18, 0.9, 0.8)).rect(P([form.size * 0.25, form.size * 0.75]))

    visualTest(draw, "quickstart_color_picker")


def test_quickstart_color_scheme(visualTest):
    """Color scheme palette circles."""

    @frame(bg="#00000000", size=(150, 100))
    def draw(form):
        cube = colorCube(steps=24)
        variant = randomizer(8)
        palette = colorScheme(cube, 9, 6, 0.5, variant)
        form.colorMap(palette[:, :, 0], P([0, 0], form.size), style="circle")

    visualTest(draw, "quickstart_color_scheme")


def test_quickstart_color_interpolation(visualTest):
    """Batched color interpolation paths."""

    @frame(bg="#00000000", size=(100, 100))
    def draw(form):
        cc = colorCube(steps=100)
        # Indices are (H, C, L) - hue, chroma, lightness
        startColors = [P(80, 28, 99), P(50, 90, 50)]
        endColors = [P(10, 99, 70), P(80, 30, 10)]
        indices = colorInterpolation(cc, start=startColors, end=endColors, steps=8, asPath=True)
        form.colorMap(indices, P([0, 0], form.size))

    visualTest(draw, "quickstart_color_interpolation")


def test_quickstart_gradient(visualTest):
    """Sweep gradient with stops."""

    @frame(size=(100, 100))
    def draw(form):
        colors = [0xFF2099FF, 0xFFFF66FF, 0x20CCE0FF]
        shader = gradient(P([[50, 50]]), colors, "sweep", stops=[0.2, 0.5, 0.8])
        form.fill(shader).rect(P([[0, 0], form.size]))

    visualTest(draw, "quickstart_gradient")


def test_quickstart_steps_shaping(visualTest):
    """Points with cubicOut shaping."""

    # Original: 300x300, steps([40, 150], [260, 150], count=20)
    # Scaled: 100x100, steps([13, 50], [87, 50], count=20)
    @frame(bg="000", size=(100, 100))
    def draw(form):
        points = steps([13, 50], [form.width - 13, 50], count=20, shaping="cubicOut")
        form.fillOnly("f0f5fa").points(points, 2)

    visualTest(draw, "quickstart_steps_shaping")


def test_quickstart_polygon_recursive(visualTest):
    """Recursive polygon pattern."""

    # Original: 300x300, radius=40, center=[150,150], points radius=3
    # Scaled: 100x100, radius=13.3, center=[50,50], points radius=1
    @frame(bg="000", size=(100, 100))
    def draw(form):
        makePolygon = regularPolygon(TBD, sides=4, radius=13, closed=True)

        def getNext(form, pts):
            results = []
            for pt in pts:
                next = makePolygon(pt)
                form.strokeOnly("f0f5fa", 1).lines(next)
                results.append(next)
            return (form, np.vstack(results))

        def makeDot(form, pts):
            form.fillOnly("fff").points(pts, 1)
            return (form, pts)

        makeAll = F(getNext) >> makeDot >> getNext

        poly = makePolygon([50, 50])
        form.strokeOnly("f0f5fa", 1).lines(poly)
        makeAll(form, poly)

    visualTest(draw, "quickstart_polygon_recursive")


def test_quickstart_lines_animation(visualTest):
    """Animated lines with dynamic shaping."""

    # Original: 300x300, 60 frames, 20 steps, width 10-4
    # Scaled: 100x100, 5 frames, 7 steps, width 3.3-1.3
    @frames(bg="000", size=(100, 100), numFrames=5)
    def draw(form, index):
        # Simulate frames 0, 15, 30, 45, 60 from original
        originalIndex = index * 15
        top = steps(0, form.width, 7, shaping=lambda t, a, b: a + (b - a) * (t * t * originalIndex / 30.0))
        top = expand(top[:, np.newaxis], 0)
        lines = np.stack([top, top + [0, form.height]], axis=1)
        form.strokeOnly("f0f5fa", 3.3 - originalIndex / 30.0).lines(lines)

    visualTest(draw, "quickstart_lines_animation")


def test_quickstart_growing_dot(visualTest):
    """Growing dot animation."""

    # Original: default size (512x512), 60 frames, radius = 10 + index
    # Scaled: 100x100, 5 frames, radius = 2 + index*2
    @frames(size=(100, 100), numFrames=5)
    def draw(form, index, radius):
        form.fillOnly("000").point(form.size / 2, radius + index * 2)

    visualTest(lambda: draw(radius=2), "quickstart_growing_dot")


def test_quickstart_points_partial_animation(visualTest):
    """Animated points with partial application."""
    # Original: 300x300, 60 frames, steps from [40,150] to dynamic end
    # Scaled: 100x100, 5 frames
    pointsFn = steps([13, 50], shaping="cubicOut")

    @frames(bg="000", size=(100, 100), numFrames=5)
    def draw(form, index):
        # Simulate frames 0, 15, 30, 45, 60 from original
        originalIndex = index * 15
        count = int(originalIndex / 2.0)
        end = [form.width - 13, 13 + originalIndex / 60.0 * (form.height - 26)]
        if count > 0:
            form.fillOnly("f0f5fa").points(pointsFn(end, count), 2)

    visualTest(draw, "quickstart_points_partial_animation")


def test_quickstart_color_disc_animation(visualTest):
    """Color disc animation with partial application."""
    # Original: 300x300, 60 frames, two nested circles with color palette
    # Scaled: 100x100, 5 frames
    from ptspy import partial

    @partial(placeholder=TBD)
    def colorDisc(form, color, box):
        form.fillOnly(color).circle(box)

    cc = colorCube(steps=100)
    # Indices are (H, C, L) - hue, chroma, lightness
    palette = colorInterpolation(cc, P(0, 28, 99), P(50, 70, 50), steps=60, asPath=True)

    first = colorDisc(box=P([3, 3], [97, 97]))
    second = colorDisc(form=TBD, color=TBD, box=P([27, 27], [73, 73]))

    @frames(bg="f0f5fa", size=(100, 100), numFrames=5)
    def draw(form, index):
        # Simulate frames 0, 15, 30, 45, 59 from original
        originalIndex = min(index * 15, 59)
        first(form, palette[originalIndex])
        second(form=form, color=palette[59 - originalIndex])

    visualTest(draw, "quickstart_color_disc_animation")


def test_quickstart_polygon_progress_animation(visualTest):
    """Polygon progress animation with pipeline."""
    # Original: 300x300, 20 frames, radius=30, center=[150,150]
    # Scaled: 100x100, 5 frames, radius=10, center=[50,50]
    makePolygon = regularPolygon(TBD, TBD, radius=10, closed=True)

    def getNext(pts, sides):
        if len(pts.shape) > 2:
            pts = pts.reshape(-1, 2)
        return (P([makePolygon(center, sides) for center in pts]), sides)

    def makeDot(pts, sides):
        return (pts.reshape(-1, 2), sides)

    makeAll = pipeline(getNext, getNext, getNext, makeDot, lazy=True)
    progress = list(makeAll(P(50, 50), 6))

    @frames(bg="000", size=(100, 100), numFrames=5)
    def draw(form, index):
        # Sample 5 frames from the progress
        i = int(index * len(progress) / 5) % len(progress)
        if len(progress[i][0].shape) > 2:
            form.strokeOnly("f0f5fa").lines(progress[i][0].reshape(-1, 2))
        else:
            form.fillOnly("f0f5fa").points(progress[i][0], 2)

    visualTest(draw, "quickstart_polygon_progress_animation")


# =============================================================================
# Stamps examples (scaled down from 600px to 150px)
# =============================================================================


def _perforate(size, hole=2.5):
    """Create perforation pattern scaled for smaller stamps."""
    import skia

    steps_arr = populate(21, increment=7.5)  # scaled from 30 to 7.5 (150/600 = 0.25)
    edges = skia.Path()
    for s in steps_arr:
        edges.addCircle(s, 0, hole)
        edges.addCircle(0, s, hole)
        edges.addCircle(size[0], s, hole)
        edges.addCircle(s, size[1], hole)
    return edges


def test_stamp_0_circle(visualTest):
    """Stamp 0: Circle with perforations."""
    padding = 6.25  # scaled from 25

    @frame(bg="fff", size=(150, 150))
    def draw(form):
        bounds = P([padding, padding], [150 - padding, 150 - padding])
        form.fillOnly("fff").rect(bounds)
        form.strokeOnly("000", 0.5).circle(bounds)
        form.erase(_perforate(form.size))

    visualTest(draw, "stamp_0_circle")


def test_stamp_1_points(visualTest):
    """Stamp 1: Points with different shapings."""
    padding = 6.25  # scaled from 25
    boundSize = 150 - padding * 2

    @frame(bg="fff", size=(150, 150))
    def draw(form):
        bounds = P([padding, padding], [150 - padding, 150 - padding])

        # Original: 100, 200, 300, 400 in 600px -> 25, 50, 75, 100 in 150px
        # Original radius: 20->5, 2->0.5, 60->15
        one = steps([25, padding * 2.5, 5], [25, boundSize, 0.5], count=20, shaping="linear")
        two = steps([50, padding * 2, 0.5], [50, boundSize - padding * 2, 15], count=10, shaping="quadraticOut")
        three = steps([75, padding * 2.5, 5], [75, boundSize, 0.5], count=20, shaping="cubicInOut")
        four = steps(
            [100, padding * 2, 0.5],
            [100, boundSize - padding * 2, 15],
            count=10,
            shaping=lambda t, a, b: a + (b - a) * (t * t * t),
        )

        form.fillOnly("000").rect(bounds)
        form.fillOnly("fff").points(one).points(two).points(three).points(four)
        form.erase(_perforate(form.size))

    visualTest(draw, "stamp_1_points")


def test_stamp_2_lines(visualTest):
    """Stamp 2: Lines with cubicIn shaping."""
    # Original: steps(25, 575, 30), strokeOnly("fff", 3) on 600x600
    # Scaled: steps(6.25, 143.75, 8), strokeOnly("fff", 0.75) on 150x150
    padding = 6.25

    @frame(bg="fff", size=(150, 150))
    def draw(form):
        bounds = P([padding, padding], [150 - padding, 150 - padding])

        top = steps(padding, 150 - padding, 8, shaping="cubicIn")
        top = expand(top[:, np.newaxis], 0)
        lines = np.stack([top + [0, bounds[0][1]], top + [padding * 2, bounds[1][1]]], axis=1)

        form.fillOnly("000").rect(bounds)
        form.strokeOnly("fff", 0.75).lines(lines)
        form.erase(_perforate(form.size))

    visualTest(draw, "stamp_2_lines")


def test_stamp_3_color(visualTest):
    """Stamp 3: Color cube interpolation."""
    padding = 6.25

    @frame(bg="ffffff", size=(150, 150))
    def draw(form):
        bounds = P([padding, padding], [150 - padding, 150 - padding])

        cc = colorCube(steps=100)
        # Indices are (H, C, L) - hue, chroma, lightness
        start = P(0, 99, 90)
        end = P(30, 19, 60)

        indices = colorInterpolation(cc, start, end, steps=30, asPath=False)
        first = np.squeeze(indices, axis=0)
        form.image(first[25, :, :], bounds)

        indices = colorInterpolation(cc, start, end, steps=12, asPath=True)
        form.colorMap(indices, [bounds[0] + [12.5, 0.25], bounds[1] - [31.25, 0.25]])

        form.erase(_perforate(form.size))

    visualTest(draw, "stamp_3_color")


# =============================================================================
# Form.polygons tests
# =============================================================================


def test_polygons_variable_length(visualTest):
    """Polygons with different vertex counts (triangle, square, pentagon, hexagon)."""

    @frame(bg="fff", size=(100, 100))
    def draw(form):
        poly = regularPolygon(center=TBD, radius=20, sides=TBD, closed=True)
        centers = P([25, 25], [75, 25], [25, 75], [75, 75])
        shapes = [poly(center=c, sides=s) for c, s in zip(centers, [3, 4, 5, 6])]
        form.strokeOnly("000", 2).polygons(shapes)

    visualTest(draw, "polygons_variable_length")


# =============================================================================
# Form transformation tests
# =============================================================================


def test_transforms_all(visualTest):
    """Canvas transformations: translate, rotate, scale, skew, concat, resetMatrix."""
    import skia

    @frame(bg="f5f5f5", size=(200, 200))
    def draw(form):
        # 1. Translate - red rect moved from origin
        form.save()
        form.translate(20, 20)
        form.fillOnly("e74c3c").rect([[0, 0], [30, 20]])
        form.restore()

        # 2. Rotate around pivot - blue rect rotated 30 degrees
        form.save()
        form.rotate(30, pivot=[100, 50])
        form.fillOnly("3498db").rect([[85, 40], [115, 60]])
        form.restore()

        # 3. Scale uniform - green rect at 1.5x
        form.save()
        form.translate(160, 20).scale(1.5)
        form.fillOnly("2ecc71").rect([[0, 0], [20, 20]])
        form.restore()

        # 4. Skew - purple rect with horizontal shear
        form.save()
        form.translate(40, 90).skew(0.3, 0)
        form.fillOnly("9b59b6").rect([[0, 0], [40, 25]])
        form.restore()

        # 5. Chained transforms - orange rect with translate + rotate + scale
        form.save()
        form.translate(125, 110).rotate(45).scale(0.8)
        form.fillOnly("f39c12").rect([[-15, -15], [15, 15]])
        form.restore()

        # 6. concat with custom matrix - teal rect with matrix rotation
        form.save()
        m = skia.Matrix()
        m.setRotate(15, 40, 165)
        form.concat(m)
        form.fillOnly("1abc9c").rect([[25, 150], [55, 180]])
        form.restore()

        # 7. resetMatrix - dark rect proves transforms were cleared
        form.save()
        form.translate(500, 500).rotate(90).scale(3)
        form.resetMatrix()
        form.fillOnly("34495e").rect([[140, 150], [180, 180]])
        form.restore()

    visualTest(draw, "transforms_all")


# =============================================================================
# Bezier and Cardinal curve tests
# =============================================================================


def test_bezier_open(visualTest):
    """Open cubic Bezier curve with two segments."""

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        pts = P([10, 80], [40, 10], [60, 10], [80, 50], [100, 90], [130, 90], [140, 20])
        form.strokeOnly("000", 2).bezier(pts)
        # Draw control points for reference
        form.fillOnly("e74c3c").points(pts, 2)

    visualTest(draw, "bezier_open")


def test_bezier_closed(visualTest):
    """Closed cubic Bezier curve forming a loop."""

    @frame(bg="fff", size=(100, 100))
    def draw(form):
        pts = P([50, 10], [90, 30], [80, 50], [50, 50], [20, 50], [10, 30], [50, 10])
        form.fillOnly("3498db").stroke("000", 2).bezier(pts, closed=True)

    visualTest(draw, "bezier_closed")


def test_cardinal_open(visualTest):
    """Open cardinal spline through control points (tension=0.25)."""

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        pts = P([10, 50], [40, 10], [75, 90], [110, 30], [140, 60])
        form.strokeOnly("000", 2).cardinal(pts, tension=0.25)
        # Draw control points for reference
        form.fillOnly("e74c3c").points(pts, 3)

    visualTest(draw, "cardinal_open")


def test_cardinal_closed(visualTest):
    """Closed cardinal spline forming a smooth loop."""

    @frame(bg="fff", size=(100, 100))
    def draw(form):
        pts = P([50, 10], [85, 30], [85, 70], [50, 90], [15, 70], [15, 30])
        form.fillOnly("2ecc7188").stroke("000", 2).cardinal(pts, tension=0.25, closed=True)
        form.fillOnly("e74c3c").points(pts, 2)

    visualTest(draw, "cardinal_closed")


def test_cardinal_tension_comparison(visualTest):
    """Cardinal splines at different tensions overlaid for comparison."""

    @frame(bg="fff", size=(200, 120))
    def draw(form):
        pts = P([15, 60], [55, 15], [100, 100], [145, 30], [185, 70])
        # tension=0.25 (loose)
        form.strokeOnly("3498db", 1.5).cardinal(pts, tension=0.25)
        # tension=0.5 (Catmull-Rom)
        form.strokeOnly("000", 2).cardinal(pts, tension=0.5)
        # tension=0.85 (tight)
        form.strokeOnly("e74c3c", 1.5).cardinal(pts, tension=0.85)
        # Control points
        form.fillOnly("333").points(pts, 2.5)

    visualTest(draw, "cardinal_tension_comparison")


def test_sampleBezier_visual(visualTest):
    """Bezier curve with 3 sampled sub-points per segment shown as dots."""

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        pts = P([10, 80], [40, 10], [60, 10], [80, 50], [100, 90], [130, 90], [140, 20])
        # Draw curve
        form.strokeOnly("000", 2).bezier(pts)
        # Sample 3 extra sub-points per segment (4 = anchor + 3 interior)
        samples = sampleBezier(pts, numSamples=4)
        form.fillOnly("3498db").points(samples, 3)
        # Draw original control points on top
        form.fillOnly("e74c3c").points(pts, 2.5)

    visualTest(draw, "sample_bezier")


def test_sampleCardinal_visual(visualTest):
    """Cardinal spline with 3 sampled sub-points per segment shown as dots."""

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        pts = P([10, 50], [40, 10], [75, 90], [110, 30], [140, 60])
        # Draw curve
        form.strokeOnly("000", 2).cardinal(pts, tension=0.25)
        # Sample 3 extra sub-points per segment (4 = anchor + 3 interior)
        samples = sampleCardinal(pts, tension=0.25, numSamples=4)
        form.fillOnly("3498db").points(samples, 3)
        # Draw original control points on top
        form.fillOnly("e74c3c").points(pts, 3)

    visualTest(draw, "sample_cardinal")


# =============================================================================
# Hermite spline tests
# =============================================================================


def test_hermite_open(visualTest):
    """Open hermite spline with explicit tangents."""

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        pts = P([10, 50], [40, 10], [75, 90], [110, 30], [140, 60])
        tangents = P([30, -40], [40, 0], [0, 40], [40, 0], [-30, -40])
        form.strokeOnly("000", 2).hermite(pts, tangents)
        form.fillOnly("e74c3c").points(pts, 3)

    visualTest(draw, "hermite_open")


def test_hermite_closed(visualTest):
    """Closed hermite spline forming a smooth loop."""

    @frame(bg="fff", size=(100, 100))
    def draw(form):
        pts = P([50, 10], [85, 30], [85, 70], [50, 90], [15, 70], [15, 30])
        tangents = P([30, 10], [0, 30], [-30, 10], [-30, -10], [0, -30], [30, -10])
        form.fillOnly("2ecc7188").stroke("000", 2).hermite(pts, tangents, closed=True)
        form.fillOnly("e74c3c").points(pts, 2)

    visualTest(draw, "hermite_closed")


def test_sampleHermite_visual(visualTest):
    """sampleBezier(hermiteToBezier(...)) with 3 sub-points per segment."""

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        pts = P([10, 50], [40, 10], [75, 90], [110, 30], [140, 60])
        tangents = P([30, -40], [40, 0], [0, 40], [40, 0], [-30, -40])
        bezPts = hermiteToBezier(pts, tangents)
        form.strokeOnly("000", 2).bezier(bezPts)
        samples = sampleBezier(bezPts, numSamples=4)
        form.fillOnly("3498db").points(samples, 3)
        form.fillOnly("e74c3c").points(pts, 3)

    visualTest(draw, "sample_hermite")


# =============================================================================
# B-spline tests
# =============================================================================


def test_bspline_open(visualTest):
    """Open uniform cubic B-spline (approximating — curve does not pass through points)."""

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        pts = P([10, 50], [30, 10], [60, 90], [100, 10], [130, 50], [140, 80])
        form.strokeOnly("000", 2).bspline(pts)
        # Draw control points for reference
        form.fillOnly("e74c3c").points(pts, 3)

    visualTest(draw, "bspline_open")


def test_bspline_closed(visualTest):
    """Closed B-spline forming a smooth loop."""

    @frame(bg="fff", size=(100, 100))
    def draw(form):
        pts = P([50, 10], [85, 30], [85, 70], [50, 90], [15, 70], [15, 30])
        form.fillOnly("2ecc7188").stroke("000", 2).bspline(pts, closed=True)
        form.fillOnly("e74c3c").points(pts, 2)

    visualTest(draw, "bspline_closed")


def test_sampleBspline_visual(visualTest):
    """sampleBspline with 3 sub-points per segment shown as dots."""

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        pts = P([10, 50], [30, 10], [60, 90], [100, 10], [130, 50], [140, 80])
        form.strokeOnly("000", 2).bspline(pts)
        samples = sampleBspline(pts, numSamples=4)
        form.fillOnly("3498db").points(samples, 3)
        form.fillOnly("e74c3c").points(pts, 3)

    visualTest(draw, "sample_bspline")


# =============================================================================
# Shader stroke & partial clear tests
# =============================================================================


def test_stroke_shader(visualTest):
    """strokeOnly with a gradient shader draws stroked shapes with shader color."""
    import skia

    @frame(bg="fff", size=(150, 100))
    def draw(form):
        shader = skia.GradientShader.MakeLinear(
            points=[(0, 0), (150, 100)],
            colors=[0xFF3498DB, 0xFFE74C3C],
        )
        # Stroked circle with gradient shader
        form.strokeOnly(shader, width=4).circle(P([30, 50], [70, 90]))
        # Stroked rect with gradient shader
        form.strokeOnly(shader, width=3).rect(P([80, 10], [140, 60]))
        # Stroked line with gradient shader
        form.strokeOnly(shader, width=3).line(P([80, 75], [140, 90]))

    visualTest(draw, "stroke_shader")


def test_clear_partial_alpha(visualTest):
    """clear() with alpha < 1.0 fades existing content instead of replacing."""

    @frame(bg="000", size=(100, 100))
    def draw(form):
        # Draw a white filled rect on the left half
        form.fillOnly("fff").rect(P([0, 0], [50, 100]))
        # Draw a red filled rect on the right half
        form.fillOnly("e74c3c").rect(P([50, 0], [100, 100]))
        # Partial clear with black at 50% — fades both toward black
        form.clear("000", alpha=0.5)
        # Draw a bright green dot on top to show new content is unaffected
        form.fillOnly("2ecc71").point([50, 50], radius=15)

    visualTest(draw, "clear_partial_alpha")


# =============================================================================
# Noise tests
# =============================================================================


def test_poisson_disc(visualTest):
    """Poisson-disc sampling rendered as scattered points."""
    from ptspy.more import poissonDisc

    @frame(bg="fff", size=(200, 200))
    def draw(form):
        pts = poissonDisc([200, 200], 10, seed=42)
        form.fillOnly("2255cc").points(pts, 2)

    visualTest(draw, "poisson_disc")


def test_noise_skia_fractal(visualTest):
    """Skia fractal Perlin noise shader fills a rect."""
    from ptspy import perlinNoise

    @frame(bg="000", size=(100, 100))
    def draw(form):
        shader = perlinNoise(0.05, 0.05, 4, 42.0, type="fractal")
        form.fill(shader).rect(P([0, 0], form.size))

    visualTest(draw, "noise_skia_fractal")


def test_noise_skia_turbulence(visualTest):
    """Skia turbulence noise shader fills a rect."""
    from ptspy import perlinNoise

    @frame(bg="000", size=(100, 100))
    def draw(form):
        shader = perlinNoise(0.05, 0.05, 4, 42.0, type="turbulence")
        form.fill(shader).rect(P([0, 0], form.size))

    visualTest(draw, "noise_skia_turbulence")


def test_noise_custom_perlin2d(visualTest):
    """Custom Perlin2D noise rendered as a grayscale image."""
    from ptspy.more import createPerlinNoise2D
    from ptspy.core.util import arrayToImage

    @frame(bg="000", size=(100, 100))
    def draw(form):
        noise = createPerlinNoise2D((100, 100), seed=42)
        values = noise(freq=3.0)
        pixels = arrayToImage(values, min=-1.0, max=1.0)
        form.image(pixels, P([0, 0], form.size))

    visualTest(draw, "noise_custom_perlin2d")


def test_noise_custom_fbm(visualTest):
    """Custom Perlin2D fBM (multi-octave) rendered as a grayscale image."""
    from ptspy.more import createPerlinNoise2D
    from ptspy.core.util import arrayToImage

    @frame(bg="000", size=(100, 100))
    def draw(form):
        noise = createPerlinNoise2D((100, 100), seed=7)
        values = noise(freq=2.0, octaves=4, lacunarity=2.0, gain=0.5)
        pixels = arrayToImage(values, min=-1.0, max=1.0)
        form.image(pixels, P([0, 0], form.size))

    visualTest(draw, "noise_custom_fbm")


def test_noise_custom_domain(visualTest):
    """Custom Perlin2D with non-default domain range."""
    from ptspy.more.noise import getPerlin2DGrid, createPerlinNoise2D
    from ptspy.core.util import arrayToImage

    @frame(bg="000", size=(100, 100))
    def draw(form):
        noise = createPerlinNoise2D((100, 100), seed=42)
        # Replace the grid with a wider domain
        noise.grid = getPerlin2DGrid(100, 100, domain=(0.0, 20.0))
        values = noise(freq=2.0)
        pixels = arrayToImage(values, min=-1.0, max=1.0)
        form.image(pixels, P([0, 0], form.size))

    visualTest(draw, "noise_custom_domain")
