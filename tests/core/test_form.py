import numpy as np
import pytest
import skia

from ptspy.core.form import Form, frame, frames, _coerceSize


def test_coerce_size_validation():
    with pytest.raises(ValueError):
        _coerceSize((1,))
    with pytest.raises(ValueError):
        _coerceSize((0, 1))
    assert _coerceSize((2, 3)) == (2, 3)


def test_frames_rejects_non_positive_numFrames():
    with pytest.raises(ValueError):
        frames(numFrames=0)
    with pytest.raises(ValueError):
        frames(numFrames=-1)


def test_rect_circle_polygon_raise_on_wrong_ndim():
    form = Form(width=10, height=10)
    form.fill("ff0000")
    # 1D array should raise ValueError
    with pytest.raises(ValueError):
        form.rect(np.array([1, 2, 3, 4]))
    with pytest.raises(ValueError):
        form.circle(np.array([1, 2, 3, 4]))
    with pytest.raises(ValueError):
        form.polygon(np.array([1, 2, 3, 4]))


def test_clip_rejects_invalid_mode():
    form = Form(width=10, height=10)
    path = skia.Path()
    path.addCircle(5, 5, 3)
    with pytest.raises(ValueError, match="mode"):
        form.clip(path, mode="invalid")


def test_perlin_noise_rejects_invalid_type():
    from ptspy.core.form import perlinNoise

    with pytest.raises(ValueError, match="type"):
        perlinNoise(0.1, 0.1, 4, 42.0, type="invalid")


def test_frame_decorator_returns_image_and_wraps_errors():
    @frame(bg=0xFFFFFFFF, size=(4, 4))
    def draw(form):
        form.stroke(0xFF0000).point([2, 2], radius=1.0)

    img = draw()
    assert isinstance(img, skia.Image)
    assert img.width() == 4 and img.height() == 4

    external = skia.Surface(3, 3).makeImageSnapshot()

    @frame()
    def passthrough(form):
        return external

    assert passthrough() is external

    @frame()
    def explode(form):
        raise ValueError("boom")

    with pytest.raises(RuntimeError):
        explode()


def test_frame_accepts_surface_snapshot():
    surface = skia.Surface(4, 4)

    @frame()
    def draw_surface(form):
        surface.getCanvas().clear(0xFFFFFFFF)
        return surface  # has makeImageSnapshot()

    img = draw_surface()
    assert isinstance(img, skia.Image)


def test_frames_decorator_generates_array():
    @frames(size=(4, 4), numFrames=2, clearBg=True)
    def anim(form, t):
        form.stroke(0xFFFFFFFF, width=1.0)
        form.point([t + 1, t + 1], radius=0.5)

    video = anim()
    assert video.shape == (2, 4, 4, 3)
    assert video.dtype == np.uint8

    @frames(size=(2, 2), numFrames=3, startAt=5, keepAlpha=True)
    def track(form, t, log):
        log.append(t)

    log: list[int] = []
    clip = track(log)
    assert log == [5, 6, 7]
    assert clip.shape == (3, 2, 2, 4)

    with pytest.raises(ValueError):
        frames(startAt=-1)


def test_clear_alpha_fades_instead_of_replacing():
    """clear() with alpha < 1.0 blends over existing content instead of replacing."""
    form = Form(width=2, height=2)
    form.clear("fff")  # white
    form.clear("000", alpha=0.0)  # alpha=0 is a no-op
    arr = form.toArray()
    assert np.all(arr[:, :, :3] == 255), "alpha=0 should leave canvas unchanged"

    form.clear("fff")  # white again
    form.clear("000", alpha=1.0)  # full clear to black
    arr = form.toArray()
    assert np.all(arr[:, :, :3] == 0), "alpha=1.0 should fully clear"

    form.clear("fff")  # white
    form.clear("000", alpha=0.5)  # partial fade toward black
    arr = form.toArray()
    # Pixels should be darker than white but not fully black
    assert np.all(arr[:, :, :3] < 255) and np.all(arr[:, :, :3] > 0)


def test_stroke_and_fill_validation_and_state():
    form = Form(width=4, height=4)
    assert form.width == 4 and form.height == 4
    assert np.array_equal(form.size, np.array([4.0, 4.0]))

    with pytest.raises(ValueError):
        form.stroke(width=-1)

    shader = skia.GradientShader.MakeSweep(0, 0, [0xFFFFFFFF, 0xFF000000])
    form.stroke(shader, width=2.0)
    assert form.strokeColor is None
    assert form.shaderParams["Stroke"] is not None

    form.fillOnly(0x00FF00)
    assert form.fillColor == 0xFF00FF00
    assert form.strokeColor is None

    form.strokeOnly(0x0000FF, width=1.0)
    assert form.strokeColor == 0xFF0000FF
    assert form.fillColor is None


def test_image_uses_cpu_rgba_buffer_for_skia_bitmap():
    form = Form(width=2, height=2)
    table = np.array([[0xFF0000FF, 0x00FF00FF], [0x0000FFFF, 0xFFFFFFFF]], dtype=np.uint32)
    form.image(table, np.array([[0.0, 0.0], [2.0, 2.0]]), filterMode="nearest")

    pinned = form._pinnedColorMap
    assert isinstance(pinned, np.ndarray)
    assert pinned.dtype == np.uint8
    assert pinned.shape == (2, 2, 4)

    arr = form.toArray()
    assert arr.shape == (2, 2, 4)
    assert np.any(arr[..., :3] != 0)


def test_drawing_helpers_validation_and_output():
    form = Form(width=6, height=6)
    with pytest.raises(ValueError):
        form.dots(np.ones((2, 2, 2, 2)))

    with pytest.raises(ValueError):
        form.points(np.array([1, 2, 3]))

    form.clear(0x000000)
    form.stroke(0xFFFFFFFF, width=1.0)
    form.points(np.array([[1, 1], [2, 2]]), radius=0.5)
    arr = form.toArray()
    assert arr.shape == (6, 6, 4)
    assert arr.dtype == np.uint8

    form.stroke(0xFFFFFFFF, width=1.0)
    form.line(np.array([[0, 0], [5, 5]]))
    form.point([3, 3], style="rect")


def test_polygons_variable_length():
    """Test polygons with different vertex counts."""
    form = Form(width=100, height=100)
    form.clear("fff")  # White background

    # Variable-length: triangle, square, pentagon
    mixed = [
        [[10, 10], [20, 10], [15, 20]],  # 3 vertices
        [[30, 10], [40, 10], [40, 20], [30, 20]],  # 4 vertices
        [[50, 10], [60, 12], [58, 22], [52, 22], [48, 12]],  # 5 vertices
    ]
    form.strokeOnly("000", width=2).polygons(mixed)

    # Verify pixels were drawn - arr[y, x] indexing
    arr = form.toArray()
    assert arr[10, 15, 0] < 255, "Triangle edge at y=10, x=15"
    assert arr[10, 35, 0] < 255, "Square edge at y=10, x=35"


def test_polygons_empty_and_degenerate():
    """Test edge cases: empty list, degenerate polygons."""
    form = Form(width=100, height=100)
    form.clear("fff")

    # Empty list - should be no-op
    form.polygons([])

    # Mixed with degenerate (< 2 vertices) - should skip degenerate
    mixed = [
        [[10, 10]],  # Only 1 vertex - skip
        [[20, 20], [30, 30]],  # Valid line
    ]
    form.strokeOnly("000", width=2).polygons(mixed)

    # Verify line was drawn (diagonal from 20,20 to 30,30)
    arr = form.toArray()
    assert arr[25, 25, 0] < 255, "Line should be drawn at midpoint"


def test_polygons_uniform_still_works():
    """Backward compatibility: uniform (n, m, 2) arrays still work."""
    from ptspy import P

    form = Form(width=100, height=100)
    form.clear("fff")
    triangles = P([[[10, 10], [50, 50], [50, 10]], [[60, 60], [99, 99], [99, 60]]])
    form.strokeOnly("000", width=2).polygons(triangles)

    # Verify both triangles were drawn - arr[y, x] indexing
    arr = form.toArray()
    assert arr[10, 30, 0] < 255, "First triangle edge at y=10, x=30"
    assert arr[60, 80, 0] < 255, "Second triangle edge at y=60, x=80"


def test_translate_returns_self():
    """Test translate() returns self for chaining."""
    form = Form(width=10, height=10)
    result = form.translate(5, 5)
    assert result is form


def test_translate_moves_origin():
    """Test translate() moves the drawing origin."""
    form = Form(width=20, height=20)
    form.clear("fff")
    form.save()
    form.translate(10, 10)
    form.strokeOnly("000", width=2).point([0, 0], radius=2)
    form.restore()

    # Point at origin (0,0) after translate(10,10) should appear at (10,10)
    arr = form.toArray()
    assert arr[10, 10, 0] < 255, "Point should appear at (10,10)"


def test_rotate_returns_self():
    """Test rotate() returns self for chaining."""
    form = Form(width=10, height=10)
    result = form.rotate(45)
    assert result is form


def test_rotate_with_pivot():
    """Test rotate() with pivot point."""
    form = Form(width=100, height=100)
    form.clear("fff")

    # Draw a horizontal line from (60, 50) to (90, 50)
    # Rotate 90 degrees around (50, 50), line should become vertical from (50, 60) to (50, 90)
    form.save()
    form.rotate(90, pivot=[50, 50])
    form.strokeOnly("000", width=2).line([[60, 50], [90, 50]])
    form.restore()

    arr = form.toArray()
    # After 90deg rotation around (50,50), point at (60,50) goes to (50,60)
    assert arr[70, 50, 0] < 255, "Rotated line should pass through (50,70)"


def test_scale_returns_self():
    """Test scale() returns self for chaining."""
    form = Form(width=10, height=10)
    result = form.scale(2)
    assert result is form


def test_scale_uniform():
    """Test scale() with uniform scaling."""
    form = Form(width=100, height=100)
    form.clear("fff")
    form.save()
    form.scale(2)
    # A point at (10, 10) with scale(2) should appear at (20, 20)
    form.fillOnly("000").point([10, 10], radius=3)
    form.restore()

    arr = form.toArray()
    # Check area around expected position (20, 20) - scale affects radius too
    assert arr[20, 20, 0] < 255, "Scaled point should appear at (20,20)"


def test_scale_non_uniform():
    """Test scale() with different x and y factors."""
    form = Form(width=100, height=100)
    form.clear("fff")
    form.save()
    form.scale(2, 1)  # Double x, keep y
    form.strokeOnly("000", width=2).point([10, 20], radius=2)
    form.restore()

    arr = form.toArray()
    # Point at (10, 20) with scale(2, 1) should appear at (20, 20)
    assert arr[20, 20, 0] < 255, "Scaled point should appear at (20,20)"


def test_skew_returns_self():
    """Test skew() returns self for chaining."""
    form = Form(width=10, height=10)
    result = form.skew(0, 0)
    assert result is form


def test_resetMatrix_returns_self():
    """Test resetMatrix() returns self for chaining."""
    form = Form(width=10, height=10)
    form.translate(5, 5)
    result = form.resetMatrix()
    assert result is form


def test_resetMatrix_clears_transforms():
    """Test resetMatrix() resets to identity."""
    form = Form(width=50, height=50)
    form.clear("fff")

    # Apply transforms then reset
    form.translate(100, 100)
    form.rotate(45)
    form.scale(3)
    form.resetMatrix()

    # After reset, drawing at (10,10) should appear at (10,10)
    form.strokeOnly("000", width=2).point([10, 10], radius=2)

    arr = form.toArray()
    assert arr[10, 10, 0] < 255, "Point should appear at (10,10) after resetMatrix"


def test_concat_returns_self():
    """Test concat() returns self for chaining."""
    form = Form(width=10, height=10)
    m = skia.Matrix()
    result = form.concat(m)
    assert result is form


def test_concat_applies_matrix():
    """Test concat() applies custom transformation matrix."""
    form = Form(width=100, height=100)
    form.clear("fff")

    # Create a translation matrix
    m = skia.Matrix()
    m.setTranslate(20, 20)

    form.save()
    form.concat(m)
    form.strokeOnly("000", width=2).point([5, 5], radius=2)
    form.restore()

    arr = form.toArray()
    # Point at (5,5) with translate(20,20) should appear at (25,25)
    assert arr[25, 25, 0] < 255, "Point should appear at (25,25) after concat"


def test_cardinal_draws_without_error():
    """form.cardinal() should draw without error and return self."""
    form = Form(width=100, height=100)
    form.clear("fff")
    form.strokeOnly("000", width=2)
    pts = [[10, 50], [30, 10], [60, 90], [90, 50]]
    result = form.cardinal(pts)
    assert result is form

    # Also test closed variant
    result2 = form.cardinal(pts, closed=True)
    assert result2 is form

    # Verify some pixels were drawn (not all white)
    arr = form.toArray()
    assert arr.min() < 255, "Cardinal curve should draw something on the canvas"


def test_hermite_draws_without_error():
    """form.hermite() should draw without error and return self."""
    form = Form(width=100, height=100)
    form.clear("fff")
    form.strokeOnly("000", width=2)
    pts = [[10, 50], [30, 10], [60, 90], [90, 50]]
    tangents = [[20, 0], [0, -40], [0, 40], [-20, 0]]
    result = form.hermite(pts, tangents)
    assert result is form

    # Also test closed variant
    result2 = form.hermite(pts, tangents, closed=True)
    assert result2 is form

    # Verify some pixels were drawn (not all white)
    arr = form.toArray()
    assert arr.min() < 255, "Hermite curve should draw something on the canvas"


def test_transform_chaining():
    """Test that transformations can be chained fluently."""
    form = Form(width=100, height=100)
    form.clear("fff")

    # Chain multiple transforms
    form.save()
    result = form.translate(50, 50).rotate(45).scale(2)
    assert result is form
    form.restore()
