import numpy as np
import pytest

from ptspy.core import process


def double(x):
    return x * 2


def increment(x):
    return x + 1


def pair(a, b):
    return a + 1, b + 2


def test_pipeline_eager_and_lazy():
    fn = process.pipeline(double, increment)
    assert fn(3) == 7

    lazy = process.pipeline(double, increment, lazy=True)
    assert list(lazy(3)) == [6, 7]


@pytest.mark.asyncio
async def test_pipeline_async_and_trace(capsys):
    async def a_double(x):
        return x * 2

    async def a_inc(x):
        return x + 1

    traced = process.pipeline(a_double, a_inc, trace=True)
    out = await traced(2)
    assert out == 5
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["a_double: 4", "a_inc: 5"]


def test_pipeline_trace_prints(capsys):
    def inc(x):
        return x + 1

    process.pipeline(double, inc, trace=True)(3)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["double: 6", "inc: 7"]


def test_pipeline_trace_lazy_prints(capsys):
    def inc(x):
        return x + 1

    lazy = process.pipeline(double, inc, lazy=True, trace=True)
    assert list(lazy(3)) == [6, 7]
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["double: 6", "inc: 7"]


def test_reverse_pipeline_trace_prints(capsys):
    def inc(x):
        return x + 1

    process.reversePipeline(double, inc, trace=True)(3)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["inc: 4", "double: 8"]


def test_pipeline_auto_unpack_tuple():
    def returns_tuple():
        return (1, 2)

    def takes_two(a, b):
        return a + b

    def takes_tuple(pair):
        return pair[0] + pair[1]

    # Tuples are auto-unpacked as arguments to the next stage
    auto_unpacked = process.pipeline(returns_tuple, takes_two)
    assert auto_unpacked() == 3

    # Function expecting single tuple arg fails with auto-unpacking
    with pytest.raises(TypeError):
        process.pipeline(returns_tuple, takes_tuple)()


def test_pipeline_trace_callable():
    seen = []

    def tracer(name, value):
        seen.append((name, value))

    def inc(x):
        return x + 1

    traced = process.pipeline(double, inc, trace=tracer)
    assert traced(2) == 5
    assert seen == [("double", 4), ("inc", 5)]


def test_stagefunction_conflicting_options_raise():
    eager = process.pipeline(double, increment, lazy=False)
    lazy = process.pipeline(double, increment, lazy=True)
    with pytest.raises(ValueError, match="Conflicting pipeline option 'lazy'"):
        _ = process.F(eager) >> lazy


def test_stagefunction_auto_unpack():
    def returns_tuple():
        return (1, 2)

    def takes_two(a, b):
        return a + b

    # Tuples are auto-unpacked when using StageFunction (F) operators
    chain = process.F(returns_tuple) >> takes_two
    assert chain() == 3


@pytest.mark.asyncio
async def test_pipeline_async_awaits_returned_awaitable():
    async def a_double(x):
        return x * 2

    def returns_coro(x):
        return a_double(x)

    pipe = process.pipeline(a_double, returns_coro, increment)
    assert await pipe(2) == 9


def test_pipeline_sync_rejects_awaitable_return():
    async def a_double(x):
        return x * 2

    def returns_coro(x):
        return a_double(x)

    pipe = process.pipeline(returns_coro, increment)
    with pytest.raises(TypeError, match="awaitable"):
        pipe(2)


def test_reverse_pipeline_and_stage_function_ops():
    fn = process.reversePipeline(double, increment)
    assert fn(3) == 8  # increment then double

    f = process.F(double) >> increment
    assert f(4) == 9

    g = process.F(double) << increment
    assert g(3) == 8

    h = increment >> process.F(double)
    assert h(2) == 6

    # Auto-unpacking: pair returns (2, 4), which gets unpacked for the lambda
    chain = process.pipeline(pair, lambda a, b: a * b)
    assert chain(1, 2) == 8  # pair(1,2) -> (2,4) -> 2*4

    with pytest.raises(TypeError):
        process.StageFunction("not-callable")


def test_shaping_function_lookup_and_errors():
    linear = process.shapingFunction("linear")
    assert linear(0.5, 0.0, 10.0) == 5.0

    with pytest.raises(KeyError):
        process.shapingFunction("missing")

    step = process.shapingFunction("step", 4)
    assert step(0.3, 0.0, 1.0) == pytest.approx(0.25)


def test_step_factory_rejects_non_positive_steps():
    with pytest.raises(ValueError):
        process.shapingFunction("step", 0)
    with pytest.raises(ValueError):
        process.shapingFunction("step", -1)


def test_expand_before_and_axis_validation():
    arr = np.array([1, 2, 3], dtype=float)
    out = process.expand(arr, fill=0, repeat=2, before=True)
    assert np.array_equal(out, np.array([0, 0, 1, 2, 3], dtype=float))

    mat = np.array([[1, 2], [3, 4]], dtype=float)
    expanded = process.expand(mat, fill=9, repeat=1, axis=1)
    assert np.array_equal(expanded[:, -1], np.array([9, 9], dtype=float))

    with pytest.raises(ValueError):
        process.expand(arr, fill=0, repeat=-1)
    with pytest.raises(ValueError):
        process.expand(arr, fill=0, repeat=1, axis=5)
    assert np.array_equal(process.expand(arr, fill=-1, repeat=0), arr)


def test_change_dims_add_and_remove():
    arr = np.array([1, 2, 3], dtype=float)
    added = process.changeDims(arr, target=2, axis=-1)
    assert added.shape == (3, 1)

    squeezed = process.changeDims(added, target=1, axis=0)
    assert squeezed.shape == (3,)

    with pytest.raises(ValueError):
        process.changeDims(arr, target=0)

    with pytest.raises(ValueError):
        process.changeDims(np.ones((2, 2)), target=1)  # no singleton axes to drop
    with pytest.raises(ValueError):
        process.changeDims(arr, target=64)  # maxDepthCheck triggers

    mat = np.ones((2, 1, 3))
    squeezed_end = process.changeDims(mat, target=2, axis=-1)
    assert squeezed_end.shape == (2, 3)


def test_group_1d_and_nd_pad_modes_and_errors():
    one_d = process.group([1, 2, 3, 4, 5], groupSize=2)
    assert np.array_equal(one_d, np.array([[1, 2], [3, 4]], dtype=float))

    wrapped = process.group([1, 2, 3], groupSize=2, padMode="wrap")
    assert np.array_equal(wrapped, np.array([[1, 2], [3, 1]], dtype=float))

    mirrored = process.group(np.array([[1, 2], [3, 4], [5, 6]]), groupSize=2, axis=0, padMode="mirror")
    assert mirrored.shape == (2, 2, 2)

    with pytest.raises(ValueError):
        process.group([1, 2, 3], groupSize=0)
    with pytest.raises(ValueError):
        process.group(np.ones((2, 2)), groupSize=2, axis=5)
    with pytest.raises(ValueError):
        process.group([1, 2, 3], groupSize=2, padMode="invalid")

    by_col = process.group([[1, 2, 3], [4, 5, 6]], groupSize=2, axis=1, padMode="truncate")
    assert by_col.shape == (2, 1, 2)


def test_group_value_wrap_and_mirror_pad_modes():
    arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    value_padded = process.group(arr, groupSize=2, padMode="value", pad=-1, axis=0)
    assert value_padded.shape == (2, 2, 2)
    assert np.array_equal(value_padded[1, 1], np.array([-1.0, -1.0]))

    wrap_cols = process.group(np.array([[1, 2, 3], [4, 5, 6]], dtype=float), groupSize=2, axis=1, padMode="wrap")
    assert wrap_cols.shape == (2, 2, 2)
    assert np.array_equal(wrap_cols[0], np.array([[1, 2], [3, 1]], dtype=float))

    mirrored = process.group(np.array([1, 2, 3], dtype=float), groupSize=4, padMode="mirror")
    assert np.array_equal(mirrored, np.array([[1, 2, 3, 3]], dtype=float))


def test_interleave_validations_and_behavior():
    with pytest.raises(ValueError):
        process.interleave()
    with pytest.raises(ValueError):
        process.interleave(np.ones(2))
    with pytest.raises(ValueError):
        process.interleave(np.ones(2), np.ones(3))
    with pytest.raises(ValueError):
        process.interleave(np.array([1, 2, 3]))  # single array not allowed

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    inter = process.interleave(a, b, axis=1)
    assert np.array_equal(inter, np.array([[1, 5, 2, 6], [3, 7, 4, 8]]))

    inter_axis0 = process.interleave(a, b, axis=0)
    assert np.array_equal(inter_axis0, np.array([[1, 2], [5, 6], [3, 4], [7, 8]]))


def test_interleave_axis_wraps_and_dtype_from_first():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4.5, 5.5, 6.5], dtype=np.float64)
    mixed = process.interleave(a, b)
    assert mixed.dtype == a.dtype
    assert np.array_equal(mixed, np.array([1, 4, 2, 5, 3, 6], dtype=np.int32))

    axis_wrapped = process.interleave(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), axis=2)
    assert np.array_equal(axis_wrapped, process.interleave(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), axis=0))


def test_bound_clip_and_wrap_inclusive():
    clipped = process.bound([1, 5, 10], 2, 4, loopBack=False)
    assert np.array_equal(clipped, np.array([2, 4, 4]))

    wrapped = process.bound([1, 5, 10], 2, 4, loopBack=True, inclusive=True)
    assert np.array_equal(wrapped, np.array([4, 2, 4]))
    wrapped_no_inc = process.bound([1, 5, 10], 2, 4, loopBack=True, inclusive=False)
    assert np.array_equal(wrapped_no_inc, np.array([3, 3, 2]))

    with pytest.raises(ValueError):
        process.bound([1, 2], 3, 3)


def test_bound_wrap_reversed_bounds_equivalence():
    pts = np.array([1.0, 2.0, 4.5, 5.0, 6.0], dtype=float)
    forward = process.bound(pts, 2.0, 5.0, loopBack=True, inclusive=False)
    reversed_bounds = process.bound(pts, 5.0, 2.0, loopBack=True, inclusive=False)
    assert np.array_equal(forward, reversed_bounds)


def test_interpolate_path_and_apply_all_modes():
    pts = np.array([[0, 0], [10, 10], [20, 0]], dtype=float)

    mid = process.interpolate(pts, 0.5)
    assert np.array_equal(mid, np.array([[5.0, 5.0], [15.0, 5.0]]))

    path = process.interpolate(pts, [0.25, 0.75], asPath=True)
    assert np.allclose(path, np.array([[5.0, 5.0], [15.0, 5.0]]))

    all_segments = process.interpolate(pts, [0.25, 0.75], applyAll=True)
    assert all_segments.shape == (2, 2, 2)

    unclamped = process.interpolate([[0], [10]], 1.5, clamp=False)
    assert unclamped == np.array([15.0])

    with pytest.raises(ValueError):
        process.interpolate([[0, 0]], 0.5)

    # applyAll with scalar path values
    both = process.interpolate([[0], [10], [20]], [0.25, 0.75], applyAll=True)
    assert both.shape == (2, 2, 1)


def test_interpolate_indices_rounding_and_validation():
    with pytest.raises(ValueError):
        process.interpolateIndices([0, 0], [1, 1], 0)
    with pytest.raises(ValueError):
        process.interpolateIndices([0, 0], [1], 2)

    steps = process.interpolateIndices([0, 0], [4, 4], 3, rounding="floor")
    assert np.array_equal(steps, np.array([[0, 0], [2, 2], [4, 4]]))

    ceil = process.interpolateIndices([0, 0], [2, 2], 3, rounding="ceil")
    assert np.array_equal(ceil, np.array([[0, 0], [1, 1], [2, 2]]))

    rounded = process.interpolateIndices([0, 0], [2, 2], 3)
    assert np.array_equal(rounded, np.array([[0, 0], [1, 1], [2, 2]]))


def test_points_to_regions_and_validation():
    regions = process.pointsToRegions([[5, 5], [10, 10]], [2, 4], anchor=[0.5, 0.25])
    assert regions.shape == (2, 2, 2)
    assert np.array_equal(regions[0, 0], np.array([4.0, 4.0]))
    assert np.array_equal(regions[0, 1], np.array([6.0, 8.0]))

    with pytest.raises(ValueError):
        process.pointsToRegions(np.ones((2, 2, 2)), 1)
    with pytest.raises(ValueError):
        process.pointsToRegions([[1, 2]], [1], anchor=[1, 2, 3])  # broadcast mismatch
    # scalar regionSize broadcast
    scalar = process.pointsToRegions([[1, 2], [3, 4]], 2)
    assert scalar.shape == (2, 2, 2)

    empty = process.pointsToRegions(np.empty((0, 2)), 1)
    assert empty.shape == (0, 2, 2)


def test_attach_position_index_offsets_and_validation():
    arr = np.array([[10, 20], [30, 40]], dtype=float)
    result = process.attachPositionIndex(arr, offsets=[2, 3])
    assert result.shape == (2, 2, 3)
    assert np.array_equal(result[1, 1], np.array([2.0, 3.0, 40.0]))

    with pytest.raises(ValueError):
        process.attachPositionIndex(arr, offsets=[1])

    ident = process.attachPositionIndex(arr)
    assert np.array_equal(ident[..., -1], arr)


# =============================================================================
# Exhaustive group pad-mode tests
# =============================================================================

def test_group_1d_last_pad_mode():
    result = process.group([1, 2, 3], groupSize=2, padMode="last")
    assert np.array_equal(result, np.array([[1, 2], [3, 3]], dtype=float))


def test_group_1d_value_pad_mode():
    result = process.group([1, 2, 3], groupSize=2, padMode="value", pad=-1)
    assert np.array_equal(result, np.array([[1, 2], [3, -1]], dtype=float))


def test_group_1d_exact_division():
    """No remainder — all pad modes should produce the same result."""
    arr = [1, 2, 3, 4]
    expected = np.array([[1, 2], [3, 4]], dtype=float)
    for mode in ("truncate", "last", "value", "wrap", "mirror"):
        assert np.array_equal(process.group(arr, groupSize=2, padMode=mode), expected)


def test_group_nd_last_pad_mode():
    arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    result = process.group(arr, groupSize=2, padMode="last", axis=0)
    assert result.shape == (2, 2, 2)
    assert np.array_equal(result[1, 1], np.array([5, 6], dtype=float))  # last row repeated


def test_group_nd_mirror_axis1():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    result = process.group(arr, groupSize=2, padMode="mirror", axis=1)
    assert result.shape == (2, 2, 2)
    # col 3 is leftover; mirror pads with col 3 reversed = col 3 itself (single element)
    assert np.array_equal(result[0], np.array([[1, 2], [3, 3]], dtype=float))


def test_group_nd_truncate_axis1_with_remainder():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    result = process.group(arr, groupSize=2, padMode="truncate", axis=1)
    assert result.shape == (2, 1, 2)
    assert np.array_equal(result[0, 0], np.array([1, 2], dtype=float))


def test_group_nd_negative_axis():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    pos = process.group(arr, groupSize=2, padMode="truncate", axis=1)
    neg = process.group(arr, groupSize=2, padMode="truncate", axis=-1)
    assert np.array_equal(pos, neg)


def test_group_nd_value_axis1():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    result = process.group(arr, groupSize=2, padMode="value", pad=0, axis=1)
    assert result.shape == (2, 2, 2)
    assert np.array_equal(result[0], np.array([[1, 2], [3, 0]], dtype=float))


# =============================================================================
# Exhaustive interpolate mode/shape tests
# =============================================================================

def test_interpolate_pairwise_cycling():
    """Multiple t values cycle across segments."""
    pts = np.array([[0], [10], [20]], dtype=float)
    # 2 segments, 3 t values → t[0] for seg0, t[1] for seg1, t[2] cycles to seg0 (ignored, only 2 segs)
    result = process.interpolate(pts, [0.5, 0.25])
    # seg0 uses t[0]=0.5 → 5, seg1 uses t[1]=0.25 → 12.5
    assert np.allclose(result, np.array([[5.0], [12.5]]))


def test_interpolate_custom_shaping():
    """Custom shaping function (quadratic ease)."""
    pts = np.array([[0, 0], [10, 10]], dtype=float)
    result = process.interpolate(pts, 0.5, shaping=lambda t, a, b: a + (b - a) * t * t)
    assert np.allclose(result, np.array([[2.5, 2.5]]))


def test_interpolate_axis_parameter():
    """Interpolate along a non-default axis."""
    # pts shape (2, 3) — 2 coords, 3 points along axis=1
    pts = np.array([[0, 5, 10], [0, 5, 10]], dtype=float)
    result = process.interpolate(pts, 0.5, axis=1)
    # 2 segments along axis=1, each at t=0.5
    assert result.shape == (2, 2)
    assert np.allclose(result, np.array([[2.5, 7.5], [2.5, 7.5]]))


def test_interpolate_path_boundary_values():
    """asPath with t=0.0 and t=1.0 returns first and last points."""
    pts = np.array([[0, 0], [10, 10], [20, 0]], dtype=float)
    endpoints = process.interpolate(pts, [0.0, 1.0], asPath=True)
    assert np.allclose(endpoints[0], np.array([0, 0]))
    assert np.allclose(endpoints[1], np.array([20, 0]))


def test_interpolate_apply_all_values():
    """applyAll: verify actual values, not just shape."""
    pts = np.array([[0], [10], [20]], dtype=float)
    result = process.interpolate(pts, [0.0, 1.0], applyAll=True)
    # shape (2, 2, 1): 2 t values × 2 segments × 1 coord
    assert result.shape == (2, 2, 1)
    # t=0.0: seg0→0, seg1→10
    assert np.allclose(result[0], np.array([[0], [10]]))
    # t=1.0: seg0→10, seg1→20
    assert np.allclose(result[1], np.array([[10], [20]]))
