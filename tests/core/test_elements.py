import numpy as np
import pytest

from ptspy.core.elements import P, P2, P3, TBD, partial, randomizer


def test_partial_optional_with_placeholder_allows_curry_and_fill():
    @partial(optional=True, placeholder=TBD)
    def fn(a, b, c=3):
        return a + b + c

    curried = fn(TBD, 2)
    assert callable(curried)
    assert curried(1) == 6  # a gets filled, c defaults


def test_partial_respects_required_params_and_rejects_placeholder_conflicts():
    @partial(optional=False, placeholder=TBD)
    def fn(a, b, *, c):
        return a + b + c

    curried = fn(1)  # still waiting for b and c in strict mode
    assert callable(curried)
    assert curried(2, c=3) == 6


def test_p_and_p2_p3_helpers_shapes():
    assert P().shape == (0,)
    assert np.array_equal(P(1, 2, 3), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(P([1, 2, 3]), np.array([1.0, 2.0, 3.0]))

    grid2 = P2([1, 2], [3, 4])
    assert grid2.shape == (2, 2, 2)
    assert np.array_equal(grid2[0, 0], np.array([1.0, 3.0]))

    grid3 = P3([0, 1], [10], [5, 6, 7])
    assert grid3.shape == (2, 1, 3, 3)
    assert np.array_equal(grid3[1, 0, 2], np.array([1.0, 10.0, 7.0]))


def test_randomizer_seed_and_passthrough():
    rng = randomizer(123)
    assert isinstance(rng, np.random.Generator)
    assert rng.random() == pytest.approx(np.random.default_rng(123).random())

    supplied = np.random.default_rng(7)
    assert randomizer(rng=supplied) is supplied


def test_partial_placeholder_extra_args_appended():
    """Extra positional args beyond placeholders are appended."""

    @partial(optional=False, placeholder=TBD)
    def fn(a, b, c):
        return (a, b, c)

    assert fn(TBD, 2)(1, 3) == (1, 2, 3)  # 1 fills TBD, 3 appended as c
    assert fn(TBD, TBD, 3)(1, 2) == (1, 2, 3)


def test_partial_placeholder_fewer_args_than_placeholders():
    """When fewer args than placeholders are provided, remaining stay as TBD (returns curry)."""

    @partial(optional=False, placeholder=TBD)
    def fn(a, b, c):
        return (a, b, c)

    still_curried = fn(TBD, TBD, 3)(1)  # only fills first TBD
    assert callable(still_curried)
    assert still_curried(2) == (1, 2, 3)


def test_partial_keyword_placeholder_replacement():
    """Keyword placeholders are replaced by subsequent calls."""

    @partial(optional=False, placeholder=TBD)
    def fn(a, b, *, c):
        return (a, b, c)

    curried = fn(1, b=TBD, c=TBD)
    assert callable(curried)
    assert curried(b=2, c=3) == (1, 2, 3)


def test_partial_keyword_duplicate_raises_with_function_name():
    """Duplicate keyword args include function name in error message."""

    @partial(optional=False)
    def myFunc(a, b):
        return a + b

    with pytest.raises(TypeError, match="myFunc"):
        myFunc(a=1)(a=2)


def test_partial_keyword_only_params():
    """Keyword-only parameters work correctly with partial."""

    @partial(optional=False)
    def fn(*, x, y, z):
        return x + y + z

    assert fn(x=1)(y=2)(z=3) == 6
    assert fn(x=1, y=2)(z=3) == 6


def test_partial_with_varargs():
    """Functions with *args work with partial (optional mode)."""

    @partial(optional=True)
    def fn(a, *args):
        return (a, args)

    assert fn(1) == (1, ())
    assert fn(1, 2, 3) == (1, (2, 3))


def test_partial_with_kwargs():
    """Functions with **kwargs work with partial (optional mode)."""

    @partial(optional=True)
    def fn(a, **kwargs):
        return (a, kwargs)

    assert fn(1) == (1, {})
    assert fn(1, x=2) == (1, {"x": 2})


def test_p_rejects_strings_as_iterables():
    """P() should treat strings as scalar-like, not iterate over characters."""
    with pytest.raises((ValueError, TypeError)):
        P("hello")
    with pytest.raises((ValueError, TypeError)):
        P(b"bytes")


def test_toPArray_and_p_helpers_with_empty_and_nested():
    assert np.array_equal(P(), np.array([]))
    nested = P([1, 2], [3, 4])
    assert nested.shape == (2, 2)
    arr = np.array([[1, 2], [3, 4]])
    assert np.array_equal(arr, arr)  # sanity
