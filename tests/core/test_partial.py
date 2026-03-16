import inspect
import pickle

import numpy as np
import pytest

from ptspy.core.elements import TBD, partial


def test_strict_currying_requires_all_non_variadic_even_with_defaults():
    @partial(optional=False)
    def add(a, b=5):
        return a + b

    curried = add(1)  # default b is ignored until provided in strict mode
    assert callable(curried)
    assert curried(4) == 5  # b supplied now


def test_optional_currying_executes_when_required_satisfied():
    @partial(optional=True)
    def add(a, b, c=3):
        return a + b + c

    # optional=True uses defaults once required args are present
    assert add(1)(2) == 6
    # can still split across more calls
    step = add(1)
    assert callable(step)
    assert step(2, 4) == 7


def test_positional_placeholder_replacement_and_extension():
    @partial(optional=False, placeholder=TBD)
    def combine(a, b, c):
        return a * 100 + b * 10 + c

    # fill placeholder first, then supply remaining argument
    curried = combine(TBD, 2)
    assert callable(curried)
    assert curried(1, 3) == 123

    # placeholders are replaced in order, remaining args extend
    assert combine(TBD, 2, 3)(4) == 423


def test_keyword_placeholders_and_conflicts():
    @partial(placeholder=TBD)
    def build(a, b, c=0, *, d=0):
        return a + b + c + d

    fn = build(a=TBD, b=2, c=TBD, d=10)
    assert fn(a=1, c=3) == 16

    dup = build(a=TBD, b=2)
    with pytest.raises(TypeError):
        dup(a=1, b=3)  # duplicate non-placeholder keyword


def test_signature_preserved():
    @partial(placeholder=TBD, optional=True)
    def foo(x: int, y: int = 9) -> int:
        return x + y

    assert inspect.signature(foo) == inspect.signature(foo.__wrapped__)  # type: ignore[attr-defined]


def test_varargs_kwargs_supported():
    @partial(optional=False)
    def collect(a, b, *args, **kwargs):
        return a, b, args, kwargs

    out = collect(1, 2, 3, 4, k=5)
    assert out == (1, 2, (3, 4), {"k": 5})


def test_optional_mode_uses_defaults_with_placeholders():
    @partial(optional=True, placeholder=TBD)
    def config(host=TBD, port=8080, ssl=False):
        return host, port, ssl

    fn = config(host="example.com", port=TBD, ssl=True)
    assert fn(port=9000) == ("example.com", 9000, True)


def test_positional_only_and_keyword_only_arguments():
    @partial(placeholder=TBD, optional=True)
    def func(a, b, /, *, c=10):
        return a + b + c

    fn = func(1, TBD, c=5)
    assert fn(2) == 8


def test_placeholder_disabled_behaves_like_normal_currying():
    @partial(placeholder=None)
    def add(a, b):
        return a + b

    curried = add(1)
    assert callable(curried)
    assert curried(2) == 3


def test_random_placeholder_integration_matches_np_behavior():
    # Regression: placeholder with numpy scalars should still work
    @partial(placeholder=TBD)
    def scale(arr, factor):
        return np.asarray(arr) * factor

    fn = scale([1, 2, 3], TBD)
    out = fn(2)
    assert np.array_equal(out, np.array([2, 4, 6]))


def test_tbd_singleton_identity_and_reduce_roundtrip():
    reduceFn, reduceArgs = TBD.__reduce__()
    assert reduceFn(*reduceArgs) is TBD
    assert pickle.loads(pickle.dumps(TBD)) is TBD


def test_partial_matrix_positional_only_optional_and_strict():
    @partial(optional=False)
    def strictPosOnly(a, b, /, c=10):
        return a + b + c

    @partial(optional=True)
    def optionalPosOnly(a, b, /, c=10):
        return a + b + c

    # strict: needs all non-variadic params, including c
    assert strictPosOnly(1)(2)(3) == 6
    # optional: executes once required positional-only args are provided
    assert optionalPosOnly(1)(2) == 13


def test_partial_matrix_keyword_only_strict_and_optional():
    @partial(optional=False)
    def strictKwOnly(a, *, b, c=7):
        return a + b + c

    @partial(optional=True)
    def optionalKwOnly(a, *, b, c=7):
        return a + b + c

    # strict: c must be explicitly provided
    assert strictKwOnly(1)(b=2)(c=3) == 6
    # optional: c can use default
    assert optionalKwOnly(1)(b=2) == 10


def test_partial_matrix_varargs_and_repeated_partial_application():
    @partial(optional=False)
    def collectStrict(a, *args, scale=1):
        return (a + sum(args)) * scale

    @partial(optional=True, placeholder=TBD)
    def digits(a, b, c, d):
        return a * 1000 + b * 100 + c * 10 + d

    # strict + varargs: keyword-only default must be provided in strict mode
    assert collectStrict(1)(2, 3)(scale=2) == 12

    # repeated partial applications with placeholder across calls
    stage1 = digits(TBD, 2)
    stage2 = stage1(TBD, 3)
    stage3 = stage2(1)
    assert callable(stage3)
    assert stage3(4) == 1234
