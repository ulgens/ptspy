# ruff: noqa: E731
"""Pipeline composition and array processing utilities."""

from __future__ import annotations
import functools
from dataclasses import dataclass
from functools import wraps
from inspect import iscoroutinefunction, isawaitable, iscoroutine
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    ParamSpec,
    TypeVar,
    Optional,
    Literal,
    Sequence,
    Union,
    cast,
    overload,
)

from .compose import toPArray, TBD
from ptspy.typing import (
    PArrayOrScalar,
    PArray,
    PArrayLike,
    IntArray,
    NumberSequenceND,
    RoundingMode,
    ShapingFunction,
    ShapingMethod,
    Number,
    PadMode,
)
from .. import np_wrapper as np
from .elements import partial


_PS = ParamSpec("_PS")
_RT = TypeVar("_RT")

_UNSET = object()
_TRACE_PRINT = object()
_PIPELINE_SPEC_ATTR = "_pipeline_spec"


@dataclass(frozen=True)
class _PipelineSpec:
    stages: tuple[Callable[..., Any], ...]
    lazy: bool | object
    trace: object | Callable[[str, Any], Any] | None


def unpackResult(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a stage so that a tuple result from the previous stage is unpacked
    into positional arguments: ``fn(*result)`` instead of ``fn(result)``.

    This overrides the default auto-unpack behavior of pipelines, giving
    explicit control when a stage expects a tuple as a single argument.
    """
    if iscoroutinefunction(fn):

        @wraps(fn)
        async def _awrapper(result):
            if not isinstance(result, tuple):
                raise TypeError("unpackResult expects a tuple")
            return await fn(*result)

        return _awrapper

    @wraps(fn)
    def _wrapper(result):
        if not isinstance(result, tuple):
            raise TypeError("unpackResult expects a tuple")
        return fn(*result)

    return _wrapper


def _spec_from_callable(fn: Callable[..., Any]) -> _PipelineSpec:
    spec = getattr(fn, _PIPELINE_SPEC_ATTR, None)
    if spec is not None:
        return cast(_PipelineSpec, spec)
    return _PipelineSpec((fn,), _UNSET, _UNSET)


def _merge_option(name: str, left: object, right: object) -> object:
    if left is _UNSET:
        return right
    if right is _UNSET:
        return left
    if left != right:
        raise ValueError(f"Conflicting pipeline option '{name}': {left!r} vs {right!r}")
    return left


def _normalize_trace(
    trace: bool | Callable[[str, Any], Any],
) -> tuple[object | Callable[[str, Any], Any] | None, Callable[[str, Any], Any] | None]:
    if trace is False or trace is None:
        return None, None
    if trace is True:

        def _tap(name, value):
            print(f"{name}: {value!r}")
            return value

        return _TRACE_PRINT, _tap
    if callable(trace):
        return trace, trace
    raise TypeError("trace must be a bool or a callable")


def _reject_awaitable(value: Any) -> None:
    if isawaitable(value):
        if iscoroutine(value) and hasattr(value, "close"):
            value.close()
        raise TypeError("Stage returned awaitable in a sync pipeline")


def _call_sync(fn: Callable[..., Any], value: Any) -> Any:
    # Auto-unpack tuples as multiple arguments
    if isinstance(value, tuple):
        value = fn(*value)
    else:
        value = fn(value)
    _reject_awaitable(value)
    return value


async def _call_async(fn: Callable[..., Any], value: Any) -> Any:
    # Auto-unpack tuples as multiple arguments
    if isinstance(value, tuple):
        value = await fn(*value) if iscoroutinefunction(fn) else fn(*value)
    else:
        value = await fn(value) if iscoroutinefunction(fn) else fn(value)
    if isawaitable(value):
        value = await value
    return value


# Helper for pipeline functions - Return a wrapper with the same sync/async kind as *fn*.
def _traceWrapper(fn, tap):
    if iscoroutinefunction(fn):

        @wraps(fn)
        async def awrapper(*a, **kw):
            val = await fn(*a, **kw)
            tap(fn.__name__, val)
            return val

        return awrapper
    else:

        @wraps(fn)
        def swrapper(*a, **kw):
            val = fn(*a, **kw)
            tap(fn.__name__, val)
            return val

        return swrapper


def _build(
    funcs: tuple[Callable[..., Any], ...],
    *,
    reverse: bool = False,
    lazy: bool = False,
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Any]:
    """
    Helper for pipeline functions - Build a pipeline from the provided functions, either in forward or reverse order.
    """

    if not funcs:
        raise ValueError("At least one function must be provided")

    original_funcs = funcs
    if reverse:
        funcs = funcs[::-1]

    trace_spec, trace_fn = _normalize_trace(trace)

    # trace wrapping
    if trace_fn is not None:
        funcs = tuple(_traceWrapper(f, trace_fn) for f in funcs)

    has_async = any(iscoroutinefunction(f) for f in funcs)
    first, *rest = funcs

    #  lazy pipeline (sync)
    if lazy and not has_async:

        @wraps(first)
        def _gen(*args: _PS.args, **kwargs: _PS.kwargs) -> Generator[Any, None, Any]:
            value = first(*args, **kwargs)
            _reject_awaitable(value)
            yield value
            for fn in rest:
                value = _call_sync(fn, value)
                yield value

        _gen.stages = original_funcs  # type: ignore[attr-defined]
        setattr(_gen, _PIPELINE_SPEC_ATTR, _PipelineSpec(original_funcs, lazy, trace_spec))  # type: ignore[attr-defined]
        return _gen

    #  lazy pipeline (async)
    if lazy and has_async:

        @wraps(first)
        async def _agen(*args: _PS.args, **kwargs: _PS.kwargs) -> AsyncGenerator[Any, None]:
            value = await first(*args, **kwargs) if iscoroutinefunction(first) else first(*args, **kwargs)
            if isawaitable(value):
                value = await value
            yield value
            for fn in rest:
                value = await _call_async(fn, value)
                yield value

        _agen.stages = original_funcs  # type: ignore[attr-defined]
        setattr(_agen, _PIPELINE_SPEC_ATTR, _PipelineSpec(original_funcs, lazy, trace_spec))  # type: ignore[attr-defined]
        return _agen

    #  eager pipeline (sync)
    if not has_async:

        @wraps(first)
        def _fn(*args: _PS.args, **kwargs: _PS.kwargs) -> _RT:  # type: ignore[valid-type]
            value: Any = first(*args, **kwargs)
            _reject_awaitable(value)
            for fn in rest:
                value = _call_sync(fn, value)
            return value  # type: ignore[return-value]

        _fn.stages = original_funcs  # type: ignore[attr-defined]
        setattr(_fn, _PIPELINE_SPEC_ATTR, _PipelineSpec(original_funcs, lazy, trace_spec))  # type: ignore[attr-defined]
        return _fn

    #  eager pipeline (async)
    @wraps(first)
    async def _afn(*args: _PS.args, **kwargs: _PS.kwargs) -> _RT:  # type: ignore[valid-type]
        value: Any = await first(*args, **kwargs) if iscoroutinefunction(first) else first(*args, **kwargs)
        if isawaitable(value):
            value = await value
        for fn in rest:
            value = await _call_async(fn, value)
        return value  # type: ignore[return-value]

    _afn.stages = original_funcs  # type: ignore[attr-defined]
    setattr(_afn, _PIPELINE_SPEC_ATTR, _PipelineSpec(original_funcs, lazy, trace_spec))  # type: ignore[attr-defined]
    return _afn


@overload
def pipeline(
    *funcs: Callable[..., Any],
    lazy: Literal[True],
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Generator[Any, None, Any]]: ...


@overload
def pipeline(
    *funcs: Callable[..., Any],
    lazy: Literal[False] = False,
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Any]: ...


@overload
def pipeline(
    *funcs: Callable[..., Any],
    lazy: bool = False,
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Any] | Callable[..., Generator[Any, None, Any]]: ...


def pipeline(
    *funcs: Callable[..., Any],
    lazy: bool = False,
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Any] | Callable[..., Generator[Any, None, Any]]:
    """
    Chain functions left-to-right. Each output feeds into the next input like in a pipeline.

    Works with sync and async functions. Tuple returns are automatically unpacked
    as multiple arguments for the next function, enabling easy multi-value passing.
    Use `lazy=True` to yield intermediate results, or `trace=True` to debug each step.

    For operator syntax, use `F`: `F(a) >> b >> c` equals `pipeline(a, b, c)`.

    Args:
        *funcs (Callable[..., Any]): Functions to execute in sequence.
        lazy (bool): If True, yields intermediate results as a generator.
        trace (bool): If True, prints each function's name and output.

    Returns:
        Callable: A composed function (async if any input is async).

    Examples:
        >>> def double(x): return x * 2
        >>> def add_one(x): return x + 1
        >>> f = pipeline(double, add_one)
        >>> f(3)
        7  # (3 * 2) + 1

        >>> # Operator syntax
        >>> (F(double) >> add_one)(3)
        7

        >>> # Lazy evaluation
        >>> list(pipeline(double, add_one, lazy=True)(3))
        [6, 7]

        >>> # Multi-value passing via tuple unpacking
        >>> def split(x): return (x, x * 10)
        >>> def combine(a, b): return a + b
        >>> pipeline(split, combine)(5)
        55  # 5 + 50
    """
    return _build(funcs, reverse=False, lazy=lazy, trace=trace)


@overload
def reversePipeline(
    *funcs: Callable[..., Any],
    lazy: Literal[True],
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Generator[Any, None, Any]]: ...


@overload
def reversePipeline(
    *funcs: Callable[..., Any],
    lazy: Literal[False] = False,
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Any]: ...


@overload
def reversePipeline(
    *funcs: Callable[..., Any],
    lazy: bool = False,
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Any] | Callable[..., Generator[Any, None, Any]]: ...


def reversePipeline(
    *funcs: Callable[..., Any],
    lazy: bool = False,
    trace: bool | Callable[[str, Any], Any] = False,
) -> Callable[..., Any] | Callable[..., Generator[Any, None, Any]]:
    """
    Chain functions right-to-left. It has the exact same features as `pipeline()`, but executes in reverse order.

    Useful when you want to read the composition in traditional mathematical order: `f(g(x))`.

    For operator syntax, use `F`: `F(a) << b << c` equals `reversePipeline(a, b, c)`.

    Args:
        *funcs (Callable[..., Any]): Functions to compose (executed last-to-first).
        lazy (bool): If True, yields intermediate results as a generator.
        trace (bool): If True, prints each function's name and output.

    Returns:
        Callable: A composed function (async if any input is async).

    Examples:
        >>> def double(x): return x * 2
        >>> def add_one(x): return x + 1
        >>> f = reversePipeline(double, add_one)
        >>> f(3)
        8  # double(add_one(3)) = (3 + 1) * 2

        >>> # Operator syntax
        >>> (F(double) << add_one)(3)
        8
    """
    return _build(funcs, reverse=True, lazy=lazy, trace=trace)


class StageFunction:
    """
    Function wrapper enabling operator-based pipeline composition.

    Wrap a function with `F()` (alias for `StageFunction`) to chain it with
    others using `>>` (forward) or `<<` (reverse) operators. Chains flatten
    automatically, so `F(a) >> b >> c` creates a single 3-stage pipeline.

    Examples:
        >>> def double(x): return x * 2
        >>> def add_one(x): return x + 1

        >>> # Forward pipeline: double first, then add_one
        >>> (F(double) >> add_one)(3)
        7

        >>> # Reverse pipeline: add_one first, then double
        >>> (F(double) << add_one)(3)
        8

        >>> # Build reusable pipelines
        >>> process = F(double) >> add_one >> double
        >>> process(5)
        22  # ((5 * 2) + 1) * 2
    """

    __slots__ = ("fn", "_spec")

    @staticmethod
    def _spec_from(obj: "StageFunction | Callable[..., Any]") -> _PipelineSpec:
        if isinstance(obj, StageFunction):
            return obj._spec
        return _spec_from_callable(obj)

    @staticmethod
    def _merge_specs(left: _PipelineSpec, right: _PipelineSpec) -> _PipelineSpec:
        lazy = _merge_option("lazy", left.lazy, right.lazy)
        trace = _merge_option("trace", left.trace, right.trace)
        stages = (*left.stages, *right.stages)
        return _PipelineSpec(
            stages=stages,
            lazy=lazy,
            trace=trace,
        )

    @staticmethod
    def _trace_arg_from_spec(trace: object) -> bool | Callable[[str, Any], Any]:
        if trace is _UNSET or trace is None:
            return False
        if trace is _TRACE_PRINT:
            return True
        return cast(Callable[[str, Any], Any], trace)

    @staticmethod
    def _lazy_from_spec(lazy: object) -> bool:
        return False if lazy is _UNSET else cast(bool, lazy)

    def __init__(self, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError("StageFunction expects a callable")
        self.fn = fn
        self._spec = _spec_from_callable(fn)

    def __rshift__(self, other: "StageFunction | Callable[..., Any]") -> "StageFunction":
        """
        Implements the right shift (>>) operator to compose two stage functions into a single pipeline.
        Args:
            other (StageFunction | Callable[..., Any]): The stage function or callable to compose with this stage.
        Returns:
            StageFunction: A new StageFunction representing the composed pipeline.
        """
        left = self._spec
        right = self._spec_from(other)
        merged = self._merge_specs(left, right)
        return StageFunction(
            pipeline(
                *merged.stages,
                lazy=self._lazy_from_spec(merged.lazy),
                trace=self._trace_arg_from_spec(merged.trace),
            )
        )

    def __rrshift__(self, other: "StageFunction | Callable[..., Any]") -> "StageFunction":
        """
        Implements the right-hand shift operator (>>) to compose stage functions into a pipeline.
        Args:
            other (StageFunction | Callable[..., Any]): The stage function or callable to be composed with this instance.
        Returns:
            StageFunction: A new StageFunction representing the composed pipeline.
        """
        left = self._spec_from(other)
        right = self._spec
        merged = self._merge_specs(left, right)
        return StageFunction(
            pipeline(
                *merged.stages,
                lazy=self._lazy_from_spec(merged.lazy),
                trace=self._trace_arg_from_spec(merged.trace),
            )
        )

    def __lshift__(self, other: "StageFunction | Callable[..., Any]") -> "StageFunction":
        """
        Implements the left shift (<<) operator to compose two stage functions into a new StageFunction with reversed pipeline order.
        Args:
            other (StageFunction | Callable[..., Any]): The stage function or callable to compose with this stage.
        Returns:
            StageFunction: A new StageFunction representing the composition of the two functions in reverse order.
        """
        left = self._spec
        right = self._spec_from(other)
        merged = self._merge_specs(left, right)
        return StageFunction(
            reversePipeline(
                *merged.stages,
                lazy=self._lazy_from_spec(merged.lazy),
                trace=self._trace_arg_from_spec(merged.trace),
            )
        )

    def __rlshift__(self, other: "StageFunction | Callable[..., Any]") -> "StageFunction":
        """
        Implements the reverse left shift operator (<<) to compose two stage functions in reverse order.
        Args:
            other (StageFunction | Callable[..., Any]): The function or StageFunction to compose with this instance.
        Returns:
            StageFunction: A new StageFunction representing the reverse composition of the two functions.
        """
        left = self._spec_from(other)
        right = self._spec
        merged = self._merge_specs(left, right)
        return StageFunction(
            reversePipeline(
                *merged.stages,
                lazy=self._lazy_from_spec(merged.lazy),
                trace=self._trace_arg_from_spec(merged.trace),
            )
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Stage {self.fn}>"


# Short alias
F = StageFunction


# ---
# Shaping functions' internal table builder – run only once, memoised by lru_cache
# ---
def _stepFactory(steps: int, shaper: ShapingFunction) -> ShapingFunction:
    """Create a stepped/discretized version of a shaping function."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    s = 1.0 / steps
    return lambda t, a, b: shaper(np.floor(t / s) * s, a, b)


@functools.lru_cache(maxsize=1)
def _getShapingTable() -> dict[str, ShapingFunction]:
    def _power(p: float, mode: Literal["in", "out", "inOut"] = "in") -> ShapingFunction:
        if mode == "in":
            return lambda t, a, b: a + (b - a) * t**p
        if mode == "out":
            return lambda t, a, b: a + (b - a) * (1 - (1 - t) ** p)
        return lambda t, a, b: (
            a
            + (b - a)
            * np.where(
                t < 0.5,
                (2 * t) ** p / 2,
                1 - (2 * (1 - t)) ** p / 2,
            )
        )

    linear = lambda t, a, b: a + (b - a) * t
    quadraticIn = _power(2, "in")
    quadraticOut = _power(2, "out")
    quadraticInOut = _power(2, "inOut")
    cubicIn = _power(3, "in")
    cubicOut = _power(3, "out")
    cubicInOut = _power(3, "inOut")
    exponentialIn = lambda t, a, b, p=0.35: a + (b - a) * t ** (1 / p)
    exponentialOut = lambda t, a, b, p=0.35: a + (b - a) * t**p

    def cosineInOut(t: PArrayOrScalar, a: PArrayOrScalar, b: PArrayOrScalar) -> PArrayOrScalar:
        c = b - a
        t2 = t * t
        t4 = t2 * t2
        t6 = t4 * t2
        return a + c * (4 * t6 / 9 - 17 * t4 / 9 + 22 * t2 / 9)

    sigmoid = lambda t, a, b, p=10: a + (b - a) / (1 + np.exp(-p * (t - 0.5)))

    # vector-safe seat
    def seat(t: PArrayOrScalar, a: PArrayOrScalar, b: PArrayOrScalar, p: float = 0.5) -> PArrayOrScalar:
        c = b - a
        inner = (2 * t) ** (1 - p) / 2
        outer = 1 - (2 * (1 - t)) ** (1 - p) / 2
        return a + c * np.where(t < 0.5, inner, outer)

    return {name: obj for name, obj in locals().items() if callable(obj) and name not in {"_power"}}


def shapingFunction(id: ShapingMethod, params: Optional[Number | Sequence[Number]] = None) -> ShapingFunction:
    """
    Returns a shaping function based on the provided identifier.

    This function provides various shaping functions that can be used to interpolate between two values or arrays.
    Each shaping function defines a specific way of calculating intermediate values, which can create different easing effects.

    Args:
        id (str): The identifier of the shaping function. Supported values include: "linear", "quadraticIn", "quadraticOut", "quadraticInOut", "cubicIn", "cubicOut", "cubicInOut", "exponentialIn", "exponentialOut", "cosineInOut", "sigmoid", "seat", "step"
        params (Number | Sequence[Number], optional): Parameters for certain shaping functions. For "step", this is the number of steps (default 1).

    Returns:
        Callable: A function that takes the parameters `(t, a, b, *args)` where `t` is the interpolation factor, `a` and `b`
        are the start and end values or arrays, and `*args` are optional additional parameters.

    Raises:
        KeyError: If no shaping function matches the provided identifier.

    Examples:
        >>> func = shapingFunction("linear")
        >>> func(0.5, P(0, 0), P(10, 10))
            array([5., 5.])

        >>> func = shapingFunction("quadraticIn")
        >>> func(0.5, 0, 10)
        2.5
    """
    table = _getShapingTable()

    # Handle "step" specially since it needs params
    if id == "step":
        steps = 1 if params is None else int(cast(Number, params))
        inner_shaper = table.get("linear", table["linear"])
        return _stepFactory(steps, inner_shaper)

    try:
        return table[id]
    except KeyError:
        available = sorted(list(table.keys()) + ["step"])
        raise KeyError(f"No shaping function named '{id}'. Available: {', '.join(available)}")


@partial(optional=True, placeholder=TBD)
def expand(
    pts: NumberSequenceND,
    fill: NumberSequenceND,
    repeat: int = 1,
    before: bool = False,
    axis: int = -1,
) -> PArray:
    """
    Extend an array by appending or prepending repeated values along an axis.

    Expands the given array by inserting repeated instances of `fill` along the specified axis.
    You can control the insertion position (before or after existing values), how many times
    the fill values repeat, and along which axis expansion occurs.

    Args:
        pts (NumberSequenceND): Input array-like data to expand.
        fill (NumberSequenceND): Number or array used for expansion, broadcast as needed.
        repeat (int, optional): Number of times to repeat the fill values. Defaults to 1.
        before (bool, optional): If True, insert fill values before existing data. Defaults to False.
        axis (int, optional): Axis along which to expand. Negative values count from the end.

    Returns:
        PArray: Expanded array with fill values inserted.


    Examples:
        >>> expand([1, 2, 3], 0, 3)
        array([1, 2, 3, 0, 0, 0])

        >>> expand([1, 2, 3], fill=0, repeat=2, before=True)
        array([0, 0, 1, 2, 3])

        >>> expand([[1, 2], [3, 4]], fill=9, repeat=1)
        array([[1, 2, 9],
               [3, 4, 9]])

        >>> expand([[[1], [2]], [[3], [4]]], fill=[[[0]], [[9]]], repeat=2)
        array([[[1., 0., 0.],
                [2., 0., 0.]],
               [[3., 9., 9.],
                [4., 9., 9.]]])
    """
    if repeat < 0:
        raise ValueError("repeat must be non-negative")

    arr = toPArray(pts, gpu=True)
    ndim = arr.ndim

    axis = axis if axis >= 0 else ndim + axis
    if not (0 <= axis < ndim):
        raise ValueError("axis out of range")
    if repeat == 0:
        return arr

    if ndim == 1:  # ---- 1-D fast path
        pad = np.broadcast_to(fill, (repeat,)).astype(arr.dtype, copy=False)
        pieces = (pad, arr) if before else (arr, pad)
        return np.concatenate(pieces, axis=0)

    # ---- N-D (N ≥ 2)
    pad_shape = list(arr.shape)
    pad_shape[axis] = repeat
    pad = np.broadcast_to(fill, pad_shape).astype(arr.dtype, copy=False)
    pieces = (pad, arr) if before else (arr, pad)
    return np.concatenate(pieces, axis=axis)


@partial(optional=True, placeholder=TBD)
def changeDims(
    arr: NumberSequenceND,
    target: int,
    axis: int = 0,
    maxDepthCheck: bool = True,
) -> PArray:
    """
    Adjust an array to have exactly `target` number of dimensions.

    Adds or removes singleton dimensions (axes of size 1) from the array to match the desired
    dimensionality. Use the `axis` parameter to specify where adjustments occur, similar to NumPy.
    Positive values count from the beginning, negative values count from the end.

    Args:
        arr (NumberSequenceND): Input array-like data.
        target (int): Desired number of dimensions (must be ≥ 1).
        axis (int, optional): Where to add or remove singleton axes. Negative values count from the end.
        maxDepthCheck (bool, optional): If True, enforces an upper limit (32) on the target dimensions.

    Returns:
        PArray: Array reshaped to exactly `target` dimensions.

    Raises:
        ValueError: If `target` is invalid or the requested dimensions cannot be achieved.

    Examples:
        >>> changeDims([1, 2, 3], 2)
        [[1, 2, 3]]

        >>> changeDims([[1], [2], [3]], 1)
        [1, 2, 3]

        >>> changeDims([0,0,0], target=2, axis=-1)
        [[0.], [0.], [0.]]
    """
    if target < 1:
        raise ValueError("`target` depth must be at least 1")

    if maxDepthCheck and target > 32:
        raise ValueError("`target` depth too large")

    a = toPArray(arr, gpu=True)
    ndim = a.ndim

    # no change needed
    if ndim == target:
        return a

    # add singleton axes
    if ndim < target:
        dims_to_add = target - ndim

        # Determine insertion position
        if axis < 0:
            # Negative axis: insert from end
            # -1 means insert at end, -2 means insert at second-to-last position
            insert_pos = ndim + axis + 1
            insert_pos = max(0, insert_pos)  # Clamp to valid range
        else:
            # Non-negative axis: insert from start
            # 0 means insert at beginning, 1 means insert after first dimension, etc.
            insert_pos = min(axis, ndim)

        # Build new shape by inserting singleton dimensions
        shape_list = list(a.shape)
        for _ in range(dims_to_add):
            shape_list.insert(insert_pos, 1)

        return a.reshape(tuple(shape_list))

    # remove singleton axes
    squeezeNeeded = ndim - target

    # Find singleton axes - we need to find consecutive groups
    singleton_indices = [i for i, s in enumerate(a.shape) if s == 1]

    if len(singleton_indices) < squeezeNeeded:
        raise ValueError("Cannot reach requested dims — missing singleton axes")

    # For removal, prefer working from start or end based on axis
    from_start = axis >= 0

    # Find the best consecutive group of singletons
    if from_start:
        # Look for consecutive singletons starting from the beginning of singleton_indices
        # Find the longest consecutive sequence from the start
        candidates = []
        for i in range(len(singleton_indices)):
            if i == 0 or singleton_indices[i] == singleton_indices[i - 1] + 1:
                candidates.append(singleton_indices[i])
            else:
                break

        # If we don't have enough consecutive from start, take first available ones
        if len(candidates) < squeezeNeeded:
            candidates = singleton_indices[:squeezeNeeded]
    else:
        # Look for consecutive singletons from the end
        # Find the longest consecutive sequence from the end
        candidates = []
        for i in range(len(singleton_indices) - 1, -1, -1):
            if i == len(singleton_indices) - 1 or singleton_indices[i] == singleton_indices[i + 1] - 1:
                candidates.append(singleton_indices[i])
            else:
                break
        candidates.reverse()  # Put back in ascending order

        # If we don't have enough consecutive from end, take last available ones
        if len(candidates) < squeezeNeeded:
            candidates = singleton_indices[-squeezeNeeded:]

    squeezeAxes = tuple(sorted(candidates[:squeezeNeeded]))
    squeezed = np.squeeze(a, axis=squeezeAxes)

    if squeezed.ndim != target:
        # Shouldn’t happen, but guard anyway
        raise ValueError("Squeeze failed to reach requested dimensionality")

    return squeezed


@partial(optional=True, placeholder=TBD)
def group(
    pts: NumberSequenceND,
    groupSize: int,
    padMode: PadMode = "truncate",
    pad: float = 0.0,
    axis: int = 0,
) -> PArray:
    """
    Divide an array into groups along a specified axis, handling leftover items flexibly.

    Splits the array along the given axis into equally sized groups (`groupSize`). If the array length
    isn't divisible by the group size, the leftover elements are managed according to `padMode`.

    For 1D input, the result is 2D with shape `(numGroups, groupSize)` and `axis` is ignored.

    Args:
        pts (NumberSequenceND): Input array-like data (1D or higher).
        groupSize (int): Number of elements per group. Must be positive.
        padMode (PadMode, optional): Determines handling of leftover elements --
            - "truncate": Discard leftovers (default).
            - "last": Repeat the last item.
            - "value": Use a fixed padding value (`pad`).
            - "wrap": Restart from the beginning.
            - "mirror": Repeat elements in reverse order.
        pad (float, optional): Padding value used if `padMode="value"`. Defaults to 0.0.
        axis (int, optional): Axis along which to group elements. Negative values count from the end. Ignored for 1D input.

    Returns:
        PArray: Array with an additional grouping dimension inserted at the specified axis.

    Raises:
        ValueError: If inputs are invalid, such as incorrect axis or unknown pad mode.

    Examples:
        >>> group([1, 2, 3, 4, 5], groupSize=2) # default padMode is "truncate"
        [[1, 2], [3, 4]]

        >>> group([[1, 2], [3, 4], [5, 6]], groupSize=2, padMode="value", pad=0)
        [[[1, 2], [3, 4]], [[5, 6], [0, 0]]]

        >>> group([[1, 2, 3], [4, 5, 6], [7, 8, 9]], groupSize=2, axis=1, padMode="wrap")
        [[[1, 2], [3, 1]], [[4, 5], [6, 4]], [[7, 8], [9, 7]]]
    """

    if groupSize <= 0:
        raise ValueError("groupSize must be positive")

    # Convert to array and check dims
    arr = toPArray(pts, gpu=True)

    # If 1-D, treat it as a special case: produce a 2-D result
    if arr.ndim == 1:
        n = arr.shape[0]
        tail = n % groupSize

        if tail != 0:
            if padMode == "truncate":
                data = arr[:-tail]
            else:
                missing = groupSize - tail
                if padMode == "last":
                    block = np.repeat(arr[-1:], missing)
                elif padMode == "value":
                    block = np.full((missing,), pad, dtype=arr.dtype)
                elif padMode == "wrap":
                    reps = (missing - 1) // n + 1
                    block = np.concatenate([arr] * reps)[:missing]
                elif padMode == "mirror":
                    mirrored = arr[::-1]
                    reps = (missing - 1) // n + 1
                    block = np.concatenate([mirrored] * reps)[:missing]
                else:
                    raise ValueError(f"Unknown pad mode: {padMode!r}")
                data = np.concatenate((arr, block))
        else:
            data = arr

        # Now reshape into shape (numBlocks, groupSize)
        return data.reshape(-1, groupSize)

    if arr.ndim < 2:
        raise ValueError("input should have at least two dimensions")

    # Normalize `axis` into [0..arr.ndim-1]
    axis = axis if (axis >= 0 and axis < arr.ndim) else arr.ndim + axis
    if not (0 <= axis < arr.ndim):
        raise ValueError("axis out of range")

    # Validate padMode
    valid_modes = {"truncate", "last", "value", "wrap", "mirror"}
    if padMode not in valid_modes:
        raise ValueError(f"Unknown pad mode: {padMode!r}")

    # “Swapped” has the chosen axis up front, so we can work on dimension‐0 easily.
    swapped = np.moveaxis(arr, axis, 0)  # shape: (N, …rest…)
    n = swapped.shape[0]
    tail = n % groupSize

    if tail != 0:
        if padMode == "truncate":
            # Drop the last `tail` items
            swapped = swapped[:-tail]

        else:
            missing = groupSize - tail

            if padMode == "last":
                block = np.repeat(swapped[-1:], missing, axis=0)

            elif padMode == "value":
                pad_shape = (missing, *swapped.shape[1:])
                block = np.full(pad_shape, pad, dtype=swapped.dtype)

            elif padMode == "wrap":
                # We need at least `missing` entries, so repeat `swapped` ceil(missing/n) times
                reps = (missing - 1) // n + 1
                block = np.concatenate([swapped] * reps, axis=0)[:missing]

            elif padMode == "mirror":
                mirrored = np.flip(swapped, axis=0)
                reps = (missing - 1) // n + 1
                block = np.concatenate([mirrored] * reps, axis=0)[:missing]

            else:
                # Should never happen
                raise AssertionError("Unhandled padMode")

            swapped = np.concatenate((swapped, block), axis=0)

    # Now reshape `swapped` into blocks of length `groupSize`.
    # New shape = (numBlocks, groupSize, …rest…) where numBlocks = swapped.shape[0] // groupSize.
    grouped = swapped.reshape(-1, groupSize, *swapped.shape[1:])

    # Finally, move the first two axes (block‐index and group‐size) back into the original `axis` position.
    # grouped currently has axes: [block_idx, groupSize, *original_axes_except_axis].
    # We want final to have: [… original_axes_before_axis …, block_idx, groupSize, … axes_after_axis …].
    # That is equivalent to moving axis 0 → `axis`, and axis 1 → `axis+1`.
    grouped = np.moveaxis(grouped, [0, 1], [axis, axis + 1])

    return grouped


def interleave(*arrays: PArray, axis: int = 0) -> PArray:
    """
    Combine multiple arrays by alternating their elements along a given axis.

    Takes arrays with identical shapes and merges them by interleaving their elements.
    Elements from each array alternate along the specified `axis`.

    Args:
        *arrays (PArray): 2 or more input arrays of identical shape.
        axis (int, optional): Axis along which to interleave. Negative values count from the end.

    Returns:
        PArray: Array with interleaved elements, expanded along the chosen axis.

    Raises:
        ValueError: If fewer than two arrays are provided or their shapes differ.

    Examples:
        >>> a = P(1, 3, 5)
        >>> b = P(2, 4, 6)
        >>> interleave(a, b)
        [1, 2, 3, 4, 5, 6]

        >>> x = P([1, 2], [3, 4])
        >>> y = P([5, 6], [7, 8])
        >>> interleave(x, y, axis=0)
        [[1, 2], [5, 6], [3, 4], [7, 8]]

        >>> interleave(x, y, axis=1)
        [[1, 5, 2, 6], [3, 7, 4, 8]]
    """
    if not arrays:
        raise ValueError("interleave() requires at least 2 arrays")

    k = len(arrays)
    if k == 1:
        raise ValueError("interleave() requires at least 2 arrays")

    ref_shape = arrays[0].shape
    ndim = len(ref_shape)

    axis = axis % ndim  # normalise negative axis
    if any(arr.shape != ref_shape for arr in arrays):
        raise ValueError("all input arrays must have the same shape")

    # prepare output and fill
    out_shape = list(ref_shape)
    out_shape[axis] *= k
    out = np.empty(out_shape, dtype=arrays[0].dtype)

    for idx, arr in enumerate(arrays):
        slicer = [slice(None)] * out.ndim
        slicer[axis] = slice(idx, None, k)
        out[tuple(slicer)] = arr

    return out


@partial(optional=True, placeholder=TBD)
def bound(
    pts: NumberSequenceND,
    bound1: PArrayOrScalar,
    bound2: PArrayOrScalar,
    *,
    loopBack: bool = False,
    inclusive: bool = False,
) -> PArray:
    """
    Limit or wrap array values to stay within given bounds.

    Ensures array elements fall within the range defined by `bound1` and `bound2`. The function
    automatically detects min/max regardless of their order. You can either clip out-of-bound values
    or wrap them around using modular arithmetic.

    Args:
        pts (NumberSequenceND): Input array to constrain.
        bound1 (PArrayOrScalar): One boundary of the allowed range (min or max).
        bound2 (PArrayOrScalar): Other boundary of the allowed range.
        loopBack (bool, optional): If False, clips values to bounds; if True, wraps around. Default False.
        inclusive (bool, optional): If True, includes the upper bound in the wrapping range. Default False.

    Returns:
        PArray: Array with values constrained or wrapped within specified bounds.

    Examples:
        >>> bound([1, 2, 3, 4, 5], 2, 4)
        [2., 2., 3., 4., 4.]

        >>> bound([1, 2, 3, 4, 5], 2, 4, loopBack=True)
        [3., 2., 3., 4., 3.]

        >>> bound([1, 2, 3, 4, 5], 2, 4, loopBack=True, inclusive=True)
        [4., 2., 3., 4., 2.]
    """

    ptsArr = np.asarray(pts)
    b1 = np.asarray(bound1, dtype=ptsArr.dtype)
    b2 = np.asarray(bound2, dtype=ptsArr.dtype)

    lo = np.minimum(b1, b2)
    hi = np.maximum(b1, b2)
    span = hi - lo
    if np.any(span <= 0):
        raise ValueError("Bounds must differ and define a positive span element-wise")

    # Fast clip
    if not loopBack:
        return np.clip(ptsArr, lo, hi)

    # Wrap
    if inclusive:
        span = span + 1

    # Broadcast once (views, no extra memory)
    loB = np.broadcast_to(lo, ptsArr.shape)
    spanB = np.broadcast_to(span, ptsArr.shape)

    out = ptsArr.copy()
    mask = (ptsArr < loB) | (ptsArr > (loB + spanB - 1 if inclusive else hi))
    if mask.any():
        out[mask] = ((ptsArr[mask] - loB[mask]) % spanB[mask]) + loB[mask]
    return out


def _expandTrailing(x: PArray, ndims: int) -> PArray:
    """Add `ndims` trailing singleton axes so x broadcasts with points."""
    return x[(...,) + (None,) * ndims]


@partial(optional=True, placeholder=TBD)
def interpolate(
    pts: NumberSequenceND,
    t: PArrayOrScalar,
    *,
    asPath: bool = False,
    applyAll: bool = False,
    shaping: Union[str, ShapingFunction] = "linear",
    axis: int = 0,
    clamp: bool = True,
) -> PArrayOrScalar:
    """
    Calculate interpolated points between given coordinates.

    Interpolates values smoothly between input points (`pts`) using `t` parameters.
    Supports customizable interpolation shapes (linear, quadratic, cubic, etc.), clamping, and
    different operational modes for flexibility.

    Args:
        pts (NumberSequenceND): Array of input points to interpolate between. Requires at least two.
        t (PArrayOrScalar): Interpolation parameter(s) from 0 to 1. Can be a single scalar or an array of values.
        asPath (bool, optional): If True, treats points as a continuous path. Default is False.
        applyAll (bool, optional): If True and `asPath` is False, all values of `t` applies to all point segments. Default is False.
        shaping (str or ShapingFunction, optional): Interpolation shape function name or callable.
        axis (int, optional): Axis along which input points are interpreted. Default is 0.
        clamp (bool, optional): If True, restricts `t` to [0, 1]. Default is True.

    Returns:
        PArrayOrScalar: Interpolated points matching input dimensions.

    Raises:
        ValueError: If fewer than two points are provided.

    Examples:
        >>> interpolate([[0, 0], [10, 10]], 0.5)
        [5., 5.]

        >>> interpolate([[0, 0], [10, 10], [20, 0]], [0.25, 0.75], asPath=True)
        [[ 5.,  5.], [15.,  5.]]

        >>> interpolate([[0, 0], [10, 10], [20, 0]], [0.25, 0.75], applyAll=True)
        [[[ 2.5,  2.5], [12.5,  7.5]], [[ 7.5,  7.5], [17.5,  2.5]]]
    """

    ptsArr = np.asarray(pts, dtype=float)
    ptsArr = np.moveaxis(ptsArr, axis, 0)
    if ptsArr.shape[0] < 2:
        raise ValueError("need ≥2 points")

    tArr = np.asarray(t, dtype=float).ravel()  # always 1-D
    if clamp:
        tArr = np.clip(tArr, 0.0, 1.0)

    easeFn = shapingFunction(shaping) if isinstance(shaping, str) else shaping
    coordDims = ptsArr.ndim - 1

    if asPath:  # one long path
        segCount = ptsArr.shape[0] - 1
        idx = np.clip((tArr * segCount).astype(int), 0, segCount - 1)
        localTau = _expandTrailing(tArr * segCount - idx, coordDims)
        out = easeFn(localTau, ptsArr[idx], ptsArr[idx + 1])
        return out  # path axis removed

    # pairwise mode --------
    segA, segB = ptsArr[:-1], ptsArr[1:]
    if tArr.size == 1:  # the same one t for all segments
        localTau = _expandTrailing(tArr, coordDims)
        out = easeFn(localTau, segA, segB)
    elif applyAll:  # all t's apply to all segments
        localTau = _expandTrailing(tArr, coordDims)[:, None]
        out = easeFn(localTau, segA[None], segB[None])
    else:  # each segment takes one t (cycling)
        idx = np.arange(segA.shape[0]) % tArr.size
        localTau = _expandTrailing(tArr[idx], coordDims)
        out = easeFn(localTau, segA, segB)

    return np.moveaxis(out, 0, axis)


@partial(optional=True, placeholder=TBD)
def interpolateIndices(
    start: NumberSequenceND,
    end: NumberSequenceND,
    steps: int,
    rounding: RoundingMode = "round",
) -> IntArray:
    """
    Interpolate between two points in an n-dimensional index table and return the interpolated indices.

    This function generates a series of interpolated indices between a start and an end point in an n-dimensional space.
    The resulting indices can be used to retrieve values from an index table or multidimensional array.

    Args:
        start (PArray): The starting point in n-dimensional space.
        end (PArray): The ending point in n-dimensional space.
        steps (int): The number of interpolation steps, including the start and end points.
        rounding (RoundingMode, optional): rounding method for the resulting int indices.

    Returns:
        IntArray: An int32 array of shape (steps, len(start)) containing the interpolated indices.

    Examples:
        >>> interpolateIndices(P(0,0), P(4, 4), 5)
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

        >>> table = np.arange(16).reshape(4, 4)
        >>> indices = interpolateIndices(P(0, 0), P(3, 3), 4)
        >>> table[indices[:, 0], indices[:, 1]] # or table(tuple(indices[0])) for a single value
        [0,  5, 10, 15]
    """

    if not (isinstance(steps, int) and steps >= 1):
        raise ValueError("steps must be a positive integer")

    startArr = toPArray(start, gpu=True)
    endArr = toPArray(end, gpu=True)

    if startArr.shape != endArr.shape:
        raise ValueError("start and end must have identical shape")

    # interpolation
    if steps == 1:
        result = startArr[np.newaxis, ...].copy()
    else:
        delta = (endArr - startArr) / (steps - 1)

        # t has shape (steps, 1, 1, …) to broadcast over all dims of startArr
        t = np.arange(steps, dtype=float).reshape((steps,) + (1,) * startArr.ndim)
        result = startArr + delta * t

    if rounding == "floor":
        result = np.floor(result)
    elif rounding == "ceil":
        result = np.ceil(result)
    else:  # default 'round'
        result = np.rint(result)

    return result.astype(int)


def _pointsToRegionsParam(name: str, param, targetShape: tuple) -> np.ndarray:
    """
    Broadcast param to targetShape, raising ValueError if impossible.
    """
    arr = np.asarray(param, float)
    try:
        return np.broadcast_to(arr, targetShape)
    except ValueError:
        raise ValueError(f"{name} cannot broadcast to shape {targetShape}")


@partial(optional=True, placeholder=TBD)
def pointsToRegions(
    points: NumberSequenceND,
    regionSize: PArrayOrScalar,
    anchor: PArrayLike = 0.0,
) -> PArray:
    """
    Create axis-aligned rectangular regions around given points.

    Each input point expands into a rectangular region defined by `regionSize`.
    The position of the point within each region is controlled by the fractional `anchor`.
    An anchor of 0 places the point at the region's minimum corner (eg, left or top),
    1 places it at the maximum corner, and intermediate values position it proportionally.

    Args:
        points (NumberSequenceND): Array of points, shape `(n, d)` or `(n,)` for 1D points.
        regionSize (PArrayOrScalar): Size of each region, supporting:
            - scalar: uniform size across all dimensions and points.
            - (d,): size per dimension, identical for all points.
            - (n, d): individual size per point.
        anchor (PArrayLike, optional): Fractional position of points within their regions, default `0`. Same broadcast rules as `regionSize`.

    Returns:
        PArray: Array of regions shaped `(n, 2, d)`, with each region's start (min corner) and end (max corner) coordinates.

    Examples:
        >>> pointsToRegions([[5, 5, 0]], [2, 4, 6], anchor=[.5, .5, 0])
        [[[4., 3., 0.],   # min corner
          [6., 7., 6.]]]  # max corner
    """

    pts = np.asarray(points, float)
    # support 1D points as (n,) -> (n,1)
    if pts.ndim == 1:
        pts = pts[:, None]
    if pts.ndim != 2:
        raise ValueError("points must be of shape (n, d)")
    n, d = pts.shape

    size = _pointsToRegionsParam("regionSize", regionSize, (n, d))
    anch = _pointsToRegionsParam("anchor", anchor, (n, d))

    start = pts - anch * size
    end = start + size

    return np.stack((start, end), axis=1)


def attachPositionIndex(arr: PArray, offsets: Optional[NumberSequenceND] = None) -> PArray:
    """
    Get positional indices attached to each coordinate in the array.

    Each element in the output includes its positional indices (optionally scaled by `offsets`) followed
    by the original array value.

    Args:
        arr (PArray): Input array of any dimensionality.
        offsets (NumberSequenceND, optional): Scale factors for coordinates along each axis.
            Must match the number of array dimensions.

    Returns:
        PArray: Float64 array with shape `(*arr.shape, arr.ndim + 1)`, where the last axis
        contains `[x0, x1, …, xn-1, value]` for each element.

    Examples:
        >>> arr = P(10, 20, 30)
        >>> attachPositionIndex(arr)
        array([[0., 10.],
               [1., 20.],
               [2., 30.]])

        >>> arr = P([10, 20], [30, 40])
        >>> attachPositionIndex(arr)
        array([[[0., 0., 10.],
                [0., 1., 20.]],
               [[1., 0., 30.],
                [1., 1., 40.]]])
    """
    shape = arr.shape
    ndim = arr.ndim

    # Default offsets to 1.0 if not provided
    if offsets is None:
        offsets = [1.0] * ndim
    elif len(offsets) != ndim:
        raise ValueError(f"offsets length ({len(offsets)}) must match array dimensions ({ndim})")

    # Create coordinate grids for each dimension
    coords = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")

    # Create result array with shape (*original_shape, ndim + 1)
    resultShape = shape + (ndim + 1,)
    result = np.zeros(resultShape)

    # Fill coordinate dimensions
    for i in range(ndim):
        result[..., i] = coords[i] * offsets[i]

    # Fill the value dimension
    result[..., -1] = arr

    return result
