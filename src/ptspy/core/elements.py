"""Core array helpers and partial-application utilities."""

from ptspy.typing import PArrayLike, PArray, NumberSequenceND
import functools
import inspect
from .. import np_wrapper as np
from typing import (
    Any,
    Callable,
    Optional,
    Dict,
    Tuple,
)


# Create a sentinel type that can represent any type for partial function placeholders
class _TBDType:
    """
    A singleton sentinel object used as a placeholder in partial functions.

    Compare this sentinel by identity (`is`), not by equality (`==`).
    This type is compatible with any other type for static type checking.
    """

    def __repr__(self) -> str:
        return "TBD"

    def __reduce__(self):
        """Preserve singleton identity across pickle/copy reconstruction."""
        return (_getTBDSingleton, ())


# Create a singleton instance that can be used anywhere
TBD: Any = _TBDType()


def _getTBDSingleton() -> Any:
    """Return the module-level TBD singleton."""
    return TBD


def partial(
    *, optional: bool = False, placeholder: object | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Partial function decorator that supports both positional and keyword arguments.

    This decorator transforms a function so that it can be partially applied. It returns a new function that can be
    called with the remaining arguments. If `optional` is `True`, the function can be called using default values provided by
    optional parameters, allowing for partial application with some defaults filled in. If `optional` is `False`,
    all parameters must be provided before the function is executed.

    To skip an argument, you can specify a `placeholder` object (eg, using the global `TBD` object: `@partial(placeholder=TBD)`.
    This allows you to specify which arguments to skip (eg, `fn(TBD, 1)`).

    Args:
        optional (bool, optional): If `True`, allows the function to be called with default values for any missing arguments. Default is `False`.
        placeholder (object, optional): An object used as a placeholder for arguments that are not yet provided. Use `TBD` as the placeholder. Default is None.

    Returns:
        Callable: A partially applied version of the original function.

    Examples:
        >>> @partial(optional=True)
        ... def add(a, b, c=3):
        ...     return a + b + c
        >>> add(1)(2)
        6 # because c defaults to 3 and optional is True

        >>> @partial(optional=True, placeholder=TBD) # using a placeholder
        ... def add(a, b, c=3):
        ...     return a + b + c
        >>> add(TBD, 2, 10)(0)
        12 # a=0, b=2, c=10
    """

    use_ph = placeholder is not None
    ph = placeholder  # shorthand inside decorator

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        params = tuple(sig.parameters.values())

        # Pre-compute parameter sets once (performance optimization)
        required = frozenset(
            p.name for p in params if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        )
        non_variadic = frozenset(p.name for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))

        # -------- helpers --------------------------------------------------

        def mergeKeywords(base: Dict[str, Any], inc: Dict[str, Any]) -> Dict[str, Any]:
            dup = set(base) & set(inc)
            if dup:
                raise TypeError(
                    f"{fn.__name__}() got multiple values for keyword argument(s): {', '.join(sorted(dup))}"
                )
            return {**base, **inc}

        def fillPlaceholders(existing: Tuple[Any, ...], incoming: Tuple[Any, ...]) -> Tuple[Any, ...]:
            """Replace placeholders in positional args with incoming values."""
            if not use_ph:
                return existing + incoming

            out = list(existing)
            it = iter(incoming)
            for i, val in enumerate(out):
                if val is ph:
                    try:
                        out[i] = next(it)
                    except StopIteration:
                        break
            out.extend(it)
            return tuple(out)

        def fillKeywordPlaceholders(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            """Replace keyword placeholders and merge non-conflicting keywords."""
            if not use_ph:
                return mergeKeywords(existing, incoming)

            result = dict(existing)

            # Replace placeholders with incoming values
            for key, value in existing.items():
                if value is ph and key in incoming:
                    result[key] = incoming[key]

            # Add non-conflicting new keywords
            for key, value in incoming.items():
                if key not in existing:
                    result[key] = value
                elif existing[key] is not ph and key in incoming:
                    # This is a conflict for non-placeholder values
                    raise TypeError(f"{fn.__name__}() got multiple values for keyword argument: {key}")

            return result

        def hasPlaceholdersLeft(a: Tuple[Any, ...], kw: Dict[str, Any]) -> bool:
            if not use_ph:
                return False
            return any(x is ph for x in a) or any(v is ph for v in kw.values())

        def ready(a: Tuple[Any, ...], kw: Dict[str, Any]) -> bool:
            """Check if we have enough arguments to execute the function."""
            if hasPlaceholdersLeft(a, kw):
                return False
            try:
                bound = sig.bind_partial(*a, **kw)
            except TypeError:
                return False

            supplied = set(bound.arguments)
            return required.issubset(supplied) if optional else non_variadic.issubset(supplied)

        def makeCurry(a: Tuple[Any, ...], kw: Dict[str, Any]):
            @functools.wraps(fn)
            def _curried(*na, **nkw):
                new_a = fillPlaceholders(a, na)
                new_kw = fillKeywordPlaceholders(kw, nkw)

                if ready(new_a, new_kw):
                    bound = sig.bind_partial(*new_a, **new_kw)
                    bound.apply_defaults()
                    return fn(*bound.args, **bound.kwargs)

                return makeCurry(new_a, new_kw)

            _curried.__signature__ = sig  # type: ignore[attr-defined]
            return _curried

        return makeCurry((), {})

    return decorator


def P(*args: PArrayLike) -> PArray:
    """
    Create an array of floats from array-like inputs.

    Accepts either individual numeric elements or collections (lists, tuples, arrays). If multiple arguments are given, each becomes an element or row in the resulting array. If a single collection is provided, its contents form the array directly.

    Args:
        *args (PArrayLike): Numeric elements or array-like objects to convert.

    Returns:
        PArray: An array with float values.

    Examples:
        >>> P(1, 2, 3)
        array([1., 2., 3.])

        >>> P([1, 2, 3])
        array([1., 2., 3.])

        >>> P([1, 2, 3], [3, 4, 5])
        array([[1., 2., 3.],
               [3., 4., 5.]])
    """
    if len(args) == 0:
        return np.array([], dtype=np.float32)

    first = args[0] if hasattr(args[0], "__iter__") and not isinstance(args[0], (str, bytes)) else args

    return np.array(args if len(args) > 1 else first, dtype=np.float32)


@partial(placeholder=TBD)
def P2(xs: PArrayLike, ys: PArrayLike) -> PArray:
    """
    Create a 2D grid of coordinate points from x and y arrays using ``xy`` indexing.

    Given arrays of x and y coordinates, this function generates a structured grid. Each point on the grid pairs an x-value with a y-value, arranged into a 2D array.

    Args:
        xs (PArrayLike): Array-like sequence of x-coordinates.
        ys (PArrayLike): Array-like sequence of y-coordinates.

    Returns:
        PArray: An array shaped (len(ys), len(xs), 2), containing coordinate pairs.

    Examples:
        >>> P2([1, 2, 3], [4, 5])
        array([[[1., 4.],
                [2., 4.],
                [3., 4.]],

               [[1., 5.],
                [2., 5.],
                [3., 5.]]])
    """
    X, Y = np.meshgrid(P(xs), P(ys))
    return np.dstack((X, Y))


@partial(placeholder=TBD)
def P3(xs: PArrayLike, ys: PArrayLike, zs: PArrayLike) -> PArray:
    """
    Create a 3D grid of points from arrays of x, y, and z coordinates.

    Given sequences of coordinates along three axes, this function produces a structured, cube-shaped grid. Each point in the grid combines one x, one y, and one z value, arranged in a 3D array.

    Args:
        xs (PArrayLike): Array-like sequence of x-coordinates.
        ys (PArrayLike): Array-like sequence of y-coordinates.
        zs (PArrayLike): Array-like sequence of z-coordinates.

    Returns:
        PArray: An array with shape (len(xs), len(ys), len(zs), 3), containing 3D coordinate points.
            Uses ``ij`` indexing (first axis = xs, second = ys, third = zs).

    Examples:
        >>> xs = [0, 1, 2]
        >>> ys = [0, 1]
        >>> zs = [0, 1, 2, 3]
        >>> points = P3(xs, ys, zs)
        >>> points.shape
        (3, 2, 4, 3)
    """
    X, Y, Z = np.meshgrid(P(xs), P(ys), P(zs), indexing="ij")
    return np.stack((X, Y, Z), axis=-1)


def toPArray(pts: NumberSequenceND, gpu: bool = False) -> PArray:
    """
    Convert input to a PArray (float32 ndarray). Returns a view when possible, copies only when dtype conversion is needed.

    By default, returns a CPU numpy array — suitable for Skia rendering and the
    common numpy-only workflow.  When ``gpu=True``, returns an array on the active
    backend (e.g. cupy for GPU-accelerated computation).

    Note: ``P()`` always creates arrays on the active backend. At Skia boundaries
    (drawing methods), use ``toPArray(x)`` (default CPU) rather than ``P()`` to
    avoid unnecessary GPU round-trips.

    Args:
        pts (NumberSequenceND): A list of points.
        gpu (bool): If True, return an array on the active backend. Default False.

    Returns:
        PArray: A PArray containing the points.
    """
    if gpu:
        return np.asarray(pts, dtype=np.float32)
    return np.asnumpy(pts, dtype=np.float32)


def randomizer(
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.random.Generator:
    """
    Initialize a random number generator with an optional seed and return it.

    Args:
        seed (int, optional): An integer seed for the random number generator. Default is None.
        rng (np.random.Generator, optional): Return this generator if provided.
            If `None` and `seed` is given, a new generator is created.
            Otherwise a new default generator is returned.

    Returns:
        np.random.Generator: a Generator instance.

    Examples:
        >>> rng = randomizer(42)
        >>> rng.random()
        0.7739560485559633

        >>> rng = randomizer()
        >>> rng.random()
        0.4388784397520523
    """

    if rng is not None:
        return rng

    elif seed is not None:
        return np.random.default_rng(seed)

    else:
        return np.random.default_rng()
