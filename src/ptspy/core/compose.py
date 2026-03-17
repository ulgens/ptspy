"""Composition and array-construction helpers."""

from .elements import randomizer, toPArray, partial, TBD
from .process import shapingFunction
from ptspy.typing import PArrayOrScalar, PArrayLike, PArray, IntArrayOrScalar, NumberSequenceND, ShapingFunction
from .. import np_wrapper as np
from typing import cast, Union, Optional


@partial(optional=True, placeholder=TBD)
def populate(shape: IntArrayOrScalar, start=0, increment=1) -> PArray:
    """
    Create an array of a given shape filled with sequential values.

    The array is populated starting from `start`, incrementing by `increment` with each element. If `increment` is zero, the entire array is filled uniformly with the `start` value.

    Args:
        shape (IntArrayOrScalar): Desired shape of the resulting array.
        start (Number, optional): The initial value to populate from (default is 0).
        increment (Number, optional): Increment between consecutive elements (default is 1). If it's 0, the array is filled with the `start` value.

    Returns:
        PArray: An array of specified shape, containing sequential values.

    Examples:
        >>> populate((2, 3))
        array([[0., 1., 2.],
               [3., 4., 5.]])

        >>> populate((2, 3), start=1, increment=2)
        array([[ 1.,  3.,  5.],
               [ 7.,  9., 11.]])

        >>> populate((2, 3), start=5, increment=0)
        array([[5., 5., 5.],
               [5., 5., 5.]])
    """

    numElements = int(np.prod(shape))

    if increment == 0:
        return np.full(shape, start, dtype=np.float32)

    # build by index then scale+shift
    seq = start + increment * np.arange(numElements, dtype=np.float32)
    return seq.reshape(shape)


@partial(optional=True, placeholder=TBD)
def steps(
    start: NumberSequenceND,
    end: NumberSequenceND,
    count: int,
    inclusive: bool = True,
    shaping: Union[str, ShapingFunction] = "linear",
) -> PArray:
    """
    Generate evenly spaced points between `start` and `end`.

    Produces an array of interpolated points from the given `start` to the `end` value. You specify how many points (`count`) you'd like. By default, both endpoints are included.

    Optionally, use one of the predefined shaping functions or pass your own. The steps will then be interpolated by the shaping function.

    Args:
        start (NumberSequenceND): Starting value or array of starting points.
        end (NumberSequenceND): Ending value or array of ending points.
        count (int): Number of points to generate. If 0, returns an empty array.
        inclusive (bool, optional): If True, includes start and end points. If False, excludes both endpoints. Default is True.
        shaping (str, optional): Interpolation style ('linear', etc.) or a custom [ShapingFunction](`typing/ShapingFunction`). Defaults to 'linear'.

    Returns:
        PArray: Array of interpolated points from `start` to `end`.

    Examples:
        >>> steps(0, 10, 5)
        array([ 0.,  2.5,  5.,  7.5, 10.])

        >>> steps([0, 0], [10, 5], 3)
        array([[ 0. ,  0. ],
               [ 5. ,  2.5],
               [10. ,  5. ]])

        >>> steps([1,2,3], [5], 3)
        array([[1. , 2. , 3. ],
               [3. , 3.5, 4. ],
               [5. , 5. , 5. ]])
    """

    if count < 0:
        raise ValueError("steps must be >= 0")

    startArr = toPArray(start, gpu=True)
    endArr = toPArray(end, gpu=True)

    # broadcasted base shape (raises if not broadcastable)
    baseShape = np.broadcast_shapes(np.shape(startArr), np.shape(endArr))

    # number of points to generate
    n = count if inclusive else count + 2
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if not inclusive:
        t = t[1:-1]

    if baseShape != ():
        t = t.reshape((t.size,) + (1,) * len(baseShape))  # (n, 1, 1, ..., 1)

    func = shapingFunction(shaping) if isinstance(shaping, str) else shaping
    return cast(PArray, func(t, startArr, endArr))


@partial(optional=False, placeholder=TBD)
def gridPoints(repeat: IntArrayOrScalar, spacing: PArrayOrScalar) -> PArray:
    """
    Generate a grid of points arranged in regular intervals across dimensions.

    Creates a multi-dimensional grid defined by the number of repeated steps and the spacing
    between points. Alternatively, using a table analogy, think of `repeat` as the number
    of rows or columns, and `spacing` as cell sizes. For simpler 2D or 3D grids, you can
    also use [`P2`](`core/P2`) or [`P3`](`core/P3`)

    Args:
        repeat (IntArrayOrScalar): Number of points along each dimension. Scalar implies one dimension.
        spacing (PArrayOrScalar): Distance between adjacent points. Scalar applies to all dimensions.

    Returns:
        PArray: Array of shape (total_points, dims). Axes are ordered to match `repeat`/`spacing` — first dimension varies slowest, last varies fastest (``ij`` indexing).


    Examples:
        >>> gridPoints(3, 2)
        array([[0],
               [2],
               [4]])

        >>> gridPoints([2, 3], [1, 2])
        array([[0, 0],
               [1, 0],
               [0, 2],
               [1, 2],
               [0, 4],
               [1, 4]])

        >>> gridPoints([3, 2, 2, 1], [100, 10, 1, 1])
        array([[  0,   0,   0,   0],
               [  0,   0,   1,   0],
               [  0,  10,   0,   0],
               [  0,  10,   1,   0],
               [100,   0,   0,   0],
               [100,   0,   1,   0],
               [100,  10,   0,   0],
               [100,  10,   1,   0],
               [200,   0,   0,   0],
               [200,   0,   1,   0],
               [200,  10,   0,   0],
               [200,  10,   1,   0]])
    """
    repeater = toPArray(np.atleast_1d(repeat), gpu=True)
    if np.any(repeater < 1):
        raise ValueError("repeat values must be >= 1")
    spacer = toPArray(np.atleast_1d(spacing), gpu=True)
    dims = repeater.size

    # If spacing is a single value but repeat has multiple dims, broadcast
    if len(spacer) == 1 and dims > 1:
        spacer = np.full((dims,), spacer[0], dtype=float)
    elif len(spacer) != dims:
        raise ValueError(f"Incompatible: repeat has {dims} dims, but spacing has {spacer.size}.")

    # Build axes. r is the number of points along that dimension and s is the cell size
    # e.g. if r=3 and s=5 => axis = [0, 5, 10]
    axes = [np.arange(r, dtype=np.float32) * s for r, s in zip(repeater, spacer)]

    # Create the mesh with 'ij' so that the first axis corresponds to repeat_arr[0].
    # Note: if memory becomes an issue, we can build the mesh by looping through dims
    mesh = np.meshgrid(*axes, indexing="ij")

    # mesh is a list of length dims; each element has shape (r0, r1, ..., r_{dims-1})
    # We stack them along a new last axis => shape (..., dims)
    stacked = np.stack(mesh, axis=-1)

    # Flatten to shape (total_points, dims)
    return stacked.reshape(-1, dims)


@partial(optional=True, placeholder=TBD)
def random(
    shape: Optional[PArrayLike] = None,
    max=1.0,
    min=0.0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> PArray:
    """
    Create an array of random numbers within a specified range.

    Generates uniformly distributed random floats between `min` (inclusive) and `max` (exclusive). If `shape` is provided, returns an array with that shape; otherwise, returns a single float. An optional `seed` ensures reproducible results.

    Args:
        shape (PArrayLike, optional): Shape of the array to generate. If None, returns a scalar float32.
        max (float, optional): Upper bound for random numbers (exclusive). Defaults to 1.0.
        min (float, optional): Lower bound for random numbers (inclusive). Defaults to 0.0.
        seed (int, optional): Seed for reproducibility. Defaults to None.
        rng (np.random.Generator, optional): If provided, this generator is used
            for sampling. If `None` and `seed` is given, a new generator is
            created with that seed. Otherwise a new default generator is used.

    Returns:
        PArray: Array of random floats within `min` and `max` or a single float if `shape` is None.

    Examples:
        >>> random()
        0.77395605

        >>> random((2, 3))
        array([[0.5488135 , 0.71518937, 0.60276338],
               [0.54488318, 0.4236548 , 0.64589411]])

        >>> random((2, 3), max=10, min=5)
        array([[5.53716201, 8.3015781 , 6.28250546],
               [9.57844397, 6.65108716, 9.55465934]])
    """

    rand = randomizer(seed, rng)

    if shape is None:
        return rand.random(dtype=np.float32)

    return np.asarray(rand.uniform(min, max, shape), dtype=np.float32)


@partial(optional=True, placeholder=TBD)
def irregular(
    maxValue: float, steps: int, rng: np.random.Generator = None, shaping: Union[str, ShapingFunction] = "linear"
) -> PArray:
    """
    Generate irregularly spaced numbers between 0 and a maximum value.

    Produces an array of `steps` points randomly spaced between 0 and `maxValue`. Points are
    unevenly spaced yet well-distributed across the range.

    Args:
        maxValue (float): Upper limit for generated values.
        steps (int): Number of points to create.
        rng (np.random.Generator, optional): RNG to use. If None, a new default RNG is created.
        shaping (str or ShapingFunction, optional): Shaping function name or callable for jitter.

    Returns:
        PArray: Array of irregularly spaced floats in [0, `maxValue`) (upper bound exclusive).

    Examples:
        >>> irregular(10, 5)
        array([0.7, 2.4, 5.1, 7.2, 9.5])
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if maxValue < 0:
        raise ValueError("maxValue must be >= 0")
    if steps == 0:
        return np.empty((0,), dtype=np.float32)

    randGen = rng or np.random.default_rng()

    scale = np.float32(maxValue) / steps
    bins = np.arange(steps, dtype=np.float32)  # 0..steps-1
    jitter = randGen.random(steps)  # 0-1

    shaper = shapingFunction(shaping) if isinstance(shaping, str) else shaping
    u = np.asarray(shaper(jitter, 0.0, 1.0), dtype=np.float32)

    # Keep u normalized to ensure each sample stays within its bin
    eps = np.finfo(np.float32).eps
    u = np.clip(u, 0.0, 1.0 - eps)

    result = (bins + u) * scale
    return result.astype(np.float32, copy=False)
