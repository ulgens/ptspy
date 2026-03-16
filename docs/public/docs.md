# Pts.py API Documentation

This documentation is auto-generated from source code docstrings.

## Type Definitions

### Number
Either an int value or a float value
```python
Number = int | float | np.intp | np.float32 | np.float64
```

### StrokeJoinCap
Names for stroke joins and caps
```python
StrokeJoinCap = Literal["bevel", "round", "miter", "last"]
```

### PointStyle
Visual styles for rendering points
```python
PointStyle = Literal["circle", "rect"]
```

### PArray
A numpy-like array of floats (float32 for GPU compatibility and Skia alignment)
```python
PArray = NDArray[np.float32]
```

### PArrayOrScalar
Either a float value or a numpy-like array of floats
```python
PArrayOrScalar = float | np.float32 | PArray
```

### IntArray
A numpy-like array of ints
```python
IntArray = NDArray[np.intp]
```

### IntArrayOrScalar
Either an int value or a numpy-like array of ints
```python
IntArrayOrScalar = int | np.intp | IntArray
```

### IntArrayOrList
Either a numpy-like array or a python sequence of ints
```python
IntArrayOrList = IntArray | Sequence[int]
```

### NumberSequence1D
A numpy-like 1D array, list or tuple of floats or ints
```python
NumberSequence1D = PArray | Sequence[Number]
```

### NumberSequence2D
A numpy-like 2D array, list or tuple of floats or ints
```python
NumberSequence2D = PArray | Sequence[Sequence[Number]]
```

### NumberSequenceND
A numpy-like ND array, list or tuple of floats or ints
```python
NumberSequenceND = "PArray | Sequence[Number | NumberSequenceND]"
```

### PArrayLike
Same as ArrayLike from numpy.typing
```python
PArrayLike = ArrayLike
```

### ColorLike
A flexible color type: Either an RGB/RGBA array, an RGB/RGBA int, or a string
```python
ColorLike = int | NumberSequence1D | str
```

### ColorOrShader
A color value or a shader
```python
ColorOrShader = ColorLike | skia.Shader
```

### ShapingFunction
A shaping function for interpolation
```python
ShapingFunction = Callable[[PArrayOrScalar, PArrayOrScalar, PArrayOrScalar], PArrayOrScalar]
```

### ShapingMethod
Shaping method identifiers for interpolation
```python
ShapingMethod = Literal[
```

### GradientStyle
Gradient styles for color gradients
```python
GradientStyle = Literal["linear", "radial", "sweep", "conical"]
```

### PadMode
Padding modes for various operations
```python
PadMode = Literal["last", "value", "wrap", "mirror", "truncate"]
```

### RoundingMode
Rounding modes for various operations
```python
RoundingMode = Literal["floor", "ceil", "round"]
```

# Package: core

## Functions

### reset

Switch the active numeric backend.

Imports `library_name` and assigns it to the global `_np_module` , letting you swap between libraries with NumPy-compatible APIs (e.g., `numpy` and `cupy` ).

Overrides are cleared on reset to avoid carrying backend-specific override behavior into a different numeric backend.

**Arguments:**
- `library_name` (str): The name of the library to import and set as `_np_module` .

**Example:**
```python
>>> reset('numpy')
>>> type(_np_module)
<class 'module'>

>>> reset('cupy')
>>> type(_np_module)
<class 'module'>
```

### override

Override a function in the current numerical library with a custom implementation.

Replaces a function on the active backend so calls to `name` will use your implementation instead.

If `name` does not exist on the active backend, a warning is emitted and the override is still stored.

**Arguments:**
- `name` (str): The name of the function you want to override.
- `func` (Callable): The custom callable that will override the original.

**Example:**
```python
def custom_linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    print(f"Custom linspace called with start={start}, stop={stop}, num={num}")
    return _np_module.linspace(start, stop, num, endpoint, retstep, dtype, axis)

override('linspace', custom_linspace)

# Now, when linspace is called, it uses the custom_linspace implementation

# Restore the original backend implementation:
override('linspace', getattr(_np_module, 'linspace'))
```

# Package: more

## Functions

### poissonDisc

Generate blue-noise distributed points via Bridson's Poisson-Disc sampling.

The algorithm uses a background grid with 5^d neighbor offsets, so dimensions above ~4 become expensive. For d=3 there are 125 neighbor cells checked per candidate; for d=5 there are 3125.

Returns an (N, d) array where every pair of points is at least `radius` apart.

**Arguments:**
- `bounds` (NumberSequenceND): Domain extent. 1-D for size-only (origin at zero), 2D for a `[min, max]` bounding rectangle. Higher dimensions can be expensive.
- `radius` (flaot): Minimum distance between any two points.
- `k` (int, Optional): Candidate samples per active point (quality knob). Default 15.
- `seed` (int, Optional): RNG seed for reproducibility.
- `rng` (np.random.Generator, Optional): Pass a pre-existing `numpy.random.Generator` if needed.

**Returns:** PArray - Array of shape `(N, d)` with the sampled points.

### createPerlinNoise2D

Create a 2D Perlin noise generator with the specified grid size.

This function creates a PerlinNoise object that can generate 2D Perlin noise patterns. The noise generator maintains its own permutation table and gradient vectors, allowing for consistent noise generation across multiple calls.

**Arguments:**
- `size` (IntArrayOrList): The dimensions of the noise grid as [width, height].
- `seed` (Optional[int], optional): Random seed for reproducible noise patterns.
- `grads` (Optional[np.ndarray], optional): Custom gradient table. If None, uses the default 8-direction gradient table.

**Returns:** PerlinNoise - An instance of the Perlin noise generator

**Example:**
```python
>>> noise = createPerlinNoise2D((256, 256), seed=123)
>>> value = noise(freq=2.0, octaves=2) # same as noise.generate(...)
```

### skiaPerlinNoise

Generates a Perlin noise shader using skia.

This function creates a Perlin noise shader using either fractal noise or turbulence noise. The generated shader can be used for various visual effects in graphics and design. You can then use `form.fill` to apply the shader.

**Arguments:**
- `freqX` (float): The frequency of the noise in the X direction.
- `freqY` (float): The frequency of the noise in the Y direction.
- `octaves` (int): The number of octaves to use for the noise. Higher values result in more detail.
- `seed` (float): The seed value for randomization. Changing this value will result in different noise patterns.
- `tileSize` (skia.ISize, optional): The tile size for the noise. If provided, the noise will repeat seamlessly at this size.
- `type` (str, optional): The type of Perlin noise to generate. Can be "fractal" for fractal noise or "turbulence" for turbulence noise. Default is "fractal".

**Returns:** skia.Shader - A Skia shader object that generates Perlin noise.

**Example:**
```python
>>> shader = skiaPerlinNoise(0.01, 0.01, 2, 42.0)
>>> isinstance(shader, skia.Shader)
True

>>> shader_turbulence = skiaPerlinNoise(0.01, 0.01, 2, 42.0, type="turbulence")
>>> isinstance(shader_turbulence, skia.Shader)
True
```

## Classes

## PerlinNoise

2D Perlin noise generator class. Typically created via `createPerlinNoise2D` function.

### PerlinNoise.resetGrid

Reset the internal grid to its original state.

### PerlinNoise.generate

Sample one step of 2D Perlin noise or FBM over the internal grid with an optional offset. If octaves > 1, FBM is used.

**Arguments:**
- `offset` (NumberSequence1D, optional): Offset to apply to the internal grid before sampling. Use this to animate or shift the noise pattern.
- `freq` (float, NumberSequence1D, optional): Base frequency either as a scalar or set x and y separately.
- `octaves` (int, optional): Number of noise layers to sum.
- `lacunarity` (float, optional): Frequency multiplier per octave.
- `gain` (float, optional): Amplitude multiplier per octave.
- `period` (Optional[IntArrayOrList], optional): Tiling period for Perlin sampling.
- `autoTile` (bool, optional): If True, adjusts per-octave periods to preserve seamless tiling across octaves. Ignored if `period` is None.

**Returns:** PArray - noise values

**Example:**
```python
>>> noise = createPerlinNoise2D((256, 256))
>>> # t can be a frame index or time value
>>> frame = noise.generate(offset=P(t * 0.1, t * 0.1), freq=0.4, octaves=1)
>>> # shorthand, same as above:
>>> frame = noise(offset=P(t * 0.1, t * 0.1), freq=0.4, octaves=1)
```

### PerlinNoise.sample

Sample 2D Perlin noise.

**Arguments:**
- `pts` (PArray): Sample coordinates of shape `(..., 2)` .
- `freq` (float, NumberSequence1D, optional): Frequency either as a scalar or set x and y separately.
- `period` (Optional[IntArrayOrList], optional): Tiling period for Perlin sampling.
- `autoTile` (bool, optional): If True, adjusts per-octave periods to preserve seamless tiling across octaves. Ignored if `period` is None.

**Returns:** PArray - noise values

**Example:**
```python
>>> values = noise.sample(P2(np.linspace(0,1,128), np.linspace(0,1,128)))
>>> values.shape
(128, 128)
```

### PerlinNoise.sampleFBM

Sample 2D Perlin fractal Brownian motion (fBM) over a set of points.

**Arguments:**
- `pts` (PArray): Sample coordinates of shape `(..., 2)` .
- `freq` (float, NumberSequence1D, optional): Base frequency either as a scalar or set x and y separately.
- `octaves` (int, optional): Number of noise layers to sum.
- `lacunarity` (float, optional): Frequency multiplier per octave.
- `gain` (float, optional): Amplitude multiplier per octave.
- `period` (Optional[IntArrayOrList], optional): Tiling period for Perlin sampling.
- `autoTile` (bool, optional): If True, adjusts per-octave periods to preserve seamless tiling across octaves. Ignored if `period` is None.

**Returns:** PArray - fBM values

**Example:**
```python
>>> fbm = noise.sampleFBM(P2(np.linspace(0,1,128), np.linspace(0,1,128)))
>>> fbm.shape
(128, 128)
```

# Package: core

## Functions

### partial

Partial function decorator that supports both positional and keyword arguments.

This decorator transforms a function so that it can be partially applied. It returns a new function that can be called with the remaining arguments. If `optional` is `True` , the function can be called using default values provided by optional parameters, allowing for partial application with some defaults filled in. If `optional` is `False` , all parameters must be provided before the function is executed.

To skip an argument, you can specify a `placeholder` object (eg, using the global `TBD` object: `@partial(placeholder=TBD)` . This allows you to specify which arguments to skip (eg, `fn(TBD, 1)` ).

**Arguments:**
- `optional` (bool, optional): If `True` , allows the function to be called with default values for any missing arguments. Default is `False` .
- `placeholder` (object, optional): An object used as a placeholder for arguments that are not yet provided. Use `TBD` as the placeholder. Default is None.

**Returns:** Callable - A partially applied version of the original function.

**Example:**
```python
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
```

### P

Create an array of floats from array-like inputs.

Accepts either individual numeric elements or collections (lists, tuples, arrays). If multiple arguments are given, each becomes an element or row in the resulting array. If a single collection is provided, its contents form the array directly.

**Arguments:**
- `*args` (PArrayLike): Numeric elements or array-like objects to convert.

**Returns:** PArray - An array with float values.

**Example:**
```python
>>> P(1, 2, 3)
array([1., 2., 3.])

>>> P([1, 2, 3])
array([1., 2., 3.])

>>> P([1, 2, 3], [3, 4, 5])
array([[1., 2., 3.],
       [3., 4., 5.]])
```

### P2

Create a 2D grid of coordinate points from x and y arrays using ` `xy` ` indexing.

Given arrays of x and y coordinates, this function generates a structured grid. Each point on the grid pairs an x-value with a y-value, arranged into a 2D array.

**Arguments:**
- `xs` (PArrayLike): Array-like sequence of x-coordinates.
- `ys` (PArrayLike): Array-like sequence of y-coordinates.

**Returns:** PArray - An array shaped (len(ys), len(xs), 2), containing coordinate pairs.

**Example:**
```python
>>> P2([1, 2, 3], [4, 5])
array([[[1., 4.],
        [2., 4.],
        [3., 4.]],

       [[1., 5.],
        [2., 5.],
        [3., 5.]]])
```

### P3

Create a 3D grid of points from arrays of x, y, and z coordinates.

Given sequences of coordinates along three axes, this function produces a structured, cube-shaped grid. Each point in the grid combines one x, one y, and one z value, arranged in a 3D array.

**Arguments:**
- `xs` (PArrayLike): Array-like sequence of x-coordinates.
- `ys` (PArrayLike): Array-like sequence of y-coordinates.
- `zs` (PArrayLike): Array-like sequence of z-coordinates.

**Returns:** PArray - An array with shape (len(xs), len(ys), len(zs), 3), containing 3D coordinate points. Uses ` `ij` ` indexing (first axis = xs, second = ys, third = zs).

**Example:**
```python
>>> xs = [0, 1, 2]
>>> ys = [0, 1]
>>> zs = [0, 1, 2, 3]
>>> points = P3(xs, ys, zs)
>>> points.shape
(3, 2, 4, 3)
```

### toPArray

Convert input to a PArray (float32 ndarray). Returns a view when possible, copies only when dtype conversion is needed.

**Arguments:**
- `pts` (NumberSequenceND): A list of points.

**Returns:** PArray - A PArray containing the points.

### randomizer

Initialize a random number generator with an optional seed and return it.

**Arguments:**
- `seed` (int, optional): An integer seed for the random number generator. Default is None.
- `rng` (np.random.Generator, optional): Return this generator if provided. If `None` and `seed` is given, a new generator is created. Otherwise a new default generator is returned.

**Returns:** np.random.Generator - a Generator instance.

**Example:**
```python
>>> rng = randomizer(42)
>>> rng.random()
0.7739560485559633

>>> rng = randomizer()
>>> rng.random()
0.4388784397520523
```

### toRadian

Convert degrees to radians.

**Arguments:**
- `degrees` (PArrayOrScalar): The angle in degrees. Can be a single scalar value or an array of values.

**Returns:** PArrayOrScalar - The angle in radians, with the same shape as the input.

**Example:**
```python
>>> toRadian(180)
3.141592653589793

>>> toRadian(P(0, 90, 180, 270, 360))
array([0.        , 1.57079633, 3.14159265, 4.71238898, 6.28318531])
```

### toDegree

Convert radians to degrees.

**Arguments:**
- `radians` (PArrayOrScalar): The angle in radians. Can be a single scalar value or an array of values.

**Returns:** PArrayOrScalar - The angle in degrees, with the same shape as the input.

**Example:**
```python
>>> toDegree(3.141592653589793)
180.0

>>> toDegree(P(0, 1.57079633, 3.14159265, 4.71238898, 6.28318531))
array([  0.,  90., 180., 270., 360.])
```

### regularPolygon

Generate the vertices of a regular polygon.

Computes evenly spaced points around a circle centered at `center` with the given

`radius` and number of `sides` . The first vertex is at angle 0 (positive x-axis) and vertices proceed counter-clockwise. When radius is 1 and the center is [0, 0], the polygon is inscribed in a unit circle, which is useful for normalization and scaling.

**Arguments:**
- `center` (NumberSequence1D): Polygon center coordinates `[x, y]` .
- `sides` (int): Number of sides (and vertices) of the polygon. Must be at least 3.
- `radius` (int): Distance from center to each vertex.
- `closed` (bool, optional): If `True` , the polygon is closed by repeating the first vertex.

**Returns:** PArray - Array of shape `(sides, 2)` containing the polygon's vertex coordinates.

**Example:**
```python
>>> regularPolygon([0, 0], sides=4, radius=1)
array([[ 1. ,  0. ],
       [ 0. ,  1. ],
       [-1. ,  0. ],
       [ 0. , -1. ]])
```

### translate2D

Translate (shift) 2D points by an offset vector.

**Arguments:**
- `pts` (NumberSequenceND): Points with shape (..., 2).
- `offset` (NumberSequenceND): Translation vector [dx, dy].

**Returns:** PArray - Translated points with the same shape as input.

**Example:**
```python
>>> translate2D(P(1, 2), P(10, 20))
array([11., 22.])
```

### rotate2D

Rotate 2D points clockwise by an angle in radians (screen coordinates, +y down).

**Arguments:**
- `pts` (NumberSequenceND): Points with shape (..., 2).
- `angle` (float): Rotation angle in radians (clockwise in screen space).
- `origin` (Optional[NumberSequenceND]): Center of rotation [x, y]. None means [0, 0].

**Returns:** PArray - Rotated points with the same shape as input.

**Example:**
```python
>>> rotate2D(P(1, 0), PI / 2)
array([0., 1.])
```

### scale2D

Scale 2D points by a factor from an origin.

**Arguments:**
- `pts` (NumberSequenceND): Points with shape (..., 2).
- `factor` (NumberSequenceND): Scalar for uniform scale, or [sx, sy] for non-uniform.
- `origin` (Optional[NumberSequenceND]): Center of scaling [x, y]. None means [0, 0].

**Returns:** PArray - Scaled points with the same shape as input.

**Example:**
```python
>>> scale2D(P(1, 2), 3)
array([3., 6.])
```

### skew2D

Skew (shear) 2D points by factors [sx, sy].

**Arguments:**
- `pts` (NumberSequenceND): Points with shape (..., 2).
- `shear` (NumberSequenceND): Shear factors [sx, sy].
- `origin` (Optional[NumberSequenceND]): Center of shear [x, y]. None means [0, 0].

**Returns:** PArray - Skewed points with the same shape as input.

**Example:**
```python
>>> skew2D(P(1, 0), P(0.5, 0))
array([1., 0.])
```

### reflect2D

Reflect 2D points across a line through the origin (or a given point) at the specified angle.

The reflection axis is a line at `angle` radians from the positive x-axis.

`angle=0` reflects across the x-axis (flips y) and `angle=PI/2` reflects across the y-axis (flips x).

**Arguments:**
- `pts` (NumberSequenceND): Points with shape (..., 2).
- `angle` (float): Angle of the reflection axis in radians. Defaults to 0 (x-axis).
- `origin` (Optional[NumberSequenceND]): Center of reflection [x, y]. None means [0, 0].

**Returns:** PArray - Reflected points with the same shape as input.

**Example:**
```python
>>> reflect2D(P(1, 1), 0)
array([ 1., -1.])
```

### transform2D

Apply a general affine transform to 2D points.

**Arguments:**
- `pts` (NumberSequenceND): Points with shape (..., 2).
- `matrix` (NumberSequenceND): Either a (2, 2) linear matrix or a (3, 3) affine matrix.

**Returns:** PArray - Transformed points with the same shape as input.

**Example:**
```python
>>> transform2D(P(1, 0), np.eye(2))
array([1., 0.])
```

### hermiteToBezier

Convert Hermite spline data (positions + tangent vectors) into cubic Bezier control points.

**Arguments:**
- `pts` (NumberSequence2D): Shape (n, 2) — positions the curve passes through.
- `tangents` (NumberSequence2D): Shape (n, 2) — tangent vectors at each point.
- `closed` (bool): If True, adds a final segment back to the first point.

**Returns:** PArray - array of shape (m, 2) like `[start, cp1, cp2, end, cp1, cp2, end, ...]`

### cardinalToBezier

Convert cardinal spline control points into cubic Bezier control points.

**Arguments:**
- `pts` (NumberSequence2D): Shape (n, 2) — the points the curve passes through.
- `tension` (PArrayOrScalar): Scalar (default 0.5 = Catmull-Rom) or array of shape (n,) for per-point tension. 1.0 produces straight lines.
- `closed` (bool): If True, wraps tangents and adds a final segment back to the start.

**Returns:** PArray - array of shape (m, 2) like `[start, cp1, cp2, end, cp1, cp2, end, ...]`

### sampleCardinal

Evaluate points along a cardinal spline curve using the Hermite basis matrix.

**Arguments:**
- `pts` (NumberSequence2D): Shape (n, 2) — control points.
- `tension` (PArrayOrScalar): Scalar or per-point array of shape (n,).
- `numSamples` (int): Samples per segment.
- `closed` (bool): If True, the curve loops back to the start.

**Returns:** PArray - Shape (totalSamples, 2) of evaluated positions.

### bsplineToBezier

Convert uniform cubic B-spline control points into cubic Bezier control points.

**Arguments:**
- `pts` (NumberSequence2D): Shape (n, 2) B-spline control points. Minimum 4 for open, 3 for closed.
- `closed` (bool): If True, wraps control points and adds a final segment back to the start.

**Returns:** PArray - array of shape (m, 2) like `[start, cp1, cp2, end, cp1, cp2, end, ...]`

### sampleBspline

Evaluate points along a uniform cubic B-spline using the basis matrix.

**Arguments:**
- `pts` (NumberSequence2D): Shape (n, 2) — B-spline control points. Minimum 4 for open, 3 for closed.
- `numSamples` (int): Samples per segment.
- `closed` (bool): If True, the curve loops back to the start.

**Returns:** PArray - Shape (totalSamples, 2) of evaluated positions.

### sampleBezier

Sample points along a cubic Bezier curve.

**Arguments:**
- `pts` (NumberSequence2D): array of shape (m, 2) like `[start, cp1, cp2, end, cp1, cp2, end, ...]`
- `numSamples` (int): Samples per segment.

**Returns:** PArray - Shape (totalSamples, 2) of evaluated positions.

### unpackResult

Wrap a stage so that a tuple result from the previous stage is unpacked into positional arguments: ` `fn(*result)` ` instead of ` `fn(result)` `.

This overrides the default auto-unpack behavior of pipelines, giving explicit control when a stage expects a tuple as a single argument.

### pipeline

Chain functions left-to-right. Each output feeds into the next input like in a pipeline.

Works with sync and async functions. Tuple returns are automatically unpacked as multiple arguments for the next function, enabling easy multi-value passing. Use `lazy=True` to yield intermediate results, or `trace=True` to debug each step.

For operator syntax, use `F` : `F(a) >> b >> c` equals `pipeline(a, b, c)` .

**Arguments:**
- `*funcs` (Callable[..., Any]): Functions to execute in sequence.
- `lazy` (bool): If True, yields intermediate results as a generator.
- `trace` (bool): If True, prints each function's name and output.

**Returns:** Callable - A composed function (async if any input is async).

**Example:**
```python
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
```

### reversePipeline

Chain functions right-to-left. It has the exact same features as `pipeline()` , but executes in reverse order.

Useful when you want to read the composition in traditional mathematical order: `f(g(x))` .

For operator syntax, use `F` : `F(a) << b << c` equals `reversePipeline(a, b, c)` .

**Arguments:**
- `*funcs` (Callable[..., Any]): Functions to compose (executed last-to-first).
- `lazy` (bool): If True, yields intermediate results as a generator.
- `trace` (bool): If True, prints each function's name and output.

**Returns:** Callable - A composed function (async if any input is async).

**Example:**
```python
>>> def double(x): return x * 2
>>> def add_one(x): return x + 1
>>> f = reversePipeline(double, add_one)
>>> f(3)
8  # double(add_one(3)) = (3 + 1) * 2

>>> # Operator syntax
>>> (F(double) << add_one)(3)
8
```

### shapingFunction

Returns a shaping function based on the provided identifier.

This function provides various shaping functions that can be used to interpolate between two values or arrays. Each shaping function defines a specific way of calculating intermediate values, which can create different easing effects.

**Arguments:**
- `id` (str): The identifier of the shaping function. Supported values include: "linear", "quadraticIn", "quadraticOut", "quadraticInOut", "cubicIn", "cubicOut", "cubicInOut", "exponentialIn", "exponentialOut", "cosineInOut", "sigmoid", "seat", "step"
- `params` (Number, Sequence[Number], optional): Parameters for certain shaping functions. For "step", this is the number of steps (default 1).

**Returns:** Callable - A function that takes the parameters `(t, a, b, *args)` where `t` is the interpolation factor, `a` and `b` are the start and end values or arrays, and `*args` are optional additional parameters.

**Example:**
```python
>>> func = shapingFunction("linear")
>>> func(0.5, P(0, 0), P(10, 10))
    array([5., 5.])

>>> func = shapingFunction("quadraticIn")
>>> func(0.5, 0, 10)
2.5
```

### expand

Extend an array by appending or prepending repeated values along an axis.

Expands the given array by inserting repeated instances of `fill` along the specified axis. You can control the insertion position (before or after existing values), how many times the fill values repeat, and along which axis expansion occurs.

**Arguments:**
- `pts` (NumberSequenceND): Input array-like data to expand.
- `fill` (NumberSequenceND): Number or array used for expansion, broadcast as needed.
- `repeat` (int, optional): Number of times to repeat the fill values. Defaults to 1.
- `before` (bool, optional): If True, insert fill values before existing data. Defaults to False.
- `axis` (int, optional): Axis along which to expand. Negative values count from the end.

**Returns:** PArray - Expanded array with fill values inserted.

**Example:**
```python
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
```

### changeDims

Adjust an array to have exactly `target` number of dimensions.

Adds or removes singleton dimensions (axes of size 1) from the array to match the desired dimensionality. Use the `axis` parameter to specify where adjustments occur, similar to NumPy. Positive values count from the beginning, negative values count from the end.

**Arguments:**
- `arr` (NumberSequenceND): Input array-like data.
- `target` (int): Desired number of dimensions (must be ≥ 1).
- `axis` (int, optional): Where to add or remove singleton axes. Negative values count from the end.
- `maxDepthCheck` (bool, optional): If True, enforces an upper limit (32) on the target dimensions.

**Returns:** PArray - Array reshaped to exactly `target` dimensions.

**Example:**
```python
>>> changeDims([1, 2, 3], 2)
[[1, 2, 3]]

>>> changeDims([[1], [2], [3]], 1)
[1, 2, 3]

>>> changeDims([0,0,0], target=2, axis=-1)
[[0.], [0.], [0.]]
```

### group

Divide an array into groups along a specified axis, handling leftover items flexibly.

Splits the array along the given axis into equally sized groups ( `groupSize` ). If the array length isn't divisible by the group size, the leftover elements are managed according to `padMode` .

For 1D input, the result is 2D with shape `(numGroups, groupSize)` and `axis` is ignored.

**Arguments:**
- `pts` (NumberSequenceND): Input array-like data (1D or higher).
- `groupSize` (int): Number of elements per group. Must be positive.
- `padMode` (PadMode, optional): Determines handling of leftover elements -- - "truncate": Discard leftovers (default). - "last": Repeat the last item. - "value": Use a fixed padding value ( `pad` ). - "wrap": Restart from the beginning. - "mirror": Repeat elements in reverse order.
- `pad` (float, optional): Padding value used if `padMode="value"` . Defaults to 0.0.
- `axis` (int, optional): Axis along which to group elements. Negative values count from the end. Ignored for 1D input.

**Returns:** PArray - Array with an additional grouping dimension inserted at the specified axis.

**Example:**
```python
>>> group([1, 2, 3, 4, 5], groupSize=2) # default padMode is "truncate"
[[1, 2], [3, 4]]

>>> group([[1, 2], [3, 4], [5, 6]], groupSize=2, padMode="value", pad=0)
[[[1, 2], [3, 4]], [[5, 6], [0, 0]]]

>>> group([[1, 2, 3], [4, 5, 6], [7, 8, 9]], groupSize=2, axis=1, padMode="wrap")
[[[1, 2], [3, 1]], [[4, 5], [6, 4]], [[7, 8], [9, 7]]]
```

### interleave

Combine multiple arrays by alternating their elements along a given axis.

Takes arrays with identical shapes and merges them by interleaving their elements. Elements from each array alternate along the specified `axis` .

**Arguments:**
- `*arrays` (PArray): 2 or more input arrays of identical shape.
- `axis` (int, optional): Axis along which to interleave. Negative values count from the end.

**Returns:** PArray - Array with interleaved elements, expanded along the chosen axis.

**Example:**
```python
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
```

### bound

Limit or wrap array values to stay within given bounds.

Ensures array elements fall within the range defined by `bound1` and `bound2` . The function automatically detects min/max regardless of their order. You can either clip out-of-bound values or wrap them around using modular arithmetic.

**Arguments:**
- `pts` (NumberSequenceND): Input array to constrain.
- `bound1` (PArrayOrScalar): One boundary of the allowed range (min or max).
- `bound2` (PArrayOrScalar): Other boundary of the allowed range.
- `loopBack` (bool, optional): If False, clips values to bounds; if True, wraps around. Default False.
- `inclusive` (bool, optional): If True, includes the upper bound in the wrapping range. Default False.

**Returns:** PArray - Array with values constrained or wrapped within specified bounds.

**Example:**
```python
>>> bound([1, 2, 3, 4, 5], 2, 4)
[2., 2., 3., 4., 4.]

>>> bound([1, 2, 3, 4, 5], 2, 4, loopBack=True)
[3., 2., 3., 4., 3.]

>>> bound([1, 2, 3, 4, 5], 2, 4, loopBack=True, inclusive=True)
[4., 2., 3., 4., 2.]
```

### interpolate

Calculate interpolated points between given coordinates.

Interpolates values smoothly between input points ( `pts` ) using `t` parameters. Supports customizable interpolation shapes (linear, quadratic, cubic, etc.), clamping, and different operational modes for flexibility.

**Arguments:**
- `pts` (NumberSequenceND): Array of input points to interpolate between. Requires at least two.
- `t` (PArrayOrScalar): Interpolation parameter(s) from 0 to 1. Can be a single scalar or an array of values.
- `asPath` (bool, optional): If True, treats points as a continuous path. Default is False.
- `applyAll` (bool, optional): If True and `asPath` is False, all values of `t` applies to all point segments. Default is False.
- `shaping` (str or ShapingFunction, optional): Interpolation shape function name or callable.
- `axis` (int, optional): Axis along which input points are interpreted. Default is 0.
- `clamp` (bool, optional): If True, restricts `t` to [0, 1]. Default is True.

**Returns:** PArrayOrScalar - Interpolated points matching input dimensions.

**Example:**
```python
>>> interpolate([[0, 0], [10, 10]], 0.5)
[5., 5.]

>>> interpolate([[0, 0], [10, 10], [20, 0]], [0.25, 0.75], asPath=True)
[[ 5.,  5.], [15.,  5.]]

>>> interpolate([[0, 0], [10, 10], [20, 0]], [0.25, 0.75], applyAll=True)
[[[ 2.5,  2.5], [12.5,  7.5]], [[ 7.5,  7.5], [17.5,  2.5]]]
```

### interpolateIndices

Interpolate between two points in an n-dimensional index table and return the interpolated indices.

This function generates a series of interpolated indices between a start and an end point in an n-dimensional space. The resulting indices can be used to retrieve values from an index table or multidimensional array.

**Arguments:**
- `start` (PArray): The starting point in n-dimensional space.
- `end` (PArray): The ending point in n-dimensional space.
- `steps` (int): The number of interpolation steps, including the start and end points.
- `rounding` (RoundingMode, optional): rounding method for the resulting int indices.

**Returns:** IntArray - An int32 array of shape (steps, len(start)) containing the interpolated indices.

**Example:**
```python
>>> interpolateIndices(P(0,0), P(4, 4), 5)
[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])

>>> table = np.arange(16).reshape(4, 4)
>>> indices = interpolateIndices(P(0, 0), P(3, 3), 4)
>>> table[indices[:, 0], indices[:, 1]] # or table(tuple(indices[0])) for a single value
[0,  5, 10, 15]
```

### pointsToRegions

Create axis-aligned rectangular regions around given points.

Each input point expands into a rectangular region defined by `regionSize` . The position of the point within each region is controlled by the fractional `anchor` . An anchor of 0 places the point at the region's minimum corner (eg, left or top), 1 places it at the maximum corner, and intermediate values position it proportionally.

**Arguments:**
- `points` (NumberSequenceND): Array of points, shape `(n, d)` or `(n,)` for 1D points.
- `regionSize` (PArrayOrScalar): Size of each region, supporting: - scalar: uniform size across all dimensions and points. - (d,): size per dimension, identical for all points. - (n, d): individual size per point.
- `anchor` (PArrayLike, optional): Fractional position of points within their regions, default `0` . Same broadcast rules as `regionSize` .

**Returns:** PArray - Array of regions shaped `(n, 2, d)` , with each region's start (min corner) and end (max corner) coordinates.

**Example:**
```python
>>> pointsToRegions([[5, 5, 0]], [2, 4, 6], anchor=[.5, .5, 0])
[[[4., 3., 0.],   # min corner
  [6., 7., 6.]]]  # max corner
```

### attachPositionIndex

Get positional indices attached to each coordinate in the array.

Each element in the output includes its positional indices (optionally scaled by `offsets` ) followed by the original array value.

**Arguments:**
- `arr` (PArray): Input array of any dimensionality.
- `offsets` (NumberSequenceND, optional): Scale factors for coordinates along each axis. Must match the number of array dimensions.

**Returns:** PArray - Float64 array with shape `(*arr.shape, arr.ndim + 1)` , where the last axis contains `[x0, x1, …, xn-1, value]` for each element.

**Example:**
```python
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
```

### validateNumberSequence1D

Validate and coerce an input as a 1D float32 vector.

**Arguments:**
- `value` (NumberSequence1D): Input values to validate.
- `name` (str, optional): Field name used in error messages. Defaults to `"value"` .
- `length` (int, optional): If provided, enforce exact vector length.
- `minLength` (int, optional): If provided, enforce minimum vector length.
- `finite` (bool, optional): If True, reject NaN and infinity values.

**Returns:** PArray - A validated float32 1D array.

**Example:**
```python
>>> validateNumberSequence1D([1, 2, 3], length=3)
array([1., 2., 3.], dtype=float32)
```

### validateNumberSequence2D

Validate and coerce an input as a 2D float32 table.

**Arguments:**
- `value` (NumberSequence2D): Input values to validate.
- `name` (str, optional): Field name used in error messages. Defaults to `"value"` .
- `cols` (int, optional): If provided, enforce exact number of columns.
- `minRows` (int, optional): If provided, enforce minimum number of rows.
- `finite` (bool, optional): If True, reject NaN and infinity values.

**Returns:** PArray - A validated float32 2D array.

**Example:**
```python
>>> validateNumberSequence2D([[0, 1], [2, 3]], cols=2)
array([[0., 1.],
       [2., 3.]], dtype=float32)
```

### validateShapeND

Generic shape validator for public APIs.

**Arguments:**
- `value` (ArrayLike): Input value to validate and coerce.
- `name` (str, optional): Field name used in error messages. Defaults to `"value"` .
- `ndim` (int, optional): If provided, enforce exact number of dimensions.
- `lastDim` (int, optional): If provided, enforce exact size of the last dimension.
- `dtype` (Any, optional): Target dtype used by `np.asarray` . Defaults to `np.float32` .
- `finite` (bool, optional): If True, reject NaN and infinity values.

**Returns:** NDArray[Any] - A validated NumPy-like array.

**Example:**
```python
>>> validateShapeND([[1, 2], [3, 4]], ndim=2, lastDim=2)
array([[1., 2.],
       [3., 4.]], dtype=float32)
```

### validateColorLike

Validate common color inputs used by drawing APIs.

Accepted formats: int / numpy integer, hex-like string, 1D sequence or array with 3 (RGB) or 4 (RGBA) numeric channels

**Arguments:**
- `value` (ColorLike): Color value to validate.
- `name` (str, optional): Field name used in error messages. Defaults to `"color"` .

**Returns:** ColorLike - The original color value if validation succeeds.

**Example:**
```python
>>> validateColorLike("#ffaa11")
'#ffaa11'
>>> validateColorLike([255, 170, 17, 128])
[255, 170, 17, 128]
```

### pretty

Format a multi-dimensional array into a readable string.

Returns a clearly indented string with an optional header, useful for inspecting nested arrays.

**Arguments:**
- `arr` (NumberSequenceND): Array-like data to format. header (str, optional): A title displayed above the formatted array. indent (int, optional): Spaces used for indentation. Defaults to 0.

**Returns:** str - A neatly formatted string representation of the array.

**Example:**
```python
        >>> pretty([[1, 2], [3, 4]], header="2D Array")
        '

2D Array
[
  [1, 2],
  [3, 4]
]'

```

### toRGBA

Converts an ARGB color value to RGBA format.

This function reorders the bytes of a 32-bit color value from ARGB (Alpha, Red, Green, Blue) to RGBA (Red, Green, Blue, Alpha).

**Arguments:**
- `argb` (IntArrayOrScalar): A single ARGB color value or an array of ARGB values.

**Returns:** int, np.ndarray - The corresponding RGBA value(s). If the input is an integer, returns an integer. If the input is an array, returns an array of the same shape as the input.

**Example:**
```python
>>> toRGBA(0xFF112233)
0x112233FF

>>> toRGBA(np.array([0xFF112233, 0xAA445566]))
array([0x112233FF, 0x445566AA])
```

### toARGB

Converts an RGBA or RGB color value to ARGB format.

This function reorders the bytes of a 32-bit color value from RGBA (Red, Green, Blue, Alpha) to ARGB (Alpha, Red, Green, Blue). If the input is in RGB format (i.e., the alpha channel is not present), set `hasAlpha` to False.

**Arguments:**
- `rgba` (IntArrayOrScalar): A single RGBA or RGB color value or an array of such values.
- `hasAlpha` (bool): A flag indicating whether the input is in RGB format. Default is True (indicating RGBA).

**Returns:** int, np.ndarray - The corresponding ARGB value(s). If the input is an integer, returns an integer. If the input is an array, returns an array of the same shape as the input.

**Example:**
```python
>>> toARGB(0x112233FF)
0xFF112233

>>> toARGB(np.array([0x112233FF, 0x445566AA]))
array([0xFF112233, 0xAA445566])
```

### toARGBInt

Converts a color represented as an RGBA integer, a string, or an array-like object into an ARGB integer suitable for use in `skia-python` functions.

Accepts int (RGB or RGBA), hex string (` `"#RGB"` `, ` `"#RGBA"` `, ` `"#RRGGBB"` `, ` `"#RRGGBBAA"` `),
or a 3/4-element sequence. Short hex like ` `"#abc"` ` is expanded to ` `"#aabbcc"` `.

For int inputs < 0xFFFFFF, the value is assumed to be RGB with alpha=255 (use ` `hasAlpha=True` `
to override). A single-element list like ` `[0x112233]` ` is treated as a scalar int.

**Arguments:**
- `*args` (int, ColorLike): The color input. Can be: a single integer, a hex string or a list/tuple of RGBA components.
- `hasAlpha` (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

**Returns:** int - The corresponding ARGB integer.

**Example:**
```python
>>> toARGBInt(0x112233)
0xFF112233

>>> toARGBInt("#112233FF")
0xFF112233

>>> toARGBInt([17, 34, 51])
0xFF112233

>>> toARGBInt(17, 34, 51, 128)
0x80112233
```

### getColor

Retrieves a color from a 3D array of colors in HCL space at specified indices.

This function allows you to extract a color from a 3D array (such as one generated by `colorCube()` ) using a set of indices. Optionally, an alpha value can be provided to include transparency in the resulting color.

**Arguments:**
- `cube` (IntArray): A 3D array of colors, such as from `colorCube()` . Axes are (H, C, L).
- `indices` (IntArray): A set of indices specifying which color to retrieve (hue, chroma, lightness). Can be a single set of indices or an array of indices.
- `alpha` (int, optional): An optional alpha value to apply to the color. If provided, the returned color will be in ARGB format with the specified alpha.

**Returns:** int, IntArray - The color at the specified indices. If `alpha` is provided, returns an integer in ARGB format; otherwise, returns the raw color value(s).

**Example:**
```python
>>> hcl = np.array([[[255, 128, 64], [128, 64, 32]], [[0, 0, 0], [255, 255, 255]]])
>>> indices = np.array([0, 1, 2])
>>> getColor(hcl, indices)
32

>>> getColor(hcl, indices, alpha=128)
2147483680
```

### imageToArray

Convert a skia.Image to a float32 array normalized to [0.0, 1.0].

**Arguments:**
- `image` (skia.Image): The skia.Image to convert.
- `asGrayscale` (bool, optional): If True, returns a luminance array using ITU-R BT.601 weights (0.299 R + 0.587 G + 0.114 B).

**Returns:** PArray - Array of shape (height, width, 4) in RGBA order, or (height, width) if ` `asGrayscale` `.

### arrayToImage

Convert a numeric array to packed 32-bit pixels.

Normalizes values to [0, 255] (optionally using `min` / `max` ), then packs channels into 32-bit integers.

**Arguments:**
- `imageData` (PArray): Grayscale (H, W), RGB (H, W, 3), or RGBA (H, W, 4) array.
- `min` (float, optional): Lower bound for normalization. Defaults to `imageData.min()` .
- `max` (float, optional): Upper bound for normalization. Defaults to `imageData.max()` .
- `alwaysRGBA` (bool, optional): If True, force ARGB output with alpha=255 for gray/RGB.

**Returns:** IntArray - Packed pixels as RGB or ARGB integers.

**Example:**
```python
>>> gray = np.array([[0.0, 1.0]])
>>> arrayToImage(gray)
array([[       0, 16777215]], dtype=uint32)

>>> rgb = np.array([[[0.0, 0.5, 1.0]]])
>>> arrayToImage(rgb, alwaysRGBA=True)
array([[4278222847]], dtype=uint32)
```

### frame

Decorate a function to draw and return a Skia image.

This decorator provides a drawing context with size, background color, and passed a `Form` instance to the first parameter of the decorated function. The decorated function is called as `func(form, *args, **kwargs)` , and returns a skia.Image as default, but your drawing function can return something else convertible to an image.

**Arguments:**
- `bg` (ColorLike, optional): Background color of the frame. Defaults to `"e5e5e7"` .
- `size` (tuple[int, int], optional): Width and height of the frame in pixels. This can be accessed as `form.width` , `form.height` and `form.size` . Defaults to `(200, 200)` .

**Returns:** Callable - A decorator that wraps a function, injecting a `Form` instance as its first argument. The wrapped function returns a `skia.Image` . If the decorated function returns a `skia.Image` or a snapshot-able object (e.g. `skia.Surface` ), that is used directly; otherwise the Form's canvas is snapshot automatically.

**Example:**
```python
>>> @frame(bg=0xFFFFFF, size=(256, 256))
... def draw(form, r):
...     form.circle(128, 128, r)
>>> img = draw(r=100)  # returns a skia.Image
```

### frames

Decorate a function to render multiple frames as an array for animation or video. Like `frame` , this decorator provides a drawing context with size, background color, and passes two parameters: an instance of `Form` class, and the current frame index.

The decorated function is called as `func(form, index, *args, **kwargs)` for each frame, and returns an array of shape `(numFrames, height, width, channels)` , with RGBA or RGB channels based on `keepAlpha` .

**Arguments:**
- `bg` (ColorLike, optional): Background color for frames. Defaults to a light gray `"F0F3F9"` .
- `size` (Sequence[int], optional): Width and height of each frame. Defaults to `(200, 200)` .
- `numFrames` (int, optional): Total number of frames to render. Defaults to `30` .
- `clearBg` (bool, optional): If True, clears background before each frame. Defaults to `True` .
- `startAt` (int, optional): Initial frame index for rendering sequences. Defaults to `0` .
- `keepAlpha` (bool, optional): If True, retains alpha channel in output frames. Defaults to `False` .

**Returns:** Callable - A decorated function that receives a `Form` instance and frame index (int) as its first two arguments. Returns a uint8 array of shape ` `(numFrames, height, width, channels)` ` where channels is 4 (RGBA) if ` `keepAlpha=True` `, else 3 (RGB).

### perlinNoise

Generates a Perlin noise shader.

This function creates a Perlin noise shader using either fractal noise or turbulence noise. The generated shader can be used for various visual effects in graphics and design. You can then use `form.fill` to apply the shader.

**Arguments:**
- `freqX` (float): The frequency of the noise in the X direction.
- `freqY` (float): The frequency of the noise in the Y direction.
- `octaves` (int): The number of octaves to use for the noise. Higher values result in more detail.
- `seed` (float): The seed value for randomization. Changing this value will result in different noise patterns.
- `tileSize` (skia.ISize, optional): The tile size for the noise. If provided, the noise will repeat seamlessly at this size.
- `type` (str, optional): The type of Perlin noise to generate. Can be "fractal" for fractal noise or "turbulence" for turbulence noise. Default is "fractal".

**Returns:** skia.Shader - A Skia shader object that generates Perlin noise.

**Example:**
```python
>>> shader = perlinNoise(0.1, 0.1, 4, 42.0)
>>> isinstance(shader, skia.Shader)
True

>>> shader_turbulence = perlinNoise(0.1, 0.1, 4, 42.0, type="turbulence")
>>> isinstance(shader_turbulence, skia.Shader)
True
```

### populate

Create an array of a given shape filled with sequential values.

The array is populated starting from `start` , incrementing by `increment` with each element. If `increment` is zero, the entire array is filled uniformly with the `start` value.

**Arguments:**
- `shape` (IntArrayOrScalar): Desired shape of the resulting array.
- `start` (Number, optional): The initial value to populate from (default is 0).
- `increment` (Number, optional): Increment between consecutive elements (default is 1). If it's 0, the array is filled with the `start` value.

**Returns:** PArray - An array of specified shape, containing sequential values.

**Example:**
```python
>>> populate((2, 3))
array([[0., 1., 2.],
       [3., 4., 5.]])

>>> populate((2, 3), start=1, increment=2)
array([[ 1.,  3.,  5.],
       [ 7.,  9., 11.]])

>>> populate((2, 3), start=5, increment=0)
array([[5., 5., 5.],
       [5., 5., 5.]])
```

### steps

Generate evenly spaced points between `start` and `end` .

Produces an array of interpolated points from the given `start` to the `end` value. You specify how many points ( `count` ) you'd like. By default, both endpoints are included.

Optionally, use one of the predefined shaping functions or pass your own. The steps will then be interpolated by the shaping function.

**Arguments:**
- `start` (NumberSequenceND): Starting value or array of starting points.
- `end` (NumberSequenceND): Ending value or array of ending points.
- `count` (int): Number of points to generate. If 0, returns an empty array.
- `inclusive` (bool, optional): If True, includes start and end points. If False, excludes both endpoints. Default is True.
- `shaping` (str, optional): Interpolation style ('linear', etc.) or a custom `ShapingFunction` . Defaults to 'linear'.

**Returns:** PArray - Array of interpolated points from `start` to `end` .

**Example:**
```python
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
```

### gridPoints

Generate a grid of points arranged in regular intervals across dimensions.

Creates a multi-dimensional grid defined by the number of repeated steps and the spacing between points. Alternatively, using a table analogy, think of `repeat` as the number of rows or columns, and `spacing` as cell sizes. For simpler 2D or 3D grids, you can also use ``P2`` or ``P3``

**Arguments:**
- `repeat` (IntArrayOrScalar): Number of points along each dimension. Scalar implies one dimension.
- `spacing` (PArrayOrScalar): Distance between adjacent points. Scalar applies to all dimensions.

**Returns:** PArray - Array of shape (total_points, dims). Axes are ordered to match `repeat` / `spacing` — first dimension varies slowest, last varies fastest (` `ij` ` indexing).

**Example:**
```python
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
```

### random

Create an array of random numbers within a specified range.

Generates uniformly distributed random floats between `min` (inclusive) and `max` (exclusive). If `shape` is provided, returns an array with that shape; otherwise, returns a single float. An optional `seed` ensures reproducible results.

**Arguments:**
- `shape` (PArrayLike, optional): Shape of the array to generate. If None, returns a scalar float32.
- `max` (float, optional): Upper bound for random numbers (exclusive). Defaults to 1.0.
- `min` (float, optional): Lower bound for random numbers (inclusive). Defaults to 0.0.
- `seed` (int, optional): Seed for reproducibility. Defaults to None.
- `rng` (np.random.Generator, optional): If provided, this generator is used for sampling. If `None` and `seed` is given, a new generator is created with that seed. Otherwise a new default generator is used.

**Returns:** PArray - Array of random floats within `min` and `max` or a single float if `shape` is None.

**Example:**
```python
>>> random()
0.77395605

>>> random((2, 3))
array([[0.5488135 , 0.71518937, 0.60276338],
       [0.54488318, 0.4236548 , 0.64589411]])

>>> random((2, 3), max=10, min=5)
array([[5.53716201, 8.3015781 , 6.28250546],
       [9.57844397, 6.65108716, 9.55465934]])
```

### irregular

Generate irregularly spaced numbers between 0 and a maximum value.

Produces an array of `steps` points randomly spaced between 0 and `maxValue` . Points are unevenly spaced yet well-distributed across the range.

**Arguments:**
- `maxValue` (float): Upper limit for generated values.
- `steps` (int): Number of points to create.
- `rng` (np.random.Generator, optional): RNG to use. If None, a new default RNG is created.
- `shaping` (str or ShapingFunction, optional): Shaping function name or callable for jitter.

**Returns:** PArray - Array of irregularly spaced floats in [0, `maxValue` ) (upper bound exclusive).

**Example:**
```python
>>> irregular(10, 5)
array([0.7, 2.4, 5.1, 7.2, 9.5])
```

### cubeHCL

Generate a 3D grid of HCL color values.

This function creates a 3D grid of HCL (Hue, Chroma, Lightness) values by interpolating between the provided ranges.

**Arguments:**
- `steps` (int, optional): Number of points along each H, C, and L axis (must be >= 2). Defaults to 36.
- `h_range` (tuple, optional): Min and max values for hue (in radians). Defaults to (0, TWO_PI).
- `c_range` (tuple, optional): Min and max values for chroma. Defaults to (0, 100).
- `l_range` (tuple, optional): Min and max values for lightness. Defaults to (0, 100).
- `includeHueMax` (bool, optional): If True, includes hue endpoint (e.g., to duplicate 0 and 2π). Default is False.

**Returns:** PArray - An array shaped `(steps, steps, steps, 3)` , containing HCL coordinates with axes (H, C, L).

**Example:**
```python
>>> hcl_cube = cubeHCL(10)
>>> hcl_cube.shape
(10, 10, 10, 3)

>>> hcl_cube[0, 0, 0]
array([0., 0., 0.])
```

### colorCube

Generates a 3D cube of RGB integers by sampling the HCL color space (hue, chroma, and lightness).

This function creates a 3D array of shape `(steps, steps, steps)` where each element is an RGB integer. The RGB values are generated by interpolating through the HCL color space and converting the results to a structured cube of RGB. The cube axes are ordered as (H, C, L) - hue on axis 0, chroma on axis 1, lightness on axis 2.

**Arguments:**
- `steps` (int, optional): Number of samples per H, C, and L axis. Defaults to 36. Large values may require significant memory.
- `observer` (Literal["2", "10", "R"], optional): Standard observer angle (2, 10, or R). Defaults to "10".
- `illuminant` (str, optional): The illuminant used for color conversion. Defaults to "D65".
- `gamutMap` (Literal["soft", "clip"], optional): Gamut mapping strategy. "soft" uses smooth shoulder compression for gradual transitions; "clip" uses hard clipping. Defaults to "soft".
- `gamutThreshold` (float, optional): For soft mapping, start compression at this fraction of gamut (0.85 = 85%). Lower values compress earlier for smoother results. Defaults to 0.85.
- `gamutStrength` (float, optional): For soft mapping, compression intensity. Higher values produce more aggressive compression. Defaults to 1.5.

**Returns:** IntArray - An integer array of shape `(steps, steps, steps)` , with RGB values packed as 0xRRGGBB. Axes are (hue, chroma, lightness).

**Example:**
```python
>>> cube = colorCube(10)
>>> cube.shape
(10, 10, 10)

>>> cube[0, 0, 0]
0x123456  # Example output representing an RGB color

>>> # Hard clipping (legacy behavior)
>>> cube = colorCube(10, gamutMap="clip")

>>> # Aggressive soft compression
>>> cube = colorCube(10, gamutThreshold=0.7, gamutStrength=3.0)
```

### colorInterpolation

Interpolates between two colors in a color cube.

This function interpolates colors within a 3D array of colors (a color cube) between a specified start and end color. The interpolation can be performed along a straight path between the two colors, or within a sub-cube of the color cube, depending on the `asPath` argument.

**Arguments:**
- `cube` (IntArray): A 3D array of colors, typically generated by the `colorCube()` function.
- `start` (PArray): The starting color as (hue, chroma, lightness) indices. It can be a batch of colors or a single color.
- `end` (PArray): The ending color as (hue, chroma, lightness) indices. It can be a batch of colors or a single color.
- `steps` (int, optional): Number of interpolation steps (integer >= 1). Effective per-axis/path sample count is `N = steps + (1 if includeEnd else 0)` . Default is 10.
- `asPath` (bool, optional): If True, interpolates colors along a straight line path between `start` and `end` with O(N) storage. If False, samples a dense sub-cube with O(N^3) storage. Default is True.
- `includeEnd` (bool, optional): If True, include endpoint sample(s) from `end` . If False, endpoint is excluded: path mode returns exactly `steps` samples and sub-cube mode returns `steps` samples per axis. Default is True.

**Returns:** IntArray - Interpolated colors with shape determined by `asPath` (see table above).

**Example:**
```python
>>> cube = colorCube(36)
>>> start = P(10, 20, 30)  # (hue, chroma, lightness) indices
>>> end = P(25, 35, 5)
>>> colors = colorInterpolation(cube, start, end, steps=5)
>>> colors.shape
(6,)

>>> colors_as_subcube = colorInterpolation(cube, start, end, steps=5, asPath=False)
>>> colors_as_subcube.shape
(6, 6, 6)
```

### colorPicker

Pick a color from a color cube using normalized hue, chroma, and lightness.

Retrieves a color from a 3D HCL color cube. Inputs are treated as normalized values and mapped to indices in the cube. Values outside the expected range are clamped to valid cube bounds. Returns either the RGB integer or the cube indices.

**Arguments:**
- `cube` (IntArray): A 3D color array, typically from `colorCube()` . Axes are (H, C, L).
- `hue` (float): Normalized hue (typically in [0, 1], inclusive; clamped if outside).
- `chroma` (float): Normalized chroma (typically in [0, 1], inclusive; clamped if outside).
- `lightness` (float): Normalized lightness (typically in [0, 1], inclusive; clamped if outside).
- `asArray` (bool, optional): If True, returns indices array; if False, returns RGB integer. Default is False.

**Returns:** int, PArray - RGB integer (when asArray=False) or array of [h, c, l] indices (when asArray=True).

**Example:**
```python
>>> cube = colorCube(36)
>>> color = colorPicker(cube, 0.5, 0.5, 0.5)
>>> color
0x123456  # RGB color value

>>> indices = colorPicker(cube, 0.5, 0.5, 0.5, asArray=True)
>>> indices
array([18, 18, 18])
```

### colorScheme

Generates a color scheme from a 3D array of colors in HCL space. Experimental.

This function creates a color scheme by sampling from a 3D HCL color cube. The scheme is designed to vary in lightness and chroma, with the top part of the scheme getting darker and less saturated, while the bottom part gets brighter. The function returns an array of shape `(hues, swatches, 4)` , where each entry contains the RGB integer, and the indices from the HCL cube corresponding to hue, chroma, and lightness.

**Arguments:**
- `cube` (IntArray): A 3D array of colors, typically generated from the `colorCube()` function. Axes are (H, C, L).
- `hues` (int, optional): The number of different hues in the scheme. Default is 8.
- `swatches` (int, optional): The number of swatches per hue. Default is 5.
- `intensity` (float, optional): Controls the variation in lightness and chroma. Lower values create more contrast. Default is 0.4.
- `variant` (np.random.Generator, optional): If an rng is provided, uses irregular spacing for the hue samples. Default is None.

**Returns:** PArray - An array of shape `(hues, swatches, 4)` where the last dimension contains: - RGB integer color - Hue index - Chroma index - Lightness index

**Example:**
```python
>>> cube = colorCube(36)
>>> scheme = colorScheme(cube, hues=6, swatches=4)
>>> scheme.shape
(6, 4, 4)

>>> scheme[0, 0]  # Example of the first color's data
array([0x123456, 0, 12, 18])  # [RGB integer, h-index, c-index, l-index]
```

### gradient

Creates a gradient shader based on the specified kind and color stops. You can then use [Form.fill](Form.fill) to fill the gradient.

This function generates a gradient shader, either linear, radial, sweep, or conical, depending on the `kind` argument.

The gradient is defined by the `pts` for the start and end points:

- linear: 2 points [start, end]

- radial: 2 points [center, radius]

- sweep: 1 point [center]

- conical: 2 points [start_center, end_center]

**Arguments:**
- `pts` (PArray): An array of points defining the gradient's start and end positions.
- `colors` (List[ColorLike]): A list of colors to be used in the gradient. These can be RGBA integers or other color formats.
- `kind` (str, optional): The type of gradient. Can be "linear", "radial", "sweep", or "conical". Default is "linear".
- `stops` (NumberSequence1D, optional): A sequence of stops for the gradient colors. If None, the colors are evenly spaced.

**Returns:** skia.Shader - A Skia shader object that can be used for drawing the gradient.

**Example:**
```python
>>> pts = P([0, 0], [100, 100])
>>> colors = [0xFF0000FF, 0xFFFF00FF, 0xFF00FFFF]
>>> shader = gradient(pts, colors, kind="linear")
>>> form.fill(shader)
```

## Classes

## Matrix2D

Fluent builder for 2D affine transform matrices.

Chains transform calls left-to-right to build a 3x3 affine matrix. The result can be used in transform2D function.

### Matrix2D.translate

Translate by offset.

**Arguments:**
- `offset` (NumberSequenceND): Translation vector `[dx, dy]` .

**Returns:** Matrix2D - Self, for chaining.

### Matrix2D.rotate

Rotate clockwise (in screen space) by angle in radians.

**Arguments:**
- `angle` (float): Rotation angle in radians.
- `origin` (NumberSequenceND, optional): Center of rotation. Defaults to `[0, 0]` .

**Returns:** Matrix2D - Self, for chaining.

### Matrix2D.scale

Scale uniformly by a scalar or non-uniformly by `[sx, sy]` .

**Arguments:**
- `factor` (NumberSequenceND): Scalar for uniform scale, or `[sx, sy]` for per-axis.
- `origin` (NumberSequenceND, optional): Center of scaling. Defaults to `[0, 0]` .

**Returns:** Matrix2D - Self, for chaining.

### Matrix2D.skew

Skew by shear factors.

**Arguments:**
- `shear` (NumberSequenceND): Shear factors `[sx, sy]` .
- `origin` (NumberSequenceND, optional): Center of shearing. Defaults to `[0, 0]` .

**Returns:** Matrix2D - Self, for chaining.

### Matrix2D.reflect

Reflect across a line at a given angle from the x-axis.

**Arguments:**
- `angle` (float): Angle of the reflection axis in radians. Defaults to 0 (x-axis).
- `origin` (NumberSequenceND, optional): Center of reflection. Defaults to `[0, 0]` .

**Returns:** Matrix2D - Self, for chaining.

### Matrix2D.inverse

Return a new Matrix2D with the inverted transform.

**Returns:** Matrix2D - A new instance whose matrix is the inverse of this one.

### Matrix2D.m

The 3x3 affine matrix.

**Returns:** PArray - The accumulated 3x3 affine transform matrix.

## Rectangle

A class for various static helper methods related to rectangles.

### Rectangle.size

Calculate the size of a rectangle.

Takes a rectangle defined by two corner points and returns its width and height.

**Arguments:**
- `rect` (PArray): An array of shape (2, 2) where rect[0] is one corner and rect[1] is the opposite corner.

**Returns:** PArray - An array of shape (2,) containing [width, height].

**Example:**
```python
>>> rect = P([0, 0], [10, 20])
>>> size = Rectangle.size(rect)
>>> size
array([10, 20])
```

### Rectangle.center

Calculate the center point of a rectangle.

Takes a rectangle defined by two corner points and returns its center.

**Arguments:**
- `rect` (PArray): An array of shape (2, 2) where rect[0] is one corner and rect[1] is the opposite corner.

**Returns:** PArray - An array of shape (2,) containing [x, y] coordinates of the center.

**Example:**
```python
>>> rect = P([0, 0], [10, 20])
>>> center = Rectangle.center(rect)
>>> center
array([5.0, 10.0])
```

### Rectangle.fromTopLeft

Create a rectangle from the top-left corner and size.

This method takes the top-left point of a rectangle and its size (width and height), and returns the rectangle defined by the top-left and bottom-right points.

**Arguments:**
- `topLeft` (PArray): A 2D array representing the top-left point of the rectangle.
- `size` (PArrayOrScalar): A 2D array or scalar representing the size of the rectangle. If a scalar is provided, it is used for both width and height.

**Returns:** PArray - A 2D array where the first row is the top-left point and the second row is the bottom-right point.

**Example:**
```python
>>> topLeft = P(0, 0)
>>> size = P(10, 20)
>>> rect = Rectangle.fromTopLeft(topLeft, size)
>>> rect
array([[ 0,  0],
       [10, 20]])
```

### Rectangle.fromCenter

Create a bounding box from a center point and size.

**Arguments:**
- `center` (PArray): The center point of the bounding box.
- `size` (PArrayOrScalar): The size of the bounding box. Can be a scalar or an array specifying the size along each dimension.

**Returns:** P - A bounding box with coordinates calculated from the center and size.

**Example:**
```python
>>> center = P(5, 5)
>>> size = 4
>>> bbox = Rectangle.fromCenter(center, size)
>>> bbox
P(array([3., 3.]), array([7., 7.]))

>>> size = P(4, 6)
>>> bbox = Rectangle.fromCenter(center, size)
>>> bbox
P(array([3., 2.]), array([7., 8.]))
```

### Rectangle.strips

Split a rectangle into strips along an axis based on the `ratio` parameter.

**Arguments:**
- `rect` (NumberSequenceND): Either a rectangular size `[w, h]` , or a bounding rectangle `[[x0, y0], [x1, y1]]` .
- `ratio` (PArrayOrScalar): Fraction(s) of the span. Scalar for equal splits, array for proportional splits.
- `axis` (int): 0 = vertical strips (along x), 1 = horizontal (along y).

**Returns:** PArray - an array of (number of strips, 2, 2)

**Example:**
```python
>>> Rectangle.strips([100, 60], 0.25).shape
(4, 2, 2)
>>> Rectangle.strips([100, 60], [0.2, 0.5, 0.1]).shape
(4, 2, 2)
```

### Rectangle.helper

Determine the bounds of a 2D rectangle and provide semantic access to its corners. Assumes screen coordinates (y increases downward).

**Arguments:**
- `data` (NumberSequenceND): Either [width, height] (origin at [0,0]) or [[x1, y1], [x2, y2]].

**Returns:** SimpleNamespace - A namespace with attributes 'tl' (top left), 'tr' (top right), 'bl' (bottom left), 'br' (bottom right), and 'center'. It also contains 'top', 'left', 'right', and 'bottom' segments, plus 'size'.

**Example:**
```python
>>> data = [10, 5]
>>> bounds = Rectangle.helper(data)
>>> bounds.tl
P([0, 0])
>>> bounds.br
P([10, 5])

>>> data = [[2, 3], [5, 7]]
>>> bounds = Rectangle.helper(data)
>>> bounds.tl
P([2, 3])
>>> bounds.br
P([5, 7])
```

## Circle

A class for various static helper methods related to circles.

### Circle.fromCenter

Create a bounding box from a center point and radius.

**Arguments:**
- `center` (PArray): The center point of the bounding box.
- `radius` (PArrayOrScalar): The radius of the bounding box. Can be a scalar or an array specifying the radius along each dimension.

**Returns:** P - A bounding box with coordinates calculated from the center and radius.

**Example:**
```python
>>> center = P(5, 5)
>>> radius = 2
>>> bbox = Circle.fromCenter(center, radius)
>>> bbox
P([3, 3], [7, 7])

>>> radius = P(2, 3)
>>> bbox = Circle.fromCenter(center, radius)
>>> bbox
P([3, 2], [7, 8])
```

## StageFunction

Function wrapper enabling operator-based pipeline composition.

Wrap a function with `F()` (alias for `StageFunction` ) to chain it with others using `>>` (forward) or `<<` (reverse) operators. Chains flatten automatically, so `F(a) >> b >> c` creates a single 3-stage pipeline.

## Form

Skia-backed drawing surface for 2D graphics.

Form wraps a Skia canvas and provides methods to draw shapes, text, and images. Configure stroke and fill colors, then chain drawing calls. Most methods return

`self` for fluent chaining.

Typically created via the `@frame` decorator or `Form.setup()` :

@frame(size=(400, 400)) def draw(form): form.fill("#ff0000").circle([[50, 50], [150, 150]])

Or manually:

form = Form(width=400, height=400) form.fill("#ff0000").circle([[50, 50], [150, 150]]) img = form.toImage()

### Form.createRasterCanvas

Create a skia.Canvas that uses raster as its backend. For convenience, you can also use `@frame` decorator instead.

This method creates a `skia.Surface` with the specified width and height, and returns the canvas associated with that surface.

**Arguments:**
- `width` (int): The width of the canvas.
- `height` (int): The height of the canvas.

**Returns:** skia.Canvas - A canvas object for raster drawing.

**Example:**
```python
>>> canvas = Form.createRasterCanvas(800, 600)
```

### Form.setup

Create a Form object with a raster canvas backend.

**Arguments:**
- `width` (int): The width of the canvas in pixels.
- `height` (int): The height of the canvas in pixels.

**Returns:** Form - A Form object initialized with a raster canvas backend.

**Example:**
```python
>>> form = Form.setup(800, 600)
>>> # Use the form object for drawing or other operations.
```

### Form.setupGL

Create a Form object with an OpenGL canvas backend.

**Arguments:**
- `width` (int): The width of the canvas in pixels.
- `height` (int): The height of the canvas in pixels.
- `window` (Any, optional): An existing OpenGL window context. If None, a new window will be created. Default is None.

**Returns:** Form - A Form object initialized with an OpenGL canvas backend.

**Example:**
```python
>>> form = Form.setupGL(800, 600)
>>> # Use the form object for OpenGL rendering.
```

### Form.width

Get the width of the canvas.

**Returns:** int - The width of the canvas in pixels.

### Form.height

Get the height of the canvas.

**Returns:** int - The height of the canvas in pixels.

### Form.size

Get the size of the canvas as [width, height].

**Returns:** PArray - A PArray of [width, height]

### Form.stroke

Configure stroke properties for subsequent drawing operations.

Sets attributes like stroke color, line width, line cap style, line join style, and miter limit. If no argument is provided, the stroke color and shader are cleared (disables stroke).

**Arguments:**
- `color` (ColorOrShader, optional): The stroke color. Can be an RGB/RGBA array-like object ([255,0,0,255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff) or a skia Shader. Default is None.
- `width` (float, optional): The stroke width. Default is None.
- `cap` (StrokeJoinCap, optional): The stroke cap style ("bevel", "round", "miter", or "last"). Default is None.
- `join` (StrokeJoinCap, optional): The stroke join style ("bevel", "round", "miter", or "last"). Default is None.
- `miter` (float, optional): The stroke miter limit. Default is None.
- `hasAlpha` (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.stroke("#ff0000", width=2.0, cap="round", join="miter", miter=10.0)
>>> # Use the form object for drawing with the specified stroke properties.
```

### Form.fill

Configure fill properties for subsequent drawing operations. Accepts a color value or a ` `skia.Shader` `. If no argument is provided, the fill color and shader are cleared (disables fill).

**Arguments:**
- `color` (ColorOrShader, optional): The fill color or a [skia.Shader](https://kyamagu.github.io/skia-python/reference/skia.Shader.html) object. Color can be an RGB/RGBA array-like object ([255,0,0,255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff). Default is None.
- `hasAlpha` (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

**Returns:** Form - self.

**Example:**
```python
>>> form.fill(0xFF00FFFF)  # Set fill color to pink
>>> form.fill(skia.Shader.MakeLinearGradient(...))  # Set fill shader
```

### Form.fillOnly

Set the fill color and disable the stroke for the form instance. See `fill` and `stroke` for details.

**Arguments:**
- `color` (ColorOrShader, optional): The fill color or a [skia.Shader](https://kyamagu.github.io/skia-python/reference/skia.Shader.html) object. Color can be an RGB/RGBA array-like object ([255,0,0,255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff). Default is None.
- `hasAlpha` (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.fillOnly("#ff0000")  # Set fill color to red and disable stroke
```

### Form.strokeOnly

Set the stroke color and disable the fill for the form instance. See `fill` and `stroke` for details.

**Arguments:**
- `color` (ColorOrShader, optional): The stroke color or a skia.Shader object. Color can be an RGB/RGBA array-like object ([255,0,0,255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff). Default is None.
- `width` (float, optional): The stroke width. Default is None.
- `cap` (StrokeJoinCap, optional): The stroke cap style. Can be "bevel", "round", "miter", or "last". Default is None.
- `join` (StrokeJoinCap, optional): The stroke join style. Can be "bevel", "round", "miter", or "last". Default is None.
- `miter` (float, optional): The stroke miter limit. Default is None.
- `hasAlpha` (bool, optional): Optionally set whether the color value has alpha channel. This is only necessary for RGBA color integer less than 0xffffff, where heuristics can become ambiguous.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.strokeOnly("#00ff00", width=2.0, cap="round", join="miter", miter=10.0)  # Set stroke properties and disable fill
```

### Form.clear

Clear the canvas with the specified color.

When alpha is 1.0 (default), fully replaces all pixels. When alpha < 1.0, draws the color at that opacity over the existing content, creating a fade/trail effect.

**Arguments:**
- `color` (ColorLike): The color to clear the canvas with. Can be an RGB/RGBA array-like object ([255, 0, 0, 255]), an RGB/RGBA string ("#ff0000ff"), or an RGB/RGBA int (0xff0000ff).
- `alpha` (float): Opacity of the clear operation, from 0.0 (no-op) to 1.0 (full clear). Default is 1.0.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.clear("#ffffff")  # Clear the canvas with white color
>>> form.clear("000", alpha=0.1)  # Fade toward black (motion trail effect)
```

### Form.surface

Get the Skia surface associated with the canvas.

**Returns:** skia.Surface - The surface of the canvas.

### Form.dots

Draw dots at the specified points on the canvas. For simple visualization of points, this method is faster than using `point()` .

**Arguments:**
- `pts` (NumberSequence1D): An array of points. Can be of shape (n, 2) or (n, 2, 2).

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> points = P([10, 10], [20, 20], [30, 30])
>>> form.dots(points)  # Draw dots at the specified points
```

### Form.point

Draw a point on the canvas with the specified style.

**Arguments:**
- `pt` (NumberSequence1D): The coordinates of the point. Should be a 2-element array or list [x, y].
- `radius` (float, optional): The radius of the point. Default is 5.0.
- `style` (PointStyle, optional): The style of the point. Can be "circle" or "rect". Default is "circle".

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.point([50, 50], radius=10.0, style="circle")  # Draw a circle point
>>> form.point([100, 100], radius=5.0, style="rect")  # Draw a rectangle point
```

### Form.points

Draw multiple points on the canvas with the specified style.

**Arguments:**
- `pts` (NumberSequence1D): Array of shape (n, 2) for [x, y] positions, or (n, 3) where the third column is a per-point radius override.
- `radius` (float, optional): The radius of each point. Default is 5.0.
- `style` (PointStyle, optional): The style of each point. Can be "circle" or "rect". Default is "circle".

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> points = P([50, 50], [100, 100], [150, 150])
>>> form.points(points, radius=10.0, style="circle")  # Draw multiple circle points
```

### Form.line

Draw lines on the canvas.

This method draws a continuous polyline if the input is of shape (n, 2), or multiple line segments if the input is of shape (n, 2, 2).

**Arguments:**
- `pts` (NumberSequence1D): An array of points. Can be of shape (n, 2) to draw a continuous polyline, or shape (n, 2, 2) to draw multiple line segments.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> continuous_line = P([10, 10], [20, 20], [30, 30])
>>> form.line(continuous_line)  # Draw a continuous polyline
>>> segments = P([[[10, 10], [20, 20]], [[30, 30], [40, 40]]])
>>> form.line(segments)  # Draw multiple line segments
```

### Form.lines

Alias for `line()` . Reads better when drawing multiple independent segments via shape (n, 2, 2).

### Form.rect

Draw a rectangle on the canvas.

This method draws a rectangle using the coordinates of two diagonal corners.

**Arguments:**
- `pts` (NumberSequence2D): An array of points specifying the diagonal corners of the rectangle. Should be of shape (2, 2).

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> corners = P([10, 10], [50, 50])
>>> form.rect(corners)  # Draw a rectangle from (10, 10) to (50, 50)
```

### Form.rects

Draw multiple rectangles on the canvas.

This method draws multiple rectangles using the coordinates of their diagonal corners.

**Arguments:**
- `pts` (NumberSequence2D): An array of points specifying the diagonal corners of each rectangle. Should be of shape (n, 2, 2).

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> rectangles = P([[[10, 10], [50, 50]], [[60, 60], [100, 100]]])
>>> form.rects(rectangles)  # Draw two rectangles
```

### Form.circle

Draw a circle or ellipse inscribed in a bounding rectangle.

The shape is defined by two diagonal corners. A square bounding box produces a circle; a rectangular one produces an ellipse.

**Arguments:**
- `pts` (NumberSequence2D): Diagonal corners of the bounding rectangle as an array of shape (2, 2).

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.circle([[10, 10], [50, 50]])
```

### Form.circles

Draw multiple circles or ellipses inscribed in bounding rectangles. Each shape is defined by two diagonal corners.

**Arguments:**
- `pts` (NumberSequence2D): Bounding rectangles in an array of shape (n, 2, 2).

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> bounds = np.array([[[10, 10], [50, 50]], [[60, 60], [100, 100]]])
>>> form.circles(bounds)  # Draw two circles
```

### Form.ellipse

Draw an ellipse on the canvas.

This method draws an ellipse using the coordinates of two diagonal corners of the bounding rectangle. Optionally, it can draw a rotated ellipse or an elliptical arc.

**Arguments:**
- `pts` (NumberSequence2D): An array of points specifying the diagonal corners of the bounding rectangle. Should be of shape (2, 2).
- `rotation` (float, optional): The rotation angle in degrees. Default is None.
- `startAngle` (float, optional): The starting angle in degrees for the arc. Default is None.
- `endAngle` (float, optional): The ending angle in degrees for the arc. Default is None.
- `counterClockwise` (bool, optional): Whether the arc is drawn counterclockwise. Default is False.
- `wedge` (bool, optional): Whether to draw a wedge instead of an arc. Default is False.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> corners = P([10, 10], [50, 50])
>>> form.ellipse(corners)  # Draw an ellipse
>>> form.ellipse(corners, rotation=45)  # Draw a rotated ellipse
>>> form.ellipse(corners, startAngle=0, endAngle=180)  # Draw an arc
```

### Form.ellipses

Draw multiple ellipses on the canvas.

This method draws multiple ellipses using the coordinates of their diagonal corners of the bounding rectangles. Optionally, it can draw rotated ellipses or elliptical arcs.

**Arguments:**
- `pts` (NumberSequence2D): An array of points specifying the diagonal corners of each bounding rectangle. Should be of shape (n, 2, 2).
- `rotation` (float, optional): The rotation angle in degrees for each ellipse. Default is None.
- `startAngle` (float, optional): The starting angle in degrees for each arc. Default is None.
- `endAngle` (float, optional): The ending angle in degrees for each arc. Default is None.
- `counterClockwise` (bool, optional): Whether each arc is drawn counterclockwise. Default is False.
- `wedge` (bool, optional): Whether to draw wedges instead of arcs. Default is False.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> ellipses = P([[[10, 10], [50, 50]], [[60, 60], [100, 100]]])
>>> form.ellipses(ellipses)  # Draw multiple ellipses
>>> form.ellipses(ellipses, rotation=45)  # Draw multiple rotated ellipses
>>> form.ellipses(ellipses, startAngle=0, endAngle=180)  # Draw multiple arcs
```

### Form.polygon

Draw a polygon on the canvas.

This method draws a polygon using the provided points. Optionally, the polygon can be closed to form a complete shape.

**Arguments:**
- `pts` (NumberSequence2D): An array of points specifying the vertices of the polygon. Should be of shape (n, 2).
- `closed` (bool, optional): Whether to close the polygon by connecting the last point to the first. Default is True.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> vertices = P([10, 10], [50, 50], [50, 10])
>>> form.polygon(vertices)  # Draw a closed triangle
>>> form.polygon(vertices, closed=False)  # Draw an open polyline
```

### Form.polygons

Draw multiple polygons on the canvas in a single draw call.

**Arguments:**
- `pts` (NumberSequenceND): an array of polygons, of shape `(n, m, 2)`
- `closed` (bool): Whether to close each polygon. Default is True.

**Returns:** Form - self.

**Example:**
```python
>>> form.polygons(P([[[10, 10], [50, 50], [50, 10]], [[60, 60], [100, 100], [100, 60]]]))
>>> form.polygons([[[0,0], [10,0], [5,10]], [[20,0], [30,0], [30,10], [20,10]]])
```

### Form.bezier

Draw a 2D Bézier curve on the canvas.

This method draws a Bézier curve using the provided control points. Optionally, the curve can be closed to form a complete shape.

Expects points in flat format: ` `[start, cp1, cp2, end, cp1, cp2, end, ...]` `.
If ` `(len(pts) - 1)` ` is not divisible by 3, trailing points are silently dropped.

**Arguments:**
- `pts` (NumberSequence2D): An array of points specifying the control points of the Bézier curve. Should be of shape (n, 2).
- `closed` (bool, optional): Whether to close the curve by connecting the last point to the first. Default is False.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> control_points = P([10, 10], [20, 30], [30, 10], [40, 30])
>>> form.bezier(control_points)  # Draw an open Bézier curve
>>> form.bezier(control_points, closed=True)  # Draw a closed Bézier curve
```

### Form.cardinal

Draw a cardinal spline through the given control points.

Converts cardinal spline points to cubic Bezier control points via ` `cardinalToBezier` `, then delegates to ` `bezier()` `.

**Arguments:**
- `pts` (NumberSequence2D): Shape (n, 2) — the points the curve passes through.
- `tension` (float): Tension parameter (0–1). 0.5 is Catmull-Rom, 1.0 is straight lines.
- `closed` (bool): If True, the curve loops back to the first point.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.cardinal([[10, 10], [50, 80], [90, 10], [130, 80]])
```

### Form.hermite

Draw a Hermite spline through the given points with explicit tangent vectors.

Converts positions and tangents to cubic Bezier control points via ` `hermiteToBezier` `, then draws the resulting path.

**Arguments:**
- `pts` (NumberSequence2D): Shape (n, 2) — positions the curve passes through.
- `tangents` (NumberSequence2D): Shape (n, 2) — tangent vectors at each point.
- `closed` (bool): If True, the curve loops back to the first point.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.hermite([[10, 50], [50, 10], [90, 50]], [[20, 0], [0, -20], [-20, 0]])
```

### Form.bspline

Draw a uniform cubic B-spline through the given control points.

Unlike cardinal/hermite splines, B-splines are *approximating* — the curve does **not** pass through the control points.

Converts B-spline control points to cubic Bezier control points via ` `bsplineToBezier` `, then draws the resulting path.

**Arguments:**
- `pts` (NumberSequence2D): Shape (n, 2) — B-spline control points. Minimum 4 for open, 3 for closed.
- `closed` (bool): If True, the curve loops back to the start.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.bspline([[10, 50], [40, 10], [75, 90], [110, 30], [140, 60]])
```

### Form.font

Set the font style for text drawing.

**Arguments:**
- `size` (int, optional): The size of the font. Default is 12.
- `name` (str, optional): The name of the font. Default is "Arial".
- `bold` (bool, optional): Whether the font is bold. Ignored if `weight` is specified. Default is False.
- `italic` (bool, optional): Whether the font is italic. Default is False.
- `weight` (int, optional): The font weight (100-900). Common values: 100 (Thin), 200 (ExtraLight), 300 (Light), 400 (Normal), 500 (Medium), 600 (SemiBold), 700 (Bold), 800 (ExtraBold), 900 (Black). If specified, overrides the `bold` parameter. Default is None.
- `width` (int, optional): The font width (1-9). Values: 1 (UltraCondensed), 2 (ExtraCondensed), 3 (Condensed), 4 (SemiCondensed), 5 (Normal), 6 (SemiExpanded), 7 (Expanded), 8 (ExtraExpanded), 9 (UltraExpanded). Default is None (Normal).

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.font(size=16, name="Times New Roman", bold=True)  # Set font to 16pt Times New Roman, bold
>>> form.font(size=20, name="Roboto", weight=500)  # Set font to 20pt Roboto Medium
>>> form.font(size=18, name="Inter", weight=600, italic=True)  # Set font to 18pt Inter SemiBold Italic
```

### Form.text

Draw text on the canvas at the specified position.

**Arguments:**
- `pt` (NumberSequence1D): The position to draw the text. Should be a 2-element array or list [x, y].
- `txt` (str): The text string to draw.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.font(size=16, name="Arial", bold=True)
>>> form.text([50, 50], "Hello, World!")  # Draw text at position (50, 50)
```

### Form.colorMap

Draw a color map that visualizes a 2D array of colors within a specified rectangle, using either rectangles or circles to represent each cell in the array.

This is currently non-vectorized but can draw different styles (rectangles or circles). For large arrays, use `image` for better performance.

**Arguments:**
- `table` (IntArray): A 2D array of colors of shape (n, m).
- `rect` (NumberSequence2D): The rectangle in which to draw the color map. Should be of shape (2, 2).
- `style` (PointStyle, optional): The style of each cell. Can be "rect" or "circle". Default is "rect".
- `useCeiling` (bool, optional): Whether to use the ceiling value for cell sizes. Default is False.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> table = P([0xFF0000, 0x00FF00], [0x0000FF, 0xFFFF00]).astype("uint32")
>>> rect = P([10, 10], [110, 110])
>>> form.colorMap(table, rect, style="rect")  # Draw a color map with rectangles
```

### Form.image

Draw a color table into a rectangle as a bitmap image.

This is the vectorized version of `colorMap` , optimized for large arrays.

**Arguments:**
- `table` (IntArray): A 2D array of colors of shape (n, m).
- `rect` (NumberSequence2D): The rectangle in which to draw the color map. Should be of shape (2, 2).
- `filterMode` (str, optional): The filtering mode for the color map. Can be "nearest" or "linear". Default is "nearest".

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> table = P([0xFF0000, 0x00FF00], [0x0000FF, 0xFFFF00]).astype("uint32")
>>> rect = P([10, 10], [110, 110])
>>> form.image(table, rect)
```

### Form.toImage

Create an image snapshot of the current canvas.

**Returns:** skia.Image - An image snapshot of the current canvas, or None if the surface is not available.

**Example:**
```python
>>> form = Form()
>>> image = form.toImage()
>>> # Use the image for saving to a file or other operations.
```

### Form.toArray

Convert the current canvas to a NumPy array.

**Returns:** np.ndarray - A uint8 array of shape (height, width, 4) in RGBA channel order.

**Example:**
```python
>>> form = Form()
>>> arr = form.toArray()
>>> # Use the array for image processing or other operations.
```

### Form.save

Save the current canvas state (transformation matrix, clip region, etc.) onto a stack. Use `restore()` to pop the state back. This is useful when you want to temporarily modify the canvas state and then return to the previous state.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.save()
>>> form.clip(path)
>>> # ... draw something with clipping
>>> form.restore()
```

### Form.restore

Restore the canvas state to the most recently saved state from the stack. This undoes any transformations or clipping applied after the corresponding `save()` .

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.save()
>>> form.clip(path)
>>> form.restore()  # Remove the clip
```

### Form.translate

Translate (move) the canvas origin by (dx, dy).

**Arguments:**
- `dx` (float): Horizontal translation offset.
- `dy` (float): Vertical translation offset.

**Returns:** Form - self.

**Example:**
```python
>>> form.save()
>>> form.translate(50, 50)
>>> form.point([0, 0], 3)  # Drawn at (50, 50)
>>> form.restore()
```

### Form.rotate

Rotate the canvas by degrees, optionally around a pivot point.

**Arguments:**
- `degrees` (float): Rotation angle in degrees (clockwise).
- `pivot` (Optional[NumberSequence1D]): Center of rotation [x, y]. If None, rotates around origin.

**Returns:** Form - self.

**Example:**
```python
>>> form.save()
>>> form.rotate(45, pivot=[100, 100])
>>> form.rect([[75, 75], [125, 125]])
>>> form.restore()
```

### Form.scale

Scale the canvas by (sx, sy). If sy is None, scales uniformly by sx.

**Arguments:**
- `sx` (float): Horizontal scale factor.
- `sy` (Optional[float]): Vertical scale factor. If None, uses sx for uniform scaling.

**Returns:** Form - self.

**Example:**
```python
>>> form.save()
>>> form.translate(100, 100).scale(2)
>>> form.circle([[0, 0]])  # Drawn at 2x size
>>> form.restore()
```

### Form.skew

Skew the canvas. Values are tangent factors, not degrees (e.g. ` `sx=1` ` shears x by 45 degrees).

**Arguments:**
- `sx` (float): Horizontal skew factor (tan of the angle).
- `sy` (float): Vertical skew factor (tan of the angle).

**Returns:** Form - self.

**Example:**
```python
>>> form.save()
>>> form.translate(100, 100).skew(15, 0)
>>> form.rect([[-20, -20], [20, 20]])
>>> form.restore()
```

### Form.resetMatrix

Reset the transformation matrix to identity. Does not affect the clip stack.

**Returns:** Form - self.

**Example:**
```python
>>> form.translate(50, 50).rotate(45)
>>> form.resetMatrix()  # Back to identity
```

### Form.concat

Post-multiply the current transformation matrix by the given matrix (` `current = current * matrix` `).

**Arguments:**
- `matrix` (skia.Matrix): The transformation matrix to apply.

**Returns:** Form - self.

**Example:**
```python
>>> m = skia.Matrix()
>>> m.setRotate(45, 100, 100)
>>> form.concat(m)
```

### Form.clip

Set a clipping region using a path. All subsequent drawing operations will be confined to this region until `restore()` is called.

**Arguments:**
- `path` (skia.Path): The path defining the clipping region.
- `mode` (str, optional): The clipping mode. Can be "intersect" or "difference". Default is "intersect".
- `antialias` (bool, optional): Whether to apply antialiasing to the clip edge. Default is True.

**Returns:** Form - self.

**Example:**
```python
>>> circle = skia.Path()
>>> circle.addCircle(100, 100, 50)
>>> form.save().clip(circle)
>>> form.fillOnly("blue").rect([[0, 0], [200, 200]])
>>> form.restore()
```

### Form.erase

Erase pixels to fully transparent (both color and alpha) within the current clip region or a specified region. Useful for punching transparent holes in existing drawings.

**Arguments:**
- `region` (skia.Path, optional): The region to erase. If None, erases within the current clip region. Default is None.

**Returns:** Form - self.

**Example:**
```python
>>> form = Form()
>>> form.fillOnly("blue").rect([[0, 0], [200, 200]])
>>> circle = skia.Path()
>>> circle.addCircle(100, 100, 50)
>>> form.save().clip(circle).erase().restore()  # Punch a hole
```
