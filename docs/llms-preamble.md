# Pts.py Guide

## Install

```python
pip install ptspy
from ptspy import *
```

Requires Python 3.10+. Uses numpy + skia-python.

## Drawing

Use `@frame` for a single image and `@frames` for animation. The first parameter `form` is a `Form` instance (Skia-backed drawing surface).

```python
@frame(bg="000", size=(300, 300))
def draw(form, radius):
    points = regularPolygon(form.size / 2, 6, radius)
    form.fillOnly("fff").points(points, radius / 2)

draw(100)
```

```python
@frames(bg="000", size=(300, 300), numFrames=60)
def draw(form, index, radius):
    form.fillOnly("fff").point(form.size / 2, radius + index)

video = draw(radius=10)
```

Key `Form` methods: `point()`, `points()`, `line()`, `lines()`, `rect()`, `circle()`, `polygon()`, `image()`, `colorMap()`.
Style methods (chainable): `fill()`, `fillOnly()`, `stroke()`, `strokeOnly()`, `noFill()`, `noStroke()`.
Transform methods: `save()`, `restore()`, `rotate()`, `translate()`, `scale()`.

## Points

`P()`, `P2()`, `P3()` are shorthands for creating numpy float32 arrays.

| Constructor | Description | Example | Shape |
|---|---|---|---|
| `P(1, 2, 3)` | 1D point | `array([1., 2., 3.])` | `(3,)` |
| `P((1,2), (3,4))` | Multiple points | `array([[1.,2.],[3.,4.]])` | `(2, 2)` |
| `P2([1,2], [4,5])` | 2D grid of points | Grid from axes | `(2, 2, 2)` |
| `P3(xs, ys, zs)` | 3D grid of points | Cube from axes | `(n, n, n, 3)` |

All numpy operations work on points: `pts + 10`, `pts * 2`, `np.mean(pts, axis=0)`.

## Partial Application

Most ptspy functions support partial application. Use `@partial` to create your own. Use `TBD` as a placeholder.

```python
# Built-in: steps() with incomplete parameters returns a function
pointsFn = steps([40, 150], shaping="cubicOut")
result = pointsFn([260, 150], 10)  # complete it later

# Custom partial function
@partial(placeholder=TBD)
def createShape(form, center, sides, rotation):
    poly = regularPolygon(center=center, radius=50, sides=sides, closed=True)
    form.save()
    form.rotate(rotation, pivot=center)
    form.fillOnly("fff", 2).polygon(poly)
    form.restore()

# Partially apply — set sides now, form and rotation later
shape = createShape(form=TBD, center=P(150, 150), sides=6, rotation=TBD)
shape(form=myForm, rotation=45)  # complete it
```

## Pipelines

Compose functions so output of one becomes input to the next. Use `pipeline()` or the `>>` operator.

```python
def a(x, n):
    return P(x, x * n)

def b(p):
    return (P(p, p + 100), 5)  # tuple → multiple args to next

def c(path, repeat):
    return steps(path[0], path[1], repeat)

# Three equivalent ways:
fn = pipeline(a, b, c)
fn = F(a) >> b >> c
fn = F(c) << b << a

result = fn(10, 2)
```

Options: `lazy=True` returns intermediate results as a generator. `trace=True` prints each stage.

## Colors

```python
# Create a color cube (hue × chroma × lightness)
cube = colorCube(steps=25)

# Pick a color by hue, chroma, lightness (0–1 range)
color = colorPicker(cube, hue=0.2, chroma=0.9, lightness=0.7)

# Generate a color scheme
palette = colorScheme(cube, hues=9, swatches=9, intensity=0.5)

# Interpolate between colors
indices = colorInterpolation(cube, start=P(80, 28, 99), end=P(10, 99, 70), steps=8)

# Create gradients (linear, radial, sweep, conical)
shader = gradient(P([[100, 100]]), colors=[0xFF2099FF, 0xFFFF66FF], style="sweep")
```

Color values: hex strings (`"ff0000"`, `"fff"`), ARGB ints (`0xFFFF0000`), or RGB/RGBA arrays.

## Module Map

| Module | Purpose | Key Exports |
|---|---|---|
| `core.elements` | Point constructors, partial application | `P`, `P2`, `P3`, `TBD`, `@partial` |
| `core.process` | Pipeline composition, array processing | `pipeline`, `F`, `steps`, `interpolate`, `shaping` |
| `core.form` | Skia drawing surface | `Form`, `@frame`, `@frames` |
| `core.compose` | Array construction helpers | `populate`, `steps`, `gridPoints`, `regularPolygon` |
| `core.colors` | Color manipulation | `colorCube`, `colorPicker`, `colorScheme`, `gradient` |
| `core.util` | General utilities | `randomizer`, `random`, `pretty` |
| `core.typing` | Type aliases | `PArray`, `ColorLike`, `NumberSequenceND` |

---

