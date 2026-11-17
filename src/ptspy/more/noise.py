"""
Noise generation module for ptspy.

This module provides various noise generation functions, primarily focused on Perlin noise
implementation. It includes both custom Perlin noise generators and Skia-based noise shaders
for creating natural-looking patterns, textures, and procedural content.


Example:
    >>> # Create a 2D Perlin noise generator
    >>> noise = createPerlinNoise2D((100, 100), seed=42)
    >>> values = noise()
    >>>
    >>> # Create a Skia noise shader
    >>> shader = skiaPerlinNoise(0.01, 0.01, 4, 42.0)
"""

from __future__ import annotations

from ptspy.typing import IntArrayOrList, NumberSequence1D, PArray
from .. import np_wrapper as np
from ptspy import partial, randomizer, P2, P
import skia


def createPerlinNoise2D(
    size: IntArrayOrList,
    seed: int | None = None,
    *,
    grads: np.ndarray | None = None,
) -> PerlinNoise:
    """
    Create a 2D Perlin noise generator with the specified grid size.

    This function creates a PerlinNoise object that can generate 2D Perlin noise
    patterns. The noise generator maintains its own permutation table and gradient
    vectors, allowing for consistent noise generation across multiple calls.

    Args:
        size (IntArrayOrList): The dimensions of the noise grid as [width, height].
        seed (Optional[int], optional): Random seed for reproducible noise patterns.
        grads (Optional[np.ndarray], optional): Custom gradient table. If None, uses the default 8-direction gradient table.

    Returns:
        PerlinNoise: An instance of the Perlin noise generator

    Examples:
        >>> noise = createPerlinNoise2D((256, 256), seed=123)
        >>> value = noise(freq=2.0, octaves=2) # same as noise.generate(...)
    """
    grid = getPerlin2DGrid(size[0], size[1])
    permTab = _makePerm(seed)
    gtab = grads if grads is not None else _gradTable2D()
    return PerlinNoise(grid, permTab, gtab, seed)


@partial(optional=True)
def skiaPerlinNoise(
    freqX: float, freqY: float, octaves: int, seed: float, tileSize: skia.ISize = None, type: str = "fractal"
) -> skia.Shader:
    """
    Generates a Perlin noise shader using skia.

    This function creates a Perlin noise shader using either fractal noise or turbulence noise. The generated shader can be used
    for various visual effects in graphics and design. You can then use [form.fill](`core/Form.fill`) to apply the shader.

    Args:
        freqX (float): The frequency of the noise in the X direction.
        freqY (float): The frequency of the noise in the Y direction.
        octaves (int): The number of octaves to use for the noise. Higher values result in more detail.
        seed (float): The seed value for randomization. Changing this value will result in different noise patterns.
        tileSize (skia.ISize, optional): The tile size for the noise. If provided, the noise will repeat seamlessly at this size.
        type (str, optional): The type of Perlin noise to generate. Can be "fractal" for fractal noise or "turbulence" for turbulence noise. Default is "fractal".

    Returns:
        skia.Shader: A Skia shader object that generates Perlin noise.

    Examples:
        >>> shader = skiaPerlinNoise(0.01, 0.01, 2, 42.0)
        >>> isinstance(shader, skia.Shader)
        True

        >>> shader_turbulence = skiaPerlinNoise(0.01, 0.01, 2, 42.0, type="turbulence")
        >>> isinstance(shader_turbulence, skia.Shader)
        True
    """

    if type not in ("fractal", "turbulence"):
        raise ValueError(f"type must be 'fractal' or 'turbulence', got {type!r}")
    if type == "turbulence":
        return skia.PerlinNoiseShader.MakeTurbulence(freqX, freqY, octaves, seed, tileSize)
    return skia.PerlinNoiseShader.MakeFractalNoise(freqX, freqY, octaves, seed, tileSize)


class PerlinNoise:
    """
    2D Perlin noise generator class. Typically created via `createPerlinNoise2D` function.
    """

    __slots__ = ("grid", "perm", "grads", "seed")

    def __init__(self, grid: PArray, perm: PArray, grads: PArray, seed: int | None):
        self.grid = grid
        self.perm = perm
        self.grads = grads
        self.seed = seed

    def __repr__(self) -> str:
        return f"PerlinNoise(seed={self.seed}, perm.shape={self.perm.shape}, grads.shape={self.grads.shape})"

    def __call__(
        self,
        offset: NumberSequence1D = [0.1, 0.1],
        freq: float | NumberSequence1D = 1.0,
        *,
        octaves: int = 1,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        period: IntArrayOrList | None = None,
        autoTile: bool = False,
    ) -> PArray:
        """Shorthand for `generate()` — accepts the same arguments."""
        return self.generate(
            offset=offset,
            freq=freq,
            octaves=octaves,
            lacunarity=lacunarity,
            gain=gain,
            period=period,
            autoTile=autoTile,
        )

    def resetGrid(self):
        """Reset the internal grid to its original state."""
        self.grid = getPerlin2DGrid(self.grid.shape[0], self.grid.shape[1])

    def generate(
        self,
        offset: NumberSequence1D = [0.1, 0.1],
        freq: float | NumberSequence1D = 1.0,
        *,
        octaves: int = 1,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        period: IntArrayOrList | None = None,
        autoTile: bool = False,
    ):
        """
        Sample one step of 2D Perlin noise or FBM over the internal grid with an optional offset. If octaves > 1, FBM is used.

        Args:
            offset (NumberSequence1D, optional): Offset to apply to the internal grid before sampling. Use this to animate or shift the noise pattern.
            freq (float | NumberSequence1D, optional): Base frequency either as a scalar or set x and y separately.
            octaves (int, optional): Number of noise layers to sum.
            lacunarity (float, optional): Frequency multiplier per octave.
            gain (float, optional): Amplitude multiplier per octave.
            period (Optional[IntArrayOrList], optional): Tiling period for Perlin sampling.
            autoTile (bool, optional): If True, adjusts per-octave periods to preserve seamless tiling across octaves. Ignored if `period` is None.

        Returns:
            PArray: noise values

        Examples:
            >>> noise = createPerlinNoise2D((256, 256))
            >>> # t can be a frame index or time value
            >>> frame = noise.generate(offset=P(t * 0.1, t * 0.1), freq=0.4, octaves=1)
            >>> # shorthand, same as above:
            >>> frame = noise(offset=P(t * 0.1, t * 0.1), freq=0.4, octaves=1)
        """

        if octaves <= 1:
            return self.sample(self.grid + offset, freq=freq, period=period, autoTile=autoTile)
        else:
            return self.sampleFBM(
                self.grid + offset,
                freq=freq,
                octaves=octaves,
                lacunarity=lacunarity,
                gain=gain,
                period=period,
                autoTile=autoTile,
            )

    def sample(
        self,
        pts: PArray,
        freq: float | NumberSequence1D = 1.0,
        *,
        period: IntArrayOrList | None = None,
        autoTile: bool = False,
    ) -> PArray:
        """
        Sample 2D Perlin noise.

        Args:
            pts (PArray): Sample coordinates of shape `(..., 2)`.
            freq (float | NumberSequence1D, optional): Frequency either as a scalar or set x and y separately.
            period (Optional[IntArrayOrList], optional): Tiling period for Perlin sampling.
            autoTile (bool, optional): If True, adjusts per-octave periods to preserve seamless tiling across octaves. Ignored if `period` is None.

        Returns:
            PArray: noise values

        Examples:
            >>> values = noise.sample(P2(np.linspace(0,1,128), np.linspace(0,1,128)))
            >>> values.shape
            (128, 128)
        """

        fx, fy = _asFreqTuple(freq)
        periodUse = period
        if autoTile and period is None:
            px = max(1, int(round(fx)))
            py = max(1, int(round(fy)))
            fx, fy = float(px), float(py)
            periodUse = (px, py)

        return perlin2D(
            pts,
            freq=(fx, fy),
            period=periodUse,
            perm=self.perm,
            grads=self.grads,
            seed=None,
        )

    def sampleFBM(
        self,
        pts: PArray,
        freq: float | NumberSequence1D = 1.0,
        *,
        octaves: int = 4,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        period: IntArrayOrList | None = None,
        autoTile: bool = False,
    ) -> PArray:
        """
        Sample 2D Perlin fractal Brownian motion (fBM) over a set of points.

        Args:
            pts (PArray): Sample coordinates of shape `(..., 2)`.
            freq (float | NumberSequence1D, optional): Base frequency either as a scalar or set x and y separately.
            octaves (int, optional): Number of noise layers to sum.
            lacunarity (float, optional): Frequency multiplier per octave.
            gain (float, optional): Amplitude multiplier per octave.
            period (Optional[IntArrayOrList], optional): Tiling period for Perlin sampling.
            autoTile (bool, optional): If True, adjusts per-octave periods to preserve seamless tiling across octaves. Ignored if `period` is None.

        Returns:
            PArray: fBM values

        Examples:
            >>> fbm = noise.sampleFBM(P2(np.linspace(0,1,128), np.linspace(0,1,128)))
            >>> fbm.shape
            (128, 128)
        """

        fx0, fy0 = _asFreqTuple(freq)
        total = np.zeros(pts.shape[:-1])
        amp = 1.0
        maxAmp = 0.0
        fx, fy = fx0, fy0

        for i in range(octaves):
            (fxi, fyi), periodI = _perOctaveFreqAndPeriod(fx, fy, period, i, lacunarity, autoTile)
            total += amp * perlin2D(
                pts,
                freq=(fxi, fyi),
                period=periodI,
                perm=self.perm,
                grads=self.grads,
                seed=None,
            )
            maxAmp += amp
            amp *= gain
            fx *= lacunarity
            fy *= lacunarity

        return total / (maxAmp if maxAmp > 0 else 1)


# -----------------------------------
# Internal functions


def _fade(t: PArray) -> PArray:
    """Perlin's improved smoothstep: 6t^5 - 15t^4 + 10t^3 (Ken Perlin, "Improving Noise", SIGGRAPH 2002)."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: PArray, b: PArray, t: PArray) -> PArray:
    return a + t * (b - a)


def _makePerm(seed: int | None) -> PArray:
    """Build a permutation table: 256 entries shuffled, then duplicated to 512 to avoid index wrapping."""
    r = randomizer(seed)
    p = np.arange(256, dtype=np.int32)
    r.shuffle(p)

    return np.concatenate([p, p]).astype(np.int32)


def _gradTable2D() -> PArray:
    """8 gradient directions (axes + diagonals), all scaled by 1/sqrt(2) so diagonals have unit length."""
    return P([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]) * (1.0 / np.sqrt(2.0))


def _hash2D(ix: PArray, iy: PArray, perm: PArray) -> PArray:
    """Hash 2D lattice coords via the 256-entry permutation table (indices masked to 0-255)."""
    return perm[(perm[ix & 255] + (iy & 255)) & 255]


def _dot2(g: PArray, dx: PArray, dy: PArray) -> PArray:
    """Dot product of gradient vectors `g` (..., 2) with offset scalars `dx`, `dy` (broadcast-compatible)."""
    return g[..., 0] * dx + g[..., 1] * dy


def _asFreqTuple(freq) -> tuple[float, float]:
    """Normalize freq to (fx, fy): accepts a scalar (applied to both axes) or a 2-element sequence."""
    return (freq, freq) if isinstance(freq, (int, float)) else freq


def _perOctaveFreqAndPeriod(
    fx: float,
    fy: float,
    basePeriod: IntArrayOrList | None,
    i: int,
    lacunarity: float,
    autoTile: bool,
) -> tuple[tuple[float, float], IntArrayOrList | None]:
    """Compute per-octave freq and tiling period. When ``autoTile`` is True, the period scales with
    ``lacunarity**i`` so each octave tiles at an integer multiple of the base period, preserving seamless tiling."""
    if not autoTile:
        return (fx, fy), basePeriod
    if basePeriod is None:
        px = max(1, int(round(fx)))
        py = max(1, int(round(fy)))
        return (float(px), float(py)), (px, py)
    scale = lacunarity**i
    px = max(1, int(round(basePeriod[0] * scale)))
    py = max(1, int(round(basePeriod[1] * scale)))
    return (fx, fy), (px, py)


def getPerlin2DGrid(w: int, h: int, domain: tuple[float, float] = (0.0, 10.0)) -> PArray:
    """
    Generate a 2D grid for Perlin noise sampling.

    Creates a coordinate grid spanning the given domain in both dimensions, providing the spatial coordinates for noise evaluation.

    Args:
        w (int): Width of the grid (number of samples along x-axis).
        h (int): Height of the grid (number of samples along y-axis).
        domain (Tuple[float, float], optional): The (min, max) range for both axes. Default is (0, 10).

    Data:
        internal: true

    Returns:
        PArray: A 2D array of shape (h, w, 2).
    """
    xs = np.linspace(domain[0], domain[1], w)
    ys = np.linspace(domain[0], domain[1], h)
    return P2(xs, ys)


@partial(optional=True)
def perlin2D(
    grid: PArray,
    *,
    freq: float | NumberSequence1D = 1.0,
    seed: int | None = None,
    period: IntArrayOrList | None = None,
    perm: PArray | None = None,
    grads: PArray | None = None,
) -> PArray:
    """
    Vectorized 2D Perlin noise. Returns values in ~[-1, 1].

    Args:
        grid (PArray): Sample coordinates of shape `(..., 2)`.
        freq (float | NumberSequence1D, optional): Base frequency either as a scalar or set x and y separately.
        seed (int, optional): Seed used only if 'perm' is not provided
        period (IntArrayOrList, optional): Tiling period for Perlin sampling.
        perm (PArray, optional): Optional permutation table (256*2 int32) to reuse across frames
        grads (PArray, optional): Optional gradient table (k, 2). Default is 8 canonical directions.

    Data:
        internal: true

    Returns:
        PArray: pts[...,0] with float noise
    """
    if grid.shape[-1] != 2:
        raise ValueError("perlin2D expects pts of shape (..., 2)")

    px, py = period if period else (None, None)
    permTab = perm if perm is not None else _makePerm(seed)
    gtab = grads if grads is not None else _gradTable2D()

    fx, fy = (freq, freq) if isinstance(freq, (int, float)) else freq

    # Scale to lattice space
    sx = grid[..., 0] * fx
    sy = grid[..., 1] * fy

    # Integer lattice corners
    ix0 = np.floor(sx).astype(np.int32)
    iy0 = np.floor(sy).astype(np.int32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Relative local coords inside cell
    fx = sx - ix0
    fy = sy - iy0

    # Optional tiling: wrap lattice integers before hashing
    if px is not None:
        ix0 = ix0 % px
        ix1 = ix1 % px
    if py is not None:
        iy0 = iy0 % py
        iy1 = iy1 % py

    # Hash → gradient index
    nGrads = len(gtab)
    h00 = _hash2D(ix0, iy0, permTab) % nGrads
    h10 = _hash2D(ix1, iy0, permTab) % nGrads
    h01 = _hash2D(ix0, iy1, permTab) % nGrads
    h11 = _hash2D(ix1, iy1, permTab) % nGrads

    # Gather gradients (broadcast along pts shape)
    g00 = gtab[h00]
    g10 = gtab[h10]
    g01 = gtab[h01]
    g11 = gtab[h11]

    # Corner offset vectors
    # top-left (ix0, iy0) uses ( fx,  fy)
    # top-right(ix1, iy0) uses (fx-1, fy)
    # bot-left(ix0, iy1) uses ( fx, fy-1)
    # bot-right(ix1,iy1) uses (fx-1,fy-1)
    n00 = _dot2(g00, fx, fy)
    n10 = _dot2(g10, fx - 1, fy)
    n01 = _dot2(g01, fx, fy - 1)
    n11 = _dot2(g11, fx - 1, fy - 1)

    # Fade and bilerp
    u = _fade(fx)
    v = _fade(fy)
    nx0 = _lerp(n00, n10, u)
    nx1 = _lerp(n01, n11, u)
    nxy = _lerp(nx0, nx1, v)

    # Empirical normalization to ~[-1,1] is already decent with chosen grads
    return nxy.astype(np.float32)
