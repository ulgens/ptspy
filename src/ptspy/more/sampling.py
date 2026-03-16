"""Spatial sampling algorithms for ptspy.

Provides blue-noise point distributions using Poisson-Disc sampling (Bridson's algorithm),
generalized to N dimensions.

Example:
    >>> from ptspy.more import poissonDisc
    >>> pts = poissonDisc([200, 200], 10, seed=42)
    >>> pts.shape  # (N, 2) where N depends on packing
"""
from __future__ import annotations

import itertools

from ptspy.typing import PArray, NumberSequenceND
from .. import np_wrapper as np
from ptspy import partial, randomizer, TBD
from ptspy.core.elements import toPArray


@partial(optional=True, placeholder=TBD)
def poissonDisc(
    bounds: NumberSequenceND,
    radius: float,
    k: int = 15,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> PArray:
    """Generate blue-noise distributed points via Bridson's Poisson-Disc sampling.

    The algorithm uses a background grid with 5^d neighbor offsets, so dimensions above ~4 become expensive.
    For d=3 there are 125 neighbor cells checked per candidate; for d=5 there are 3125.

    Returns an (N, d) array where every pair of points is at least `radius` apart.

    Args:
        bounds (NumberSequenceND): Domain extent. 1-D for size-only (origin at zero), 2D
            for a `[min, max]` bounding rectangle. Higher dimensions can be expensive.
        radius (float): Minimum distance between any two points.
        k (int, Optional): Candidate samples per active point (quality knob). Default 15.
        seed (int, Optional): RNG seed for reproducibility.
        rng (np.random.Generator, Optional): Pass a pre-existing `numpy.random.Generator` if needed.

    Returns:
        PArray: Array of shape `(N, d)` with the sampled points.

    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    bounds_arr = np.atleast_1d(np.asarray(bounds, dtype=np.float64))

    # Interpret bounds shape: (2, d) = [min, max], (d,) = size from origin 0
    if bounds_arr.ndim == 2 and bounds_arr.shape[0] == 2:
        origin_arr = bounds_arr[0]
        size_arr = bounds_arr[1] - bounds_arr[0]
    elif bounds_arr.ndim == 1 and len(bounds_arr) > 0:
        origin_arr = np.zeros_like(bounds_arr)
        size_arr = bounds_arr
    else:
        raise ValueError(f"bounds must be 1-D size or (2, d) bounding box, got shape {bounds_arr.shape}")

    if np.any(size_arr <= 0):
        raise ValueError(f"bounds size must be positive in every dimension, got {size_arr.tolist()}")

    d = len(size_arr)
    gen = randomizer(seed, rng)

    # --- grid setup (flat 1-D array for vectorized lookup) ---
    cellSize = radius / np.sqrt(d)
    gridShape = np.ceil(size_arr / cellSize).astype(np.intp)
    gridStrides = np.concatenate([np.cumprod(gridShape[::-1])[::-1][1:], [1]]).astype(np.intp)
    gridSize = int(np.prod(gridShape))
    flatGrid = np.full(gridSize, -1, dtype=np.intp)

    # Precompute neighbor offset vectors ([-2..+2]^d) as flat-index deltas
    neighborNd = np.array(list(itertools.product(range(-2, 3), repeat=d)), dtype=np.intp)
    neighborFlatDeltas = neighborNd @ gridStrides  # flat offset per neighbor

    # --- pre-allocated point storage ---
    # Upper-bound estimate: area / circle-packing lower bound (generous 2x headroom)
    estimatedMax = max(int(np.prod(size_arr) / (radius ** d) * 4), 64)
    pointsBuf = np.empty((estimatedMax, d), dtype=np.float64)
    nPoints = 0
    active: list[int] = []

    radiusSq = radius * radius

    def _insert(pt: np.ndarray) -> None:
        nonlocal nPoints, pointsBuf
        idx = nPoints
        if idx >= len(pointsBuf):
            pointsBuf = np.concatenate([pointsBuf, np.empty((len(pointsBuf), d), dtype=np.float64)])
        pointsBuf[idx] = pt
        cellIdx = (pt / cellSize).astype(np.intp)
        flatGrid[int(cellIdx @ gridStrides)] = idx
        active.append(idx)
        nPoints += 1

    # --- seed point ---
    _insert(gen.uniform(0, size_arr))

    # --- main loop ---
    while active:
        activeIdx = gen.integers(len(active))
        centerIdx = active[activeIdx]
        center = pointsBuf[centerIdx]

        # Generate k candidates in annulus [radius, 2*radius] — vectorized
        directions = gen.standard_normal((k, d))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        directions /= norms
        radii = gen.uniform(radius, 2.0 * radius, size=(k, 1))
        candidates = center + directions * radii

        # Bounds check — vectorized
        inBounds = np.all((candidates >= 0) & (candidates < size_arr), axis=1)

        found = False
        for ci in np.flatnonzero(inBounds):
            candidate = candidates[ci]
            cellIdx = (candidate / cellSize).astype(np.intp)
            cellFlat = int(cellIdx @ gridStrides)

            # Compute flat indices for all neighbors, then filter valid ones
            nbFlat = cellFlat + neighborFlatDeltas
            nbNd = cellIdx + neighborNd
            valid = np.all((nbNd >= 0) & (nbNd < gridShape), axis=1)
            nbFlat = nbFlat[valid]

            # Batch grid lookup
            ptIndices = flatGrid[nbFlat]
            occupied = ptIndices[ptIndices >= 0]

            if len(occupied) > 0:
                # Vectorized distance check against all occupied neighbors
                diffs = pointsBuf[occupied] - candidate
                distsSq = np.einsum('ij,ij->i', diffs, diffs)
                if np.any(distsSq < radiusSq):
                    continue

            _insert(candidate)
            found = True
            break

        if not found:
            active[activeIdx] = active[-1]
            active.pop()

    result = pointsBuf[:nPoints].astype(np.float32) + origin_arr.astype(np.float32)
    return toPArray(result, gpu=True)
