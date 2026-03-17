"""Performance benchmarks for core ptspy functions."""

import pytest
import numpy as np
from ptspy.core.elements import P, P2, P3
from ptspy.core.compose import steps, gridPoints
from ptspy.core.process import interpolate, bound
from ptspy.core.colors import colorCube


# ── P (1D point construction) ──────────────────────────────────────────────


@pytest.mark.parametrize("n", [10, 100, 1000], ids=["n=10", "n=100", "n=1000"])
def test_bench_P(benchmark, n):
    data = list(range(n))
    benchmark.group = "P"
    benchmark(P, data)


# ── P2 (2D grid construction) ──────────────────────────────────────────────


@pytest.mark.parametrize("n", [10, 50, 100], ids=["10x10", "50x50", "100x100"])
def test_bench_P2(benchmark, n):
    xs = list(range(n))
    ys = list(range(n))
    benchmark.group = "P2"
    benchmark(P2, xs, ys)


# ── P3 (3D grid construction) ──────────────────────────────────────────────


@pytest.mark.parametrize("n", [5, 10, 20], ids=["5³", "10³", "20³"])
def test_bench_P3(benchmark, n):
    xs = list(range(n))
    ys = list(range(n))
    zs = list(range(n))
    benchmark.group = "P3"
    benchmark(P3, xs, ys, zs)


# ── steps (keyframe generation) ────────────────────────────────────────────


@pytest.mark.parametrize("count", [10, 100, 1000], ids=["count=10", "count=100", "count=1000"])
def test_bench_steps(benchmark, count):
    benchmark.group = "steps"
    benchmark(steps, 0, 10, count)


# ── gridPoints (multi-dim grid) ───────────────────────────────────────────


@pytest.mark.parametrize("n", [10, 20, 50], ids=["[10,10]", "[20,20]", "[50,50]"])
def test_bench_gridPoints(benchmark, n):
    benchmark.group = "gridPoints"
    benchmark(gridPoints, [n, n], [1, 1])


# ── interpolate (asPath mode) ─────────────────────────────────────────────


@pytest.mark.parametrize("nt", [10, 100, 1000], ids=["t=10", "t=100", "t=1000"])
def test_bench_interpolate_asPath(benchmark, nt):
    pts = np.array([[0.0, 0.0], [5.0, 10.0], [10.0, 0.0], [15.0, 10.0]])
    t = np.linspace(0, 1, nt)
    benchmark.group = "interpolate-asPath"
    benchmark(interpolate, pts, t, asPath=True)


# ── interpolate (pairwise mode) ───────────────────────────────────────────


@pytest.mark.parametrize("nt", [10, 100, 1000], ids=["t=10", "t=100", "t=1000"])
def test_bench_interpolate_pairwise(benchmark, nt):
    pts = np.linspace([0.0, 0.0], [10.0, 10.0], nt + 1)
    t = np.full(nt, 0.5)
    benchmark.group = "interpolate-pairwise"
    benchmark(interpolate, pts, t)


# ── bound (clip mode) ─────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [100, 1000, 10000], ids=["n=100", "n=1000", "n=10000"])
def test_bench_bound_clip(benchmark, n):
    data = np.random.default_rng(42).uniform(-10, 20, n)
    benchmark.group = "bound-clip"
    benchmark(bound, data, 0, 10)


# ── bound (loopBack mode) ─────────────────────────────────────────────────


@pytest.mark.parametrize("n", [100, 1000, 10000], ids=["n=100", "n=1000", "n=10000"])
def test_bench_bound_loopBack(benchmark, n):
    data = np.random.default_rng(42).uniform(-10, 20, n)
    benchmark.group = "bound-loopBack"
    benchmark(bound, data, 0, 10, loopBack=True)


# ── colorCube (HCL→RGB pipeline) ──────────────────────────────────────────


@pytest.mark.parametrize("s", [10, 20, 36], ids=["steps=10", "steps=20", "steps=36"])
def test_bench_colorCube(benchmark, s):
    benchmark.group = "colorCube"
    benchmark(colorCube, s)
