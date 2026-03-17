"""Performance benchmarks comparing numpy vs cupy backends.

These benchmarks measure the same operations under both backends to highlight
where GPU acceleration provides speedups. Operations that benefit from cupy
are typically large-scale vectorized computations. Operations bound by Python
overhead or Skia interop show less improvement.

Run with:
    pytest tests/benchmarks/test_backend_benchmark.py -v --benchmark-group-by=group
    pytest tests/benchmarks/test_backend_benchmark.py -v --benchmark-sort=name

CuPy tests are automatically skipped if CuPy is unavailable or CUDA is unusable.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ptspy.core.form import Form

# Import np_wrapper directly to avoid pulling the full ptspy package during backend-only benchmarks.
_spec = importlib.util.spec_from_file_location(
    "np_wrapper", str(Path(__file__).resolve().parents[2] / "src" / "ptspy" / "np_wrapper.py")
)
npw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(npw)

try:
    import cupy  # type: ignore[import-untyped]

    HAS_CUPY = True
    try:
        HAS_CUDA = bool(cupy.cuda.is_available())
    except Exception:
        HAS_CUDA = False
except ImportError:
    HAS_CUPY = False
    HAS_CUDA = False

CUDA_SKIP_REASON = "CUDA GPU not available" if HAS_CUPY else "CuPy not installed"
requires_cupy_cuda = pytest.mark.skipif(not HAS_CUDA, reason=CUDA_SKIP_REASON)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_backend(backend_name):
    """Switch np_wrapper to the given backend and return the module."""
    npw.reset(backend_name)
    return npw


def _teardown_backend():
    npw.reset("numpy")


@pytest.fixture(scope="session", autouse=True)
def _warmup_cupy_once():
    """Initialize the CUDA context once so the first CuPy benchmark is not polluted by startup cost."""
    if not HAS_CUDA:
        yield
        return

    _setup_backend("cupy")
    arr = cupy.arange(1, dtype=cupy.float32)
    cupy.cuda.Stream.null.synchronize()
    del arr
    _teardown_backend()
    yield
    _teardown_backend()


# ---------------------------------------------------------------------------
# 1. Array creation — linspace / arange
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1_000, 100_000, 1_000_000], ids=["1K", "100K", "1M"])
class TestLinspace:
    """Benchmark linspace under both backends."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def test_numpy_linspace(self, benchmark, n):
        _setup_backend("numpy")
        benchmark.group = f"linspace-{n}"
        benchmark(npw.linspace, 0.0, 1.0, n)

    @requires_cupy_cuda
    def test_cupy_linspace(self, benchmark, n):
        _setup_backend("cupy")
        benchmark.group = f"linspace-{n}"

        def run():
            result = npw.linspace(0.0, 1.0, n)
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)


# ---------------------------------------------------------------------------
# 2. Element-wise math (sin, cos, exp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [10_000, 100_000, 1_000_000], ids=["10K", "100K", "1M"])
class TestElementwiseMath:
    """Benchmark vectorized trig/math operations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def test_numpy_elementwise_math(self, benchmark, n):
        _setup_backend("numpy")
        benchmark.group = f"sin_cos_exp-{n}"
        data = npw.linspace(0.0, 6.28, n)

        def run():
            return npw.sin(data) * npw.cos(data) + npw.exp(-data)

        benchmark(run)

    @requires_cupy_cuda
    def test_cupy_elementwise_math(self, benchmark, n):
        _setup_backend("cupy")
        benchmark.group = f"sin_cos_exp-{n}"
        data = npw.linspace(0.0, 6.28, n)

        def run():
            result = npw.sin(data) * npw.cos(data) + npw.exp(-data)
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)


# ---------------------------------------------------------------------------
# 3. Matrix operations — meshgrid + reshape (like P2/gridPoints)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [50, 200, 500], ids=["50x50", "200x200", "500x500"])
class TestMeshgrid:
    """Benchmark meshgrid creation (core of P2/gridPoints)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def test_numpy_meshgrid(self, benchmark, n):
        _setup_backend("numpy")
        benchmark.group = f"meshgrid-{n}x{n}"
        xs = npw.linspace(0, 1, n)
        ys = npw.linspace(0, 1, n)

        def run():
            gx, gy = npw.meshgrid(xs, ys)
            return npw.stack([gx.ravel(), gy.ravel()], axis=1)

        benchmark(run)

    @requires_cupy_cuda
    def test_cupy_meshgrid(self, benchmark, n):
        _setup_backend("cupy")
        benchmark.group = f"meshgrid-{n}x{n}"
        xs = npw.linspace(0, 1, n)
        ys = npw.linspace(0, 1, n)

        def run():
            gx, gy = npw.meshgrid(xs, ys)
            result = npw.stack([gx.ravel(), gy.ravel()], axis=1)
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)


# ---------------------------------------------------------------------------
# 4. Bitwise operations (like image() pixel packing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(256, 256), (1024, 1024), (2048, 2048)], ids=["256x256", "1024x1024", "2048x2048"])
class TestBitwisePixelPacking:
    """Benchmark ARGB→RGBA bitwise unpacking (used in Form.image())."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def test_numpy_bitwise_pixel_packing(self, benchmark, size):
        _setup_backend("numpy")
        h, w = size
        benchmark.group = f"pixel_pack-{h}x{w}"
        arr = npw.random.default_rng(42).integers(0, 0xFFFFFFFF, (h, w), dtype=npw.uint32)

        def run():
            rgba = npw.empty((h, w, 4), dtype=npw.uint8)
            rgba[..., 0] = (arr >> 16) & 0xFF
            rgba[..., 1] = (arr >> 8) & 0xFF
            rgba[..., 2] = arr & 0xFF
            rgba[..., 3] = 255
            return rgba

        benchmark(run)

    @requires_cupy_cuda
    def test_cupy_bitwise_pixel_packing(self, benchmark, size):
        _setup_backend("cupy")
        h, w = size
        benchmark.group = f"pixel_pack-{h}x{w}"
        arr = npw.random.default_rng(42).integers(0, 0xFFFFFFFF, (h, w), dtype=npw.uint32)

        def run():
            rgba = npw.empty((h, w, 4), dtype=npw.uint8)
            rgba[..., 0] = (arr >> 16) & 0xFF
            rgba[..., 1] = (arr >> 8) & 0xFF
            rgba[..., 2] = arr & 0xFF
            rgba[..., 3] = 255
            cupy.cuda.Stream.null.synchronize()
            return rgba

        benchmark(run)


# ---------------------------------------------------------------------------
# 5. Skia image upload boundary (Form.image)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(128, 128), (512, 512)], ids=["128x128", "512x512"])
class TestFormImage:
    """Benchmark Form.image() with a CPU color table under both backends.

    This isolates the Skia boundary path and shows whether a CuPy-active session
    still handles host-side image data efficiently.
    """

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def _make_inputs(self, size):
        h, w = size
        table = np.random.default_rng(42).integers(0, 0x01000000, (h, w), dtype=np.uint32) | np.uint32(0xFF000000)
        rect = np.array([[0.0, 0.0], [float(w), float(h)]], dtype=np.float32)
        return table, rect

    def test_numpy_form_image(self, benchmark, size):
        _setup_backend("numpy")
        benchmark.group = f"form_image_cpu_table-{size[0]}x{size[1]}"
        table, rect = self._make_inputs(size)
        form = Form(width=size[1], height=size[0])

        def run():
            form.image(table, rect, filterMode="nearest")
            return form

        benchmark(run)

    @requires_cupy_cuda
    def test_cupy_form_image(self, benchmark, size):
        _setup_backend("cupy")
        benchmark.group = f"form_image_cpu_table-{size[0]}x{size[1]}"
        table, rect = self._make_inputs(size)
        form = Form(width=size[1], height=size[0])

        def run():
            form.image(table, rect, filterMode="nearest")
            return form

        benchmark(run)


# ---------------------------------------------------------------------------
# 6. Linear algebra — matrix multiply (like transform2D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1_000, 10_000, 100_000], ids=["1K", "10K", "100K"])
class TestMatmul:
    """Benchmark matrix multiply (core of transform2D affine transforms)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def test_numpy_matmul(self, benchmark, n):
        _setup_backend("numpy")
        benchmark.group = f"matmul-{n}"
        pts = npw.random.default_rng(42).standard_normal((n, 3)).astype(npw.float32)
        matrix = npw.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=npw.float32)

        def run():
            return pts @ matrix.T

        benchmark(run)

    @requires_cupy_cuda
    def test_cupy_matmul(self, benchmark, n):
        _setup_backend("cupy")
        benchmark.group = f"matmul-{n}"
        pts = npw.random.default_rng(42).standard_normal((n, 3)).astype(npw.float32)
        matrix = npw.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=npw.float32)

        def run():
            result = pts @ matrix.T
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)


# ---------------------------------------------------------------------------
# 7. asnumpy transfer cost (GPU→CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1_000, 100_000, 1_000_000], ids=["1K", "100K", "1M"])
class TestAsnumpyTransfer:
    """Benchmark GPU→CPU transfer cost via asnumpy()."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def test_numpy_noop(self, benchmark, n):
        """Baseline: asnumpy on a numpy array (should be near-zero cost)."""
        _setup_backend("numpy")
        benchmark.group = f"asnumpy-{n}"
        data = npw.linspace(0, 1, n)
        benchmark(npw.asnumpy, data)

    @requires_cupy_cuda
    def test_cupy_transfer(self, benchmark, n):
        """GPU→CPU transfer cost for cupy arrays."""
        _setup_backend("cupy")
        benchmark.group = f"asnumpy-{n}"
        data = npw.linspace(0, 1, n)
        cupy.cuda.Stream.null.synchronize()
        benchmark(npw.asnumpy, data)


# ---------------------------------------------------------------------------
# 8. Einsum (used in spline sampling)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("segments,samples", [(10, 20), (100, 50), (500, 100)], ids=["10x20", "100x50", "500x100"])
class TestEinsum:
    """Benchmark einsum (core of sampleCardinal/sampleBspline)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def test_numpy_einsum(self, benchmark, segments, samples):
        _setup_backend("numpy")
        benchmark.group = f"einsum-{segments}x{samples}"
        TM = npw.random.default_rng(42).standard_normal((samples, 4)).astype(npw.float32)
        G = npw.random.default_rng(42).standard_normal((segments, 4, 2)).astype(npw.float32)

        def run():
            return npw.einsum("si,nid->nsd", TM, G)

        benchmark(run)

    @requires_cupy_cuda
    def test_cupy_einsum(self, benchmark, segments, samples):
        _setup_backend("cupy")
        benchmark.group = f"einsum-{segments}x{samples}"
        TM = npw.random.default_rng(42).standard_normal((samples, 4)).astype(npw.float32)
        G = npw.random.default_rng(42).standard_normal((segments, 4, 2)).astype(npw.float32)

        def run():
            result = npw.einsum("si,nid->nsd", TM, G)
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)


# ---------------------------------------------------------------------------
# 9. Color conversion (like arrayToImage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(100, 100), (500, 500), (1000, 1000)], ids=["100x100", "500x500", "1000x1000"])
class TestColorConversion:
    """Benchmark float→uint8 color conversion (like arrayToImage)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def test_numpy_color_conversion(self, benchmark, size):
        _setup_backend("numpy")
        h, w = size
        benchmark.group = f"color_conv-{h}x{w}"
        data = npw.random.default_rng(42).standard_normal((h, w, 3)).astype(npw.float32)

        def run():
            minVal = npw.min(data)
            maxVal = npw.max(data)
            pixel = (data - minVal) / (maxVal - minVal + 1e-9)
            rgb = npw.rint(pixel * 255).astype(npw.uint32)
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            return (r << 16) | (g << 8) | b

        benchmark(run)

    @requires_cupy_cuda
    def test_cupy_color_conversion(self, benchmark, size):
        _setup_backend("cupy")
        h, w = size
        benchmark.group = f"color_conv-{h}x{w}"
        data = npw.random.default_rng(42).standard_normal((h, w, 3)).astype(npw.float32)

        def run():
            minVal = npw.min(data)
            maxVal = npw.max(data)
            pixel = (data - minVal) / (maxVal - minVal + 1e-9)
            rgb = npw.rint(pixel * 255).astype(npw.uint32)
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            result = (r << 16) | (g << 8) | b
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)


# ---------------------------------------------------------------------------
# 10. HCL→LAB→RGB pipeline (colorCube core computation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [10, 24, 36], ids=["10", "24", "36"])
class TestColorCubePipeline:
    """Benchmark the vectorized HCL→LAB→RGB color space conversion.

    This is the compute-heavy core of colorCube(): building a 3D grid of
    HCL values and converting them through LAB to sRGB. Large cubes (36^3)
    create ~46K color conversions — a good GPU workload.
    """

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def _run_pipeline(self, steps):
        TWO_PI = 6.283185307179586
        h = npw.linspace(0, TWO_PI, steps, endpoint=False)
        c = npw.linspace(0, 100, steps)
        l = npw.linspace(0, 100, steps)
        H, C, L = npw.meshgrid(h, c, l, indexing="ij")
        hcl = npw.stack([H, C, L], axis=-1)

        # HCL → LAB (polar to cartesian)
        lab = npw.stack(
            [
                hcl[..., 2],
                hcl[..., 1] * npw.cos(hcl[..., 0]),
                hcl[..., 1] * npw.sin(hcl[..., 0]),
            ],
            axis=-1,
        )

        # LAB → XYZ → linear RGB (simplified)
        fy = (lab[..., 0] + 16) / 116
        fx = lab[..., 1] / 500 + fy
        fz = fy - lab[..., 2] / 200
        delta = 6.0 / 29.0
        X = 0.95047 * npw.where(fx > delta, fx**3, 3 * delta**2 * (fx - 4 / 29))
        Y = 1.00000 * npw.where(fy > delta, fy**3, 3 * delta**2 * (fy - 4 / 29))
        Z = 1.08883 * npw.where(fz > delta, fz**3, 3 * delta**2 * (fz - 4 / 29))

        r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
        g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
        b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

        rgb = npw.clip(npw.stack([r, g, b], axis=-1), 0, 1)
        return npw.rint(rgb * 255).astype(npw.uint8)

    def test_numpy_color_cube_pipeline(self, benchmark, steps):
        _setup_backend("numpy")
        benchmark.group = f"color_cube-{steps}"
        benchmark(self._run_pipeline, steps)

    @requires_cupy_cuda
    def test_cupy_color_cube_pipeline(self, benchmark, steps):
        _setup_backend("cupy")
        benchmark.group = f"color_cube-{steps}"

        def run():
            result = self._run_pipeline(steps)
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)


# ---------------------------------------------------------------------------
# 11. Gamut mapping — soft shoulder compression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [10_000, 100_000, 1_000_000], ids=["10K", "100K", "1M"])
class TestGamutMapping:
    """Benchmark soft gamut mapping (Reinhard-style shoulder compression).

    This operation runs on every pixel of every color conversion and uses
    heavy np.where/np.clip branching — a good test of GPU conditional ops.
    """

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def _run_gamut(self, n):
        data = npw.random.default_rng(42).standard_normal((n, 3)).astype(npw.float32) * 1.5
        t, s = 0.85, 1.5
        x = data
        above1 = x > 1
        inBlend = (x > t) & (x <= 1)
        below0 = x < 0
        overflow = 1 + (x - 1) / (1 + s * (x - 1))
        blendT = (x - t) / (1 - t)
        blend = t + (1 - t) * blendT / (1 + s * blendT * (1 - blendT))
        underflow = x / (1 - s * x)
        result = npw.where(above1, overflow, x)
        result = npw.where(inBlend, blend, result)
        result = npw.where(below0, underflow, result)
        return npw.clip(result, 0, 1)

    def test_numpy_gamut_mapping(self, benchmark, n):
        _setup_backend("numpy")
        benchmark.group = f"gamut_map-{n}"
        benchmark(self._run_gamut, n)

    @requires_cupy_cuda
    def test_cupy_gamut_mapping(self, benchmark, n):
        _setup_backend("cupy")
        benchmark.group = f"gamut_map-{n}"

        def run():
            result = self._run_gamut(n)
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)


# ---------------------------------------------------------------------------
# 12. Spline basis — Bernstein + einsum (sampleBezier core)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("segments,samples", [(50, 50), (200, 100), (500, 200)], ids=["50x50", "200x100", "500x200"])
class TestSplineSampling:
    """Benchmark Bernstein basis evaluation + einsum (sampleBezier core).

    This is the inner loop of all spline rendering: build Bernstein weights,
    stack geometry matrices, and contract via einsum.
    """

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        _teardown_backend()

    def _run_spline(self, segments, samples):
        t = npw.linspace(0, 1, samples, endpoint=False, dtype=npw.float32)
        u = 1.0 - t
        B = npw.stack([u**3, 3.0 * u**2 * t, 3.0 * u * t**2, t**3], axis=1)

        rng = npw.random.default_rng(42)
        G = rng.standard_normal((segments, 4, 2)).astype(npw.float32)
        return npw.einsum("si,nid->nsd", B, G).reshape(-1, 2)

    def test_numpy_spline_sampling(self, benchmark, segments, samples):
        _setup_backend("numpy")
        benchmark.group = f"spline-{segments}x{samples}"
        benchmark(self._run_spline, segments, samples)

    @requires_cupy_cuda
    def test_cupy_spline_sampling(self, benchmark, segments, samples):
        _setup_backend("cupy")
        benchmark.group = f"spline-{segments}x{samples}"

        def run():
            result = self._run_spline(segments, samples)
            cupy.cuda.Stream.null.synchronize()
            return result

        benchmark(run)
