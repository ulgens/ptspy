"""Tests for CuPy backend compatibility.

These tests validate that all ptspy core operations produce correct results when
run through np_wrapper, and that Skia boundary conversions are handled properly.

CuPy-specific tests are gated behind `HAS_CUPY` and skipped otherwise.

Run with:
    pytest tests/test_cupy_compat.py -v
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Import np_wrapper directly to avoid skia dependency
_spec = importlib.util.spec_from_file_location(
    "np_wrapper", str(Path(__file__).resolve().parents[1] / "src" / "ptspy" / "np_wrapper.py")
)
npw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(npw)


try:
    import cupy  # type: ignore[import-untyped]

    HAS_CUPY = True
    HAS_CUDA = cupy.cuda.is_available()
except ImportError:
    HAS_CUPY = False
    HAS_CUDA = False


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure we're on numpy backend after each test."""
    yield
    npw.reset("numpy")


# ---------------------------------------------------------------------------
# 1. asnumpy edge cases
# ---------------------------------------------------------------------------


class TestAsnumpyEdgeCases:
    """Test asnumpy with various input types and edge cases."""

    def test_scalar_input(self):
        result = npw.asnumpy(42)
        assert isinstance(result, np.ndarray)
        assert result.item() == 42

    def test_empty_array(self):
        result = npw.asnumpy(np.array([]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_nested_list(self):
        result = npw.asnumpy([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_3d_array(self):
        arr = np.ones((3, 4, 5), dtype=np.float32)
        result = npw.asnumpy(arr)
        assert result.shape == (3, 4, 5)
        assert result.dtype == np.float32

    def test_uint32_array(self):
        """Pixel data often uses uint32."""
        arr = np.array([0xFF0000, 0x00FF00, 0x0000FF], dtype=np.uint32)
        result = npw.asnumpy(arr)
        assert result.dtype == np.uint32
        np.testing.assert_array_equal(result, arr)

    def test_dict_not_treated_as_gpu_array(self):
        """Dicts have .get() but asnumpy should not call it."""
        result = npw.asnumpy({"key": 1})
        assert isinstance(result, np.ndarray)

    def test_asnumpy_is_idempotent(self):
        arr = np.array([1.0, 2.0, 3.0])
        result1 = npw.asnumpy(arr)
        result2 = npw.asnumpy(result1)
        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# 2. fromcpu edge cases
# ---------------------------------------------------------------------------


class TestFromhostEdgeCases:
    def test_with_different_dtypes(self):
        for dtype in [np.float32, np.float64, np.int32, np.uint8, np.uint32]:
            host = np.array([1, 2, 3], dtype=dtype)
            result = npw.fromcpu(host)
            np.testing.assert_array_equal(result, host)

    def test_with_scalar(self):
        result = npw.fromcpu(np.float32(3.14))
        assert float(result) == pytest.approx(3.14, abs=1e-5)

    def test_with_empty(self):
        result = npw.fromcpu(np.array([]))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. isarray edge cases
# ---------------------------------------------------------------------------


class TestIsarrayEdgeCases:
    def test_0d_array(self):
        assert npw.isarray(np.array(5)) is True

    def test_bool_array(self):
        assert npw.isarray(np.array([True, False])) is True

    def test_complex_array(self):
        assert npw.isarray(np.array([1 + 2j, 3 + 4j])) is True

    def test_numpy_matrix(self):
        # np.matrix is deprecated but still exists
        with pytest.warns(PendingDeprecationWarning):
            m = np.matrix([[1, 2], [3, 4]])
        assert npw.isarray(m) is True

    def test_memoryview_false(self):
        assert npw.isarray(memoryview(b"hello")) is False

    def test_bytes_false(self):
        assert npw.isarray(b"hello") is False


# ---------------------------------------------------------------------------
# 4. Backend switching
# ---------------------------------------------------------------------------


class TestBackendSwitching:
    def test_reset_numpy_is_default(self):
        npw.reset("numpy")
        arr = npw.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)

    def test_getattr_works_after_reset(self):
        npw.reset("numpy")
        assert npw.float32 == np.float32
        assert npw.uint32 == np.uint32
        assert callable(npw.linspace)
        assert callable(npw.meshgrid)

    def test_reset_clears_overrides(self):
        called = [False]

        def custom_linspace(*a, **kw):
            called[0] = True
            return np.linspace(*a, **kw)

        npw.override("linspace", custom_linspace)
        npw.linspace(0, 1, 5)
        assert called[0] is True

        npw.reset("numpy")
        called[0] = False
        npw.linspace(0, 1, 5)
        assert called[0] is False

    def test_override_nonexistent_warns(self):
        with pytest.warns(RuntimeWarning, match="does not exist"):
            npw.override("nonexistent_function_xyz", lambda: None)

    def test_override_type_validation(self):
        with pytest.raises(TypeError):
            npw.override(123, lambda: None)
        with pytest.raises(TypeError):
            npw.override("test", "not_callable")


# ---------------------------------------------------------------------------
# 5. Roundtrip integrity
# ---------------------------------------------------------------------------


class TestRoundtripIntegrity:
    """Ensure asnumpy -> fromcpu -> asnumpy preserves data exactly."""

    @pytest.mark.parametrize("shape", [(10,), (5, 3), (2, 3, 4), (2, 2, 2, 3)])
    def test_roundtrip_shapes(self, shape):
        original = np.random.default_rng(42).random(shape).astype(np.float32)
        result = npw.asnumpy(npw.fromcpu(original))
        np.testing.assert_array_equal(result, original)

    def test_roundtrip_integer_array(self):
        original = np.array([0xFF0000, 0x00FF00, 0x0000FF], dtype=np.uint32)
        result = npw.asnumpy(npw.fromcpu(original))
        np.testing.assert_array_equal(result, original)

    def test_roundtrip_reverse(self):
        """fromcpu -> asnumpy preserves data."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        backend_arr = npw.fromcpu(original)
        back = npw.asnumpy(backend_arr)
        np.testing.assert_array_equal(back, original)


# ---------------------------------------------------------------------------
# 6. Core ptspy operations via np_wrapper
# ---------------------------------------------------------------------------


class TestCoreOperations:
    """Verify that key ptspy operations work through np_wrapper."""

    def test_linspace(self):
        result = npw.linspace(0, 1, 10, dtype=np.float32)
        assert result.shape == (10,)
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[-1], 1.0)

    def test_meshgrid(self):
        xs = npw.linspace(0, 1, 3)
        ys = npw.linspace(0, 1, 4)
        X, Y = npw.meshgrid(xs, ys)
        assert X.shape == (4, 3)
        assert Y.shape == (4, 3)

    def test_stack(self):
        a = npw.array([1, 2, 3])
        b = npw.array([4, 5, 6])
        result = npw.stack([a, b], axis=0)
        assert result.shape == (2, 3)

    def test_dstack(self):
        a = npw.ones((3, 4))
        b = npw.zeros((3, 4))
        result = npw.dstack([a, b])
        assert result.shape == (3, 4, 2)

    def test_einsum(self):
        """Einsum is used in spline sampling."""
        TM = npw.random.default_rng(42).standard_normal((20, 4)).astype(npw.float32)
        G = npw.random.default_rng(42).standard_normal((10, 4, 2)).astype(npw.float32)
        result = npw.einsum("si,nid->nsd", TM, G)
        assert result.shape == (10, 20, 2)

    def test_random_generator(self):
        rng = npw.random.default_rng(42)
        vals = rng.random(10)
        assert vals.shape == (10,)

    def test_bitwise_operations(self):
        """Pixel packing uses bitwise ops."""
        arr = npw.array([0xFF112233], dtype=npw.uint32)
        r = (arr >> 16) & 0xFF
        g = (arr >> 8) & 0xFF
        b = arr & 0xFF
        assert int(r[0]) == 0x11
        assert int(g[0]) == 0x22
        assert int(b[0]) == 0x33

    def test_broadcast_shapes(self):
        s = npw.broadcast_shapes((3, 1), (1, 4))
        assert s == (3, 4)

    def test_atleast_1d(self):
        result = npw.atleast_1d(5.0)
        assert result.ndim == 1

    def test_where(self):
        arr = npw.array([1, -2, 3, -4])
        result = npw.where(arr > 0, arr, 0)
        np.testing.assert_array_equal(npw.asnumpy(result), [1, 0, 3, 0])

    def test_clip(self):
        arr = npw.array([-1, 0.5, 2.0])
        result = npw.clip(arr, 0, 1)
        np.testing.assert_array_equal(npw.asnumpy(result), [0, 0.5, 1.0])

    def test_linalg_norm(self):
        arr = npw.array([[3.0, 4.0]])
        result = float(npw.linalg.norm(arr))
        assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 7. Simulated backend-switch test (exercises the switching path)
# ---------------------------------------------------------------------------


class TestBackendSwitch:
    """Test that np_wrapper can switch backends and operations still work."""

    def test_switch_and_basic_ops(self):
        npw.reset("numpy")
        a = npw.array([1.0, 2.0, 3.0])
        assert npw.isarray(a)
        np.testing.assert_allclose(npw.asnumpy(a), [1.0, 2.0, 3.0])

    def test_override_and_restore(self):
        """Test that override works and is cleared on reset."""
        results = []

        def traced_linspace(*args, **kwargs):
            results.append("traced")
            return np.linspace(*args, **kwargs)

        npw.override("linspace", traced_linspace)
        npw.linspace(0, 1, 5)
        assert len(results) == 1

        npw.reset("numpy")
        npw.linspace(0, 1, 5)
        assert len(results) == 1  # not incremented


# ---------------------------------------------------------------------------
# 7b. CuPy backend-switch tests (no CUDA needed — only cupy module import)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
class TestCupyBackendSwitch:
    """Test backend switching to cupy without requiring CUDA hardware."""

    def teardown_method(self):
        npw.reset("numpy")

    def test_reset_cupy_sets_module(self):
        """reset('cupy') should import and set cupy as the backend module."""
        npw.reset("cupy")
        import cupy

        assert npw._np_module is cupy

    def test_isarray_numpy_under_cupy(self):
        """isarray should recognize numpy arrays even when cupy is active."""
        npw.reset("cupy")
        assert npw.isarray(np.array([1, 2])) is True

    def test_asnumpy_passthrough_under_cupy(self):
        """asnumpy should pass through numpy arrays when cupy is active."""
        npw.reset("cupy")
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = npw.asnumpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_asnumpy_with_dtype_under_cupy(self):
        """asnumpy with dtype should convert numpy arrays when cupy is active."""
        npw.reset("cupy")
        result = npw.asnumpy([1, 2, 3], dtype=np.float32)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_reset_back_to_numpy(self):
        """Switching cupy→numpy restores numpy behavior."""
        npw.reset("cupy")
        npw.reset("numpy")
        arr = npw.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)

    def test_overrides_cleared_on_cupy_reset(self):
        """Overrides should be cleared when switching to cupy."""
        npw.override("linspace", lambda *a, **kw: None)
        npw.reset("cupy")
        assert npw._overrides == {}

    def test_getattr_resolves_cupy_types(self):
        """__getattr__ should resolve cupy types."""
        npw.reset("cupy")
        import cupy

        assert npw.float32 is cupy.float32
        assert npw.uint32 is cupy.uint32


# ---------------------------------------------------------------------------
# 8. Conditional CuPy integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
class TestCupyIntegration:
    """End-to-end tests with actual CuPy backend."""

    def setup_method(self):
        npw.reset("cupy")

    def teardown_method(self):
        npw.reset("numpy")

    def test_array_creation(self):
        arr = npw.array([1.0, 2.0, 3.0])
        assert type(arr).__module__.startswith("cupy")
        assert arr.shape == (3,)

    def test_linspace_on_gpu(self):
        result = npw.linspace(0, 1, 100)
        assert type(result).__module__.startswith("cupy")
        host = npw.asnumpy(result)
        assert isinstance(host, np.ndarray)
        np.testing.assert_allclose(host[0], 0.0)
        np.testing.assert_allclose(host[-1], 1.0)

    def test_meshgrid_on_gpu(self):
        xs = npw.linspace(0, 1, 10)
        ys = npw.linspace(0, 1, 10)
        X, Y = npw.meshgrid(xs, ys)
        assert type(X).__module__.startswith("cupy")
        result = npw.stack([X.ravel(), Y.ravel()], axis=1)
        host = npw.asnumpy(result)
        assert host.shape == (100, 2)

    def test_vectorized_math_on_gpu(self):
        data = npw.linspace(0, 6.28, 1000)
        result = npw.sin(data) * npw.cos(data) + npw.exp(-data)
        host = npw.asnumpy(result)
        assert host.shape == (1000,)

    def test_matmul_on_gpu(self):
        pts = npw.random.default_rng(42).standard_normal((100, 3)).astype(npw.float32)
        matrix = npw.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=npw.float32)
        result = pts @ matrix.T
        host = npw.asnumpy(result)
        assert host.shape == (100, 3)

    def test_einsum_on_gpu(self):
        TM = npw.random.default_rng(42).standard_normal((20, 4)).astype(npw.float32)
        G = npw.random.default_rng(42).standard_normal((10, 4, 2)).astype(npw.float32)
        result = npw.einsum("si,nid->nsd", TM, G)
        assert type(result).__module__.startswith("cupy")
        host = npw.asnumpy(result)
        assert host.shape == (10, 20, 2)

    def test_bitwise_pixel_packing_on_gpu(self):
        arr = npw.array([0xFF112233, 0xAA445566], dtype=npw.uint32)
        r = (arr >> 16) & 0xFF
        g = (arr >> 8) & 0xFF
        b = arr & 0xFF
        hr, hg, hb = npw.asnumpy(r), npw.asnumpy(g), npw.asnumpy(b)
        np.testing.assert_array_equal(hr, [0x11, 0x44])
        np.testing.assert_array_equal(hg, [0x22, 0x55])
        np.testing.assert_array_equal(hb, [0x33, 0x66])

    def test_asnumpy_roundtrip(self):
        host_orig = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu = npw.fromcpu(host_orig)
        assert type(gpu).__module__.startswith("cupy")
        back = npw.asnumpy(gpu)
        assert isinstance(back, np.ndarray)
        np.testing.assert_array_equal(back, host_orig)

    def test_isarray_both_backends(self):
        gpu_arr = npw.array([1, 2, 3])
        np_arr = np.array([1, 2, 3])
        assert npw.isarray(gpu_arr) is True
        assert npw.isarray(np_arr) is True
        assert npw.isarray([1, 2, 3]) is False

    def test_random_generator_on_gpu(self):
        rng = npw.random.default_rng(42)
        vals = rng.random(100)
        assert type(vals).__module__.startswith("cupy")
        host = npw.asnumpy(vals)
        assert host.shape == (100,)
        assert np.all((host >= 0) & (host < 1))

    def test_color_conversion_pipeline(self):
        """Simulate the arrayToImage pipeline on GPU."""
        data = npw.random.default_rng(42).standard_normal((100, 100, 3)).astype(npw.float32)
        minVal = npw.min(data)
        maxVal = npw.max(data)
        pixel = (data - minVal) / (maxVal - minVal + 1e-9)
        rgb = npw.rint(pixel * 255).astype(npw.uint32)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        packed = (r << 16) | (g << 8) | b
        host = npw.asnumpy(packed)
        assert host.shape == (100, 100)
        assert host.dtype == np.uint32

    def test_gpu_transfer_overhead_is_bounded(self):
        """Ensure asnumpy on a 1M element array completes in reasonable time."""
        import time

        data = npw.linspace(0, 1, 1_000_000)
        cupy.cuda.Stream.null.synchronize()

        start = time.perf_counter()
        host = npw.asnumpy(data)
        elapsed = time.perf_counter() - start

        assert host.shape == (1_000_000,)
        # Should complete in under 1 second even on slow machines
        assert elapsed < 1.0
