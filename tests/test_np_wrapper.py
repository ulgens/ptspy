"""Tests for the np_wrapper interop helpers."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

# Import np_wrapper directly to avoid pulling in skia via ptspy.__init__
_spec = importlib.util.spec_from_file_location(
    'np_wrapper', str(Path(__file__).resolve().parents[1] / "src" / "ptspy" / "np_wrapper.py")
)
npw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(npw)


def _install_fake_backend(name="fakebackend_npw"):
    """Install a minimal numpy-like backend for post-reset interoperability tests."""
    module = types.ModuleType(name)

    class FakeArray:
        def __init__(self, data):
            self._data = np.asarray(data)
            self.ndim = self._data.ndim
            self.dtype = self._data.dtype
            self.shape = self._data.shape

        def get(self):
            return self._data.copy()

        def __array__(self, dtype=None):
            raise TypeError("implicit conversion forbidden; use get()")

    def asarray(a, dtype=None):
        if isinstance(a, FakeArray):
            return FakeArray(a._data.astype(dtype) if dtype is not None else a._data)
        return FakeArray(np.asarray(a, dtype=dtype))

    module.ndarray = FakeArray
    module.asarray = asarray
    module.array = asarray
    module.float32 = np.float32
    sys.modules[name] = module
    return name, module


class TestAsnumpy:
    def test_returns_numpy_array_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = npw.asnumpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_returns_same_numpy_array_without_dtype_conversion(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = npw.asnumpy(arr)
        assert result is arr

    def test_returns_numpy_array_from_list(self):
        result = npw.asnumpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_preserves_dtype(self):
        arr = np.array([1, 2], dtype=np.float32)
        result = npw.asnumpy(arr)
        assert result.dtype == np.float32

    def test_preserves_shape(self):
        arr = np.ones((3, 4, 2), dtype=np.float32)
        result = npw.asnumpy(arr)
        assert result.shape == (3, 4, 2)

    def test_dict_not_treated_as_gpu_array(self):
        """Dicts have .get() but should not be treated as GPU arrays."""
        result = npw.asnumpy({"a": 1})
        assert isinstance(result, np.ndarray)  # wrapped as object array, not .get() called

    def test_dtype_conversion(self):
        """asnumpy(x, dtype=float32) converts dtype."""
        result = npw.asnumpy([1, 2, 3], dtype=np.float32)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_dtype_preserves_existing(self):
        """asnumpy without dtype preserves original dtype."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = npw.asnumpy(arr)
        assert result.dtype == np.int64


class TestFromhost:
    def test_returns_backend_array(self):
        host = np.array([1.0, 2.0])
        result = npw.fromcpu(host)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, host)

    def test_from_list(self):
        result = npw.fromcpu([1, 2, 3])
        assert isinstance(result, np.ndarray)

    def test_preserves_multidimensional(self):
        host = np.ones((2, 3), dtype=np.float32)
        result = npw.fromcpu(host)
        assert result.shape == (2, 3)


class TestIsarray:
    def test_numpy_array_true(self):
        assert npw.isarray(np.array([1, 2, 3])) is True

    def test_numpy_scalar_array_true(self):
        assert npw.isarray(np.array(5)) is True

    def test_numpy_empty_array_true(self):
        assert npw.isarray(np.array([])) is True

    def test_list_false(self):
        assert npw.isarray([1, 2, 3]) is False

    def test_tuple_false(self):
        assert npw.isarray((1, 2, 3)) is False

    def test_scalar_int_false(self):
        assert npw.isarray(42) is False

    def test_scalar_float_false(self):
        assert npw.isarray(3.14) is False

    def test_string_false(self):
        assert npw.isarray("hello") is False

    def test_none_false(self):
        assert npw.isarray(None) is False

    def test_dict_false(self):
        assert npw.isarray({}) is False


class TestReset:
    def test_reset_to_numpy(self):
        npw.reset('numpy')
        arr = npw.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)

    def test_helpers_work_after_reset(self):
        npw.reset('numpy')
        assert npw.isarray(npw.array([1]))
        np.testing.assert_array_equal(npw.asnumpy(npw.array([1, 2])), [1, 2])

    def test_reset_clears_overrides(self):
        npw.override('linspace', lambda *a, **kw: None)
        npw.reset('numpy')
        # Should use real numpy linspace now
        result = npw.linspace(0, 1, 5)
        assert len(result) == 5

    def test_getattr_delegates_to_backend(self):
        """Verify __getattr__ still works for standard numpy operations."""
        npw.reset('numpy')
        arr = npw.linspace(0, 1, 10)
        assert len(arr) == 10
        assert npw.float32 == np.float32

    def test_getattr_raises_for_missing(self):
        with pytest.raises(AttributeError):
            npw.nonexistent_function_xyz



class TestRoundtrip:
    """Test that asnumpy/fromcpu form a correct roundtrip."""

    def test_roundtrip_1d(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = npw.fromcpu(npw.asnumpy(original))
        np.testing.assert_array_equal(result, original)

    def test_roundtrip_2d(self):
        original = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = npw.fromcpu(npw.asnumpy(original))
        np.testing.assert_array_equal(result, original)

    def test_roundtrip_preserves_dtype(self):
        for dtype in [np.float32, np.float64, np.int32, np.uint8]:
            original = np.array([1, 2, 3], dtype=dtype)
            result = npw.asnumpy(npw.fromcpu(original))
            assert result.dtype == dtype


# Conditional cupy tests
try:
    import cupy  # type: ignore[import-untyped]
    HAS_CUPY = True
    HAS_CUDA = cupy.cuda.is_available()
except ImportError:
    HAS_CUPY = False
    HAS_CUDA = False


class TestBackendSwitch:
    """Test backend switching with cupy module (no CUDA needed)."""

    def setup_method(self):
        npw.reset('numpy')

    def teardown_method(self):
        npw.reset('numpy')

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_reset_to_cupy_sets_backend(self):
        """reset('cupy') should set _np_module to cupy."""
        npw.reset('cupy')
        import cupy
        assert npw._np_module is cupy

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_isarray_numpy_still_true_under_cupy(self):
        """numpy arrays should pass isarray even when cupy is active."""
        npw.reset('cupy')
        assert npw.isarray(np.array([1, 2])) is True

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_asnumpy_numpy_array_under_cupy(self):
        """asnumpy should still handle numpy arrays when cupy is active."""
        npw.reset('cupy')
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = npw.asnumpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_reset_back_to_numpy(self):
        """Switching cupy→numpy should restore normal behavior."""
        npw.reset('cupy')
        npw.reset('numpy')
        arr = npw.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_getattr_delegates_to_cupy(self):
        """__getattr__ should resolve cupy attributes."""
        npw.reset('cupy')
        import cupy
        assert npw.float32 is cupy.float32

    def test_asnumpy_handles_arrays_from_previous_backend_explicitly(self):
        """Arrays created before reset() should still round-trip via explicit .get()."""
        backend_name, fake = _install_fake_backend()

        npw.reset(backend_name)
        arr = npw.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert isinstance(arr, fake.ndarray)

        npw.reset('numpy')
        result = npw.asnumpy(arr)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_isarray_recognizes_arrays_from_previous_backend_after_reset(self):
        """Validation should still recognize arrays created before switching backends."""
        backend_name, fake = _install_fake_backend("fakebackend_npw_isarray")

        npw.reset(backend_name)
        arr = npw.array([1.0, 2.0], dtype=np.float32)
        assert isinstance(arr, fake.ndarray)

        npw.reset('numpy')
        assert npw.isarray(arr) is True


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
class TestCupyInterop:
    """Tests requiring actual CUDA hardware."""

    def setup_method(self):
        npw.reset('cupy')

    def teardown_method(self):
        npw.reset('numpy')

    def test_asnumpy_from_cupy(self):
        import cupy
        arr = cupy.array([1.0, 2.0, 3.0])
        result = npw.asnumpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_fromcpu_to_cupy(self):
        import cupy
        host = np.array([1.0, 2.0])
        result = npw.fromcpu(host)
        assert isinstance(result, cupy.ndarray)

    def test_isarray_cupy_true(self):
        import cupy
        assert npw.isarray(cupy.array([1, 2])) is True

    def test_roundtrip_cupy(self):
        import cupy
        host = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu = npw.fromcpu(host)
        assert isinstance(gpu, cupy.ndarray)
        back = npw.asnumpy(gpu)
        assert isinstance(back, np.ndarray)
        np.testing.assert_array_equal(back, host)

    def test_cupy_array_creation(self):
        """Standard numpy-like array creation should work via the wrapper."""
        arr = npw.array([1, 2, 3])
        import cupy
        assert isinstance(arr, cupy.ndarray)

    def test_cupy_linspace(self):
        import cupy
        result = npw.linspace(0, 1, 10)
        assert isinstance(result, cupy.ndarray)
        assert len(result) == 10
