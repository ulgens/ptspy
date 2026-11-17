"""Numpy-like wrapper that supports backend swapping and overrides."""

import importlib
import warnings
from typing import Any
from collections.abc import Callable
import numpy as _cpu_numpy  # Permanent reference to CPU numpy, independent of active backend

# Default to numpy
_np_module = _cpu_numpy
_overrides = {}
_known_array_types: tuple[type, ...] = (_cpu_numpy.ndarray,)
_known_device_array_types: tuple[type, ...] = ()


def _registerArrayType(arrayType: type):
    """Remember backend ndarray types so post-reset arrays stay explicitly supported."""
    global _known_array_types, _known_device_array_types
    if arrayType in _known_array_types:
        return
    _known_array_types = _known_array_types + (arrayType,)
    if arrayType is not _cpu_numpy.ndarray:
        _known_device_array_types = _known_device_array_types + (arrayType,)


def asnumpy(a, dtype=None):
    """Convert any array to a CPU numpy array."""
    if type(a) is _cpu_numpy.ndarray and dtype is None:
        return a
    if isinstance(a, _cpu_numpy.ndarray):
        return _cpu_numpy.asarray(a, dtype=dtype)
    if _known_device_array_types and isinstance(a, _known_device_array_types):
        a = a.get()
    elif hasattr(a, "get") and callable(a.get) and hasattr(a, "ndim") and hasattr(a, "dtype") and hasattr(a, "shape"):
        # Fallback for array types that look numpy-like but have not been activated via reset() yet.
        a = a.get()
    return _cpu_numpy.asarray(a, dtype=dtype)


def fromcpu(a):
    """Convert a CPU numpy array to the active backend."""
    return _np_module.asarray(a)


def isarray(x):
    """Check if x is an ndarray from any supported backend."""
    if isinstance(x, _known_array_types):
        return True
    return hasattr(x, "ndim") and hasattr(x, "dtype") and hasattr(x, "shape") and not isinstance(x, (list, tuple))


def reset(library_name: str):
    """
    Switch the active numeric backend.

    Imports `library_name` and assigns it to the global `_np_module`, letting you swap
    between libraries with NumPy-compatible APIs (e.g., `numpy` and `cupy`).

    Overrides are cleared on reset to avoid carrying backend-specific override behavior
    into a different numeric backend.

    Args:
        library_name (str): The name of the library to import and set as `_np_module`.

    Examples:
        >>> reset('numpy')
        >>> type(_np_module)
        <class 'module'>

        >>> reset('cupy')
        >>> type(_np_module)
        <class 'module'>
    """
    global _np_module, _overrides
    mod = importlib.import_module(library_name)
    if not hasattr(mod, "ndarray") or not hasattr(mod, "asarray"):
        raise ValueError(f"'{library_name}' is not a numpy-compatible backend")
    if _np_module is not mod and _np_module is not _cpu_numpy:
        warnings.warn(
            f"Switching backend from '{_np_module.__name__}' to '{library_name}'. "
            "Arrays created under the previous backend may trigger transfers when reused.",
            stacklevel=2,
        )
    _registerArrayType(mod.ndarray)
    _np_module = mod
    _overrides = {}


def override(name: str, func: Callable[..., Any]):
    """
    Override a function in the current numerical library with a custom implementation.

    Replaces a function on the active backend so calls to `name` will use your implementation instead.

    If `name` does not exist on the active backend, a warning is emitted and the override is still stored.

    Args:
        name (str): The name of the function you want to override.
        func (Callable): The custom callable that will override the original.

    Raises:
        TypeError: If `name` is not a string or if `func` is not callable.


    Examples:
        def custom_linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
            print(f"Custom linspace called with start={start}, stop={stop}, num={num}")
            return _np_module.linspace(start, stop, num, endpoint, retstep, dtype, axis)

        override('linspace', custom_linspace)

        # Now, when linspace is called, it uses the custom_linspace implementation

        # Restore the original backend implementation:
        override('linspace', getattr(_np_module, 'linspace'))
    """
    if not isinstance(name, str):
        raise TypeError("override name must be a string")
    if not callable(func):
        raise TypeError("override func must be callable")

    if not hasattr(_np_module, name):
        warnings.warn(
            f"Override target '{name}' does not exist on backend '{_np_module.__name__}'",
            RuntimeWarning,
            stacklevel=2,
        )

    _overrides[name] = func


def __getattr__(name: str):
    """
    Custom attribute access function for backend dispatch.

    Lookup precedence:
    1. Return override in `_overrides` when present.
    2. Otherwise resolve the attribute on the active backend module.

    Raises:
        AttributeError: If neither an override nor a backend attribute exists.

    Data:
        internal: true
    """
    if name in _overrides:
        return _overrides[name]
    try:
        return getattr(_np_module, name)
    except AttributeError as exc:
        raise AttributeError(f"Attribute '{name}' not found in overrides or backend '{_np_module.__name__}'") from exc
