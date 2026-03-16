# Pts.py

Pts.py is a Python library that explores principles of visual composability in code. Built on numpy and skia-python, targeting Python 3.10+.

## Module Structure

Source code is under `src/ptspy/`:

**Core modules** (`src/ptspy/core`):

- `elements.py`: Point constructors (`P`, `P2`, `P3`), `TBD` placeholder, `@partial` decorator for currying
- `process.py`: Pipeline composition (`pipeline`, `F`, `>>` operator), array processing, shaping/interpolation
- `form.py`: Skia-backed drawing surface (`Form` class, `@frame` and `@frames` decorators)
- `compose.py`: Array construction helpers (`populate`, `steps`, `gridPoints`, `regularPolygon`)
- `colors.py`: Color manipulation (`colorCube`, `colorPicker`, `colorScheme`, `gradient`)
- `geom.py`: Geometry constants and helper classes (`Rectangle`, `Circle`)
- `util.py`: General utilities including Skia property helpers

**Types**:

- `typing.py`: Type aliases (`PArray`, `ColorLike`, `NumberSequenceND`, etc.)

**Work in progress** (not stable):

- `np_wrapper.py`: Numpy-like wrapper for backend swapping (numpy, cupy)
- `more/`: Additional utilities (noise functions)

## Key Abstractions

- **Points**: `P()`, `P2()`, `P3()` create numpy float32 arrays. All numpy operations apply.
- **Functional**: Simple functional programming features like `@partial` decorator + `TBD` placeholder enable currying. `pipeline(a, b, c)` or `F(a) >> b >> c` chains functions. Supports `lazy=True` and `trace=True`.
- **Form/Frame**: `@frame` for static images, `@frames` for animation. First param is always `form` (Skia drawing surface).

## Code Style

- camelCase naming (not snake_case)
- 120 character line length
- Type hints from `ptspy.typing`
- Numpy vectorized operations over loops
- Functional programming style, pure functions where possible

## Build & Test

```bash
# Install
pip install -e '.[dev]'

# Run all tests
pytest

# Run specific tests
pytest tests/core/test_elements.py
pytest tests/core/test_elements.py::test_partial_optional_with_placeholder_allows_curry_and_fill

# Visual regression tests
pytest tests/visual/
pytest tests/visual/ --update-groundtruth
pytest tests/visual/ --visual-tolerance=2

# Lint and format
ruff check .
ruff format

# Type checking
pyright

# Generate docs (JSON + Markdown + llms-full.txt)
./docs.sh
```

## Test Structure

- Tests mirror source structure under `tests/`
- `tests/conftest.py` adds `src/` to sys.path
- Naming: `test_<feature>_<scenario>`
- Use pytest fixtures and parametrization
