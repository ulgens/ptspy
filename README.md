# Pts.py

Pts.py is an experimental librar for for creative coding, visualization and play. It explores principles of visual composability in code. I started this series of explorations, first with an idea about a simple point ([Pt](https://github.com/williamngan/pt), 2015), then an idea about connecting them, ([Pts.js](https://github.com/williamngan/pts), 2017). Starting anew in Pts.py (2025), this is again both a software library and a speculative playground.

## Install python

Pts.py requires Python 3.10+. If you're new to python and don't have 3.10+ (check your version with `python --version`), or you want to create a virtual environment, create it via `conda` or `uv`.

### Option 1: setup with `uv`

Install [uv](https://github.com/astral-sh/uv) and create a virtual environment:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### Option 2: setup with `conda`

First download and install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

```bash
conda create --name my_env python=3.11
conda activate my_env
```

## Get Pts.py

The simplest way is to install via `pip install ptspy`, or `uv pip install ptspy` if you're using `uv`.

To develop from source and run notebooks in this repo, first clone this repo and then set up with `uv` or `pip`.

Recommended reproducible setup with `uv`:

```bash
uv sync --extra dev
source .venv/bin/activate
```

If you also want the optional OpenGL stack:

```bash
uv sync --extra dev --extra opengl
source .venv/bin/activate
```

If you also want the optional cupy switching feature, first install the corresponding cupy package.

```bash
# if using Nvidia CUDA 12.x
uv pip install cupy-cuda12x

```

Equivalent `pip` setup:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Add the OpenGL extra if needed:

```bash
python -m pip install -e '.[dev,opengl]'
```

### Run notebooks

To run a jupyter notebook in `/notebooks`, first add your venv to the list of kernels:

```
python -m ipykernel install --user --name=your_env_name --display-name "your_env_name"
```

Then start `jupyter notebook`, open a notebook and then select the correct kernel.

Note: `mediapy` is used in notebook for displaying videos. If getting BrokenPipe error in a conda env, install `conda install ffmpeg`

## Quick start

After your environment is set up, start with the [quick start guide](https://ptspy.org/start), or browse the source [notebooks on GitHub](https://github.com/williamngan/ptspy/tree/main/notebooks).

## Testing

You can run the tests and lint the code via these commands.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_elements.py

# Run specific test function
pytest tests/core/test_elements.py::test_partial_optional_with_placeholder_allows_curry_and_fill

# Check code with ruff
ruff check .

# Format code with ruff
ruff format

# Type checking (pyright configured in pyrightconfig.json)
pyright

# Build package artifacts
python -m build

# Verify README / metadata render correctly for PyPI
twine check dist/*

# Visual regression tests
pytest tests/visual/                        # Run visual tests
pytest tests/visual/ --update-groundtruth   # Add/replace groundtruth files
pytest tests/visual/ --rebuild-groundtruth  # Wipe and regenerate all groundtruth
pytest tests/visual/ --visual-tolerance=2   # Allow pixel tolerance (default: 0)

```

There is also a convenience script for the main test suites:

```bash
./tests.sh unit
./tests.sh visual
./tests.sh bench
./tests.sh all
```

### Benchmark baselines

Use the benchmark mode in `tests.sh` to save and compare local performance baselines on the same machine.

```bash
# Run all benchmarks
./tests.sh bench

# Save a benchmark baseline with an automatic name
./tests.sh bench --save

# Compare against the latest compatible saved baseline
./tests.sh bench --compare

# Compare against a named baseline
./tests.sh bench --compare --with=barcelona_gpu_baseline

# Fail if mean performance regresses by more than 8%
./tests.sh bench --compare --threshold=8%
./tests.sh bench --compare --with=barcelona_gpu_baseline --threshold=8%

# Use a slower, more stable measurement mode for release verification
./tests.sh bench --save --precise
./tests.sh bench --compare --precise
./tests.sh bench --compare --precise --threshold=5%
```

## Generate docs

To regenerate the API documentation (JSON + Markdown) from source docstrings:

```bash
./docs.sh
```

## Notes on contribution

The best way to contribute to this project would be to try it out and give feedback. Please feel free to create an issue or contact me in social media. PRs are closed until this project reaches a stable milestone. See CONTRIBUTING.md for more.

## Citation

If you use Pts.py in your research or creative work, please cite it as follows:

```
Ngan, W. (2026). Pts.py: Visual Thinking in Code [Computer software]. https://ptspy.org
```

BibTeX:

```bibtex
@software{ptspy2026,
  author = {Ngan, William},
  title = {Pts.py: Visual Thinking in Code},
  year = {2026},
  url = {https://ptspy.org},
  repository = {https://github.com/williamngan/ptspy}
}
```

# License

Copyright 2026-present William Ngan and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
