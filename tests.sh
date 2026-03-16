#!/usr/bin/env bash
# Run ptspy test suites
set -e
cd "$(dirname "$0")"

# Use the project venv if no python is on PATH
if ! command -v python &>/dev/null && [ -x .venv/bin/python ]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi

usage() {
  echo "Usage: ./tests.sh [unit|visual|bench|all]"
  echo ""
  echo "  unit    Run unit tests (default)"
  echo "  visual  Run visual regression tests"
  echo "  bench   Run performance benchmarks"
  echo "  all     Run all test suites"
  echo ""
  echo "Benchmark options:"
  echo "  ./tests.sh bench --save"
  echo "  ./tests.sh bench --compare"
  echo "  ./tests.sh bench --compare --with=NAME --threshold=8%"
  echo "  ./tests.sh bench --save --precise"
  echo "  ./tests.sh bench --compare --precise --threshold=5%"
  exit 1
}

BENCHMARK_STORAGE="${BENCHMARK_STORAGE:-file://./.benchmarks}"
BENCHMARK_RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-./tests/results}"

benchmark_storage_dir() {
  case "$BENCHMARK_STORAGE" in
    file://*)
      printf '%s\n' "${BENCHMARK_STORAGE#file://}"
      ;;
    *)
      echo "Error: only file:// benchmark storage is supported by tests.sh helpers." >&2
      exit 1
      ;;
  esac
}

benchmark_default_name() {
  python - <<'PY'
from datetime import datetime
import platform
import socket
import subprocess

host = socket.gethostname().split('.')[0]
py = ''.join(platform.python_version_tuple()[:2])
stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
try:
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    git_sha = "nogit"
print(f"{host}_py{py}_{stamp}_{git_sha}")
PY
}

benchmark_report_stamp() {
  python - <<'PY'
from datetime import datetime
print(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
PY
}

benchmark_current_meta() {
  local benchmark_mode="$1"
  python - "$benchmark_mode" <<'PY'
import json
import platform
import socket
import sys

import numpy as np

try:
    import cupy  # type: ignore[import-untyped]
    cupy_version = cupy.__version__
    try:
        cuda_available = bool(cupy.cuda.is_available())
    except Exception:
        cuda_available = False
except Exception:
    cupy_version = None
    cuda_available = False

benchmark_mode = sys.argv[1]

print(json.dumps({
    "hostname": socket.gethostname().split('.')[0],
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "cupy_version": cupy_version,
    "cuda_available": cuda_available,
    "benchmark_mode": benchmark_mode,
}))
PY
}

benchmark_find_saved_by_name() {
  local save_name="$1"
  local storage_dir
  storage_dir="$(benchmark_storage_dir)"
  python - "$save_name" "$storage_dir" <<'PY'
import sys
from pathlib import Path

save_name = sys.argv[1]
storage_dir = Path(sys.argv[2])
files = sorted(
    p for p in storage_dir.rglob(f"*_{save_name}.json")
    if not p.name.endswith(".meta")
)
if files:
    print(files[-1])
PY
}

benchmark_find_latest_compatible() {
  local current_meta_json="$1"
  local storage_dir
  storage_dir="$(benchmark_storage_dir)"
  python - "$current_meta_json" "$storage_dir" <<'PY'
import json
import sys
from pathlib import Path

current = json.loads(sys.argv[1])
storage_dir = Path(sys.argv[2])
best = None

for path in storage_dir.rglob("*.json"):
    if path.name.endswith(".meta"):
        continue

    meta_path = path.with_suffix(".meta")
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            compatible = (
                meta.get("hostname") == current["hostname"]
                and meta.get("python_version") == current["python_version"]
                and meta.get("cuda_available") == current["cuda_available"]
                and (meta.get("benchmark_mode") or "standard") == current["benchmark_mode"]
            )
        else:
            data = json.loads(path.read_text())
            machine = data.get("machine_info", {})
            compatible = (
                machine.get("node", "").split(".")[0] == current["hostname"]
                and machine.get("python_version") == current["python_version"]
                and current["benchmark_mode"] == "standard"
            )
    except Exception:
        continue

    if not compatible:
        continue

    stat = path.stat()
    candidate = (stat.st_mtime, path)
    if best is None or candidate > best:
        best = candidate

if best is not None:
    print(best[1])
PY
}

benchmark_compare_selector() {
  local benchmark_file="$1"
  python - "$benchmark_file" <<'PY'
import sys
from pathlib import Path

name = Path(sys.argv[1]).stem
counter = name.split("_", 1)[0]
print(counter)
PY
}

benchmark_write_meta() {
  local benchmark_file="$1"
  local save_name="$2"
  local current_meta_json="$3"
  python - "$benchmark_file" "$save_name" "$current_meta_json" <<'PY'
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

benchmark_file = Path(sys.argv[1])
save_name = sys.argv[2]
meta = json.loads(sys.argv[3])
try:
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    git_sha = None

meta.update({
    "save_name": save_name,
    "benchmark_file": str(benchmark_file),
    "git_commit": git_sha,
    "created_at": datetime.now().isoformat(timespec="seconds"),
})

meta_path = benchmark_file.with_suffix(".meta")
meta_path.write_text(json.dumps(meta, indent=2) + "\n")
print(meta_path)
PY
}

MODE="${1:-unit}"
shift 2>/dev/null || true

case "$MODE" in
  unit)
    python -m pytest tests/ --ignore=tests/visual --ignore=tests/benchmarks -q "$@"
    ;;
  visual)
    python -m pytest tests/visual/ "$@"
    ;;
  bench)
    BENCH_SAVE=0
    BENCH_COMPARE=0
    BENCH_PRECISE=0
    BENCH_WITH=""
    BENCH_THRESHOLD=""
    BENCH_GROUP_BY="group"
    BENCH_GROUP_BY_SET=0
    BENCH_PYTEST_ARGS=()

    while [ "$#" -gt 0 ]; do
      case "$1" in
        --save)
          BENCH_SAVE=1
          ;;
        --compare)
          BENCH_COMPARE=1
          ;;
        --precise)
          BENCH_PRECISE=1
          ;;
        --with=*)
          BENCH_WITH="${1#*=}"
          ;;
        --threshold=*)
          BENCH_THRESHOLD="${1#*=}"
          ;;
        --group-by=*)
          BENCH_GROUP_BY="${1#*=}"
          BENCH_GROUP_BY_SET=1
          ;;
        *)
          BENCH_PYTEST_ARGS+=("$1")
          ;;
      esac
      shift
    done

    if [ "$BENCH_SAVE" -eq 1 ] && [ "$BENCH_COMPARE" -eq 1 ]; then
      echo "Error: --save and --compare cannot be used together."
      exit 1
    fi

    if [ -n "$BENCH_WITH" ] && [ "$BENCH_COMPARE" -ne 1 ]; then
      echo "Error: --with=NAME requires --compare."
      exit 1
    fi

    if [ -n "$BENCH_THRESHOLD" ] && [ "$BENCH_COMPARE" -ne 1 ]; then
      echo "Error: --threshold=VALUE requires --compare."
      exit 1
    fi

    if [ "$BENCH_GROUP_BY_SET" -eq 0 ] && { [ "$BENCH_COMPARE" -eq 1 ] || [ "$BENCH_PRECISE" -eq 1 ]; }; then
      BENCH_GROUP_BY="fullfunc"
    fi

    BENCHMARK_ARGS=(--benchmark-group-by="$BENCH_GROUP_BY" --benchmark-storage="$BENCHMARK_STORAGE")
    BENCHMARK_MODE="standard"
    BENCH_THRESHOLD_METRIC="mean"
    REPORT_STAMP="$(benchmark_report_stamp)"
    REPORT_JSON="/tmp/ptspy_benchmark_${REPORT_STAMP}.json"
    REPORT_HTML="${BENCHMARK_RESULTS_DIR}/benchmark_${REPORT_STAMP}.html"

    if [ "$BENCH_PRECISE" -eq 1 ]; then
      BENCHMARK_MODE="precise"
      BENCH_THRESHOLD_METRIC="median"
      BENCHMARK_ARGS+=(
        --benchmark-warmup=on
        --benchmark-min-time=0.02
        --benchmark-max-time=1.0
        --benchmark-min-rounds=12
      )
      echo "Running in precise mode: warmup enabled, longer timing window, stricter attribution output."
    fi

    if [ "$BENCH_COMPARE" -eq 1 ] || [ "$BENCH_PRECISE" -eq 1 ]; then
      BENCHMARK_ARGS+=(--benchmark-name=long --benchmark-sort=fullname)
    fi

    CURRENT_META_JSON="$(benchmark_current_meta "$BENCHMARK_MODE")"

    if [ "$BENCH_SAVE" -eq 1 ]; then
      SAVE_NAME="$(benchmark_default_name)"
      BENCHMARK_ARGS+=(--benchmark-save="$SAVE_NAME")
      echo "Saving benchmark baseline as: $SAVE_NAME"
    elif [ "$BENCH_COMPARE" -eq 1 ]; then
      if [ -n "$BENCH_WITH" ]; then
        COMPARE_FILE="$(benchmark_find_saved_by_name "$BENCH_WITH")"
      else
        COMPARE_FILE="$(benchmark_find_latest_compatible "$CURRENT_META_JSON")"
      fi

      if [ -z "$COMPARE_FILE" ]; then
        echo "Error: no compatible saved benchmark baseline found."
        exit 1
      fi

      COMPARE_SELECTOR="$(benchmark_compare_selector "$COMPARE_FILE")"
      BENCHMARK_ARGS+=(--benchmark-compare="$COMPARE_SELECTOR")
      echo "Comparing against baseline: $COMPARE_FILE"

      if [ -n "$BENCH_THRESHOLD" ]; then
        BENCHMARK_ARGS+=(--benchmark-compare-fail="$BENCH_THRESHOLD_METRIC:$BENCH_THRESHOLD")
        echo "Regression threshold: $BENCH_THRESHOLD_METRIC:$BENCH_THRESHOLD"
      fi
    fi

    mkdir -p "$BENCHMARK_RESULTS_DIR"
    BENCHMARK_ARGS+=(--benchmark-json="$REPORT_JSON")

    set +e
    python -m pytest tests/benchmarks/ -v "${BENCHMARK_ARGS[@]}" "${BENCH_PYTEST_ARGS[@]}"
    BENCH_EXIT_CODE=$?
    set -e

    if [ -f "$REPORT_JSON" ]; then
      HTML_ARGS=(
        tests/benchmarks/render_benchmark_html.py
        --input "$REPORT_JSON"
        --output "$REPORT_HTML"
        --metric "$BENCH_THRESHOLD_METRIC"
      )
      if [ -n "$COMPARE_FILE" ]; then
        HTML_ARGS+=(--compare "$COMPARE_FILE")
      fi
      if [ -n "$BENCH_THRESHOLD" ]; then
        HTML_ARGS+=(--threshold "$BENCH_THRESHOLD")
      fi
      HTML_FILE="$(python "${HTML_ARGS[@]}")"
      echo "Benchmark HTML report: $HTML_FILE"
    fi

    if [ "$BENCH_SAVE" -eq 1 ]; then
      BENCHMARK_FILE="$(benchmark_find_saved_by_name "$SAVE_NAME")"
      if [ -n "$BENCHMARK_FILE" ]; then
        META_FILE="$(benchmark_write_meta "$BENCHMARK_FILE" "$SAVE_NAME" "$CURRENT_META_JSON")"
        echo "Saved benchmark metadata in: $META_FILE"
      fi
    fi
    rm -f "$REPORT_JSON"
    exit "$BENCH_EXIT_CODE"
    ;;
  all)
    echo "=== Unit tests ==="
    python -m pytest tests/ --ignore=tests/visual --ignore=tests/benchmarks -q
    echo ""
    echo "=== Visual tests ==="
    python -m pytest tests/visual/
    echo ""
    echo "=== Benchmarks ==="
    python -m pytest tests/benchmarks/ -v
    ;;
  *)
    usage
    ;;
esac
