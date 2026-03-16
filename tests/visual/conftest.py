"""Visual regression testing fixtures for ptspy."""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from PIL import Image
import skia

# Paths
VISUAL_DIR = Path(__file__).parent
GROUNDTRUTH_DIR = VISUAL_DIR / "groundtruth"


def _imageToArray(img: skia.Image) -> np.ndarray:
    """Convert skia.Image to numpy array (RGBA uint8)."""
    surface = skia.Surface(img.width(), img.height())
    surface.getCanvas().drawImage(img, 0, 0)
    return surface.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)


def _savePng(arr: np.ndarray, path: Path) -> None:
    """Save numpy array as PNG."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        Image.fromarray(arr, "RGBA").save(path)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        Image.fromarray(arr, "RGB").save(path)
    else:
        Image.fromarray(arr).save(path)


def _loadPng(path: Path) -> np.ndarray:
    """Load PNG as numpy array."""
    return np.array(Image.open(path))


def _saveMp4(frames: np.ndarray, path: Path, fps: int = 10) -> None:
    """Save frames as MP4 for human viewing (not used for comparison)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write frames as PNGs
        for i, frame in enumerate(frames):
            _savePng(frame, Path(tmpdir) / f"frame_{i:04d}.png")

        # Use ffmpeg to create MP4 (try libopenh264, fallback to any h264)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", f"{tmpdir}/frame_%04d.png",
            "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
            str(path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)


def _compareArrays(
    actual: np.ndarray, expected: np.ndarray, tolerance: int = 0
) -> tuple[bool, Optional[np.ndarray]]:
    """Compare two arrays with optional tolerance. Returns (match, diff)."""
    if actual.shape != expected.shape:
        return False, None

    if tolerance == 0:
        match = np.array_equal(actual, expected)
    else:
        diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        match = np.all(diff <= tolerance)

    if match:
        return True, None

    diffArr = np.abs(actual.astype(np.int16) - expected.astype(np.int16)).astype(np.uint8)
    return False, diffArr


def _compareSequences(
    actual: np.ndarray, expected: np.ndarray, tolerance: int = 0
) -> tuple[bool, int, Optional[np.ndarray]]:
    """Compare video sequences frame by frame. Returns (match, failingFrame, diff)."""
    if actual.shape != expected.shape:
        return False, -1, None

    for i in range(actual.shape[0]):
        match, diff = _compareArrays(actual[i], expected[i], tolerance)
        if not match:
            return False, i, diff

    return True, -1, None


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--update-groundtruth",
        action="store_true",
        default=False,
        help="Update groundtruth (add/replace without removing old files)",
    )
    parser.addoption(
        "--rebuild-groundtruth",
        action="store_true",
        default=False,
        help="Rebuild groundtruth (wipe directory and regenerate all)",
    )
    parser.addoption(
        "--visual-tolerance",
        type=int,
        default=0,
        help="Pixel tolerance for visual comparison (0 = exact match)",
    )


def pytest_configure(config):
    """Clear groundtruth directory when rebuilding."""
    if config.getoption("--rebuild-groundtruth", default=False):
        import shutil
        if GROUNDTRUTH_DIR.exists():
            shutil.rmtree(GROUNDTRUTH_DIR)
        GROUNDTRUTH_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def visualTest(request):
    """
    Fixture for visual regression tests.

    Usage:
        def test_circle(visualTest):
            @frame(size=(100, 100), bg="white")
            def draw(form):
                form.fillOnly("red").circle([[25, 25], [75, 75]])

            visualTest(draw, "circle_red")
    """
    updateGroundtruth = (
        request.config.getoption("--update-groundtruth", default=False) or
        request.config.getoption("--rebuild-groundtruth", default=False)
    )
    tolerance = request.config.getoption("--visual-tolerance", default=0)

    def _test(drawFn: Callable, name: str, toleranceOverride: Optional[int] = None):
        tol = toleranceOverride if toleranceOverride is not None else tolerance

        # Generate actual output
        result = drawFn()

        if isinstance(result, skia.Image):
            actual = _imageToArray(result)
            gtPath = GROUNDTRUTH_DIR / f"{name}.png"
            isSequence = False
        else:
            # numpy array from @frames
            actual = result
            gtPath = GROUNDTRUTH_DIR / f"{name}.npz"
            isSequence = True

        if updateGroundtruth:
            gtPath.parent.mkdir(parents=True, exist_ok=True)
            if isSequence:
                np.savez_compressed(gtPath, frames=actual)
                # Also save MP4 for human viewing
                mp4Path = gtPath.with_suffix(".mp4")
                try:
                    _saveMp4(actual, mp4Path)
                except Exception:
                    pass  # MP4 is optional, don't fail if ffmpeg unavailable
            else:
                _savePng(actual, gtPath)
            return

        # Load and compare
        if not gtPath.exists():
            pytest.fail(
                f"Groundtruth not found: {gtPath}. Run with --update-groundtruth to create."
            )

        if isSequence:
            expected = np.load(gtPath)["frames"]
            match, failFrame, diff = _compareSequences(actual, expected, tolerance=tol)
            if not match:
                failDir = VISUAL_DIR / "failures"
                failDir.mkdir(exist_ok=True)
                _savePng(actual[failFrame], failDir / f"{name}_frame{failFrame}_actual.png")
                _savePng(expected[failFrame], failDir / f"{name}_frame{failFrame}_expected.png")
                if diff is not None:
                    _savePng(diff, failDir / f"{name}_frame{failFrame}_diff.png")
                pytest.fail(
                    f"Visual mismatch for sequence '{name}' at frame {failFrame}. See {failDir}"
                )
        else:
            expected = _loadPng(gtPath)
            match, diff = _compareArrays(actual, expected, tolerance=tol)
            if not match:
                failDir = VISUAL_DIR / "failures"
                failDir.mkdir(exist_ok=True)
                _savePng(actual, failDir / f"{name}_actual.png")
                _savePng(expected, failDir / f"{name}_expected.png")
                if diff is not None:
                    _savePng(diff, failDir / f"{name}_diff.png")
                pytest.fail(f"Visual mismatch for '{name}'. See {failDir}")

    return _test
