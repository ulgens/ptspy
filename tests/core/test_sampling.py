"""Tests for ptspy.more.sampling — Poisson-Disc sampling."""

import numpy as np
import pytest
from ptspy.more.sampling import poissonDisc


# ---------------------------------------------------------------------------
# Basic output shape and dimensionality
# ---------------------------------------------------------------------------


def test_poissonDisc_2d_default():
    """2D sampling returns (N, 2) array."""
    pts = poissonDisc([100, 100], 10, seed=42)
    assert pts.ndim == 2
    assert pts.shape[1] == 2
    assert pts.dtype == np.float32
    assert len(pts) > 0


def test_poissonDisc_3d():
    """3D sampling returns (N, 3) array."""
    pts = poissonDisc([50, 50, 50], 15, seed=42)
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert len(pts) > 0


def test_poissonDisc_1d():
    """1D sampling returns (N, 1) array."""
    pts = poissonDisc([100], 5, seed=42)
    assert pts.ndim == 2
    assert pts.shape[1] == 1
    assert len(pts) > 0


# ---------------------------------------------------------------------------
# Core guarantees
# ---------------------------------------------------------------------------


def test_poissonDisc_minimum_distance():
    """All pairwise distances must be >= radius."""
    radius = 8.0
    pts = poissonDisc([100, 100], radius, seed=7)
    # Broadcast pairwise distance check
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    assert np.all(dists >= radius - 1e-5), f"Min pairwise distance {dists.min():.6f} < {radius}"


def test_poissonDisc_bounds():
    """All points must lie within [0, bounds)."""
    bounds = [80, 120]
    pts = poissonDisc(bounds, 10, seed=3)
    assert np.all(pts >= 0)
    assert np.all(pts[:, 0] < bounds[0])
    assert np.all(pts[:, 1] < bounds[1])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_poissonDisc_seed_reproducibility():
    """Same seed produces identical output."""
    a = poissonDisc([100, 100], 10, seed=99)
    b = poissonDisc([100, 100], 10, seed=99)
    np.testing.assert_array_equal(a, b)


def test_poissonDisc_different_seeds():
    """Different seeds produce different output."""
    a = poissonDisc([100, 100], 10, seed=1)
    b = poissonDisc([100, 100], 10, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Point count sanity
# ---------------------------------------------------------------------------


def test_poissonDisc_point_count_reasonable():
    """Point count should be within a plausible range for given area and radius."""
    bounds = [200, 200]
    radius = 10
    pts = poissonDisc(bounds, radius, seed=42)
    area = 200 * 200
    # Loose bounds: theoretical max ≈ area / (pi * (r/2)^2), practical is ~60-80% of hex packing
    maxTheoretical = area / (np.pi * (radius / 2) ** 2)
    assert len(pts) > maxTheoretical * 0.15, f"Too few points: {len(pts)}"
    assert len(pts) < maxTheoretical * 1.5, f"Too many points: {len(pts)}"


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_poissonDisc_invalid_radius_zero():
    with pytest.raises(ValueError, match="radius must be positive"):
        poissonDisc([100, 100], 0, seed=1)


def test_poissonDisc_invalid_radius_negative():
    with pytest.raises(ValueError, match="radius must be positive"):
        poissonDisc([100, 100], -5, seed=1)


def test_poissonDisc_invalid_k():
    with pytest.raises(ValueError, match="k must be positive"):
        poissonDisc([100, 100], 10, k=0, seed=1)


def test_poissonDisc_empty_bounds():
    with pytest.raises(ValueError):
        poissonDisc([], 10, seed=1)


def test_poissonDisc_negative_bounds():
    with pytest.raises(ValueError, match="positive"):
        poissonDisc([100, -50], 10, seed=1)


# ---------------------------------------------------------------------------
# RNG parameter
# ---------------------------------------------------------------------------


def test_poissonDisc_rng_parameter():
    """Passing an explicit rng works and respects its state."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = poissonDisc([100, 100], 10, rng=rng1)
    b = poissonDisc([100, 100], 10, rng=rng2)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Partial application
# ---------------------------------------------------------------------------


def test_poissonDisc_partial():
    """poissonDisc supports partial application via @partial decorator."""
    sampler = poissonDisc([100, 100])
    pts = sampler(10, seed=42)
    assert pts.ndim == 2
    assert pts.shape[1] == 2


# ---------------------------------------------------------------------------
# Scalar bounds
# ---------------------------------------------------------------------------


def test_poissonDisc_scalar_bounds():
    """Scalar bounds produces 1-D points."""
    pts = poissonDisc(100, 5, seed=42)
    assert pts.ndim == 2
    assert pts.shape[1] == 1
    assert len(pts) > 0
    assert np.all(pts >= 0)
    assert np.all(pts < 100)


def test_poissonDisc_scalar_bounds_minimum_distance():
    """Scalar bounds respects minimum distance constraint."""
    radius = 5.0
    pts = poissonDisc(100, radius, seed=7)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    assert np.all(dists >= radius - 1e-5)


# ---------------------------------------------------------------------------
# Bounding-box bounds [[min], [max]]
# ---------------------------------------------------------------------------


def test_poissonDisc_bbox_2d():
    """2-D bounds [[x0,y0],[x1,y1]] constrains points to that box."""
    bbox = [[50, 100], [130, 160]]
    pts = poissonDisc(bbox, 10, seed=42)
    assert np.all(pts[:, 0] >= 50)
    assert np.all(pts[:, 1] >= 100)
    assert np.all(pts[:, 0] < 130)
    assert np.all(pts[:, 1] < 160)


def test_poissonDisc_bbox_minimum_distance():
    """Bbox bounds preserve minimum distance guarantee."""
    radius = 10.0
    pts = poissonDisc([[200, 300], [300, 400]], radius, seed=7)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    assert np.all(dists >= radius - 1e-5)


def test_poissonDisc_bbox_matches_manual_shift():
    """Bbox produces same result as size-only + manual offset."""
    a = poissonDisc([80, 60], 10, seed=42)
    b = poissonDisc([[50, 100], [130, 160]], 10, seed=42)
    np.testing.assert_allclose(b, a + np.array([50, 100], dtype=np.float32), atol=1e-6)


def test_poissonDisc_bbox_inverted():
    """Bbox with min > max in any dimension raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        poissonDisc([[100, 0], [50, 100]], 10, seed=1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_poissonDisc_radius_larger_than_domain():
    """When radius > domain diameter, only one point is returned."""
    pts = poissonDisc([10, 10], 20, seed=42)
    assert len(pts) == 1
    assert pts.shape == (1, 2)


def test_poissonDisc_elongated_bounds():
    """Non-square bounds produce points that fill the elongated domain."""
    bounds = [200, 30]
    radius = 8.0
    pts = poissonDisc(bounds, radius, seed=42)
    assert np.all(pts[:, 0] < bounds[0])
    assert np.all(pts[:, 1] < bounds[1])
    # Should have more spread in x than y
    xSpread = pts[:, 0].max() - pts[:, 0].min()
    ySpread = pts[:, 1].max() - pts[:, 1].min()
    assert xSpread > ySpread * 2, "Points should spread further in the longer dimension"
    # Distance guarantee
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    assert np.all(dists >= radius - 1e-5)


def test_poissonDisc_tiny_domain():
    """Very small domain with reasonable radius returns at least 1 point."""
    pts = poissonDisc([5, 5], 3, seed=42)
    assert len(pts) >= 1
    assert np.all(pts >= 0)
    assert np.all(pts < 5)


# ---------------------------------------------------------------------------
# k parameter effect
# ---------------------------------------------------------------------------


def test_poissonDisc_higher_k_more_points():
    """Higher k (more candidates) should generally produce at least as many points."""
    ptsLowK = poissonDisc([100, 100], 8, k=5, seed=42)
    ptsHighK = poissonDisc([100, 100], 8, k=100, seed=42)
    # Higher k explores more aggressively, so should pack at least as tightly
    # Allow some tolerance since RNG paths diverge
    assert len(ptsHighK) >= len(ptsLowK) * 0.8, f"k=100 produced {len(ptsHighK)} points vs k=5 produced {len(ptsLowK)}"


# ---------------------------------------------------------------------------
# 3D minimum distance
# ---------------------------------------------------------------------------


def test_poissonDisc_3d_minimum_distance():
    """3D sampling respects minimum distance constraint."""
    radius = 12.0
    pts = poissonDisc([50, 50, 50], radius, seed=7)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    assert np.all(dists >= radius - 1e-5), f"Min pairwise distance {dists.min():.6f} < {radius}"
