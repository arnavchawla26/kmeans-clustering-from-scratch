"""Synthetic 2D dataset generation for demos and tests.

Dependency-free re-implementation of the spirit of scikit-learn's
``make_blobs``: draw isotropic Gaussian clusters ("blobs") around a set of
centers.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple, Union

Point = Tuple[float, float]

CentersSpec = Union[int, Sequence[Point]]


def make_blobs(
    n_samples: int = 300,
    centers: CentersSpec = 3,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    random_state: int | None = None,
) -> Tuple[List[Point], List[int]]:
    """Generate isotropic 2D Gaussian blobs.

    Parameters
    ----------
    n_samples:
        Total number of points to generate, spread as evenly as possible
        across the clusters.
    centers:
        Either an integer number of clusters (centers are then placed
        uniformly at random within ``center_box``), or an explicit list of
        ``(x, y)`` center coordinates.
    cluster_std:
        Standard deviation of the Gaussian noise around each center.
    center_box:
        ``(min, max)`` bounding box used to place random centers when
        ``centers`` is an integer.
    random_state:
        Optional seed for reproducibility.

    Returns
    -------
    (points, labels):
        ``points`` is a list of ``(x, y)`` tuples; ``labels`` is the ground
        -truth cluster index (0-based) each point was drawn from, in the
        same order -- handy for evaluating a clustering against the truth
        in demos, even though a real k-means use case would not have this.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if cluster_std < 0:
        raise ValueError("cluster_std must be non-negative")

    rng = random.Random(random_state)

    if isinstance(centers, int):
        if centers <= 0:
            raise ValueError("centers must be a positive integer when given as a count")
        lo, hi = center_box
        center_points: List[Point] = [
            (rng.uniform(lo, hi), rng.uniform(lo, hi)) for _ in range(centers)
        ]
    else:
        center_points = [(float(x), float(y)) for x, y in centers]
        if not center_points:
            raise ValueError("centers list must not be empty")

    n_centers = len(center_points)
    # Distribute n_samples across clusters as evenly as possible.
    base, remainder = divmod(n_samples, n_centers)
    counts = [base + (1 if i < remainder else 0) for i in range(n_centers)]

    points: List[Point] = []
    labels: List[int] = []
    for cluster_idx, (cx, cy) in enumerate(center_points):
        for _ in range(counts[cluster_idx]):
            points.append((rng.gauss(cx, cluster_std), rng.gauss(cy, cluster_std)))
            labels.append(cluster_idx)

    # Shuffle in lockstep so consumers can't accidentally rely on the
    # cluster-sorted ordering (real datasets rarely arrive pre-sorted).
    indices = list(range(len(points)))
    rng.shuffle(indices)
    points = [points[i] for i in indices]
    labels = [labels[i] for i in indices]

    return points, labels
