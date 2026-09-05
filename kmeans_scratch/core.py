"""Dependency-free k-means clustering: k-means++ seeding + Lloyd's algorithm.

Everything here operates on plain Python data: a "point" is a tuple/list of
floats of any fixed dimension, and a "dataset" is a sequence of points. No
numpy, no external dependencies -- just the standard library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Point = Sequence[float]


def squared_distance(a: Point, b: Point) -> float:
    """Squared Euclidean distance between two points of equal dimension."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _mean_point(points: Sequence[Point], dim: int) -> Tuple[float, ...]:
    n = len(points)
    sums = [0.0] * dim
    for p in points:
        for i in range(dim):
            sums[i] += p[i]
    return tuple(s / n for s in sums)


def _nearest_centroid_index(point: Point, centroids: Sequence[Point]) -> Tuple[int, float]:
    best_idx = 0
    best_dist = squared_distance(point, centroids[0])
    for idx in range(1, len(centroids)):
        d = squared_distance(point, centroids[idx])
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx, best_dist


def kmeans_plus_plus_init(
    points: Sequence[Point], k: int, rng: random.Random
) -> List[Tuple[float, ...]]:
    """Seed k centroids using the k-means++ scheme.

    Picks the first centroid uniformly at random, then repeatedly picks the
    next centroid from the remaining points with probability proportional to
    its squared distance to the nearest already-chosen centroid. This spreads
    the initial centroids out and gives noticeably better and more consistent
    results than picking k random points, especially as k grows.
    """
    n = len(points)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > n:
        raise ValueError(f"k={k} cannot exceed the number of points ({n})")

    first = rng.randrange(n)
    centroids: List[Tuple[float, ...]] = [tuple(points[first])]

    # Track, for every point, its squared distance to the nearest chosen
    # centroid so far -- updated incrementally rather than recomputed from
    # scratch on every iteration.
    closest_sq_dist = [squared_distance(p, centroids[0]) for p in points]

    while len(centroids) < k:
        total = sum(closest_sq_dist)
        if total <= 0.0:
            # All remaining points coincide with a chosen centroid (or the
            # dataset is degenerate); fall back to uniform choice among
            # points not already used as a centroid to still return k
            # distinct-as-possible centroids.
            remaining = [i for i in range(n) if tuple(points[i]) not in centroids]
            pick = rng.choice(remaining) if remaining else rng.randrange(n)
        else:
            target = rng.random() * total
            acc = 0.0
            pick = n - 1
            for i, d in enumerate(closest_sq_dist):
                acc += d
                if acc >= target:
                    pick = i
                    break

        new_centroid = tuple(points[pick])
        centroids.append(new_centroid)
        for i, p in enumerate(points):
            d = squared_distance(p, new_centroid)
            if d < closest_sq_dist[i]:
                closest_sq_dist[i] = d

    return centroids


@dataclass
class KMeans:
    """k-means clustering via k-means++ seeding and Lloyd's algorithm.

    Parameters
    ----------
    n_clusters:
        Number of clusters, k.
    n_init:
        Number of independent (differently-seeded) runs; the run with the
        lowest final inertia is kept. Mirrors the standard practice of
        re-running k-means several times since Lloyd's algorithm only finds
        a local optimum.
    max_iter:
        Maximum number of Lloyd iterations per run.
    tol:
        Convergence threshold: a run stops early once no centroid moves (in
        squared distance) more than ``tol`` between iterations.
    random_state:
        Optional seed for reproducibility.

    Attributes (set after calling ``fit``)
    ----------
    centroids_:
        The k learned centroids.
    labels_:
        Cluster index (0..k-1) assigned to each training point.
    inertia_:
        Sum of squared distances of every point to its assigned centroid,
        for the best of the ``n_init`` runs.
    n_iter_:
        Number of Lloyd iterations the best run took to converge.
    """

    n_clusters: int
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-8
    random_state: Optional[int] = None

    centroids_: List[Tuple[float, ...]] = field(default_factory=list, init=False)
    labels_: List[int] = field(default_factory=list, init=False)
    inertia_: float = field(default=float("inf"), init=False)
    n_iter_: int = field(default=0, init=False)

    def fit(self, points: Sequence[Point]) -> "KMeans":
        if len(points) == 0:
            raise ValueError("cannot fit KMeans on an empty dataset")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.n_clusters > len(points):
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot exceed the number of points "
                f"({len(points)})"
            )

        points = [tuple(float(v) for v in p) for p in points]
        dim = len(points[0])
        rng = random.Random(self.random_state)

        best_centroids: Optional[List[Tuple[float, ...]]] = None
        best_labels: Optional[List[int]] = None
        best_inertia = float("inf")
        best_n_iter = 0

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._single_run(points, dim, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        assert best_centroids is not None and best_labels is not None
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def _single_run(
        self, points: List[Tuple[float, ...]], dim: int, rng: random.Random
    ) -> Tuple[List[Tuple[float, ...]], List[int], float, int]:
        centroids = kmeans_plus_plus_init(points, self.n_clusters, rng)
        labels = [0] * len(points)

        n_iter = 0
        for iteration in range(1, self.max_iter + 1):
            n_iter = iteration
            # Assignment step.
            for i, p in enumerate(points):
                idx, _ = _nearest_centroid_index(p, centroids)
                labels[i] = idx

            # Update step.
            new_centroids: List[Tuple[float, ...]] = []
            buckets: List[List[Tuple[float, ...]]] = [[] for _ in range(self.n_clusters)]
            for p, lbl in zip(points, labels):
                buckets[lbl].append(p)

            shift = 0.0
            for c_idx in range(self.n_clusters):
                bucket = buckets[c_idx]
                if bucket:
                    new_c = _mean_point(bucket, dim)
                else:
                    # Re-seed a centroid that lost all its points mid-fit by
                    # handing it the point currently farthest from its own
                    # assigned centroid, which avoids collapsing to fewer
                    # than k clusters.
                    farthest_idx = max(
                        range(len(points)),
                        key=lambda i: squared_distance(points[i], centroids[labels[i]]),
                    )
                    new_c = points[farthest_idx]
                shift = max(shift, squared_distance(new_c, centroids[c_idx]))
                new_centroids.append(new_c)

            centroids = new_centroids
            if shift <= self.tol:
                break

        inertia = sum(
            squared_distance(p, centroids[lbl]) for p, lbl in zip(points, labels)
        )
        return centroids, labels, inertia, n_iter

    def predict(self, points: Sequence[Point]) -> List[int]:
        """Assign each point in ``points`` to its nearest learned centroid."""
        if not self.centroids_:
            raise RuntimeError("KMeans instance is not fitted yet; call fit() first")
        return [_nearest_centroid_index(p, self.centroids_)[0] for p in points]

    def fit_predict(self, points: Sequence[Point]) -> List[int]:
        self.fit(points)
        return self.labels_
