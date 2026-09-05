"""Model-selection helpers: the elbow method and silhouette analysis.

Both are standard tools for picking k in unsupervised k-means when there is
no ground-truth labeling to check against.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .core import KMeans, Point, squared_distance


def elbow_method(
    points: Sequence[Point],
    k_values: Sequence[int],
    n_init: int = 10,
    max_iter: int = 300,
    random_state: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Fit KMeans for each k in ``k_values`` and return ``(k, inertia)`` pairs.

    Inertia (sum of squared distances to the assigned centroid) always
    decreases as k grows, so there is no single "correct" answer here --
    the caller (or a human looking at a plot) looks for the "elbow" where
    the marginal decrease in inertia sharply levels off.
    """
    results: List[Tuple[int, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state)
        model.fit(points)
        results.append((k, model.inertia_))
    return results


def find_elbow_k(elbow_results: Sequence[Tuple[int, float]]) -> int:
    """Heuristically pick the elbow point from ``elbow_method`` output.

    Uses the "maximum distance to the chord" heuristic: draw a straight
    line from the first to the last (k, inertia) point, and pick the k
    whose point is farthest (perpendicular distance) from that line. This
    is a common, simple stand-in for eyeballing the elbow on a plot.
    """
    if len(elbow_results) < 3:
        # Not enough points for the geometry to be meaningful; just return
        # the smallest k tested.
        return elbow_results[0][0]

    ks = [k for k, _ in elbow_results]
    inertias = [i for _, i in elbow_results]

    # Normalize both axes to [0, 1] so the two very different scales (k is
    # small integers, inertia can be huge) don't distort the geometry.
    k_min, k_max = min(ks), max(ks)
    i_min, i_max = min(inertias), max(inertias)
    k_span = (k_max - k_min) or 1.0
    i_span = (i_max - i_min) or 1.0

    xs = [(k - k_min) / k_span for k in ks]
    ys = [(i - i_min) / i_span for i in inertias]

    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    line_len = math.hypot(x2 - x1, y2 - y1) or 1.0

    best_idx = 0
    best_dist = -1.0
    for idx in range(len(xs)):
        # Perpendicular distance from (xs[idx], ys[idx]) to the line
        # through (x1, y1)-(x2, y2), via the standard cross-product formula.
        dist = abs((y2 - y1) * xs[idx] - (x2 - x1) * ys[idx] + x2 * y1 - y2 * x1) / line_len
        if dist > best_dist:
            best_dist = dist
            best_idx = idx

    return ks[best_idx]


def silhouette_samples(points: Sequence[Point], labels: Sequence[int]) -> List[float]:
    """Per-point silhouette coefficients, in ``[-1, 1]``.

    For point ``i`` in cluster ``A``:
      a(i) = mean distance from i to every other point in A
      b(i) = min over other clusters B of (mean distance from i to every
             point in B)
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    s(i) close to 1 means i is well inside its own cluster and far from
    others; close to 0 means it sits near a cluster boundary; negative
    means it is probably in the wrong cluster. A cluster of size 1 gets
    s(i) = 0 by convention (a(i) is undefined otherwise).

    This is an O(n^2) reference implementation, intended for the small/demo
    -sized datasets this package targets rather than large-scale use.
    """
    n = len(points)
    if n != len(labels):
        raise ValueError("points and labels must be the same length")
    if n == 0:
        return []

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("silhouette score requires at least 2 clusters")

    # Precompute the full pairwise distance matrix once; reused for both
    # a(i) and b(i) below.
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(squared_distance(points[i], points[j]))
            dist[i][j] = d
            dist[j][i] = d

    cluster_members = {lbl: [idx for idx in range(n) if labels[idx] == lbl] for lbl in unique_labels}

    scores = [0.0] * n
    for i in range(n):
        own_label = labels[i]
        own_members = [j for j in cluster_members[own_label] if j != i]

        if not own_members:
            scores[i] = 0.0
            continue

        a_i = sum(dist[i][j] for j in own_members) / len(own_members)

        b_i = math.inf
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_members = cluster_members[other_label]
            if not other_members:
                continue
            mean_dist = sum(dist[i][j] for j in other_members) / len(other_members)
            b_i = min(b_i, mean_dist)

        denom = max(a_i, b_i)
        scores[i] = 0.0 if denom == 0 else (b_i - a_i) / denom

    return scores


def silhouette_score(points: Sequence[Point], labels: Sequence[int]) -> float:
    """Mean silhouette coefficient across all points. See ``silhouette_samples``."""
    samples = silhouette_samples(points, labels)
    return sum(samples) / len(samples)


def best_k_by_silhouette(
    points: Sequence[Point],
    k_values: Sequence[int],
    n_init: int = 10,
    max_iter: int = 300,
    random_state: Optional[int] = None,
) -> Tuple[int, List[Tuple[int, float]]]:
    """Fit KMeans for each k >= 2 in ``k_values`` and score it by silhouette.

    Returns ``(best_k, [(k, mean_silhouette), ...])``. k=1 is skipped since
    silhouette is undefined for a single cluster.
    """
    scored: List[Tuple[int, float]] = []
    for k in k_values:
        if k < 2:
            continue
        model = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state)
        model.fit(points)
        score = silhouette_score(points, model.labels_)
        scored.append((k, score))

    if not scored:
        raise ValueError("k_values must include at least one value >= 2")

    best_k = max(scored, key=lambda pair: pair[1])[0]
    return best_k, scored
