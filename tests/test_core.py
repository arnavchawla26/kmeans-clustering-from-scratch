import random

import pytest

from kmeans_scratch.core import KMeans, kmeans_plus_plus_init, squared_distance
from kmeans_scratch.datasets import make_blobs


def test_squared_distance_basic():
    assert squared_distance((0, 0), (3, 4)) == 25
    assert squared_distance((1, 1, 1), (1, 1, 1)) == 0


def test_squared_distance_dimension_mismatch():
    with pytest.raises(ValueError):
        squared_distance((0, 0), (0, 0, 0))


def test_kmeans_plus_plus_init_returns_k_distinct_ish_centroids():
    points = [(0.0, 0.0), (0.1, 0.0), (10.0, 10.0), (10.1, 10.0), (20.0, 0.0)]
    rng = random.Random(0)
    centroids = kmeans_plus_plus_init(points, 3, rng)
    assert len(centroids) == 3
    for c in centroids:
        assert c in [tuple(p) for p in points]


def test_kmeans_plus_plus_init_rejects_k_larger_than_n():
    with pytest.raises(ValueError):
        kmeans_plus_plus_init([(0, 0), (1, 1)], 3, random.Random(0))


def test_kmeans_fit_separates_well_separated_blobs():
    # Three tight, far-apart blobs: k-means should recover them exactly
    # (up to a permutation of cluster indices).
    points, true_labels = make_blobs(
        n_samples=90, centers=[(0, 0), (50, 0), (0, 50)], cluster_std=0.5, random_state=1
    )
    model = KMeans(n_clusters=3, n_init=5, random_state=1)
    model.fit(points)

    assert len(model.centroids_) == 3
    assert len(model.labels_) == len(points)

    # Build a mapping from predicted label -> true label using majority vote,
    # then check every point agrees with its predicted cluster under that
    # mapping (this sidesteps the arbitrary labeling of unsupervised output).
    from collections import Counter, defaultdict

    votes = defaultdict(Counter)
    for pred, true in zip(model.labels_, true_labels):
        votes[pred][true] += 1
    mapping = {pred: counter.most_common(1)[0][0] for pred, counter in votes.items()}

    mismatches = sum(
        1 for pred, true in zip(model.labels_, true_labels) if mapping[pred] != true
    )
    assert mismatches == 0


def test_kmeans_inertia_decreases_or_equal_with_more_clusters():
    points, _ = make_blobs(n_samples=60, centers=4, cluster_std=1.5, random_state=2)
    inertias = []
    for k in (1, 2, 3, 4, 5):
        model = KMeans(n_clusters=k, n_init=5, random_state=2)
        model.fit(points)
        inertias.append(model.inertia_)

    for earlier, later in zip(inertias, inertias[1:]):
        assert later <= earlier + 1e-6


def test_kmeans_predict_matches_fit_predict_on_same_points():
    points, _ = make_blobs(n_samples=40, centers=3, cluster_std=0.8, random_state=3)
    model = KMeans(n_clusters=3, n_init=5, random_state=3)
    model.fit(points)
    assert model.predict(points) == model.labels_


def test_kmeans_predict_before_fit_raises():
    model = KMeans(n_clusters=2)
    with pytest.raises(RuntimeError):
        model.predict([(0, 0)])


def test_kmeans_rejects_more_clusters_than_points():
    model = KMeans(n_clusters=5)
    with pytest.raises(ValueError):
        model.fit([(0, 0), (1, 1)])


def test_kmeans_rejects_empty_dataset():
    model = KMeans(n_clusters=1)
    with pytest.raises(ValueError):
        model.fit([])


def test_kmeans_single_cluster_is_the_mean():
    points = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0), (1.0, -2.0)]
    model = KMeans(n_clusters=1, n_init=1, random_state=0)
    model.fit(points)
    assert model.centroids_[0] == pytest.approx((1.0, 0.0))
    assert set(model.labels_) == {0}


def test_kmeans_handles_duplicate_points_without_crashing():
    points = [(1.0, 1.0)] * 10 + [(5.0, 5.0)] * 10
    model = KMeans(n_clusters=2, n_init=3, random_state=0)
    model.fit(points)
    assert len(set(model.labels_)) == 2


def test_kmeans_is_deterministic_given_a_seed():
    points, _ = make_blobs(n_samples=50, centers=3, cluster_std=1.0, random_state=7)
    m1 = KMeans(n_clusters=3, n_init=5, random_state=123)
    m1.fit(points)
    m2 = KMeans(n_clusters=3, n_init=5, random_state=123)
    m2.fit(points)
    assert m1.centroids_ == m2.centroids_
    assert m1.labels_ == m2.labels_
    assert m1.inertia_ == m2.inertia_


def test_fit_predict_returns_same_as_labels_():
    points, _ = make_blobs(n_samples=30, centers=2, cluster_std=1.0, random_state=5)
    model = KMeans(n_clusters=2, n_init=3, random_state=5)
    labels = model.fit_predict(points)
    assert labels == model.labels_
