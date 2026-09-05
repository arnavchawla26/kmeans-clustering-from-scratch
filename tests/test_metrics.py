import pytest

from kmeans_scratch.core import KMeans
from kmeans_scratch.datasets import make_blobs
from kmeans_scratch.metrics import (
    best_k_by_silhouette,
    elbow_method,
    find_elbow_k,
    silhouette_samples,
    silhouette_score,
)


def test_elbow_method_returns_one_result_per_k():
    points, _ = make_blobs(n_samples=40, centers=3, random_state=0)
    results = elbow_method(points, [1, 2, 3, 4], n_init=3, random_state=0)
    assert [k for k, _ in results] == [1, 2, 3, 4]


def test_elbow_method_inertia_is_monotonically_non_increasing():
    points, _ = make_blobs(n_samples=40, centers=3, random_state=0)
    results = elbow_method(points, [1, 2, 3, 4, 5], n_init=5, random_state=0)
    inertias = [i for _, i in results]
    assert all(a >= b - 1e-6 for a, b in zip(inertias, inertias[1:]))


def test_find_elbow_k_prefers_the_bend_on_a_synthetic_curve():
    # A textbook elbow shape: steep drop then a plateau, elbow clearly at k=3.
    results = [(1, 100.0), (2, 40.0), (3, 10.0), (4, 9.0), (5, 8.5), (6, 8.2)]
    assert find_elbow_k(results) == 3


def test_find_elbow_k_handles_too_few_points():
    assert find_elbow_k([(2, 5.0)]) == 2
    assert find_elbow_k([(1, 10.0), (2, 5.0)]) == 1


def test_silhouette_samples_length_matches_input():
    points, _ = make_blobs(n_samples=30, centers=3, cluster_std=0.5, random_state=1)
    model = KMeans(n_clusters=3, n_init=5, random_state=1).fit(points)
    scores = silhouette_samples(points, model.labels_)
    assert len(scores) == len(points)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_silhouette_score_is_high_for_well_separated_clusters():
    points, true_labels = make_blobs(
        n_samples=60, centers=[(0, 0), (100, 0), (0, 100)], cluster_std=0.5, random_state=2
    )
    score = silhouette_score(points, true_labels)
    assert score > 0.9


def test_silhouette_score_is_lower_for_overlapping_clusters():
    separated, sep_labels = make_blobs(
        n_samples=60, centers=[(0, 0), (100, 0)], cluster_std=0.5, random_state=3
    )
    overlapping, ov_labels = make_blobs(
        n_samples=60, centers=[(0, 0), (0.5, 0)], cluster_std=1.0, random_state=3
    )
    sep_score = silhouette_score(separated, sep_labels)
    ov_score = silhouette_score(overlapping, ov_labels)
    assert sep_score > ov_score


def test_silhouette_score_requires_at_least_two_clusters():
    with pytest.raises(ValueError):
        silhouette_score([(0, 0), (1, 1)], [0, 0])


def test_silhouette_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        silhouette_samples([(0, 0), (1, 1)], [0])


def test_best_k_by_silhouette_recovers_true_k():
    points, _ = make_blobs(
        n_samples=90, centers=[(0, 0), (30, 0), (0, 30)], cluster_std=1.0, random_state=4
    )
    best_k, scored = best_k_by_silhouette(points, range(2, 7), n_init=5, random_state=4)
    assert best_k == 3
    assert [k for k, _ in scored] == [2, 3, 4, 5, 6]


def test_best_k_by_silhouette_skips_k_equals_1():
    points, _ = make_blobs(n_samples=30, centers=2, random_state=5)
    _, scored = best_k_by_silhouette(points, [1, 2, 3], n_init=3, random_state=5)
    assert 1 not in [k for k, _ in scored]


def test_best_k_by_silhouette_requires_a_usable_k_value():
    with pytest.raises(ValueError):
        best_k_by_silhouette([(0, 0), (1, 1)], [1])
