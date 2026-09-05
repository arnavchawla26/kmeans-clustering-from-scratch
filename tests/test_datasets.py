import pytest

from kmeans_scratch.datasets import make_blobs


def test_make_blobs_returns_requested_sample_count():
    points, labels = make_blobs(n_samples=100, centers=4, random_state=0)
    assert len(points) == 100
    assert len(labels) == 100


def test_make_blobs_label_set_matches_center_count():
    points, labels = make_blobs(n_samples=99, centers=3, random_state=0)
    assert set(labels) == {0, 1, 2}


def test_make_blobs_uneven_sample_count_still_covers_all_centers():
    # 10 samples over 3 centers -> counts should be 4,3,3 (or similar), never zero.
    points, labels = make_blobs(n_samples=10, centers=3, random_state=1)
    from collections import Counter

    counts = Counter(labels)
    assert set(counts) == {0, 1, 2}
    assert sum(counts.values()) == 10
    assert max(counts.values()) - min(counts.values()) <= 1


def test_make_blobs_explicit_centers():
    points, labels = make_blobs(
        n_samples=40, centers=[(0.0, 0.0), (100.0, 100.0)], cluster_std=0.1, random_state=2
    )
    assert set(labels) == {0, 1}
    # With a tiny std and centers 100+ apart, every point should land near
    # its assigned center.
    for p, lbl in zip(points, labels):
        cx, cy = [(0.0, 0.0), (100.0, 100.0)][lbl]
        assert abs(p[0] - cx) < 5
        assert abs(p[1] - cy) < 5


def test_make_blobs_is_reproducible_with_seed():
    p1, l1 = make_blobs(n_samples=50, centers=3, random_state=99)
    p2, l2 = make_blobs(n_samples=50, centers=3, random_state=99)
    assert p1 == p2
    assert l1 == l2


def test_make_blobs_rejects_non_positive_samples():
    with pytest.raises(ValueError):
        make_blobs(n_samples=0)


def test_make_blobs_rejects_negative_std():
    with pytest.raises(ValueError):
        make_blobs(n_samples=10, cluster_std=-1.0)


def test_make_blobs_rejects_empty_center_list():
    with pytest.raises(ValueError):
        make_blobs(n_samples=10, centers=[])


def test_make_blobs_zero_std_places_points_exactly_at_center():
    points, labels = make_blobs(
        n_samples=6, centers=[(1.0, 2.0), (3.0, 4.0)], cluster_std=0.0, random_state=3
    )
    centers = [(1.0, 2.0), (3.0, 4.0)]
    for p, lbl in zip(points, labels):
        assert p == pytest.approx(centers[lbl])
