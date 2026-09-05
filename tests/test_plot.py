import os

import pytest

from kmeans_scratch.plot import export_svg, render_ascii


POINTS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (10.0, 10.0), (11.0, 10.0), (10.0, 11.0)]
LABELS = [0, 0, 0, 1, 1, 1]
CENTROIDS = [(0.33, 0.33), (10.33, 10.33)]


def test_render_ascii_returns_grid_of_expected_shape():
    art = render_ascii(POINTS, LABELS, centroids=CENTROIDS, width=40, height=10)
    lines = art.splitlines()
    # top border, `height` rows, bottom border, legend line
    assert len(lines) == 10 + 3
    assert lines[0] == "+" + "-" * 40 + "+"
    assert lines[-2] == "+" + "-" * 40 + "+"


def test_render_ascii_legend_mentions_every_cluster():
    art = render_ascii(POINTS, LABELS)
    legend = art.splitlines()[-1]
    assert "cluster 0" in legend
    assert "cluster 1" in legend


def test_render_ascii_empty_points():
    assert render_ascii([], []) == "(no points to plot)"


def test_render_ascii_places_centroid_marker():
    art = render_ascii(POINTS, LABELS, centroids=CENTROIDS, width=20, height=20)
    assert "@" in art


def test_export_svg_writes_a_valid_looking_file(tmp_path):
    out_path = str(tmp_path / "clusters.svg")
    result_path = export_svg(POINTS, LABELS, centroids=CENTROIDS, path=out_path, title="Test Plot")

    assert result_path == out_path
    assert os.path.exists(out_path)

    content = open(out_path, encoding="utf-8").read()
    assert content.startswith("<svg")
    assert content.strip().endswith("</svg>")
    assert content.count("<circle") == len(POINTS)
    assert "Test Plot" in content


def test_export_svg_rejects_empty_points(tmp_path):
    with pytest.raises(ValueError):
        export_svg([], [], path=str(tmp_path / "out.svg"))


def test_export_svg_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        export_svg(POINTS, [0, 1], path=str(tmp_path / "out.svg"))


def test_export_svg_without_centroids_has_no_markers(tmp_path):
    out_path = str(tmp_path / "no_centroids.svg")
    export_svg(POINTS, LABELS, path=out_path)
    content = open(out_path, encoding="utf-8").read()
    # centroid markers are drawn as <line> pairs inside a <g>; none should exist
    assert "<line" not in content


def test_export_svg_escapes_title():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "out.svg")
        export_svg(POINTS, LABELS, path=out_path, title="A & B < C")
        content = open(out_path, encoding="utf-8").read()
        assert "A &amp; B &lt; C" in content
