import csv
import json
import os
import subprocess
import sys

import pytest


def _write_points_csv(path, points, header=True):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(["x", "y"])
        for x, y in points:
            writer.writerow([x, y])


def run_cli(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "kmeans_scratch.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


BLOB_POINTS = (
    [(0.0 + i * 0.1, 0.0 + i * 0.05) for i in range(15)]
    + [(20.0 + i * 0.1, 0.0 + i * 0.05) for i in range(15)]
    + [(0.0 + i * 0.1, 20.0 + i * 0.05) for i in range(15)]
)


def test_cli_fit_writes_model_labels_and_svg(tmp_path):
    input_csv = tmp_path / "points.csv"
    _write_points_csv(input_csv, BLOB_POINTS)

    model_out = tmp_path / "model.json"
    labels_out = tmp_path / "labels.csv"
    svg_out = tmp_path / "plot.svg"

    result = run_cli(
        [
            "fit",
            "--input",
            str(input_csv),
            "--k",
            "3",
            "--seed",
            "0",
            "--model-out",
            str(model_out),
            "--labels-out",
            str(labels_out),
            "--svg-out",
            str(svg_out),
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "Fitted KMeans(k=3)" in result.stdout
    assert model_out.exists()
    assert labels_out.exists()
    assert svg_out.exists()

    payload = json.loads(model_out.read_text())
    assert payload["n_clusters"] == 3
    assert len(payload["centroids"]) == 3

    with open(labels_out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["x", "y", "cluster"]
    assert len(rows) == 1 + len(BLOB_POINTS)


def test_cli_fit_ascii_flag_prints_plot(tmp_path):
    input_csv = tmp_path / "points.csv"
    _write_points_csv(input_csv, BLOB_POINTS)

    result = run_cli(["fit", "--input", str(input_csv), "--k", "3", "--seed", "0", "--ascii"])
    assert result.returncode == 0, result.stderr
    assert "+" in result.stdout  # ASCII plot border


def test_cli_predict_uses_saved_model(tmp_path):
    input_csv = tmp_path / "points.csv"
    _write_points_csv(input_csv, BLOB_POINTS)
    model_out = tmp_path / "model.json"

    fit_result = run_cli(
        ["fit", "--input", str(input_csv), "--k", "3", "--seed", "0", "--model-out", str(model_out)]
    )
    assert fit_result.returncode == 0, fit_result.stderr

    new_points_csv = tmp_path / "new_points.csv"
    _write_points_csv(new_points_csv, [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)])

    predict_result = run_cli(["predict", "--model", str(model_out), "--input", str(new_points_csv)])
    assert predict_result.returncode == 0, predict_result.stderr

    lines = [line for line in predict_result.stdout.strip().splitlines() if line]
    assert len(lines) == 3
    labels = [int(line.split(",")[-1]) for line in lines]
    # The three probe points sit in three different, well-separated blobs,
    # so they must end up in three different clusters.
    assert len(set(labels)) == 3


def test_cli_elbow_reports_suggested_k(tmp_path):
    input_csv = tmp_path / "points.csv"
    _write_points_csv(input_csv, BLOB_POINTS)

    result = run_cli(["elbow", "--input", str(input_csv), "--kmin", "1", "--kmax", "5", "--seed", "0"])
    assert result.returncode == 0, result.stderr
    assert "Suggested k" in result.stdout


def test_cli_silhouette_reports_best_k(tmp_path):
    input_csv = tmp_path / "points.csv"
    _write_points_csv(input_csv, BLOB_POINTS)

    result = run_cli(
        ["silhouette", "--input", str(input_csv), "--kmin", "2", "--kmax", "5", "--seed", "0"]
    )
    assert result.returncode == 0, result.stderr
    assert "Best k by mean silhouette score: 3" in result.stdout


def test_cli_demo_runs_end_to_end():
    result = run_cli(["demo", "--samples", "60", "--centers", "3", "--k", "3", "--seed", "42"])
    assert result.returncode == 0, result.stderr
    assert "Fitted KMeans(k=3)" in result.stdout
    assert "Mean silhouette score" in result.stdout


def test_cli_demo_with_svg_and_elbow(tmp_path):
    svg_out = tmp_path / "demo.svg"
    result = run_cli(
        [
            "demo",
            "--samples",
            "60",
            "--centers",
            "3",
            "--k",
            "3",
            "--seed",
            "42",
            "--show-elbow",
            "--kmax-scan",
            "5",
            "--svg-out",
            str(svg_out),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "Elbow scan" in result.stdout
    assert svg_out.exists()


def test_cli_fit_rejects_missing_file():
    result = run_cli(["fit", "--input", "/no/such/file.csv", "--k", "2"])
    assert result.returncode != 0


def test_cli_no_command_shows_usage():
    result = run_cli([])
    assert result.returncode != 0
