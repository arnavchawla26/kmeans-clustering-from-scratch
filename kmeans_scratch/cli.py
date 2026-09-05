"""``kmeans`` command-line interface.

Subcommands
-----------
fit         Fit KMeans on a CSV of 2-column points; save the model and
            (optionally) a labels CSV and an SVG cluster plot.
predict     Load a previously-saved model and assign cluster labels to new
            points from a CSV.
elbow       Sweep k over a range and print inertia per k (elbow heuristic),
            with a suggested k.
silhouette  Sweep k over a range and print the mean silhouette score per k,
            with the best-scoring k.
demo        Generate a synthetic blob dataset, fit KMeans on it, print an
            elbow/silhouette summary, and render an ASCII and/or SVG plot --
            a zero-setup way to see the whole package work end to end.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import List, Optional, Tuple

from .core import KMeans, Point
from .datasets import make_blobs
from .metrics import best_k_by_silhouette, elbow_method, find_elbow_k
from .plot import export_svg, render_ascii


def read_points_csv(path: str) -> List[Point]:
    """Read 2-column (x, y) points from a CSV file.

    A header row is auto-detected: if the first row's cells don't both
    parse as floats, it is treated as a header and skipped.
    """
    points: List[Point] = []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    if not rows:
        raise ValueError(f"{path} is empty")

    start = 0
    try:
        float(rows[0][0])
        float(rows[0][1])
    except (ValueError, IndexError):
        start = 1  # first row looked non-numeric; treat as a header

    for row_num, row in enumerate(rows[start:], start=start + 1):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) < 2:
            raise ValueError(f"{path}:{row_num}: expected at least 2 columns, got {row}")
        points.append((float(row[0]), float(row[1])))

    if not points:
        raise ValueError(f"{path} contains no data rows")
    return points


def write_labels_csv(path: str, points: List[Point], labels: List[int]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "cluster"])
        for p, lbl in zip(points, labels):
            writer.writerow([p[0], p[1], lbl])


def save_model(path: str, model: KMeans) -> None:
    payload = {
        "n_clusters": model.n_clusters,
        "centroids": [list(c) for c in model.centroids_],
        "inertia": model.inertia_,
        "n_iter": model.n_iter_,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_model(path: str) -> KMeans:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    model = KMeans(n_clusters=payload["n_clusters"])
    model.centroids_ = [tuple(c) for c in payload["centroids"]]
    model.inertia_ = payload.get("inertia", float("inf"))
    model.n_iter_ = payload.get("n_iter", 0)
    return model


def _cmd_fit(args: argparse.Namespace) -> int:
    points = read_points_csv(args.input)
    model = KMeans(
        n_clusters=args.k,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    model.fit(points)

    print(f"Fitted KMeans(k={args.k}) on {len(points)} points from {args.input}")
    print(f"  inertia:  {model.inertia_:.4f}")
    print(f"  n_iter:   {model.n_iter_}")
    for idx, c in enumerate(model.centroids_):
        coords = ", ".join(f"{v:.4f}" for v in c)
        print(f"  centroid {idx}: ({coords})")

    if args.model_out:
        save_model(args.model_out, model)
        print(f"Saved model to {args.model_out}")

    if args.labels_out:
        write_labels_csv(args.labels_out, points, model.labels_)
        print(f"Saved labeled points to {args.labels_out}")

    if args.svg_out:
        export_svg(
            points,
            model.labels_,
            centroids=model.centroids_,
            path=args.svg_out,
            title=f"KMeans (k={args.k})",
        )
        print(f"Saved cluster plot to {args.svg_out}")

    if args.ascii:
        print()
        print(render_ascii(points, model.labels_, centroids=model.centroids_))

    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    points = read_points_csv(args.input)
    labels = model.predict(points)

    for p, lbl in zip(points, labels):
        print(f"{p[0]},{p[1]},{lbl}")

    if args.labels_out:
        write_labels_csv(args.labels_out, points, labels)
        print(f"Saved labeled points to {args.labels_out}", file=sys.stderr)

    return 0


def _cmd_elbow(args: argparse.Namespace) -> int:
    points = read_points_csv(args.input)
    k_values = list(range(args.kmin, args.kmax + 1))
    results = elbow_method(points, k_values, n_init=args.n_init, random_state=args.seed)

    print(f"{'k':>4}  {'inertia':>14}")
    for k, inertia in results:
        print(f"{k:>4}  {inertia:>14.4f}")

    suggested = find_elbow_k(results)
    print(f"\nSuggested k (max-distance-to-chord heuristic): {suggested}")
    return 0


def _cmd_silhouette(args: argparse.Namespace) -> int:
    points = read_points_csv(args.input)
    k_values = list(range(max(2, args.kmin), args.kmax + 1))
    best_k, scored = best_k_by_silhouette(points, k_values, n_init=args.n_init, random_state=args.seed)

    print(f"{'k':>4}  {'mean silhouette':>16}")
    for k, score in scored:
        marker = "  <-- best" if k == best_k else ""
        print(f"{k:>4}  {score:>16.4f}{marker}")

    print(f"\nBest k by mean silhouette score: {best_k}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    points, true_labels = make_blobs(
        n_samples=args.samples,
        centers=args.centers,
        cluster_std=args.std,
        random_state=args.seed,
    )
    print(f"Generated {len(points)} synthetic points around {args.centers} true blob(s) "
          f"(cluster_std={args.std}, seed={args.seed}).")

    model = KMeans(n_clusters=args.k, n_init=args.n_init, random_state=args.seed)
    model.fit(points)
    print(f"\nFitted KMeans(k={args.k}): inertia={model.inertia_:.4f}, n_iter={model.n_iter_}")

    if args.k >= 2:
        from .metrics import silhouette_score

        score = silhouette_score(points, model.labels_)
        print(f"Mean silhouette score at k={args.k}: {score:.4f}")

    if args.show_elbow:
        k_range = list(range(1, min(args.kmax_scan, len(points)) + 1))
        elbow_results = elbow_method(points, k_range, n_init=args.n_init, random_state=args.seed)
        suggested = find_elbow_k(elbow_results)
        print(f"\nElbow scan k=1..{k_range[-1]}:")
        for k, inertia in elbow_results:
            print(f"  k={k:>2}  inertia={inertia:>12.4f}")
        print(f"Suggested k (elbow heuristic): {suggested}")

    if args.ascii:
        print()
        print(render_ascii(points, model.labels_, centroids=model.centroids_))

    if args.svg_out:
        export_svg(
            points,
            model.labels_,
            centroids=model.centroids_,
            path=args.svg_out,
            title=f"kmeans demo (k={args.k}, n={len(points)})",
        )
        print(f"\nSaved cluster plot to {args.svg_out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kmeans", description="Dependency-free k-means clustering toolkit."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="Fit KMeans on a CSV of points.")
    p_fit.add_argument("--input", required=True, help="CSV file with x,y columns.")
    p_fit.add_argument("--k", type=int, required=True, help="Number of clusters.")
    p_fit.add_argument("--n-init", type=int, default=10, help="Independent runs to try (default: 10).")
    p_fit.add_argument("--max-iter", type=int, default=300, help="Max Lloyd iterations per run.")
    p_fit.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p_fit.add_argument("--model-out", default=None, help="Where to save the fitted model (JSON).")
    p_fit.add_argument("--labels-out", default=None, help="Where to save a labeled points CSV.")
    p_fit.add_argument("--svg-out", default=None, help="Where to save an SVG cluster plot.")
    p_fit.add_argument("--ascii", action="store_true", help="Print an ASCII scatter plot too.")
    p_fit.set_defaults(func=_cmd_fit)

    p_predict = sub.add_parser("predict", help="Predict cluster labels for new points.")
    p_predict.add_argument("--model", required=True, help="Path to a model JSON saved by `fit`.")
    p_predict.add_argument("--input", required=True, help="CSV file with x,y columns.")
    p_predict.add_argument("--labels-out", default=None, help="Where to save a labeled points CSV.")
    p_predict.set_defaults(func=_cmd_predict)

    p_elbow = sub.add_parser("elbow", help="Sweep k and report inertia (elbow heuristic).")
    p_elbow.add_argument("--input", required=True, help="CSV file with x,y columns.")
    p_elbow.add_argument("--kmin", type=int, default=1, help="Smallest k to try (default: 1).")
    p_elbow.add_argument("--kmax", type=int, default=10, help="Largest k to try (default: 10).")
    p_elbow.add_argument("--n-init", type=int, default=10, help="Independent runs per k.")
    p_elbow.add_argument("--seed", type=int, default=None, help="Random seed.")
    p_elbow.set_defaults(func=_cmd_elbow)

    p_sil = sub.add_parser("silhouette", help="Sweep k and report mean silhouette score.")
    p_sil.add_argument("--input", required=True, help="CSV file with x,y columns.")
    p_sil.add_argument("--kmin", type=int, default=2, help="Smallest k to try (default: 2, min 2).")
    p_sil.add_argument("--kmax", type=int, default=10, help="Largest k to try (default: 10).")
    p_sil.add_argument("--n-init", type=int, default=10, help="Independent runs per k.")
    p_sil.add_argument("--seed", type=int, default=None, help="Random seed.")
    p_sil.set_defaults(func=_cmd_silhouette)

    p_demo = sub.add_parser("demo", help="Generate synthetic blobs and cluster them end to end.")
    p_demo.add_argument("--samples", type=int, default=300, help="Number of synthetic points.")
    p_demo.add_argument("--centers", type=int, default=3, help="Number of true blob centers.")
    p_demo.add_argument("--std", type=float, default=1.0, help="Blob standard deviation.")
    p_demo.add_argument("--k", type=int, default=3, help="k to fit KMeans with.")
    p_demo.add_argument("--n-init", type=int, default=10, help="Independent KMeans runs.")
    p_demo.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p_demo.add_argument("--show-elbow", action="store_true", help="Also print an elbow-method scan.")
    p_demo.add_argument("--kmax-scan", type=int, default=8, help="Max k for the elbow scan.")
    p_demo.add_argument("--ascii", action="store_true", help="Print an ASCII scatter plot.")
    p_demo.add_argument("--svg-out", default=None, help="Where to save an SVG cluster plot.")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
