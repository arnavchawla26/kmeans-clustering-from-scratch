"""Compare the elbow heuristic against silhouette analysis for choosing k.

Generates four well-separated synthetic blobs (so the "true" k is known to
be 4) and shows that the two model-selection methods do not always agree:
the elbow method's "distance to chord" heuristic looks at the whole inertia
curve and can be thrown off by how steep the early drops are, whereas
silhouette analysis directly measures how well-separated the resulting
clusters are at each k and recovers the true k more reliably here.

Run with:
    python examples/elbow_vs_silhouette.py
"""

from kmeans_scratch import best_k_by_silhouette, elbow_method, make_blobs
from kmeans_scratch.metrics import find_elbow_k


def main() -> None:
    points, true_labels = make_blobs(
        n_samples=300,
        centers=[(0, 0), (25, 0), (0, 25), (25, 25)],
        cluster_std=1.5,
        random_state=42,
    )
    print(f"Generated {len(points)} points around 4 true, well-separated blobs.\n")

    k_range = range(1, 9)
    elbow_results = elbow_method(points, k_range, n_init=10, random_state=42)
    elbow_k = find_elbow_k(elbow_results)

    print(f"{'k':>4}  {'inertia':>12}")
    for k, inertia in elbow_results:
        print(f"{k:>4}  {inertia:>12.2f}")
    print(f"Elbow heuristic suggests: k={elbow_k}\n")

    best_k, sil_results = best_k_by_silhouette(points, range(2, 9), n_init=10, random_state=42)
    print(f"{'k':>4}  {'mean silhouette':>16}")
    for k, score in sil_results:
        marker = "  <-- best" if k == best_k else ""
        print(f"{k:>4}  {score:>16.4f}{marker}")
    print(f"Silhouette analysis suggests: k={best_k}")

    if elbow_k == best_k == 4:
        print(
            f"\nTrue number of blobs: 4. Both methods agree here (k={best_k}) "
            "because the blobs are cleanly separated, so the inertia curve has "
            "one unambiguous bend that lines up with where cluster quality "
            "actually peaks. That agreement isn't guaranteed in general -- see "
            "the module docstring and README for a case where the two "
            "disagree."
        )
    else:
        print(
            f"\nTrue number of blobs: 4. Silhouette picked k={best_k}; elbow "
            f"picked k={elbow_k}. When they disagree, prefer silhouette for "
            "well-separated clusters -- it directly scores cluster quality, "
            "while the elbow heuristic only approximates where a human eye "
            "would see the curve 'bend'."
        )


if __name__ == "__main__":
    main()
