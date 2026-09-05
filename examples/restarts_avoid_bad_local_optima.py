"""Show why KMeans defaults to multiple restarts (n_init > 1).

Lloyd's algorithm only ever finds a *local* optimum of the k-means
objective; even with k-means++ seeding, an unlucky draw can converge to a
noticeably worse clustering than another draw would. This script fits the
same dataset many times with n_init=1 (a single k-means++ seeded run) and
shows the spread of resulting inertia values, then contrasts it with the
best-of-many result n_init=10 (the package default) actually returns.

Run with:
    python examples/restarts_avoid_bad_local_optima.py
"""

from kmeans_scratch import KMeans, make_blobs


def main() -> None:
    # Blobs arranged so that a poorly-placed initial centroid can plausibly
    # split one blob and merge two others -- a classic local-optimum trap.
    points, _ = make_blobs(
        n_samples=240,
        centers=[(0, 0), (3, 0), (3, 3), (30, 30)],
        cluster_std=1.2,
        random_state=7,
    )

    single_run_inertias = []
    for seed in range(20):
        model = KMeans(n_clusters=4, n_init=1, random_state=seed)
        model.fit(points)
        single_run_inertias.append(model.inertia_)

    best_single = min(single_run_inertias)
    worst_single = max(single_run_inertias)
    mean_single = sum(single_run_inertias) / len(single_run_inertias)

    print("20 independent single-run (n_init=1) fits, same data, different seeds:")
    print(f"  best inertia:  {best_single:.2f}")
    print(f"  worst inertia: {worst_single:.2f}")
    print(f"  mean inertia:  {mean_single:.2f}")
    print(f"  spread (worst - best): {worst_single - best_single:.2f}\n")

    restarted = KMeans(n_clusters=4, n_init=20, random_state=0)
    restarted.fit(points)
    print(f"One KMeans(n_init=20) call (keeps the best of 20 internal restarts):")
    print(f"  inertia: {restarted.inertia_:.2f}")

    if worst_single - best_single > 1e-6:
        print(
            "\nThe spread above is exactly why n_init defaults to 10: a single "
            "k-means++ run can still land in a worse local optimum, and running "
            "several and keeping the best is cheap insurance against it."
        )
    else:
        print(
            "\n(On this particular dataset/seed range every run happened to "
            "find the same optimum -- try a smaller cluster_std or more "
            "overlapping centers to see more spread.)"
        )


if __name__ == "__main__":
    main()
