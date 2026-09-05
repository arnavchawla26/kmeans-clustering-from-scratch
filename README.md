# kmeans-scratch

A dependency-free implementation of k-means clustering, built from first
principles: k-means++ seeding, Lloyd's algorithm, the elbow method and
silhouette analysis for choosing k, synthetic blob-dataset generation, and
ASCII/SVG cluster-plot export. Pure Python standard library only -- no
numpy, no scikit-learn, no plotting library.

This is an unsupervised-ML showcase: given only raw points (no labels), it
partitions them into k groups and gives you two independent, principled ways
to pick k when you don't already know it.

## What's inside

- **`kmeans_scratch.core`** -- `KMeans`, seeded with k-means++ (probability
  -weighted seeding that spreads initial centroids out, rather than picking
  k random points) and fit with Lloyd's algorithm (alternating
  assign-to-nearest-centroid / recompute-centroid-as-mean steps until
  convergence). Runs `n_init` independent restarts by default and keeps the
  best (lowest-inertia) one, since Lloyd's algorithm only finds a local
  optimum.
- **`kmeans_scratch.datasets`** -- `make_blobs`, a small re-implementation of
  the idea behind scikit-learn's `make_blobs`: isotropic Gaussian clusters
  around either random or explicit centers, for demos and tests.
- **`kmeans_scratch.metrics`** -- `elbow_method` + `find_elbow_k` (inertia
  -vs-k sweep with a "maximum distance to the chord" heuristic for picking
  the bend), and `silhouette_score` / `silhouette_samples` /
  `best_k_by_silhouette` (how well-separated the resulting clusters are, per
  point and averaged, for a range of k).
- **`kmeans_scratch.plot`** -- `render_ascii` (a quick terminal scatter plot,
  no file needed) and `export_svg` (a self-contained SVG file, points
  colored by cluster with centroid markers).
- **`kmeans` CLI** -- `fit`, `predict`, `elbow`, `silhouette`, and `demo`
  subcommands tying it all together from the command line.

## Tech stack

Python 3.9+, standard library only (`random`, `math`, `csv`, `json`,
`argparse`, `dataclasses`). `pytest` for tests (dev-only dependency).

## How to run

```bash
pip install -e ".[dev]"
pytest                 # 53 tests

# End-to-end demo: generate synthetic blobs, fit KMeans, print an elbow
# scan, and save an SVG plot -- no input files needed.
kmeans demo --samples 300 --centers 4 --k 4 --seed 42 --show-elbow --ascii --svg-out demo.svg

# Or bring your own data (a CSV with x,y columns, header optional):
kmeans fit --input points.csv --k 3 --seed 0 \
    --model-out model.json --labels-out labels.csv --svg-out plot.svg

# Not sure what k to use? Sweep it two ways:
kmeans elbow --input points.csv --kmin 1 --kmax 10
kmeans silhouette --input points.csv --kmin 2 --kmax 10

# Reuse a fitted model on new points:
kmeans predict --model model.json --input new_points.csv
```

## Example output

`kmeans demo --samples 300 --centers 4 --k 4 --seed 42 --show-elbow --ascii`
against four random blob centers:

```
Generated 300 synthetic points around 4 true blob(s) (cluster_std=1.0, seed=42).

Fitted KMeans(k=4): inertia=606.2608, n_iter=3
Mean silhouette score at k=4: 0.7327

Elbow scan k=1..8:
  k= 1  inertia=  14378.6079
  k= 2  inertia=   6996.2440
  k= 3  inertia=   1535.9859
  k= 4  inertia=    606.2608
  k= 5  inertia=    540.4549
  k= 6  inertia=    480.6347
  k= 7  inertia=    429.1976
  k= 8  inertia=    381.1195
Suggested k (elbow heuristic): 3
```

Note the last line: the elbow heuristic here actually suggests k=3, not the
true k=4, even though k-means itself clusters correctly at k=4 (silhouette
score 0.73). The "distance to chord" heuristic looks at the *whole* inertia
curve, and when two of the four random blob centers happened to land close
together, the k=1->2->3 drop was steep enough to visually dominate the
k=3->4 drop. This is a real, known limitation of the elbow method, not a bug
in this implementation -- it's exactly why `elbow` and `silhouette` are
offered as two independent CLI subcommands rather than one "just tell me
k" answer. See `examples/elbow_vs_silhouette.py` for a case (fixed,
well-separated centers) where the two methods agree, and the discussion
above for why they don't always.

`examples/restarts_avoid_bad_local_optima.py` demonstrates the other design
decision worth knowing about -- why `KMeans` defaults to `n_init=10`
(re-running the whole fit that many times and keeping the best result):

```
20 independent single-run (n_init=1) fits, same data, different seeds:
  best inertia:  621.64
  worst inertia: 847.23
  mean inertia:  678.45
  spread (worst - best): 225.59

One KMeans(n_init=20) call (keeps the best of 20 internal restarts):
  inertia: 621.64
```

Even with k-means++ seeding, Lloyd's algorithm only finds a *local* optimum
-- a single run can land nearly 40% worse than the best of a handful of
restarts on the same data.

## Design notes

- **k-means++ initialization** (`core.kmeans_plus_plus_init`): the first
  centroid is picked uniformly at random; each subsequent centroid is picked
  from the remaining points with probability proportional to its squared
  distance to the nearest already-chosen centroid. This is tracked
  incrementally (an O(n) update per centroid, not O(nk) recomputed from
  scratch) and spreads centroids out, which converges faster and more
  reliably than naive random initialization.
- **Lloyd's algorithm** (`core.KMeans._single_run`): standard
  assign/update iteration to convergence (centroid movement below `tol`) or
  `max_iter`. A centroid that loses all its points mid-fit is re-seeded at
  the point currently farthest from its own assigned centroid, which avoids
  silently collapsing to fewer than k clusters.
- **Elbow heuristic** (`metrics.find_elbow_k`): normalizes both axes to
  `[0, 1]` and finds the `(k, inertia)` point with the largest perpendicular
  distance from the straight line connecting the first and last points on
  the curve -- a numeric stand-in for "eyeballing the bend."
- **Silhouette analysis** (`metrics.silhouette_samples`): the textbook
  `(b - a) / max(a, b)` per-point formula, computed via a full O(n^2)
  pairwise-distance matrix. That's the right tradeoff for the small,
  demo-sized datasets this package targets -- not for clustering millions of
  points.
- **No numpy anywhere.** Points are plain tuples of floats; distance,
  mean, and pairwise computations are all written directly against Python
  lists. This keeps the whole algorithm inspectable in ~200 lines
  (`core.py`) without vectorized code obscuring the mechanics.

## Current status

v1, functional and tested. All five modules (`core`, `datasets`, `metrics`,
`plot`, `cli`) are implemented with 53 passing tests (unit tests for the
algorithm, the synthetic data generator, both model-selection metrics, both
plot renderers, and end-to-end CLI subprocess tests), plus two runnable
example scripts with real captured output (above) illustrating the elbow
-vs-silhouette tradeoff and why multiple restarts matter. No known bugs.

Possible future extensions (not started): k-medoids / PAM as an
outlier-robust alternative; mini-batch k-means for larger datasets; support
for higher-dimensional input in the CLI (the library core is already
dimension-agnostic -- only the ASCII/SVG plotting and the CLI's 2-column CSV
format assume 2D).

## License

MIT -- see `LICENSE`.
