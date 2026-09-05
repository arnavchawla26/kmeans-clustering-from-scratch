"""kmeans_scratch: dependency-free k-means clustering from first principles.

Public API:
    KMeans              -- k-means++ initialized Lloyd's-algorithm clusterer
    make_blobs          -- synthetic 2D Gaussian-blob dataset generator
    elbow_method         -- inertia-vs-k sweep for the elbow heuristic
    silhouette_score      -- mean silhouette coefficient for a labeling
    silhouette_samples     -- per-point silhouette coefficients
    best_k_by_silhouette   -- pick k that maximizes mean silhouette
    render_ascii          -- quick-look ASCII scatter of a clustering
    export_svg            -- SVG cluster-plot export
"""

from .core import KMeans, squared_distance
from .datasets import make_blobs
from .metrics import best_k_by_silhouette, elbow_method, silhouette_samples, silhouette_score
from .plot import export_svg, render_ascii

__version__ = "0.1.0"

__all__ = [
    "KMeans",
    "squared_distance",
    "make_blobs",
    "elbow_method",
    "silhouette_score",
    "silhouette_samples",
    "best_k_by_silhouette",
    "render_ascii",
    "export_svg",
]
