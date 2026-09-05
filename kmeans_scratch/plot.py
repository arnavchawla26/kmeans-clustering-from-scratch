"""Dependency-free cluster visualization: an ASCII scatter and an SVG export.

No plotting library required -- just string formatting and basic
coordinate-to-pixel math.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .core import Point

# A palette of visually distinct colors, reused for both the ASCII glyphs
# and the SVG fills. Cycles if there are more clusters than colors.
_ASCII_GLYPHS = "o+x*#@%&aAbBcCdDeEfFgGhH"
_SVG_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
    "#CCB974",  # olive
    "#64B5CD",  # cyan
]


def _bounds(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def render_ascii(
    points: Sequence[Point],
    labels: Sequence[int],
    centroids: Optional[Sequence[Point]] = None,
    width: int = 60,
    height: int = 24,
) -> str:
    """Render a quick-look ASCII scatter plot of a 2D clustering.

    Each cluster gets a distinct glyph from ``_ASCII_GLYPHS``; centroids
    (if given) are drawn as ``@`` markers overlaid on top. Useful for a
    terminal-only glance at a result without writing any file.
    """
    if not points:
        return "(no points to plot)"

    grid = [[" " for _ in range(width)] for _ in range(height)]

    min_x, max_x, min_y, max_y = _bounds(points)
    if centroids:
        c_min_x, c_max_x, c_min_y, c_max_y = _bounds(centroids)
        min_x, max_x = min(min_x, c_min_x), max(max_x, c_max_x)
        min_y, max_y = min(min_y, c_min_y), max(max_y, c_max_y)

    x_span = (max_x - min_x) or 1.0
    y_span = (max_y - min_y) or 1.0

    def to_cell(p: Point) -> Tuple[int, int]:
        col = int((p[0] - min_x) / x_span * (width - 1))
        # Flip vertically: screen row 0 is the top, but we want larger y up.
        row = height - 1 - int((p[1] - min_y) / y_span * (height - 1))
        return row, col

    for p, lbl in zip(points, labels):
        row, col = to_cell(p)
        glyph = _ASCII_GLYPHS[lbl % len(_ASCII_GLYPHS)]
        grid[row][col] = glyph

    if centroids:
        for c in centroids:
            row, col = to_cell(c)
            grid[row][col] = "@"

    lines = ["".join(row) for row in grid]
    border = "+" + "-" * width + "+"
    body = "\n".join(f"|{line}|" for line in lines)
    legend_bits = [f"{_ASCII_GLYPHS[i % len(_ASCII_GLYPHS)]}=cluster {i}" for i in sorted(set(labels))]
    legend = "  ".join(legend_bits) + ("  @=centroid" if centroids else "")
    return f"{border}\n{body}\n{border}\n{legend}"


def export_svg(
    points: Sequence[Point],
    labels: Sequence[int],
    centroids: Optional[Sequence[Point]] = None,
    path: str = "clusters.svg",
    width: int = 640,
    height: int = 480,
    padding: int = 30,
    point_radius: float = 4.0,
    centroid_radius: float = 8.0,
    title: Optional[str] = None,
) -> str:
    """Render a 2D clustering to a self-contained SVG file and return its path.

    Points are colored by cluster label; centroids (if given) are drawn as
    black-outlined "X" markers on top. Pure string templating -- no
    plotting library involved.
    """
    if not points:
        raise ValueError("cannot export a plot with no points")
    if len(points) != len(labels):
        raise ValueError("points and labels must be the same length")

    min_x, max_x, min_y, max_y = _bounds(points)
    if centroids:
        c_min_x, c_max_x, c_min_y, c_max_y = _bounds(centroids)
        min_x, max_x = min(min_x, c_min_x), max(max_x, c_max_x)
        min_y, max_y = min(min_y, c_min_y), max(max_y, c_max_y)

    x_span = (max_x - min_x) or 1.0
    y_span = (max_y - min_y) or 1.0
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    def to_px(p: Point) -> Tuple[float, float]:
        x = padding + (p[0] - min_x) / x_span * plot_w
        # Flip vertically: SVG y grows downward, plots want y growing up.
        y = padding + plot_h - (p[1] - min_y) / y_span * plot_h
        return x, y

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    if title:
        parts.append(
            f'<text x="{width / 2}" y="{padding / 1.6}" text-anchor="middle" '
            f'font-size="16" fill="#222222">{_escape(title)}</text>'
        )

    # Plot-area border.
    parts.append(
        f'<rect x="{padding}" y="{padding}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#cccccc" stroke-width="1"/>'
    )

    for p, lbl in zip(points, labels):
        x, y = to_px(p)
        color = _SVG_COLORS[lbl % len(_SVG_COLORS)]
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{point_radius}" '
            f'fill="{color}" fill-opacity="0.75"/>'
        )

    if centroids:
        for c in centroids:
            x, y = to_px(c)
            r = centroid_radius
            parts.append(
                f'<g stroke="#000000" stroke-width="2">'
                f'<line x1="{x - r:.2f}" y1="{y - r:.2f}" x2="{x + r:.2f}" y2="{y + r:.2f}"/>'
                f'<line x1="{x - r:.2f}" y1="{y + r:.2f}" x2="{x + r:.2f}" y2="{y - r:.2f}"/>'
                f"</g>"
            )

    parts.append("</svg>")

    svg_content = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg_content)
    return path


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
