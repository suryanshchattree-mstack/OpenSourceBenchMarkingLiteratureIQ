"""UpSet plot rendering for N-way compound membership sets."""

from __future__ import annotations

from typing import Collection

import matplotlib

# Prefer Agg for Streamlit workers / CI / pytest when no interactive backend is set.
_backend = str(matplotlib.get_backend()).lower()
if _backend in {"macosx", "tkagg", "qt5agg", "qtagg"} or "interactive" in _backend:
    pass
else:
    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _figure_size(n_categories: int, n_intersections: int) -> tuple[float, float]:
    """Pick a readable figure size that still fits typical Streamlit widths."""
    width = min(16.0, max(7.0, 1.1 * max(n_intersections, 1) + 3.5))
    height = min(10.0, max(4.0, 0.55 * max(n_categories, 1) + 3.0))
    return width, height


def _element_size(n_categories: int, n_intersections: int) -> float:
    if n_intersections >= 24 or n_categories >= 6:
        return 28.0
    if n_intersections >= 12 or n_categories >= 4:
        return 32.0
    return 36.0


def render_upset(memberships: list[frozenset[str]]) -> Figure:
    """
    Build an UpSet figure from per-cluster label memberships.

    ``memberships`` is the list produced by
    :func:`core.compound_report.build_upset_memberships` (one frozenset per
    cluster). Empty input yields an empty placeholder figure.
    """
    if not memberships:
        fig = Figure(figsize=(7, 2.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title("No compound clusters to plot")
        fig.tight_layout()
        return fig

    # Lazy import so unit tests that never plot can skip the dependency path.
    from upsetplot import from_memberships, plot as upset_plot

    categories = membership_category_labels(memberships)
    # Unique non-empty intersections drive horizontal crowding.
    intersections = {frozenset(membership) for membership in memberships if membership}
    n_categories = len(categories)
    n_intersections = max(len(intersections), 1)
    fig_width, fig_height = _figure_size(n_categories, n_intersections)
    element_size = _element_size(n_categories, n_intersections)

    series = from_memberships(
        [list(membership) for membership in memberships],
        data=None,
    )
    fig = Figure(figsize=(fig_width, fig_height))
    with plt.rc_context(
        {
            "font.size": 9 if n_intersections < 16 else 8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    ):
        try:
            axes_dict = upset_plot(
                series,
                subset_size="count",
                show_counts=True,
                element_size=element_size,
                fig=fig,
            )
        except (ValueError, TypeError) as exc:
            # Known break: pandas 3 Copy-on-Write leaves NaN edgecolors in
            # upsetplot 0.9 plot_matrix → matplotlib "Invalid RGBA argument".
            plt.close(fig)
            return _fallback_intersection_figure(
                memberships,
                categories,
                reason=str(exc),
                fig_width=fig_width,
                fig_height=min(fig_height, 6.0),
            )
    fig = _figure_from_upset_axes(axes_dict)
    fig.set_size_inches(fig_width, fig_height, forward=True)
    # upsetplot's multi-axes layout is often incompatible with tight_layout.
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.08)
    return fig


def _fallback_intersection_figure(
    memberships: list[frozenset[str]],
    categories: list[str],
    *,
    reason: str,
    fig_width: float,
    fig_height: float,
) -> Figure:
    """Simple bar chart of intersection sizes when UpSetPlot cannot render."""
    from collections import Counter

    counts = Counter(
        frozenset(membership) for membership in memberships if membership
    )
    labels = [
        " ∩ ".join(sorted(key)) if key else "(empty)"
        for key in sorted(counts, key=lambda k: (-counts[k], sorted(k)))
    ]
    values = [
        counts[key]
        for key in sorted(counts, key=lambda k: (-counts[k], sorted(k)))
    ]
    fig = Figure(figsize=(fig_width, max(3.5, min(fig_height, 0.35 * len(labels) + 2.5))))
    ax = fig.add_subplot(111)
    ax.barh(range(len(values)), values, color="#4C78A8")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cluster count")
    ax.set_title(
        "Intersection sizes (UpSet fallback — pin pandas<3 on the host)"
    )
    ax.text(
        0.01,
        -0.12,
        f"UpSet render failed ({reason[:120]})",
        transform=ax.transAxes,
        fontsize=7,
        color="#666666",
        clip_on=False,
    )
    _ = categories
    fig.tight_layout()
    return fig


def _figure_from_upset_axes(axes_dict: object) -> Figure:
    """Resolve the matplotlib Figure from upsetplot's axes return value."""
    if isinstance(axes_dict, dict):
        for value in axes_dict.values():
            if hasattr(value, "figure"):
                return value.figure
            if isinstance(value, dict):
                for nested in value.values():
                    if hasattr(nested, "figure"):
                        return nested.figure
    if hasattr(axes_dict, "figure"):
        return axes_dict.figure  # type: ignore[attr-defined]
    # Fallback: current figure after plot()
    return plt.gcf()


def membership_category_labels(memberships: Collection[frozenset[str]]) -> list[str]:
    """Sorted unique labels appearing in any membership (useful for legends)."""
    labels: set[str] = set()
    for membership in memberships:
        labels.update(membership)
    return sorted(labels)


def estimate_upset_size(memberships: list[frozenset[str]]) -> tuple[float, float, float]:
    """Return (width, height, element_size) for tests / diagnostics."""
    categories = membership_category_labels(memberships)
    intersections = {frozenset(membership) for membership in memberships if membership}
    n_categories = max(len(categories), 1)
    n_intersections = max(len(intersections), 1)
    width, height = _figure_size(n_categories, n_intersections)
    return width, height, _element_size(n_categories, n_intersections)
