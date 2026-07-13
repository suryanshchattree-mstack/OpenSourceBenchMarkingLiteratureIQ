"""UpSet plot rendering for N-way compound membership sets."""

from __future__ import annotations

from collections import Counter
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


def _force_draw(fig: Figure) -> None:
    """Force a full canvas draw so deferred artist errors surface before Streamlit savefig."""
    fig.canvas.draw()


def _try_upset_plot(series, *, element_size: float, show_counts: bool) -> Figure:
    from upsetplot import plot as upset_plot

    fig = Figure()
    with plt.rc_context(
        {
            "font.size": 8 if show_counts else 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
        }
    ):
        axes_dict = upset_plot(
            series,
            subset_size="count",
            show_counts=show_counts,
            element_size=element_size,
            fig=fig,
        )
    fig = _figure_from_upset_axes(axes_dict)
    _force_draw(fig)
    return fig


def render_upset(memberships: list[frozenset[str]]) -> Figure:
    """
    Build an UpSet figure from per-cluster label memberships.

    ``memberships`` is the list produced by
    :func:`core.compound_report.build_upset_memberships` (one frozenset per
    cluster). Empty input yields an empty placeholder figure.

    On Streamlit Cloud (Python 3.14 / newer matplotlib), upsetplot can fail either
    during ``plot()`` or later during ``savefig``/draw. We force a canvas draw
    here and fall back to a simple intersection bar chart on any failure.
    """
    if not memberships:
        fig = Figure(figsize=(7, 2.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title("No compound clusters to plot")
        fig.tight_layout()
        return fig

    # Lazy import so unit tests that never plot can skip the dependency path.
    from upsetplot import from_memberships

    # Upsetplot 0.9 uses inplace fillna that breaks under pandas CoW / 3.x.
    try:
        import pandas as pd

        if hasattr(pd.options.mode, "copy_on_write"):
            pd.options.mode.copy_on_write = False
    except Exception:
        pass

    categories = membership_category_labels(memberships)
    intersections = {frozenset(membership) for membership in memberships if membership}
    n_categories = len(categories)
    n_intersections = max(len(intersections), 1)
    fig_width, fig_height = _figure_size(n_categories, n_intersections)
    element_size = _element_size(n_categories, n_intersections)

    series = from_memberships(
        [list(membership) for membership in memberships],
        data=None,
    )

    last_error: Exception | None = None
    for show_counts in (True, False):
        try:
            fig = _try_upset_plot(
                series,
                element_size=element_size,
                show_counts=show_counts,
            )
            fig.set_size_inches(fig_width, fig_height, forward=True)
            fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.08)
            return fig
        except Exception as exc:  # noqa: BLE001 — host matplotlib/pandas quirks
            last_error = exc
            plt.close("all")

    return _fallback_intersection_figure(
        memberships,
        categories,
        reason=str(last_error) if last_error else "unknown UpSet failure",
        fig_width=fig_width,
        fig_height=min(fig_height, 6.0),
    )


def _fallback_intersection_figure(
    memberships: list[frozenset[str]],
    categories: list[str],
    *,
    reason: str,
    fig_width: float,
    fig_height: float,
) -> Figure:
    """Simple bar chart of intersection sizes when UpSetPlot cannot render."""
    counts = Counter(frozenset(membership) for membership in memberships if membership)
    ordered = sorted(counts, key=lambda key: (-counts[key], sorted(key)))
    labels = [" ∩ ".join(sorted(key)) if key else "(empty)" for key in ordered]
    values = [counts[key] for key in ordered]
    fig = Figure(
        figsize=(fig_width, max(3.5, min(fig_height, 0.35 * max(len(labels), 1) + 2.5)))
    )
    ax = fig.add_subplot(111)
    ax.barh(range(len(values)), values, color="#4C78A8")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cluster count")
    ax.set_title("Intersection sizes (UpSet fallback)")
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
    _force_draw(fig)
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
