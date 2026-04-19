from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


Backend = Literal["plotly", "matplotlib"]


def plot_layer_profiles_plotly(
    layer_indices: list[int],
    series: dict[str, list[float]],
    title: str,
    out_path: Path,
    export_png: bool = False,
) -> Path:
    fig = go.Figure()
    for name, ys in series.items():
        fig.add_trace(
            go.Scatter(
                x=layer_indices,
                y=ys,
                mode="lines+markers",
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Layer index",
        yaxis_title="Activation metric",
        hovermode="x unified",
        template="plotly_white",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    if export_png:
        try:
            fig.write_image(out_path.with_suffix(".png"), scale=2)
        except Exception:
            pass
    return out_path


def plot_persona_layer_heatmap_plotly(
    persona_labels: list[str],
    layer_indices: list[int],
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    export_png: bool = False,
) -> Path:
    """matrix shape (n_personas, n_layers)."""
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[str(i) for i in layer_indices],
            y=persona_labels,
            colorscale="Viridis",
            colorbar=dict(title="Metric"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Persona",
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    if export_png:
        try:
            fig.write_image(out_path.with_suffix(".png"), scale=2)
        except Exception:
            pass
    return out_path


def matplotlib_layer_grid(
    layer_indices: list[int],
    series: dict[str, list[float]],
    title: str,
    out_path: Path,
) -> Path:
    """Plotly/kaleido 없이 정적 PNG가 필요할 때."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, ys in series.items():
        ax.plot(layer_indices, ys, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_subplot_per_persona(
    layer_indices: list[int],
    series: dict[str, list[float]],
    title: str,
    out_path: Path,
) -> Path:
    n = len(series)
    fig = make_subplots(rows=n, cols=1, shared_x=True, subplot_titles=list(series.keys()))
    for i, (name, ys) in enumerate(series.items(), start=1):
        fig.add_trace(
            go.Scatter(x=layer_indices, y=ys, mode="lines+markers", name=name),
            row=i,
            col=1,
        )
    fig.update_layout(title_text=title, template="plotly_white", height=260 * n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path
