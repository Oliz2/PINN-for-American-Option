"""Plotly charts for the pricing results (same style as the PINN interface).

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#7c4dbd"
BACKGROUND, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
METHOD_COLOR = {"lsm": C1, "brutal": C2, "binomial": C3, "PINN": C4}


def _style(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15, color=INK)),
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, height=height,
        font=dict(family="Segoe UI, system-ui, sans-serif", color=INK2, size=12),
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    return fig


def _x_labels(df: pd.DataFrame, d: int):
    s_cols = [f"S{i + 1}" for i in range(d)]
    if d == 1:
        return df["S1"]
    return df[s_cols].apply(lambda r: "(" + "; ".join(f"{x:g}" for x in r) + ")", axis=1)


def prices_by_spot(summary: pd.DataFrame, d: int, series: dict[str, str]) -> go.Figure:
    """Price for each S0: one line per series (column name -> colour key), plus European put and payoff."""
    x = _x_labels(summary, d)
    fig = go.Figure()
    for col, key in series.items():
        if col in summary.columns:
            fig.add_scatter(x=x, y=summary[col], mode="lines+markers", name=col,
                            line=dict(color=METHOD_COLOR.get(key, MUTED), width=2))
    if "European put" in summary.columns:
        fig.add_scatter(x=x, y=summary["European put"], mode="lines", name="European put",
                        line=dict(color=MUTED, dash="dot"))
    fig.add_scatter(x=x, y=summary["Payoff"], mode="lines", name="Payoff", line=dict(color=INK2, dash="dash", width=1))
    fig.update_xaxes(title="S0" if d == 1 else "S0 = (S1, ..., Sd)", type="linear" if d == 1 else "category")
    fig.update_yaxes(title="Price")
    return _style(fig, "American put price for each S0")


def differences(comp: pd.DataFrame, d: int, method_names: dict[str, str]) -> go.Figure:
    """PINN minus each reference method, for each S0."""
    x = _x_labels(comp, d)
    fig = go.Figure()
    for key, name in method_names.items():
        col = f"PINN - {name}"
        if col in comp.columns:
            fig.add_scatter(x=x, y=comp[col], mode="lines+markers", name=col,
                            line=dict(color=METHOD_COLOR.get(key, MUTED), width=2))
    fig.add_hline(y=0, line=dict(color=INK2, width=1))
    fig.update_xaxes(title="S0" if d == 1 else "S0 = (S1, ..., Sd)", type="linear" if d == 1 else "category")
    fig.update_yaxes(title="Price difference")
    return _style(fig, "PINN price minus reference price", 380)


def convergence(conv: pd.DataFrame, spot: int, names: dict, reference: float | None) -> go.Figure:
    """Running mean of the discounted cash flows as the number of paths grows (law of large numbers).

    Each point is the estimate one would get stopping at that number of paths; the binomial price, when
    available, is drawn as reference line.
    """
    fig = go.Figure()
    for method, g in conv[conv["spot"] == spot].groupby("method", sort=False):
        fig.add_scatter(x=g["paths"], y=g["price"], mode="lines", name=names.get(method, method),
                        line=dict(color=METHOD_COLOR.get(method, MUTED), width=2))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=C3, dash="dash", width=1.5), annotation_text="Binomial tree")
    fig.update_xaxes(title="Number of paths", type="log")
    fig.update_yaxes(title="Price estimate")
    return _style(fig, "Monte Carlo estimate as the number of paths grows")


def paths(paths_df: pd.DataFrame, exercises: pd.DataFrame | None, method: str | None, strike: float,
          boundary: pd.DataFrame | None, names: dict) -> go.Figure:
    """Paths of min_i S_i with the points where the chosen method exercises."""
    fig = go.Figure()
    stop = {}
    if exercises is not None and method is not None:
        e = exercises[exercises["method"] == method]
        stop = dict(zip(e["path"], e["exercise_time"]))
    xs, ys = [], []
    for k, g in paths_df.groupby("path"):
        te = stop.get(k, np.nan)
        if not np.isnan(te):
            before = g[g["t"] <= te + 1e-9]
            after = g[g["t"] >= te - 1e-9]
            fig.add_scatter(x=before["t"], y=before["S_min"], mode="lines", line=dict(color=C1, width=1),
                            opacity=0.8, showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=after["t"], y=after["S_min"], mode="lines", line=dict(color=MUTED, width=0.8),
                            opacity=0.35, showlegend=False, hoverinfo="skip")
            row = g.iloc[(g["t"] - te).abs().argmin()]
            xs.append(row["t"]); ys.append(row["S_min"])
        else:
            fig.add_scatter(x=g["t"], y=g["S_min"], mode="lines", line=dict(color=C1, width=1), opacity=0.8,
                            showlegend=False, hoverinfo="skip")
    if xs:
        fig.add_scatter(x=xs, y=ys, mode="markers", name=f"Exercise ({names.get(method, method)})",
                        marker=dict(color=C2, size=8, line=dict(color="white", width=1)))
    if boundary is not None:
        for m, g in boundary.groupby("method", sort=False):
            fig.add_scatter(x=g["t"], y=g["B_t"], mode="lines", name=f"Boundary, {names.get(m, m)}",
                            line=dict(color=C3 if m == "binomial" else C4, width=2))
    fig.add_hline(y=strike, line=dict(color=INK2, dash="dash", width=1), annotation_text="K")
    fig.update_xaxes(title="Time t")
    fig.update_yaxes(title="min_i S_i")
    return _style(fig, "Simulated paths (grey: after exercise)", 480)


def boundary(fr: pd.DataFrame, strike: float, names: dict) -> go.Figure:
    """d = 1: exercise boundary of Longstaff-Schwartz and of the binomial tree."""
    fig = go.Figure()
    for m, g in fr.groupby("method", sort=False):
        fig.add_scatter(x=g["t"], y=g["B_t"], mode="lines", name=f"B(t), {names.get(m, m)}",
                        line=dict(color=METHOD_COLOR.get(m, MUTED), width=2))
    fig.add_hline(y=strike, line=dict(color=INK2, dash="dash", width=1), annotation_text="K")
    fig.update_xaxes(title="Time t")
    fig.update_yaxes(title="S")
    return _style(fig, "Exercise boundary (exercise below the curve)")


def price_table(pt: pd.DataFrame, strike: float, names: dict) -> go.Figure:
    """Price as a function of S for each time of the Price Table (one curve per time, dash per method)."""
    fig = go.Figure()
    times = sorted(pt["t"].unique())
    palette = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#b6d2f5"]
    dashes = {"lsm": "solid", "binomial": "dash", "brutal": "dot"}
    for method, gm in pt.groupby("method", sort=False):
        for i, t in enumerate(times):
            g = gm[np.isclose(gm["t"], t)]
            fig.add_scatter(x=g["S"], y=g["price"], mode="lines+markers", name=f"{names.get(method, method)}, t = {t:.2f}",
                            line=dict(color=palette[i % len(palette)], dash=dashes.get(method, "solid")))
    s = np.sort(pt["S"].unique())
    fig.add_scatter(x=s, y=np.maximum(strike - s, 0), mode="lines", name="Payoff", line=dict(color=INK2, dash="dash"))
    fig.update_xaxes(title="S")
    fig.update_yaxes(title="Price")
    return _style(fig, "Price Table: price as a function of S for each time", 480)
