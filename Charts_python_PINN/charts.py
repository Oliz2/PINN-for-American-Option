"""Plotly charts for the PINN results (interactive; they can be saved as PNG from the chart toolbar).

Every function receives the DataFrames read by results.py and returns a plotly Figure.

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from results import grid

# Palette (same as the thesis figures): three series colours, neutral inks for text and axes
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
BACKGROUND, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = [[0.0, "#cde2fb"], [0.2, "#86b6ef"], [0.4, "#3987e5"], [0.6, "#256abf"], [0.8, "#184f95"], [1.0, "#0d366b"]]
EXERCISE = [[0.0, "#f0efec"], [1.0, "#eb6834"]]


def _style(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    """Common look of all the charts: title, colours, fonts, grid and legend."""
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15, color=INK)),
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND, height=height,
        font=dict(family="Segoe UI, system-ui, sans-serif", color=INK2, size=12),
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    return fig


def live_loss(history: list, total_epochs: int) -> go.Figure:
    """Loss during the run (from the output of pinn.exe): one curve per seed."""
    fig = go.Figure()
    if history:
        df = pd.DataFrame(history, columns=["seed", "epoch", "sol", "fb", "lr"])
        for i, (seed, g) in enumerate(df.groupby("seed", sort=False)):
            colour = [C1, C2, C3][i % 3] if i < 3 else MUTED
            fig.add_scatter(x=g["epoch"], y=g["sol"], mode="lines+markers" if len(g) < 30 else "lines",
                            name=f"FNNSol loss, seed {seed}", line=dict(color=colour, width=2), marker=dict(size=5))
    fig.update_yaxes(type="log", title="Loss (log scale)")
    # the axis follows the epochs done so far (at the start the curve would be invisible on 0..total_epochs)
    last = max((s[1] for s in history), default=0)
    fig.update_xaxes(title="Epoch", range=[0, min(max(last * 1.1, 20), max(total_epochs, 1))])
    return _style(fig, "Loss during training", 360)


def loss_evolution(h: pd.DataFrame, penalty: bool) -> go.Figure:
    """Losses used in training. Without penalty a second panel shows the free-boundary conditions,
    which then train FNNfb; with penalty they are not trained and are not shown."""
    pde = "PDE: min(−ΛP, P−ψ)" if penalty else "PDE"
    solution = [("pde", pde, C1), ("initial", "Initial (maturity)", C2),
                ("dirichlet", "Dirichlet" + (" (S = Smax and S = 0)" if penalty else ""), C3)]
    if penalty:
        fig = go.Figure()
        for col, name, colour in solution:
            fig.add_scatter(x=h["epoch"], y=h[col].where(h[col] > 0), name=name, mode="lines",
                            line=dict(color=colour, width=1.6))
        fig.update_yaxes(type="log", title_text="MSE (log scale)")
        fig.update_xaxes(title_text="Epoch")
        return _style(fig, "Training losses", 420)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("Price network FNNSol", "Free-boundary conditions"))
    for col, name, colour in solution:
        fig.add_scatter(x=h["epoch"], y=h[col].where(h[col] > 0), name=name, mode="lines",
                        line=dict(color=colour, width=1.6), row=1, col=1)
    for col, name, colour in [("fb_init", "FB_Init", C1), ("fb_dir", "FB_Dir", C2), ("fb_neu", "FB_Neu", C3)]:
        fig.add_scatter(x=h["epoch"], y=h[col].where(h[col] > 0), name=name, mode="lines",
                        line=dict(color=colour, width=1.6, dash="dot"), row=1, col=2)
    fig.update_yaxes(type="log", title_text="MSE (log scale)", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title_text="Epoch")
    fig = _style(fig, "Loss evolution", 500)
    # legend under the plots, so that it does not cover the titles of the two panels
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                      margin=dict(b=90))
    return fig


def learning_rate(h: pd.DataFrame, penalty: bool) -> go.Figure:
    """Learning rate of FNNSol; that of FNNfb only when FNNfb is trained (no penalty) and differs."""
    fig = go.Figure()
    fig.add_scatter(x=h["epoch"], y=h["lr_sol"], mode="lines", name="FNNSol", line=dict(color=C1, width=2))
    if not penalty and not np.allclose(h["lr_sol"], h["lr_fb"]):
        fig.add_scatter(x=h["epoch"], y=h["lr_fb"], mode="lines", name="FNNfb", line=dict(color=C2, width=2))
    fig.update_yaxes(type="log", title="Learning rate")
    fig.update_xaxes(title="Epoch")
    return _style(fig, "Learning rate", 320)


def boundary(b: pd.DataFrame, strike: float) -> go.Figure:
    """Exercise boundary B(t) (one curve per underlying), with the strike K as reference line."""
    fig = go.Figure()
    cols = [c for c in b.columns if c != "t"]
    for i, c in enumerate(cols):
        name = "B(t)" if c == "B_t" else c.replace("_t", "(t)")
        fig.add_scatter(x=b["t"], y=b[c], mode="lines", name=name,
                        line=dict(color=[C1, C2, C3][i % 3] if i < 3 else MUTED, width=2))
    fig.add_hline(y=strike, line=dict(color=MUTED, dash="dot", width=1), annotation_text="K",
                  annotation_position="top left")
    fig.update_xaxes(title="Time t")
    fig.update_yaxes(title="Exercise boundary")
    return _style(fig, "Free boundary B(t)")


def price_at_time_1d(surface: pd.DataFrame, strike: float, t: float = 0.0) -> go.Figure:
    """Price P(S, t) and payoff at the saved time closest to t."""
    times = np.sort(surface["t"].unique())
    tt = times[np.argmin(np.abs(times - t))]
    g = surface[np.isclose(surface["t"], tt)].sort_values("S")
    fig = go.Figure()
    fig.add_scatter(x=g["S"], y=np.maximum(strike - g["S"], 0), mode="lines", name="Payoff (K − S)⁺",
                    line=dict(color=MUTED, width=1.2))
    fig.add_scatter(x=g["S"], y=g["Price"], mode="lines", name="PINN price", line=dict(color=C1, width=2.4))
    fig.update_xaxes(title="Underlying price S")
    fig.update_yaxes(title="Option price")
    return _style(fig, f"Option price at t = {tt:.2f}")


def price_map_1d(surface: pd.DataFrame, b: pd.DataFrame | None) -> go.Figure:
    """d = 1: heat map of P(S, t) in the plane (t, S), with the exercise boundary on top."""
    ts, Ss, P = grid(surface, "t", "S", "Price")
    fig = go.Figure(go.Heatmap(x=ts, y=Ss, z=P, colorscale=SEQ, colorbar=dict(title="Price"),
                               hovertemplate="t = %{x:.2f}<br>S = %{y:.2f}<br>P = %{z:.4f}<extra></extra>"))
    if b is not None:
        fig.add_scatter(x=b["t"], y=b[b.columns[1]], mode="lines", name="Boundary B(t)", line=dict(color=C2, width=2.5))
    fig.update_xaxes(title="Time t", showgrid=False)
    fig.update_yaxes(title="Underlying price S", showgrid=False)
    fig.update_layout(hovermode="closest")
    return _style(fig, "Price P(S, t) and exercise boundary", 480)


def surface_3d(x, y, Z, x_title: str, y_title: str, title: str) -> go.Figure:
    """3D price surface (P as a function of two variables: (S, t) in 1D, (S1, S2) in 2D)."""
    fig = go.Figure(go.Surface(x=x, y=y, z=Z, colorscale=SEQ, showscale=False,
                               hovertemplate=f"{x_title} = %{{x:.2f}}<br>{y_title} = %{{y:.2f}}<br>P = %{{z:.4f}}<extra></extra>"))
    fig.update_layout(scene=dict(xaxis_title=x_title, yaxis_title=y_title, zaxis_title="Price",
                                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))), hovermode="closest")
    return _style(fig, title, 560)


def exercise_region(x, y, E, x_title: str, y_title: str, title: str) -> go.Figure:
    """0/1 map of the exercise region (1 = exercise immediately)."""
    fig = go.Figure(go.Heatmap(x=x, y=y, z=E, colorscale=EXERCISE, zmin=0, zmax=1, showscale=False,
                               hovertemplate=f"{x_title} = %{{x:.2f}}<br>{y_title} = %{{y:.2f}}<br>exercise = %{{z}}<extra></extra>"))
    fig.update_xaxes(title=x_title, showgrid=False)
    fig.update_yaxes(title=y_title, showgrid=False)
    fig.update_layout(hovermode="closest")
    return _style(fig, title + "  (orange = early exercise)", 460)


def map_2d(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """d = 2: heat map of a column of the surface in the plane (S1, S2)."""
    x, y, Z = grid(df, "S1", "S2", column)
    fig = go.Figure(go.Heatmap(x=x, y=y, z=Z, colorscale=SEQ, colorbar=dict(title="Price"),
                               hovertemplate="S1 = %{x:.2f}<br>S2 = %{y:.2f}<br>P = %{z:.4f}<extra></extra>"))
    fig.update_xaxes(title="S1", showgrid=False)
    fig.update_yaxes(title="S2", showgrid=False, scaleanchor="x")
    fig.update_layout(hovermode="closest")
    return _style(fig, title, 500)


def prices_per_seed(prices: pd.DataFrame) -> go.Figure | None:
    """With several seeds: price at every S0 for every seed, with the mean."""
    seed_cols = [c for c in prices.columns if c.startswith("Seed")]
    s_cols = [c for c in prices.columns if c.startswith("S") and not c.startswith("Seed")]
    if len(seed_cols) < 2:
        return None
    labels = ["(" + "; ".join(f"{v:.2f}" for v in row) + ")" for row in prices[s_cols].to_numpy()]
    fig = go.Figure()
    for c in seed_cols:
        fig.add_scatter(x=labels, y=prices[c], mode="markers", name=c, marker=dict(color=MUTED, size=7),
                        showlegend=False)
    fig.add_scatter(x=labels, y=prices[seed_cols].mean(axis=1), mode="markers+lines", name="Mean",
                    marker=dict(color=C1, size=10), line=dict(color=C1, width=1.5))
    fig.update_xaxes(title="S0")
    fig.update_yaxes(title="Price")
    fig.update_layout(hovermode="closest")
    return _style(fig, "Price per seed (grey) and mean (blue)")
