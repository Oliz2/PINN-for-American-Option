"""Reading the files produced by one run of pinn.exe (one folder in runs/).

Every run folder contains config.json, log.txt and the CSV files written by the C++ program
(CsvExporter, OptionPriceReport, Table3Results). This module turns them into pandas DataFrames.

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PinnRun:
    """One finished (or running) run: its folder and the configuration it was launched with."""
    folder: Path
    config: dict

    @property
    def d(self) -> int:
        """Number of underlyings."""
        return self.config["option"]["numberOfAssets"]

    @property
    def strike(self) -> float:
        return self.config["option"]["strike"]

    @property
    def maturity(self) -> float:
        return self.config["option"]["maturity"]

    @property
    def seeds(self) -> list[int]:
        return self.config["seeds"]

    def _file(self, prefix: str, seed: int | None = None, extra: str = "") -> Path | None:
        """First CSV file of the run whose name starts with prefix (and contains the seed, if given)."""
        pattern = f"{prefix}_seed{seed}{extra}_*.csv" if seed is not None else f"{prefix}*.csv"
        found = sorted(self.folder.glob(pattern))
        return found[0] if found else None

    # ----- summary tables -----
    def prices(self) -> pd.DataFrame | None:
        """Option prices at the requested S0 (one column per seed)."""
        f = self._file("option_prices")
        return pd.read_csv(f) if f else None

    def test_losses(self) -> pd.DataFrame | None:
        """Losses on the test set (format of Table 3 of the reference paper)."""
        f = self._file("table3_results")
        return pd.read_csv(f) if f else None

    # ----- files per seed -----
    def history(self, seed: int) -> pd.DataFrame | None:
        """Losses and learning rates at every epoch."""
        f = self._file("training_history", seed)
        return pd.read_csv(f) if f else None

    def boundary(self, seed: int) -> pd.DataFrame | None:
        """Exercise boundary B(t)."""
        f = self._file("boundary_curve", seed)
        return pd.read_csv(f) if f else None

    def surface_1d(self, seed: int) -> pd.DataFrame | None:
        """d = 1: price and exercise flag on the grid (S, t)."""
        f = self._file("price_surface", seed)
        return pd.read_csv(f) if f else None

    def surfaces_2d(self, seed: int) -> dict[float, pd.DataFrame]:
        """2D surfaces saved at fixed t: {t: DataFrame(S1, S2, V, Price, is_exercise)}."""
        out = {}
        for f in sorted(self.folder.glob(f"price_surface_2d_seed{seed}_t*.csv")):
            m = re.search(r"_t([0-9.]+)_d", f.name)
            if m:
                out[float(m.group(1))] = pd.read_csv(f)
        return out

    def log(self) -> str:
        """Full text output of pinn.exe."""
        f = self.folder / "log.txt"
        return f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""

    def finished(self) -> bool:
        """A run is complete when the price file exists (it is written at the end)."""
        return self.prices() is not None

    # ----- export -----
    def excel(self) -> bytes:
        """All the CSV files of the run in one Excel file, one sheet per file (plus the configuration)."""
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            pd.DataFrame([{"parameter": k, "value": json.dumps(v)} for k, v in self.config.items()]) \
                .to_excel(xw, sheet_name="configuration", index=False)
            for f in sorted(self.folder.glob("*.csv")):
                df = pd.read_csv(f)
                if len(df) > 200_000:      # large 2D surfaces stay only in the CSV files
                    continue
                df.to_excel(xw, sheet_name=_sheet_name(f.stem), index=False)
        return buf.getvalue()


def _sheet_name(stem: str) -> str:
    """Excel sheet name from a file name (at most 31 characters, no forbidden symbols)."""
    short = (stem.replace("training_history", "history").replace("boundary_curve", "boundary")
             .replace("price_surface_2d", "surf2d").replace("price_surface", "surface")
             .replace("table3_results", "test_losses").replace("option_prices", "prices"))
    return re.sub(r"[\[\]\*\?/\\:]", "", short)[:31]


def list_runs(runs_folder: Path) -> list[Path]:
    """Run folders (those with a config.json), the most recent first."""
    if not runs_folder.exists():
        return []
    return sorted([p for p in runs_folder.iterdir() if (p / "config.json").exists()], reverse=True)


def load(folder: Path) -> PinnRun:
    """Opens a run folder."""
    return PinnRun(folder=folder, config=json.loads((folder / "config.json").read_text(encoding="utf-8")))


def grid(df: pd.DataFrame, x: str, y: str, z: str):
    """From columns (x, y, z) to a grid: returns xs, ys, Z with Z[i, j] = z(xs[j], ys[i])."""
    xs = np.sort(df[x].unique())
    ys = np.sort(df[y].unique())
    Z = df.pivot_table(index=y, columns=x, values=z).reindex(index=ys, columns=xs).to_numpy()
    return xs, ys, Z


# ---------------------------------------------------------------------------------------------
# Price Table: prices on a grid chosen by the user, taken from the surface saved by the C++ program
# ---------------------------------------------------------------------------------------------
def price_table_1d(surface: pd.DataFrame, s_values, t_values, strike: float):
    """Returns (S x t matrix of prices, {t: table for that time}).

    The C++ surface has S with step 0.1 and t with step T/150: prices in S are linearly interpolated
    (exact on the grid points), the time is the closest saved one.
    """
    times = np.sort(surface["t"].unique())
    s_values = np.asarray(s_values, dtype=float)
    matrix = {}
    per_time = {}
    for t in t_values:
        tt = float(times[np.argmin(np.abs(times - t))])
        g = surface[np.isclose(surface["t"], tt)].sort_values("S")
        price = np.interp(s_values, g["S"], g["Price"])
        nearest = np.abs(g["S"].to_numpy()[:, None] - s_values[None, :]).argmin(axis=0)
        exercise = g["is_exercise"].to_numpy()[nearest].astype(int)
        payoff = np.maximum(strike - s_values, 0.0)
        label = f"t = {tt:.2f}"
        matrix[label] = price
        per_time[label] = pd.DataFrame({
            "S": s_values, "PINN price": price, "Payoff": payoff, "Time value": price - payoff,
            "Region": np.where(exercise == 1, "exercise", "continuation")})
    mat = pd.DataFrame(matrix, index=pd.Index(np.round(s_values, 4), name="S"))
    return mat, per_time


def price_table_2d(df: pd.DataFrame, s_values) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S1 x S2 table of prices (and of the exercise region) at the closest saved values of S."""
    x, y, P = grid(df, "S1", "S2", "Price")
    _, _, E = grid(df, "S1", "S2", "is_exercise")
    s_values = np.asarray(s_values, dtype=float)
    ix = np.abs(x[:, None] - s_values[None, :]).argmin(axis=0)
    iy = np.abs(y[:, None] - s_values[None, :]).argmin(axis=0)
    # P[i, j] = price at (S1 = x[j], S2 = y[i]): rows S1, columns S2 as in the C++ program
    prices = pd.DataFrame(P[np.ix_(iy, ix)].T, index=pd.Index(np.round(x[ix], 4), name="S1 \\ S2"),
                          columns=np.round(y[iy], 4))
    exercise = pd.DataFrame(E[np.ix_(iy, ix)].T.astype(int), index=prices.index, columns=prices.columns)
    return prices, exercise
