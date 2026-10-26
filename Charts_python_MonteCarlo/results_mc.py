"""Reading the files produced by one pricing run (one folder in runs/) and comparison with the PINN.

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MCRun:
    """One pricing run: its folder and its configuration. Each method returns one CSV file as a DataFrame."""
    folder: Path
    config: dict

    @property
    def d(self) -> int:
        return self.config["numberOfAssets"]

    def _csv(self, name: str) -> pd.DataFrame | None:
        """CSV file of the run, or None if it does not exist or is empty."""
        f = self.folder / name
        if not f.exists() or f.stat().st_size == 0:
            return None
        df = pd.read_csv(f)
        return df if len(df) else None

    def prices(self):
        return self._csv("mc_prices.csv")

    def convergence(self):
        return self._csv("mc_convergence.csv")

    def paths(self):
        return self._csv("mc_paths.csv")

    def exercises(self):
        return self._csv("mc_exercises.csv")

    def boundary(self):
        return self._csv("mc_boundary.csv")

    def price_table(self):
        return self._csv("mc_price_table.csv")

    def log(self) -> str:
        f = self.folder / "log.txt"
        return f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""

    def finished(self) -> bool:
        """The Java program prints DONE as its last line."""
        return "DONE" in self.log()

    def excel(self, extra: dict[str, pd.DataFrame] | None = None) -> bytes:
        """All the mc_*.csv files (and the extra tables, e.g. the PINN comparison) in one Excel file."""
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            pd.DataFrame([{"parameter": k, "value": json.dumps(v)} for k, v in self.config.items()]) \
                .to_excel(xw, sheet_name="configuration", index=False)
            for f in sorted(self.folder.glob("mc_*.csv")):
                df = pd.read_csv(f)
                if len(df) <= 200_000:
                    df.to_excel(xw, sheet_name=f.stem[:31], index=False)
            for name, df in (extra or {}).items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
        return buf.getvalue()


def list_runs(runs_folder: Path) -> list[Path]:
    """Run folders (those with a config.json), the most recent first."""
    if not runs_folder.exists():
        return []
    return sorted([p for p in runs_folder.iterdir() if (p / "config.json").exists()], reverse=True)


def load(folder: Path) -> MCRun:
    """Opens a run folder."""
    return MCRun(folder=folder, config=json.loads((folder / "config.json").read_text(encoding="utf-8")))


def spot_columns(d: int) -> list[str]:
    """Column names of the underlyings: S1, ..., Sd."""
    return [f"S{i + 1}" for i in range(d)]


def price_summary(prices: pd.DataFrame, d: int, strike: float, names: dict) -> pd.DataFrame:
    """One row per S0, one column per method, plus payoff and European price."""
    s_cols = spot_columns(d)
    wide = prices.pivot_table(index=s_cols, columns="method", values="price", sort=False)
    wide = wide[[m for m in prices["method"].unique()]].rename(columns=names)
    wide.columns.name = None
    out = wide.reset_index()
    # European price: from the tree if available (no simulation noise), otherwise from the first method
    src = "binomial" if "binomial" in prices["method"].values else prices["method"].iloc[0]
    eu = prices[prices["method"] == src][s_cols + ["european"]].rename(columns={"european": "European put"})
    out = out.merge(eu, on=s_cols, how="left")
    out.insert(len(s_cols), "Payoff", np.maximum(strike - out[s_cols].min(axis=1), 0))
    return out


def price_table_matrix(pt: pd.DataFrame, method: str) -> pd.DataFrame:
    """Price Table of one method as a matrix: rows S, columns t."""
    g = pt[pt["method"] == method]
    mat = g.pivot_table(index="S", columns="t", values="price")
    mat.columns = [f"t = {t:.2f}" for t in mat.columns]
    mat.index.name = "S"
    return mat


# ---------------------------------------------------------------------------------------------
# Comparison with a PINN run (folder Charts_python_PINN/runs)
# ---------------------------------------------------------------------------------------------
def pinn_runs(pinn_folder: Path) -> list[Path]:
    """Finished PINN runs (with an option_prices file) available for the comparison."""
    if not pinn_folder.exists():
        return []
    return sorted([p for p in pinn_folder.iterdir()
                   if (p / "config.json").exists() and any(p.glob("option_prices*.csv"))], reverse=True)


def same_option(pinn_cfg: dict, mc_cfg: dict) -> bool:
    """True if the PINN run prices the same option in the same market."""
    try:
        o, m = pinn_cfg["option"], pinn_cfg["market"]
        return (o["numberOfAssets"] == mc_cfg["numberOfAssets"]
                and np.isclose(o["strike"], mc_cfg["strike"]) and np.isclose(o["maturity"], mc_cfg["maturity"])
                and np.isclose(m["riskFreeRate"], mc_cfg["riskFreeRate"])
                and np.allclose(m["dividendYields"], mc_cfg["dividendYields"])
                and np.allclose(m["volatilityMatrix"], mc_cfg["volatilityMatrix"]))
    except (KeyError, ValueError, TypeError):
        return False


def pinn_prices(pinn_run: Path) -> tuple[dict, pd.DataFrame] | None:
    """Configuration and prices of a PINN run (column PINN = mean over the seeds)."""
    cfg_file = pinn_run / "config.json"
    files = sorted(pinn_run.glob("option_prices*.csv"))
    if not cfg_file.exists() or not files:
        return None
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    df = pd.read_csv(files[0])
    seed_cols = [c for c in df.columns if c.startswith("Seed")]
    df["PINN"] = df[seed_cols].mean(axis=1)      # with several seeds: mean over the seeds
    return cfg, df


def comparison(summary: pd.DataFrame, pinn: pd.DataFrame, d: int, t_valuation: float,
               method_names: list[str]) -> pd.DataFrame:
    """S0 present in both runs (at the same time t): PINN price, price of each method and differences."""
    s_cols = spot_columns(d)
    if any(c not in pinn.columns for c in s_cols) or f"S{d + 1}" in pinn.columns:
        return pd.DataFrame()        # different number of underlyings
    if "t" in pinn.columns:
        pinn = pinn[np.isclose(pinn["t"], t_valuation)]
    a = pinn[s_cols + ["PINN"]].copy()
    b = summary.copy()
    for c in s_cols:
        a[c] = a[c].round(6)
        b[c] = b[c].round(6)
    out = b.merge(a, on=s_cols, how="inner")
    cols = s_cols + ["Payoff", "PINN"] + [m for m in method_names if m in out.columns]
    if "European put" in out.columns:
        cols.append("European put")
    out = out[cols]
    for m in method_names:
        if m in out.columns and m != "Brutal MC":      # the brutal MC is a wrong method: no difference columns
            out[f"PINN - {m}"] = out["PINN"] - out[m]
            out[f"PINN - {m} (%)"] = np.where(out[m].abs() > 1e-12, 100 * (out["PINN"] - out[m]) / out[m], np.nan)
    return out
