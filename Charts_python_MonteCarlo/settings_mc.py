"""Default values of the Monte Carlo interface and writing of the config.properties file read by Java.

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import numpy as np

# Starting sigma matrix for 4 underlyings (the same as in the PINN interface, from the reference paper):
# for d <= 4 the top-left block is used, so that PINN and Monte Carlo start from the same market
SIGMA_DEFAULT_4D = [
    [0.05, 0.01, 0.10, 0.00],
    [0.01, 0.06, -0.03, 0.00],
    [0.10, -0.03, 0.40, 0.20],
    [0.00, 0.00, 0.20, 0.30],
]

# Pricing methods offered by the page (key = name used by the Java program)
METHODS = {
    "lsm": "Longstaff-Schwartz Monte Carlo",
    "binomial": "Binomial tree (d = 1 only)",
    "brutal": "Brutal Monte Carlo (exercise as soon as in the money)",
}
SHORT_NAMES = {"lsm": "Longstaff-Schwartz", "binomial": "Binomial tree", "brutal": "Brutal MC"}


def default_sigma(d: int) -> list[list[float]]:
    """Starting sigma matrix: block of the 4 x 4 matrix; beyond 4 assets placeholder values (0.2 on the diagonal)."""
    m = np.zeros((d, d))
    k = min(d, 4)
    m[:k, :k] = np.array(SIGMA_DEFAULT_4D)[:k, :k]
    for i in range(4, d):
        m[i, i] = 0.2
    return m.tolist()


def build_config(*, d: int, strike: float, maturity: float, r: float, dividends: list[float],
                 sigma: list[list[float]], spots: list[list[float]], valuation_time: float, methods: list[str],
                 paths: int, exercise_dates: int, degree: int, tree_steps: int, seed: int, sample_paths: int,
                 price_table: bool, table_smin: float, table_smax: float, table_step: float, table_times: int,
                 table_paths: int) -> dict:
    """All the parameters of a run; saved as config.json (for the Results tab) and as config.properties (for Java)."""
    return {
        "numberOfAssets": int(d), "strike": float(strike), "maturity": float(maturity), "riskFreeRate": float(r),
        "dividendYields": [float(q) for q in dividends],
        "volatilityMatrix": [[float(x) for x in row] for row in sigma],
        "spotPrices": [[float(x) for x in row] for row in spots], "valuationTime": float(valuation_time),
        "methods": list(methods), "numberOfPaths": int(paths), "exerciseDates": int(exercise_dates),
        "basisDegree": int(degree), "treeSteps": int(tree_steps), "seed": int(seed), "samplePaths": int(sample_paths),
        "priceTable": bool(price_table), "tableSMin": float(table_smin), "tableSMax": float(table_smax),
        "tableSStep": float(table_step), "tableTimes": int(table_times), "tablePaths": int(table_paths),
    }


def properties(config: dict) -> str:
    """Text of config.properties: vectors comma separated, matrix rows separated by ';'."""
    def vec(v):
        return ",".join(repr(float(x)) for x in v)

    def mat(m):
        return ";".join(vec(r) for r in m)

    lines = []
    for k, v in config.items():
        if k in ("volatilityMatrix", "spotPrices"):
            v = mat(v)
        elif k == "dividendYields":
            v = vec(v)
        elif k == "methods":
            v = ",".join(v)
        elif isinstance(v, bool):
            v = "true" if v else "false"
        lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"


def estimated_memory_mb(config: dict) -> float:
    """Memory of the Longstaff-Schwartz training paths (float, 4 bytes)."""
    return config["numberOfPaths"] * (config["exerciseDates"] + 1) * config["numberOfAssets"] * 4 / 1e6


def check(config: dict) -> list[str]:
    """Checks before starting: returns the list of errors (empty if everything is fine)."""
    errors = []
    d = config["numberOfAssets"]
    if len(config["dividendYields"]) != d:
        errors.append(f"There must be {d} dividend yields.")
    sig = np.array(config["volatilityMatrix"], dtype=float)
    if sig.shape != (d, d) or np.any(np.isnan(sig)):
        errors.append(f"The sigma matrix must be {d} x {d} with no empty cells.")
    if config["strike"] <= 0 or config["maturity"] <= 0:
        errors.append("Strike and maturity must be positive.")
    if not config["spotPrices"]:
        errors.append("Enter at least one initial price S0.")
    for row in config["spotPrices"]:
        if len(row) != d or any(np.isnan(row)) or any(x <= 0 for x in row):
            errors.append("Every S0 must have d positive components.")
            break
    if not 0 <= config["valuationTime"] < config["maturity"]:
        errors.append("The valuation time must be between 0 and the maturity.")
    if not config["methods"]:
        errors.append("Choose at least one method.")
    if "binomial" in config["methods"] and d != 1:
        errors.append("The binomial tree is available only for d = 1.")
    if config["numberOfPaths"] < 100:
        errors.append("At least 100 paths are needed.")
    if config["priceTable"] and (config["tableSMax"] < config["tableSMin"] or config["tableSStep"] <= 0
                                 or config["tableSMin"] <= 0):
        errors.append("Invalid Price Table range.")
    return errors
