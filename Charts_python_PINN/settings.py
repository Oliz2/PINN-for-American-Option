"""Default values of the interface and construction of the JSON file read by pinn.exe.

The JSON has the structure described in C++_Project/src/main/it.univr.experiments/JsonConfiguration.h:
every field is required, so all of them are always written here.

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import numpy as np

# Starting sigma matrix for 4 underlyings: for d <= 4 the top-left block is used.
# Market data of the reference paper (Gatta et al., 2023, Section 5.1): K = 10, T = 3, r = 0.01, q = 0 and this
# sigma matrix of the 4D case, of which the first rows and columns are used in lower dimension.
SIGMA_DEFAULT_4D = [
    [0.05, 0.01, 0.10, 0.00],
    [0.01, 0.06, -0.03, 0.00],
    [0.10, -0.03, 0.40, 0.20],
    [0.00, 0.00, 0.20, 0.30],
]

# Starting values of points, epochs and learning-rate decay for d = 1, 2, 3, 4
# (numbers of collocation points as in Table 1 of the reference paper)
DEFAULTS = {
    1: dict(pde=30000, initial=300, dirichlet=300, fb=300, epochs=4000, n_steps=20,
            sol_steps=500, sol_rate=0.9, fb_steps=500, fb_rate=0.9, separate=False),
    2: dict(pde=60000, initial=1500, dirichlet=1500, fb=1500, epochs=10000, n_steps=4,
            sol_steps=400, sol_rate=0.9, fb_steps=100, fb_rate=0.9, separate=True),
    3: dict(pde=120000, initial=4000, dirichlet=9000, fb=9000, epochs=15000, n_steps=4,
            sol_steps=320, sol_rate=0.975, fb_steps=80, fb_rate=0.975, separate=True),
    4: dict(pde=120000, initial=12000, dirichlet=12000, fb=12000, epochs=25000, n_steps=4,
            sol_steps=640, sol_rate=0.975, fb_steps=160, fb_rate=0.975, separate=True),
}

# Features of the model, always active: normalised inputs (S/K, t/T), 30% of the points near the strike,
# learning rate decaying to a final value. The penalty formulation is chosen in the interface.
MODEL_FEATURES = ["norm", "refine", "slowlr"]


def default_sigma(d: int) -> list[list[float]]:
    """Starting sigma matrix: block of the 4 x 4 matrix for d <= 4, diagonal 0.2 for the extra assets."""
    m = np.zeros((d, d))
    k = min(d, 4)
    m[:k, :k] = np.array(SIGMA_DEFAULT_4D)[:k, :k]
    # Beyond 4 assets the paper gives no data: placeholder values (volatility 0.2, uncorrelated assets),
    # only so that the table starts from a valid matrix. They can be edited in the interface.
    for i in range(4, d):
        m[i, i] = 0.2
    return m.tolist()


def default_settings(d: int) -> dict:
    """Starting points and training settings for d <= 4; for d > 4 those of d = 4 are used."""
    return dict(DEFAULTS[min(d, 4)])


def variant_name(variants: list[str]) -> str:
    """Label of the run, used in the folder name: complementarity ("penalty") or two networks."""
    return "pinn-penalty" if "penalty" in variants else "pinn-two-networks"


def build_config(*, d: int, strike: float, maturity: float, r: float, dividends: list[float],
                 sigma: list[list[float]], spots: list[list[float]], valuation_time: float,
                 smax_over_k: float, network: dict, points: dict, variants: list[str], tolerance: float,
                 final_lr: float, training: dict, test: dict, seeds: list[int], log_every: int) -> dict:
    """Dictionary ready to be saved as JSON for pinn.exe --config."""
    K = float(strike)
    return {
        "option": {"type": "put", "numberOfAssets": int(d), "strike": K, "maturity": float(maturity)},
        "market": {"riskFreeRate": float(r), "dividendYields": [float(q) for q in dividends],
                   "volatilityMatrix": [[float(x) for x in row] for row in sigma]},
        "pricing": {"spotPrices": [[float(x) for x in row] for row in spots], "valuationTime": float(valuation_time)},
        "domain": {"sMax": float(smax_over_k) * K},                        # S in [0, Smax]^d
        "network": {k: int(v) for k, v in network.items()},
        "points": {
            "pde": int(points["pde"]), "initial": int(points["initial"]), "dirichlet": int(points["dirichlet"]),
            "fbDirichlet": int(points["fb"]), "fbNeumann": int(points["fb"]), "sampling": "sobol",
            "strikeRefinementFraction": 0.3 if "refine" in variants else 0.0,   # 30% of the points near K
            "strikeRefinementHalfWidth": 0.2 * K,                               # in [0.8 K, 1.2 K]^d
        },
        "variants": {
            "name": variant_name(variants),
            "normalizeInputs": "norm" in variants,
            "hardTerminalCondition": "hard" in variants,
            "enforceIntrinsicValue": True,              # reported price >= payoff
            "constrainedFreeBoundary": "fbcap" in variants,
            "penaltyFormulation": "penalty" in variants,
            "exerciseTolerance": float(tolerance),
            "doublePrecision": "double" in variants,
        },
        "weights": {"pde": 1.0, "initial": 1.0, "dirichlet": 1.0, "fbInitial": 1.0, "fbDirichlet": 1.0, "fbNeumann": 1.0},
        "training": {
            "epochs": int(training["epochs"]),
            "solutionStepsPerBoundaryStep": int(training["n_steps"]),
            "separateBoundaryOptimizer": bool(training["separate"]),
            "solution": {"initialLearningRate": float(training["lr"]), "decaySteps": int(training["sol_steps"]),
                         "decayRate": float(training["sol_rate"])},
            "boundary": {"initialLearningRate": float(training["lr"]), "decaySteps": int(training["fb_steps"]),
                         "decayRate": float(training["fb_rate"])},
            "rmspropRho": 0.9, "rmspropEpsilon": 1e-7,  # defaults of tf.keras RMSprop
            "slowDecayToFinalLearningRate": "slowlr" in variants, "finalLearningRate": float(final_lr),
            "verbose": False, "logEvery": int(log_every),  # one summary line every log_every epochs (read by runner.py)
        },
        "testSet": {"pdePoints": int(test["pde"]), "pdeBatchSize": 20000, "boundaryPoints": int(test["boundary"])},
        "output": {
            "printPriceTable": True, "tableSMin": 0.8 * K, "tableSMax": 1.2 * K, "tableSSteps": 9, "tableTSteps": 10,
            "exportCsv": True, "curveTSteps": 600, "surfaceSMax": 1.5 * K, "surfaceSSteps": 150, "surfaceTSteps": 150,
        },
        "seeds": [int(s) for s in seeds],
    }


def check(config: dict) -> list[str]:
    """Checks before starting: returns the list of errors (empty if everything is fine)."""
    errors = []
    d = config["option"]["numberOfAssets"]
    m = config["market"]
    if len(m["dividendYields"]) != d:
        errors.append(f"There must be {d} dividend yields.")
    sig = np.array(m["volatilityMatrix"], dtype=float)
    if sig.shape != (d, d):
        errors.append(f"The sigma matrix must be {d} x {d}.")
    elif np.any(np.isnan(sig)):
        errors.append("The sigma matrix has empty cells.")
    elif np.linalg.matrix_rank(sig @ sig.T) < d:   # the covariance sigma sigma^T must be invertible
        errors.append("sigma * sigma^T is not positive definite: two underlyings are perfectly correlated.")
    if config["option"]["strike"] <= 0 or config["option"]["maturity"] <= 0:
        errors.append("Strike and maturity must be positive.")
    if not config["pricing"]["spotPrices"]:
        errors.append("Enter at least one initial price S0.")
    for row in config["pricing"]["spotPrices"]:
        if len(row) != d or any(np.isnan(row)) or any(x < 0 for x in row):
            errors.append("Every S0 must have d non-negative components.")
            break
    if not 0 <= config["pricing"]["valuationTime"] < config["option"]["maturity"]:
        errors.append("The valuation time must be between 0 and the maturity.")
    if config["domain"]["sMax"] <= config["option"]["strike"]:
        errors.append("Smax must be larger than the strike.")
    if not config["seeds"]:
        errors.append("Enter at least one seed.")
    return errors
