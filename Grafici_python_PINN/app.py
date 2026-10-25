"""PINN interface for American options.

Start (from the VS Code terminal, in the Grafici_python_PINN folder):
    python -m streamlit run app.py

The page writes a JSON file with the chosen parameters, launches pinn.exe (C++) and shows progress,
prices and charts. Every run gets its own folder in runs/, which stays available in the
"Results" tab after the page is closed.

Python does no PINN computation: the whole training and pricing is done by the C++ program.

Layout of the page:
    sidebar    path of pinn.exe and of the results folder
    tab 1      option, market, prices to compute, network and training -> "Start training"
    tab 2      results of a run: prices, Price Table, losses, exercise boundary, price surfaces
    bottom     live progress of the running computation

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts
import results
import runner
import settings as cfg

HERE = Path(__file__).resolve().parent
# Default path of the C++ executable: the build folder of C++_Progetto, next to this folder
EXE_DEFAULT = HERE.parent / "C++_Progetto" / "build" / "Release" / ("pinn.exe" if os.name == "nt" else "pinn")
RUNS_DEFAULT = HERE / "runs"

st.set_page_config(page_title="PINN - American options", page_icon="📈", layout="wide")


def _version(v: str) -> tuple:
    """"1.46.0" -> (1, 46), to compare Streamlit versions."""
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


def shaded(df: pd.DataFrame, fmt: str = "{:.4f}"):
    """Table with a colour gradient on the values (if matplotlib is installed), otherwise only formatted."""
    style = df.style.format(fmt)
    try:
        import matplotlib  # noqa: F401  (needed by pandas for background_gradient)
        return style.background_gradient(cmap="Blues", axis=None)
    except ImportError:
        return style


# Full width for charts and tables, compatible with old and new Streamlit versions
WIDE = {"width": "stretch"} if _version(st.__version__) >= (1, 46) else {"use_container_width": True}

# ---------------------------------------------------------------------------------------------
# Sidebar: where the C++ program and the results are
# ---------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    exe = Path(st.text_input("C++ program (pinn.exe)", str(EXE_DEFAULT), key="exe"))
    if exe.is_file():
        st.success("pinn.exe found")
    else:
        st.error("pinn.exe not found: build the C++ project or fix the path.")
    runs_folder = Path(st.text_input("Results folder", str(RUNS_DEFAULT), key="runs"))
    st.caption("Every run creates a subfolder with config.json, log.txt and all the CSV files.")

st.title("PINN for American options")
st.caption("American put on the minimum of d underlyings in the Black-Scholes model, "
           "priced with a physics-informed neural network (PINN).")

tab_new, tab_results = st.tabs(["New run", "Results"])

# =============================================================================================
# TAB 1 - NEW RUN
# =============================================================================================
with tab_new:
    # ---------------- 1) Option ----------------
    st.subheader("1. Option")
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Type", ["American put on the minimum of the underlyings"],
                 help="Payoff (K − min_i S_i)⁺. With d = 1 it is the standard American put.")
    d = int(c2.number_input("Number of underlyings d", min_value=1, value=1, step=1, key="d"))
    K = c3.number_input("Strike K", min_value=0.01, value=10.0, step=1.0)
    T = c4.number_input("Maturity T (years)", min_value=0.01, value=3.0, step=0.5)
    if d > 4:
        st.warning(f"With d = {d} the training will be very slow.")
    s_names = [f"S{i + 1}" for i in range(d)]

    # ---------------- 2) Market ----------------
    st.subheader("2. Market")
    c1, c2 = st.columns([1, 3])
    r = c1.number_input("Risk-free rate r", value=0.01, step=0.005, format="%.4f")
    q_df = c2.data_editor(pd.DataFrame([[0.0] * d], columns=s_names, index=["q"]), key=f"dividends_{d}", **WIDE)
    st.markdown("Volatility matrix σ (d × d): the diffusion uses σσᵀ. The default values can be edited cell by cell.")
    sigma_df = st.data_editor(pd.DataFrame(cfg.default_sigma(d), columns=s_names, index=s_names),
                              key=f"sigma_{d}", **WIDE)
    if d > 1:
        cov = sigma_df.to_numpy(dtype=float) @ sigma_df.to_numpy(dtype=float).T
        vol = np.sqrt(np.diag(cov))
        with st.expander("Resulting volatilities and correlations"):
            st.dataframe(pd.DataFrame({"volatility": vol}, index=s_names).style.format("{:.4f}"))
            st.dataframe(pd.DataFrame(cov / np.outer(vol, vol), columns=s_names, index=s_names).style.format("{:.3f}"))

    # ---------------- 3) Prices to compute ----------------
    st.subheader("3. Prices to compute")
    st.caption("Each row is an initial price S0 = (S1, ..., Sd). Add or remove rows at the bottom of the table.")
    levels = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    spot_df = st.data_editor(pd.DataFrame([[round(x * K, 4)] * d for x in levels], columns=s_names),
                             key=f"spot_{d}_{K}", num_rows="dynamic", **WIDE)
    t_val = st.number_input("Valuation time t (0 = today)", min_value=0.0, max_value=float(T) * 0.999,
                            value=0.0, step=0.1)

    # ---------------- 4) Network and training ----------------
    st.subheader("4. Neural network and training")
    start_values = cfg.default_settings(d)
    penalty = st.toggle(
        "Penalty formulation", value=True, key="penalty",
        help="On: one condition min(−ΛP, P − ψ) = 0 on all the points; the exercise boundary is obtained "
             "from the price and FNNfb is not trained. Off: two networks, FNNSol for the price (PDE only in "
             "the continuation region) and FNNfb for the exercise boundary, trained together.")
    # norm + refine + slowlr always active; "penalty" = complementarity formulation, otherwise two networks
    variants = cfg.MODEL_FEATURES + (["penalty"] if penalty else [])
    epochs_default = 1500 if d == 1 else start_values["epochs"]
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", min_value=1, value=epochs_default, step=100, key=f"epochs_{d}")
    seeds_txt = c2.text_input("Seeds (comma separated)", "5", key="seeds",
                              help="To compare several initialisations, for example 2, 3, 5, 7.")
    log_every = c3.number_input("Update the progress every N epochs", min_value=1, value=10, step=5)

    with st.expander("Advanced settings (starting values for this d)"):
        a1, a2, a3, a4 = st.columns(4)
        sol_l = a1.number_input("FNNSol layers", 1, 50, 8, key=f"soll_{d}")
        sol_u = a2.number_input("FNNSol neurons", 1, 1000, 20, key=f"solu_{d}")
        fb_l = a3.number_input("FNNfb layers", 1, 50, 3, key=f"fbl_{d}")
        fb_u = a4.number_input("FNNfb neurons", 1, 2000, 100, key=f"fbu_{d}")
        a1, a2, a3, a4 = st.columns(4)
        n_pde = a1.number_input("PDE points", 100, 10_000_000, start_values["pde"], step=1000, key=f"pde_{d}")
        n_init = a2.number_input("Points at maturity", 10, 1_000_000, start_values["initial"], key=f"init_{d}")
        n_dir = a3.number_input("Boundary points (Dirichlet)", 10, 1_000_000, start_values["dirichlet"], key=f"dir_{d}")
        n_fb = a4.number_input("Free-boundary points", 10, 1_000_000, start_values["fb"], key=f"fb_{d}")
        a1, a2, a3, a4 = st.columns(4)
        n_steps = a1.number_input("FNNSol steps per FNNfb step", 1, 100, start_values["n_steps"], key=f"nst_{d}")
        separate = a2.checkbox("Separate optimiser for FNNfb", start_values["separate"], key=f"sep_{d}")
        lr = a3.number_input("Initial learning rate", 1e-6, 1.0, 1e-2, format="%.5f")
        smax = a4.number_input("Smax / K", 1.1, 20.0, 3.0, step=0.5)
        a1, a2, a3, a4 = st.columns(4)
        sol_steps = a1.number_input("FNNSol decay: every N steps", 1, 10**7, start_values["sol_steps"], key=f"ss_{d}")
        sol_rate = a2.number_input("FNNSol decay: factor", 0.01, 1.0, start_values["sol_rate"], format="%.3f", key=f"sr_{d}")
        fb_steps = a3.number_input("FNNfb decay: every N steps", 1, 10**7, start_values["fb_steps"], key=f"fs_{d}")
        fb_rate = a4.number_input("FNNfb decay: factor", 0.01, 1.0, start_values["fb_rate"], format="%.3f", key=f"fr_{d}")
        a1, a2, a3, a4 = st.columns(4)
        final_lr = a1.number_input("Final learning rate", 1e-8, 1.0, 1e-4, format="%.6f")
        tolerance = a2.number_input("Exercise tolerance", 1e-6, 1.0, 1e-3, format="%.4f")
        test_pde = a3.number_input("Test set: PDE points", 1000, 10_000_000, 1_000_000, step=100_000)
        test_bd = a4.number_input("Test set: points per condition", 10, 1_000_000, 1000)
        st.caption("The decay 'steps' are recomputed by the C++ program so that the learning rate reaches "
                   "the final value at the last epoch.")

    # ---------------- Start ----------------
    # All the inputs above are collected into the JSON dictionary read by pinn.exe --config
    try:
        seeds = [int(x) for x in seeds_txt.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        seeds = []
    spots = spot_df.dropna(how="all").to_numpy(dtype=float).tolist()
    config = cfg.build_config(
        d=d, strike=K, maturity=T, r=r, dividends=q_df.iloc[0].tolist(), sigma=sigma_df.to_numpy(dtype=float).tolist(),
        spots=spots, valuation_time=t_val, smax_over_k=smax,
        network=dict(solHiddenLayers=sol_l, solHiddenUnits=sol_u, fbHiddenLayers=fb_l, fbHiddenUnits=fb_u),
        points=dict(pde=n_pde, initial=n_init, dirichlet=n_dir, fb=n_fb), variants=variants, tolerance=tolerance,
        final_lr=final_lr,
        training=dict(epochs=epochs, n_steps=n_steps, separate=separate, lr=lr, sol_steps=sol_steps,
                      sol_rate=sol_rate, fb_steps=fb_steps, fb_rate=fb_rate),
        test=dict(pde=test_pde, boundary=test_bd), seeds=seeds, log_every=log_every)

    st.divider()
    running = "run" in st.session_state and st.session_state.run.running
    c1, c2, _ = st.columns([1, 1, 3])
    go_clicked = c1.button("▶ Start training", type="primary", disabled=running)
    c2.button("■ Stop", disabled=not running, on_click=lambda: st.session_state.run.stop())

    if go_clicked:
        errors = cfg.check(config)
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:   # runner.start writes config.json in a new run folder and launches pinn.exe in the background
                st.session_state.run = runner.start(exe, config, runs_folder)
                st.session_state.chosen_run = st.session_state.run.folder.name
            except Exception as e:      # wrong path, permissions, ...
                st.error(str(e))

    panel = st.container()       # progress: filled at the bottom of the page

# =============================================================================================
# TAB 2 - RESULTS
# =============================================================================================
with tab_results:
    run_list = results.list_runs(runs_folder)
    if not run_list:
        st.info("No runs yet: start one from the 'New run' tab.")
    else:
        run_names = [p.name for p in run_list]
        chosen = st.session_state.get("chosen_run")
        index = run_names.index(chosen) if chosen in run_names else 0
        name = st.selectbox("Run", run_names, index=index)
        run = results.load(runs_folder / name)
        cf = run.config     # the configuration the run was launched with (config.json)
        penalty = cf["variants"]["penaltyFormulation"]

        c = st.columns(5)
        c[0].metric("Underlyings d", run.d)
        c[1].metric("Strike K", f"{run.strike:g}")
        c[2].metric("Maturity T", f"{run.maturity:g}")
        c[3].metric("Epochs", cf["training"]["epochs"])
        c[4].metric("Seed", ", ".join(map(str, run.seeds)))
        st.markdown(f"r = {cf['market']['riskFreeRate']:g} · Smax = {cf['domain']['sMax']:g}")

        if not run.finished():
            st.warning("This run is not finished (running, stopped or failed).")

        prices = run.prices()
        if prices is not None:
            st.subheader("Option price")
            st.dataframe(prices.style.format({c: "{:.6f}" for c in prices.columns if c not in ("t",)}),
                         **WIDE, hide_index=True)
            fig = charts.prices_per_seed(prices)
            if fig is not None:
                st.plotly_chart(fig, **WIDE)
            st.caption("To compare these prices with Longstaff-Schwartz and the binomial tree, open the Monte Carlo "
                       "interface (Grafici_python_MonteCarlo) and run it with the same option and S0.")

        # ---------------- Price Table: one table per time ----------------
        if run.d in (1, 2):
            st.subheader("Price Table")
            K0, T0 = float(run.strike), float(run.maturity)
            p1, p2, p3, p4 = st.columns(4)
            s_min = p1.number_input("S from", 0.0, 1.5 * K0, 0.8 * K0, step=0.5, key=f"pt_smin_{name}")
            s_max = p2.number_input("S to", 0.0, 1.5 * K0, 1.2 * K0, step=0.5, key=f"pt_smax_{name}")
            s_step = p3.number_input("S step", 0.01, K0, 0.5, step=0.1, key=f"pt_ds_{name}")
            n_times = p4.number_input("Number of times (from 0 to T)", 2, 151, 11, key=f"pt_nt_{name}",
                                      disabled=run.d == 2)
            s_values = np.round(np.arange(s_min, s_max + s_step / 2, s_step), 6)
            if run.d == 1:
                surf_pt = run.surface_1d(run.seeds[0] if len(run.seeds) == 1 else
                                         st.selectbox("Seed for the Price Table", run.seeds, key=f"pt_seed_{name}"))
                if surf_pt is None:
                    st.info("Price surface not available yet.")
                else:
                    mat, per_time = results.price_table_1d(surf_pt, s_values, np.linspace(0, T0, int(n_times)), K0)
                    subtabs = st.tabs(["All times"] + list(per_time))
                    with subtabs[0]:
                        st.caption("Rows: underlying price S. Columns: time t. Values: PINN option price.")
                        st.dataframe(shaded(mat), **WIDE)
                    for sub, (label, tab) in zip(subtabs[1:], per_time.items()):
                        with sub:
                            st.dataframe(tab.style.format({"S": "{:.2f}", "PINN price": "{:.6f}", "Payoff": "{:.4f}",
                                                           "Time value": "{:.6f}"}), hide_index=True, **WIDE)
            else:
                surf2_pt = run.surfaces_2d(run.seeds[0])
                if not surf2_pt:
                    st.info("2D surfaces not available yet.")
                else:
                    st.caption("Rows: S1. Columns: S2. In 2D the C++ program saves the surface at t = 0 and t = T/2.")
                    subtabs = st.tabs([f"t = {t2:.2f}" for t2 in surf2_pt])
                    for sub, (t2, df2) in zip(subtabs, surf2_pt.items()):
                        with sub:
                            prices2, _ = results.price_table_2d(df2, s_values)
                            st.dataframe(shaded(prices2), **WIDE)

        st.subheader("Charts")
        seed = st.selectbox("Seed to show", run.seeds) if len(run.seeds) > 1 else run.seeds[0]
        # The boundary obtained from the price (penalty) is reliable only in 1D
        show_boundary = run.d == 1 or not penalty
        if show_boundary:
            g_loss, g_bound, g_price = st.tabs(["Losses", "Exercise boundary",
                                                "Price and exercise region" if run.d == 1 else "Price"])
        else:
            g_loss, g_price = st.tabs(["Losses", "Price"])
            g_bound = None

        with g_loss:
            h = run.history(seed)
            if h is None:
                st.info("Loss history not available yet.")
            else:
                st.plotly_chart(charts.loss_evolution(h, penalty), **WIDE)
                st.plotly_chart(charts.learning_rate(h, penalty), **WIDE)

        b = run.boundary(seed)
        if g_bound is not None:
            with g_bound:
                if b is None:
                    st.info("Boundary curve not available yet.")
                else:
                    st.plotly_chart(charts.boundary(b, run.strike), **WIDE)
                    st.caption("B(0) = " + ", ".join(f"{b[c].iloc[0]:.4f}" for c in b.columns if c != "t"))

        with g_price:
            if run.d == 1:
                surf = run.surface_1d(seed)
                if surf is None:
                    st.info("Price surface not available yet.")
                else:
                    t_chosen = st.slider("Time t for the price curve", 0.0, float(run.maturity), 0.0,
                                         step=float(run.maturity) / 150)
                    st.plotly_chart(charts.price_at_time_1d(surf, run.strike, t_chosen), **WIDE)
                    st.plotly_chart(charts.price_map_1d(surf, b), **WIDE)
                    ts, Ss, P = results.grid(surf, "t", "S", "Price")
                    st.plotly_chart(charts.surface_3d(ts, Ss, P, "t", "S", "Price surface P(S, t)"), **WIDE)
                    _, _, E = results.grid(surf, "t", "S", "is_exercise")
                    st.plotly_chart(charts.exercise_region(ts, Ss, E, "Time t", "S", "Exercise region"), **WIDE)
            elif run.d == 2:
                surf2 = run.surfaces_2d(seed)
                if not surf2:
                    st.info("2D surfaces not available yet.")
                for t2, df in surf2.items():
                    st.markdown(f"**t = {t2:.2f}**")
                    x, y, P = results.grid(df, "S1", "S2", "Price")
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(charts.map_2d(df, "Price", f"Price at t = {t2:.2f}"), **WIDE)
                    c2.plotly_chart(charts.surface_3d(x, y, P, "S1", "S2", f"Price surface at t = {t2:.2f}"), **WIDE)
            else:
                st.info(f"With d = {run.d} there is no surface to draw: prices, losses and boundary B(t) remain.")

# =============================================================================================
# PROGRESS OF THE RUNNING COMPUTATION (updated every second while pinn.exe works)
# =============================================================================================
with panel:
    es = st.session_state.get("run")
    if es is not None and not hasattr(es, "history"):      # object of the old interface still in memory
        es = None
    if es is not None:
        status = st.empty()
        bar = st.progress(0.0)
        chart = st.empty()
        log_tail = st.empty()

        redraws = [0]      # every redraw of the live chart has a new key

        def draw() -> None:
            redraws[0] += 1
            elapsed = time.time() - es.start
            p = es.progress()
            eta = f", about {elapsed * (1 - p) / p / 60:.0f} min left" if 0.02 < p < 1 else ""
            seed_txt = f"seed {es.current_seed} ({min(es.seeds_done + 1, es.n_seeds)}/{es.n_seeds})" if es.current_seed is not None else ""
            loss_txt = f" · loss {es.history[-1][2]:.3e}" if es.history else ""
            status.markdown(f"**{es.folder.name}** · {es.phase} · {seed_txt} · epoch {es.epoch}/{es.epochs}{loss_txt} "
                            f"· {elapsed / 60:.1f} min{eta}")
            bar.progress(p)
            history = list(es.history)       # the reader thread may append while drawing
            if history:
                chart.plotly_chart(charts.live_loss(history, es.epochs), **WIDE, key=f"live_{redraws[0]}")
            elif es.phase == "training":
                chart.info(f"The loss chart appears at the first loss print of pinn.exe, i.e. after {es.log_every} "
                           "epochs (setting 'Update the progress every N epochs'). In 2D or more every epoch takes "
                           "a few seconds, so the first print can take a minute or more.")
            log_tail.code("\n".join(es.lines[-12:]) or "...", language=None)

        # Streamlit redraws the page from the top: this loop keeps the page busy while pinn.exe works,
        # refreshing status, progress bar, live chart and the last lines of the log once per second
        while es.running:
            draw()
            time.sleep(1.0)
        time.sleep(0.3)      # last lines of the log
        draw()
        if es.stopped:
            st.warning("Run stopped.")
        elif es.exit_code == 0:
            st.success("Run finished: open the 'Results' tab (it is already selected).")
        else:
            errs = [r for r in es.lines if "ERRORE" in r or "ERROR" in r]
            if any("invalid sto" in r for r in errs):
                st.error("This pinn.exe is the old version, which does not know --config. Rebuild the C++ project:  "
                         "cmake -S . -B build -G \"Visual Studio 17 2022\" -A x64  and then  "
                         "cmake --build build --config Release  (in the C++_Progetto folder).")
            else:
                st.error("pinn.exe stopped with an error: " + (errs[-1] if errs else f"exit code {es.exit_code}"))
