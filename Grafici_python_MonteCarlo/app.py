"""Monte Carlo and binomial tree interface for American options.

Start (from the VS Code terminal, in the Grafici_python_MonteCarlo folder):
    python -m streamlit run app.py

The page writes a config.properties file with the chosen parameters, compiles (the first time) and runs
the Java pricer in the java/ folder, and shows progress, prices and charts. Every run gets its own
folder in runs/, which stays available in the "Results" tab after the page is closed.

The prices are computed by the Java program (Longstaff-Schwartz, binomial tree, brutal Monte Carlo);
Python only prepares the configuration, follows the progress and draws tables and charts. The Results tab
can also load a run of the PINN interface with the same option and compare the prices.

Author: Alberto Oliva Medin
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts_mc as charts
import results_mc as results
import runner_mc as runner
import settings_mc as cfg

HERE = Path(__file__).resolve().parent
RUNS_DEFAULT = HERE / "runs"
# Results of the PINN interface, next to this folder: used by the PINN vs Monte Carlo comparison
PINN_RUNS_DEFAULT = HERE.parent / "Grafici_python_PINN" / "runs"

st.set_page_config(page_title="Monte Carlo - American options", page_icon="🎲", layout="wide")


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
NAMES = cfg.SHORT_NAMES

# ---------------------------------------------------------------------------------------------
# Sidebar: Java and folders
# ---------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    if "jdk" not in st.session_state:
        st.session_state.jdk = runner.find_jdk()
    jdk = st.text_input("JDK bin folder (where javac is)", key="jdk",
                        help=r"For example C:\Program Files\Java\jdk-23\bin. Empty = use java and javac from the PATH.")
    if jdk and not Path(jdk).is_dir():
        st.error("JDK folder not found.")
    elif st.button("Compile the Java pricer"):
        try:
            st.success(runner.compile_java(jdk))
        except Exception as e:
            st.error(str(e))
    else:
        st.caption("Compilation happens automatically on the first run.")
    memory = st.number_input("Maximum Java memory (MB)", 256, 65536, 2048, step=256)
    runs_folder = Path(st.text_input("Results folder", str(RUNS_DEFAULT), key="runs"))
    pinn_folder = Path(st.text_input("PINN results (for the comparison)", str(PINN_RUNS_DEFAULT), key="runs_pinn"))
    st.caption("Every run creates a subfolder with the configuration, log.txt and all the CSV files.")

st.title("Monte Carlo and binomial tree for American options")
st.caption("American put on the minimum of d underlyings in the Black-Scholes model, priced with "
           "Longstaff-Schwartz Monte Carlo, a binomial tree or the brutal Monte Carlo, and compared with the PINN.")

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
    st.caption("Each row is an initial price S0 = (S1, ..., Sd). Add or remove rows at the bottom of the table. "
               "The paths in the chart start from the row closest to the strike. "
               "Use the same S0 as in the PINN run to compare the two.")
    levels = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    spot_df = st.data_editor(pd.DataFrame([[round(x * K, 4)] * d for x in levels], columns=s_names),
                             key=f"spot_{d}_{K}", num_rows="dynamic", **WIDE)
    t_val = st.number_input("Valuation time t (0 = today)", min_value=0.0, max_value=float(T) * 0.999,
                            value=0.0, step=0.1)

    # ---------------- 4) Methods ----------------
    st.subheader("4. Pricing methods")
    available = [m for m in cfg.METHODS if m != "binomial" or d == 1]
    methods = st.multiselect("Methods", available, default=["lsm", "binomial"] if d == 1 else ["lsm"],
                             format_func=lambda k: cfg.METHODS[k], key=f"methods_{min(d, 2)}",
                             help="Longstaff-Schwartz: regression of the continuation value on a polynomial basis, "
                                  "then pricing on new paths. Binomial tree: Cox-Ross-Rubinstein tree, one underlying only. "
                                  "Brutal: exercise at the first date the option is in the money "
                                  "(naive rule, underprices the option).")
    if d > 1:
        st.caption("The binomial tree is used only with one underlying (d = 1).")
    c1, c2, c3, c4 = st.columns(4)
    n_paths = c1.number_input("Monte Carlo paths", 100, 50_000_000, 100_000, step=10_000)
    n_dates = c2.number_input("Exercise dates up to T", 1, 100_000, 300, step=50,
                              help="Equally spaced dates between 0 and T. More dates = closer to continuous exercise.")
    degree = c3.number_input("Polynomial basis degree (Longstaff-Schwartz)", 1, 10, 4 if d == 1 else 3,
                             key=f"degree_{d}", disabled="lsm" not in methods)
    seed = c4.number_input("Seed", 0, 2**31 - 1, 3141, step=1)
    c1, c2, c3, _ = st.columns(4)
    tree_steps = c1.number_input("Binomial tree steps up to T", 10, 20_000, 2000, step=100,
                                 key="tree_steps", disabled="binomial" not in methods)
    n_show = c2.number_input("Paths to draw", 0, 1000, 40, step=10)
    # number of monomials of total degree <= degree in d variables: C(d + degree, d)
    c3.metric("Basis functions", math.comb(d + int(degree), d))

    with st.expander("Price Table (prices on a grid of S and times)"):
        pt_on = st.checkbox("Also compute the Price Table", value=False,
                            help="With d > 1 the grid uses S1 = ... = Sd = S.")
        a1, a2, a3, a4, a5 = st.columns(5)
        pt_smin = a1.number_input("S from", 0.01, 1e6, 0.8 * K, step=0.5, key=f"ptmin_{K}")
        pt_smax = a2.number_input("S to", 0.01, 1e6, 1.2 * K, step=0.5, key=f"ptmax_{K}")
        pt_step = a3.number_input("S step", 0.001, 1e6, 0.5, step=0.1)
        pt_times = a4.number_input("Number of times (from 0 to T)", 1, 151, 4)
        pt_paths = a5.number_input("Monte Carlo paths per point", 100, 50_000_000, 20_000, step=10_000)
        n_points = int(np.floor((pt_smax - pt_smin) / pt_step + 1e-9)) + 1 if pt_smax >= pt_smin else 0
        st.caption(f"{n_points} values of S × {int(pt_times)} times × {max(len(methods), 1)} methods = "
                   f"{n_points * int(pt_times) * max(len(methods), 1)} prices.")

    # ---------------- Start ----------------
    spots = spot_df.dropna(how="all").to_numpy(dtype=float).tolist()
    config = cfg.build_config(
        d=d, strike=K, maturity=T, r=r, dividends=q_df.iloc[0].tolist(), sigma=sigma_df.to_numpy(dtype=float).tolist(),
        spots=spots, valuation_time=t_val, methods=methods, paths=n_paths, exercise_dates=n_dates, degree=degree,
        tree_steps=tree_steps, seed=seed, sample_paths=n_show, price_table=pt_on, table_smin=pt_smin,
        table_smax=pt_smax, table_step=pt_step, table_times=pt_times, table_paths=pt_paths)

    # Longstaff-Schwartz stores paths x dates x d floats: warn before Java runs out of memory
    mem = cfg.estimated_memory_mb(config)
    if "lsm" in methods and mem > 0.8 * memory:
        st.warning(f"Longstaff-Schwartz keeps the paths in memory: about {mem:,.0f} MB are needed, more than the "
                   f"memory given to Java ({memory} MB). Reduce paths or dates, or raise the memory.")

    st.divider()
    running = "run" in st.session_state and st.session_state.run.running
    c1, c2, _ = st.columns([1, 1, 3])
    go_clicked = c1.button("▶ Start", type="primary", disabled=running)
    c2.button("■ Stop", disabled=not running, on_click=lambda: st.session_state.run.stop())

    if go_clicked:
        errors = cfg.check(config)
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                st.session_state.run = runner.start(jdk, config, runs_folder, memory)
                st.session_state.chosen_run = st.session_state.run.folder.name
            except FileNotFoundError:
                st.error("Java not found: enter the JDK bin folder in the sidebar.")
            except Exception as e:
                st.error(str(e))

    panel = st.container()

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
        name = st.selectbox("Run", run_names, index=run_names.index(chosen) if chosen in run_names else 0)
        run = results.load(runs_folder / name)
        cf = run.config
        Kr, Tr, tv = cf["strike"], cf["maturity"], cf["valuationTime"]
        method_names = {m: NAMES.get(m, m) for m in cf["methods"]}

        c = st.columns(6)
        c[0].metric("Underlyings d", run.d)
        c[1].metric("Strike K", f"{Kr:g}")
        c[2].metric("Maturity T", f"{Tr:g}")
        c[3].metric("Monte Carlo paths", f"{cf['numberOfPaths']:,}")
        c[4].metric("Exercise dates", cf["exerciseDates"])
        c[5].metric("Seed", cf["seed"])
        details = [f"r = {cf['riskFreeRate']:g}", f"t = {tv:g}"]
        if "lsm" in cf["methods"]:
            details.append(f"basis degree {cf['basisDegree']}")
        if "binomial" in cf["methods"]:
            details.append(f"tree steps {cf.get('treeSteps', '?')}")
        st.markdown(f"**Methods:** {', '.join(method_names.values())} · " + " · ".join(details))
        if not run.finished():
            st.warning("This run is not finished (running, stopped or failed).")

        prices = run.prices()
        summary = None
        if prices is not None:
            st.subheader("American option price")
            summary = results.price_summary(prices, run.d, Kr, method_names)
            st.dataframe(summary.style.format({c: "{:.6f}" for c in summary.columns if not c.startswith("S")}),
                         **WIDE, hide_index=True)
            with st.expander("Details: exercise statistics and computing times"):
                det = prices.drop(columns=["std_error"]).assign(method=prices["method"].map(lambda m: NAMES.get(m, m)))
                st.dataframe(det.style.format({c: "{:.6f}" for c in det.columns
                                               if c != "method" and not c.startswith("S")}, na_rep="—"),
                             **WIDE, hide_index=True)
                st.caption("exercised_fraction = share of Monte Carlo paths exercised before maturity; "
                           "mean_exercise_time = mean time of early exercise. Not defined for the tree (—).")
            st.plotly_chart(charts.prices_by_spot(summary, run.d, {v: k for k, v in method_names.items()}), **WIDE)
            if "brutal" in cf["methods"]:
                st.info("The brutal Monte Carlo exercises at the first date the option is in the money: it gives up the "
                        "waiting value and can therefore end up even below the European put. Longstaff-Schwartz and "
                        "the binomial tree compare the payoff with the continuation value instead.")

        # ---------------- PINN vs Monte Carlo ----------------
        comp = None
        if summary is not None:
            st.subheader("PINN vs Monte Carlo and binomial tree")
            candidates = results.pinn_runs(pinn_folder)
            if not candidates:
                st.caption(f"No PINN run with prices found in {pinn_folder}.")
            else:
                cfgs = {p.name: json.loads((p / "config.json").read_text(encoding="utf-8")) for p in candidates}
                matching = [p.name for p in candidates if results.same_option(cfgs[p.name], cf)]
                only_matching = st.checkbox("Show only PINN runs with the same option and market", value=bool(matching))
                options = matching if only_matching else [p.name for p in candidates]
                if not options:
                    st.info("No PINN run prices the same option (same d, K, T, r, q and σ). "
                            "Untick the box above to choose one anyway.")
                else:
                    pinn_name = st.selectbox("PINN run", options, index=0,
                                             format_func=lambda n: n + ("" if n in matching else "  (different option)"))
                    loaded = results.pinn_prices(pinn_folder / pinn_name)
                    if loaded is not None:
                        pinn_cfg, pinn_df = loaded
                        if pinn_name not in matching:
                            st.warning("This PINN run prices a different option or market: the comparison is not meaningful.")
                        comp = results.comparison(summary, pinn_df, run.d, tv, list(method_names.values()))
                        if comp.empty:
                            st.info("No S0 in common between the two runs at the same time t. "
                                    "Run the Monte Carlo with the same S0 as the PINN.")
                        else:
                            st.caption(f"PINN: {pinn_cfg.get('training', {}).get('epochs', '?')} epochs, seeds "
                                       f"{', '.join(map(str, pinn_cfg.get('seeds', [])))}"
                                       " (with several seeds the PINN price is their mean).")
                            fmt = {c: "{:.6f}" for c in comp.columns if not c.startswith("S")}
                            fmt.update({c: "{:+.3f}%" for c in comp.columns if c.endswith("(%)")})
                            fmt.update({c: "{:+.6f}" for c in comp.columns if c.startswith("PINN - ") and not c.endswith("(%)")})
                            st.dataframe(comp.style.format(fmt, na_rep="—"), **WIDE, hide_index=True)
                            series = {"PINN": "PINN", **{v: k for k, v in method_names.items()}}
                            st.plotly_chart(charts.prices_by_spot(comp, run.d, series), **WIDE)
                            st.plotly_chart(charts.differences(comp, run.d, method_names), **WIDE)
                            if "binomial" in cf["methods"]:
                                st.caption("The binomial tree has no simulation noise: it is the most precise reference "
                                           "here. Longstaff-Schwartz is slightly biased low (estimated, sub-optimal "
                                           "exercise rule).")

        # ---------------- Price Table ----------------
        pt = run.price_table()
        if pt is not None:
            st.subheader("Price Table")
            for method in pt["method"].unique():
                mat = results.price_table_matrix(pt, method)
                st.markdown(f"**{NAMES.get(method, method)}**")
                times = sorted(pt["t"].unique())
                subtabs = st.tabs(["All times"] + [f"t = {t:.2f}" for t in times])
                with subtabs[0]:
                    st.caption("Rows: underlying price S. Columns: time t. Values: option price.")
                    st.dataframe(shaded(mat), **WIDE)
                for sub, t in zip(subtabs[1:], times):
                    with sub:
                        g = pt[(pt["method"] == method) & np.isclose(pt["t"], t)][["S", "price"]].copy()
                        g["Payoff"] = np.maximum(Kr - g["S"], 0)
                        g["Time value"] = g["price"] - g["Payoff"]
                        st.dataframe(g.rename(columns={"price": "Price"}).style.format("{:.6f}").format({"S": "{:.2f}"}),
                                     hide_index=True, **WIDE)
            st.plotly_chart(charts.price_table(pt, Kr, NAMES), **WIDE)

        # ---------------- Charts ----------------
        st.subheader("Charts")
        g_conv, g_paths, g_bound = st.tabs(["Convergence", "Paths and exercise", "Exercise boundary"])
        conv = run.convergence()
        bound = run.boundary()
        with g_conv:
            if conv is None or prices is None:
                st.info("Convergence is available only for the Monte Carlo methods.")
            else:
                s_cols = results.spot_columns(run.d)
                spots_c = conv.drop_duplicates("spot")
                labels = {int(rw["spot"]): "S0 = (" + "; ".join(f"{rw[c]:g}" for c in s_cols) + ")"
                          for _, rw in spots_c.iterrows()}
                which = st.selectbox("S0", list(labels), format_func=labels.get)
                ref = None
                if "binomial" in prices["method"].values:
                    row = spots_c[spots_c["spot"] == which][s_cols].iloc[0]
                    tree = prices[(prices["method"] == "binomial")]
                    hit = tree[np.all([np.isclose(tree[c], row[c]) for c in s_cols], axis=0)]
                    ref = float(hit["price"].iloc[0]) if len(hit) else None
                st.plotly_chart(charts.convergence(conv, which, NAMES, ref), **WIDE)
        with g_paths:
            paths_df = run.paths()
            mc_methods = [m for m in cf["methods"] if m != "binomial"]
            if paths_df is None or not mc_methods:
                st.info("No paths saved (only the tree was run, or paths to draw = 0).")
            else:
                m_shown = st.radio("Exercise rule to show", mc_methods, horizontal=True, format_func=lambda m: NAMES.get(m, m))
                b_shown = None
                if bound is not None and run.d == 1:
                    b_shown = bound[bound["method"].isin(["binomial"] + (["lsm"] if m_shown == "lsm" else []))]
                st.plotly_chart(charts.paths(paths_df, run.exercises(), m_shown, Kr, b_shown, NAMES), **WIDE)
                st.caption("Paths of the minimum of the underlyings, starting from the S0 closest to the strike, on the "
                           "same sample for both Monte Carlo methods. The orange dot is the date at which the rule exercises.")
        with g_bound:
            if run.d != 1:
                st.info("The exercise boundary as a curve B(t) exists only for d = 1.")
            elif bound is None:
                st.info("The boundary is available with Longstaff-Schwartz or the binomial tree.")
            else:
                st.plotly_chart(charts.boundary(bound, Kr, NAMES), **WIDE)
                st.caption("Binomial tree: highest tree node below K where exercise is optimal (the small steps come "
                           "from the discrete grid of nodes). Longstaff-Schwartz: imprecise in the first instants, "
                           "when few paths have moved away from S0.")

# =============================================================================================
# PROGRESS OF THE RUNNING COMPUTATION
# =============================================================================================
with panel:
    es: runner.Run | None = st.session_state.get("run")
    if es is not None:
        status = st.empty()
        bar = st.progress(0.0)
        table = st.empty()
        log_tail = st.empty()

        def draw() -> None:
            elapsed = time.time() - es.start
            p = es.progress()
            eta = f", about {elapsed * (1 - p) / p / 60:.1f} min left" if 0.02 < p < 1 else ""
            status.markdown(f"**{es.folder.name}** · computation {es.done}/{es.total} · {es.current} · "
                            f"{elapsed:.0f} s{eta}")
            bar.progress(p)
            snapshot = list(es.prices)       # the reader thread may append while drawing
            if snapshot:
                table.dataframe(pd.DataFrame(snapshot, columns=["Method", "S0", "Price", "European put"])
                                .style.format({"Price": "{:.6f}", "European put": "{:.6f}"}),
                                hide_index=True, **WIDE)
            log_tail.code("\n".join(es.lines[-10:]) or "...", language=None)

        while es.running:
            draw()
            time.sleep(0.5)
        time.sleep(0.3)
        draw()
        if es.stopped:
            st.warning("Run stopped.")
        elif es.exit_code == 0:
            if st.session_state.get("refreshed") != es.folder.name:     # redraw the Results tab
                st.session_state.refreshed = es.folder.name
                st.rerun()
            st.success("Run finished: open the 'Results' tab (it is already selected).")
        else:
            if any("OutOfMemory" in r for r in es.lines):
                st.error("Java ran out of memory: raise 'Maximum Java memory' or reduce paths/dates.")
            else:
                errs = [r for r in es.lines if "ERROR" in r or "Exception" in r or "Error" in r]
                st.error("The pricer stopped with an error: " + (errs[-1] if errs else f"exit code {es.exit_code}"))
