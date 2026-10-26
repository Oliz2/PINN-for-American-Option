# Monte Carlo and binomial tree interface for American options

American put on the minimum of d underlyings (Black-Scholes), priced with:

- **Longstaff-Schwartz Monte Carlo**: regression of the continuation value on a polynomial basis
  (all monomials of total degree ≤ the chosen degree, in standardised variables), then pricing on
  new, independent paths;
- **Binomial tree**: Cox-Ross-Rubinstein tree, one underlying only (d = 1);
- **Brutal Monte Carlo**: exercise at the first date the option is in the money. A naive rule that
  throws away the waiting value and therefore underprices the option, often below the European put.

One, two or all three methods can be selected. The **Results** tab compares the American prices with a
PINN run from `../Charts_python_PINN/runs` (same option and market, same S0 and time t).

## Start

In the VS Code terminal, inside this folder:

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

A JDK (11 or newer) is needed. The page looks for `javac` by itself (PATH and `C:\Program Files\Java\jdk*`);
if it does not find it, enter the JDK `bin` folder in the sidebar, for example `C:\Program Files\Java\jdk-23\bin`.
The Java sources (`java/src/it/univr/montecarlo`) are compiled to `java/out` on the first run, and again
only when they change.

## Files of each run (folder `runs/<date>_d<d>_<methods>`)

| File | Content |
|---|---|
| `config.json`, `config.properties` | chosen parameters |
| `log.txt` | output of the Java program |
| `mc_prices.csv` | price, standard error, European price, exercised fraction, mean exercise time, seconds |
| `mc_convergence.csv` | Monte Carlo estimate as the number of paths grows |
| `mc_paths.csv`, `mc_exercises.csv` | paths of the chart and exercise date of each |
| `mc_boundary.csv` | exercise boundary B(t), Longstaff-Schwartz and binomial tree (d = 1 only) |
| `mc_price_table.csv` | Price Table on a grid of S and times (if requested) |

## Notes

- Longstaff-Schwartz is slightly biased low (the estimated exercise rule is not the optimal one), and
  with a finite number of exercise dates the price is the one of a Bermudan option.
- The binomial tree has no simulation noise and is the most precise reference for d = 1.
- Longstaff-Schwartz keeps all training paths in memory: paths × (dates + 1) × d × 4 bytes.
  If Java runs out of memory, raise the value in the sidebar.
