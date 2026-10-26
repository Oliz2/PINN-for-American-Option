package it.univr.montecarlo;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Properties;

/**
 * Prices the American put on the minimum of d underlyings (driven by the Python interface).
 *
 * Usage:  java -cp out it.univr.montecarlo.MonteCarloApp config.properties output_folder
 *
 * Methods (key "methods", comma separated):
 *   lsm       Longstaff-Schwartz Monte Carlo
 *   brutal    brutal Monte Carlo (exercise as soon as in the money)
 *   binomial  Cox-Ross-Rubinstein binomial tree (d = 1 only)
 *
 * Files written to the output folder:
 *   mc_prices.csv        price, standard error, European price, exercise statistics
 *   mc_convergence.csv   Monte Carlo estimate as the number of paths grows
 *   mc_paths.csv         a few simulated paths (for the chart)
 *   mc_exercises.csv     exercise date of those paths for each Monte Carlo method
 *   mc_boundary.csv      exercise boundary (d = 1 only; Longstaff-Schwartz and binomial tree)
 *   mc_price_table.csv   prices on a grid of S and times (if requested)
 *
 * Lines printed on the standard output and read by runner_mc.py:
 *   TASK i/n | ...   progress,   PRICE | method | S0 | price (European ...)   one result,   DONE / ERROR: ...
 *
 * @author Alberto Oliva Medin
 */
public class MonteCarloApp {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);   // decimal point, not comma, in every printf
        try {
            if (args.length < 2) throw new IllegalArgumentException("Usage: MonteCarloApp config.properties output_folder");
            run(args[0], Path.of(args[1]));
            System.out.println("DONE");
        } catch (Exception e) {
            System.out.println("ERROR: " + e.getMessage());
            System.exit(1);
        }
    }

    /*
     * Reads config.properties (written by the interface), validates it, then prices every S0 with every
     * method, writing the CSV files listed in the class comment.
     */
    private static void run(String configPath, Path out) throws IOException {
        Properties p = new Properties();
        try (FileReader r = new FileReader(configPath)) { p.load(r); }
        Files.createDirectories(out);

        final int d = Integer.parseInt(req(p, "numberOfAssets"));
        final double K = Double.parseDouble(req(p, "strike"));
        final double T = Double.parseDouble(req(p, "maturity"));
        final double r = Double.parseDouble(req(p, "riskFreeRate"));
        final double[] q = vector(req(p, "dividendYields"));
        final double[][] sigma = matrix(req(p, "volatilityMatrix"));
        final double[][] spots = matrix(req(p, "spotPrices"));
        final double tVal = Double.parseDouble(req(p, "valuationTime"));
        final String[] methods = req(p, "methods").split(",");
        final int nPaths = 2 * (Integer.parseInt(req(p, "numberOfPaths")) / 2);
        final int nDates = Integer.parseInt(req(p, "exerciseDates"));
        final int degree = Integer.parseInt(req(p, "basisDegree"));
        final int treeSteps = Integer.parseInt(req(p, "treeSteps"));
        final long seed = Long.parseLong(req(p, "seed"));
        final int samplePaths = Integer.parseInt(req(p, "samplePaths"));
        final boolean table = Boolean.parseBoolean(req(p, "priceTable"));

        if (q.length != d || sigma.length != d) throw new IllegalArgumentException("Dividend yields and sigma must have d = " + d + " rows");
        for (double[] s0 : spots) if (s0.length != d) throw new IllegalArgumentException("Every S0 must have d = " + d + " components");
        if (tVal < 0 || tVal >= T) throw new IllegalArgumentException("The valuation time must be between 0 and T");
        for (String m : methods)
            if (m.equals("binomial") && d != 1) throw new IllegalArgumentException("The binomial tree is available only for d = 1");

        // ----- list of computations, for the progress bar -----
        List<double[]> tableS = new ArrayList<>();
        List<Double> tableT = new ArrayList<>();
        int tablePaths = 0;
        if (table) {
            double sMin = Double.parseDouble(req(p, "tableSMin")), sMax = Double.parseDouble(req(p, "tableSMax"));
            double step = Double.parseDouble(req(p, "tableSStep"));
            int nTimes = Integer.parseInt(req(p, "tableTimes"));
            tablePaths = 2 * (Integer.parseInt(req(p, "tablePaths")) / 2);
            for (double s = sMin; s <= sMax + 1e-9; s += step) { double[] v = new double[d]; Arrays.fill(v, s); tableS.add(v); }
            for (int k = 0; k < nTimes; k++) tableT.add(nTimes == 1 ? 0.0 : T * k / (nTimes - 1));
        }
        final int total = methods.length * (spots.length + tableS.size() * tableT.size());
        int done = 0;

        // the paths of the chart start from the S0 closest to the strike (where exercise and continuation both show)
        int ref = 0;
        for (int si = 1; si < spots.length; si++)
            if (Math.abs(K - minOf(spots[si])) < Math.abs(K - minOf(spots[ref]))) ref = si;

        // Valuation at tVal: the option has tau = T - tVal left; dates and tree steps are scaled accordingly
        final double tau = T - tVal;
        final int datesMain = Math.max(1, (int) Math.round(nDates * tau / T));
        final int stepsMain = Math.max(1, (int) Math.round(treeSteps * tau / T));

        try (PrintWriter prices = new PrintWriter(out.resolve("mc_prices.csv").toFile());
             PrintWriter conv = new PrintWriter(out.resolve("mc_convergence.csv").toFile());
             PrintWriter stops = new PrintWriter(out.resolve("mc_exercises.csv").toFile());
             PrintWriter front = new PrintWriter(out.resolve("mc_boundary.csv").toFile())) {
            prices.println("method,t," + header(d) + ",price,std_error,european,exercised_fraction,mean_exercise_time,seconds");
            conv.println("method,spot," + header(d) + ",paths,price");
            stops.println("method,path,exercise_index,exercise_time");
            front.println("method,t,B_t");

            for (String method : methods) {
                for (int si = 0; si < spots.length; si++) {
                    double[] s0 = spots[si];
                    System.out.printf("TASK %d/%d | %s | S0 = %s | t = %.4f%n", ++done, total, name(method), label(s0), tVal);
                    MultiAssetBlackScholes model = new MultiAssetBlackScholes(s0, r, q, sigma, tau, datesMain);
                    PricingResult res = method.equals("binomial")
                            ? new BinomialTree(stepsMain).price(s0, K, r, q, sigma, tau)
                            : price(method, model, s0, K, r, nPaths, seed, degree, samplePaths);
                    prices.printf("%s,%.6f,%s,%.8f,%.8f,%.8f,%.6f,%.6f,%.3f%n", method, tVal, csv(s0), res.price,
                            res.standardError, res.europeanPrice, res.exercisedFraction, res.meanExerciseTime, res.seconds);
                    for (double[] c : res.convergence)
                        conv.printf("%s,%d,%s,%d,%.8f%n", method, si, csv(s0), (long) c[0], c[1]);
                    System.out.printf("PRICE | %s | S0 = %s | %.6f (European %.6f) | %.1f s%n",
                            name(method), label(s0), res.price, res.europeanPrice, res.seconds);
                    // paths and boundary: only for the reference S0, from which the paths written below start
                    if (si == ref) {
                        for (int k = 0; k < res.sampleStops.length; k++)
                            stops.printf("%s,%d,%d,%.6f%n", method, k, res.sampleStops[k],
                                    res.sampleStops[k] < 0 ? Double.NaN : tVal + res.sampleStops[k] * model.getTimeStep());
                        if (res.boundary != null)
                            for (int j = 0; j < res.boundary.length; j++)
                                if (!Double.isNaN(res.boundary[j]) && (j > 0 || method.equals("binomial")))
                                    front.printf("%s,%.6f,%.6f%n", method, tVal + j * res.boundaryTimeStep, res.boundary[j]);
                    }
                }
            }
        }

        // ----- a few paths from the reference S0 (same seed as the pricing pass of both Monte Carlo methods) -----
        writeSamplePaths(out, spots[ref], r, q, sigma, tau, datesMain, seed, samplePaths, tVal);

        // ----- Price Table -----
        if (table) {
            try (PrintWriter pt = new PrintWriter(out.resolve("mc_price_table.csv").toFile())) {
                pt.println("method,t,S,price,std_error");
                for (String method : methods)
                    for (double[] s0 : tableS)
                        for (double t : tableT) {
                            System.out.printf("TASK %d/%d | %s | Price Table S = %.4f | t = %.4f%n", ++done, total,
                                    name(method), s0[0], t);
                            double left = T - t;
                            double pr, se;
                            if (left < 1e-12) { pr = PricingResult.putOnMin(K, s0); se = 0; }
                            else if (method.equals("binomial")) {
                                int ns = Math.max(1, (int) Math.round(treeSteps * left / T));
                                pr = new BinomialTree(ns).price(s0, K, r, q, sigma, left).price; se = 0;
                            } else {
                                int nd = Math.max(1, (int) Math.round(nDates * left / T));
                                MultiAssetBlackScholes m = new MultiAssetBlackScholes(s0, r, q, sigma, left, nd);
                                PricingResult res = price(method, m, s0, K, r, tablePaths, seed, degree, 0);
                                pr = res.price; se = res.standardError;
                            }
                            pt.printf("%s,%.6f,%.6f,%.8f,%.8f%n", method, t, s0[0], pr, se);
                        }
            }
        }
    }

    // Monte Carlo methods. The brutal one uses seed + 1_000_003, i.e. the same pricing paths as pass 2 of LSM.
    private static PricingResult price(String method, MultiAssetBlackScholes model, double[] s0, double K, double r,
                                       int nPaths, long seed, int degree, int samplePaths) {
        switch (method) {
            case "lsm":    return new LongstaffSchwartzMultiAsset(degree).price(model, s0, K, r, nPaths, seed, samplePaths);
            case "brutal": return new BrutalMonteCarlo().price(model, s0, K, r, nPaths, seed + 1_000_003L, samplePaths);
            default: throw new IllegalArgumentException("Unknown method: " + method + " (use lsm, brutal or binomial)");
        }
    }

    // mc_paths.csv: the first n paths of the pricing pass, so that the chart shows the paths the methods priced.
    private static void writeSamplePaths(Path out, double[] s0, double r, double[] q, double[][] sigma, double tau,
                                         int nDates, long seed, int n, double tVal) throws IOException {
        MultiAssetBlackScholes model = new MultiAssetBlackScholes(s0, r, q, sigma, tau, nDates);
        java.util.SplittableRandom random = new java.util.SplittableRandom(seed + 1_000_003L);
        int d = s0.length;
        double[][] plus = new double[nDates + 1][d], minus = new double[nDates + 1][d];
        try (PrintWriter w = new PrintWriter(out.resolve("mc_paths.csv").toFile())) {
            w.println("path,t," + header(d) + ",S_min");
            for (int k = 0; k < n; k += 2) {
                model.antitheticPair(random, plus, minus);
                for (int side = 0; side < 2 && k + side < n; side++) {
                    double[][] path = side == 0 ? plus : minus;
                    for (int j = 0; j <= nDates; j++)
                        w.printf("%d,%.6f,%s,%.6f%n", k + side, tVal + j * model.getTimeStep(), csv(path[j]), minOf(path[j]));
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Parsing and formatting helpers.
    // Required property (error naming the missing key).
    private static String req(Properties p, String key) {
        String v = p.getProperty(key);
        if (v == null || v.isBlank()) throw new IllegalArgumentException("Configuration: missing field '" + key + "'");
        return v.trim();
    }
    // "a,b,c" -> {a, b, c}
    private static double[] vector(String s) {
        String[] a = s.split(",");
        double[] v = new double[a.length];
        for (int i = 0; i < a.length; i++) v[i] = Double.parseDouble(a[i].trim());
        return v;
    }
    // "a,b;c,d" -> {{a, b}, {c, d}} (rows separated by semicolons)
    private static double[][] matrix(String s) {
        String[] rows = s.split(";");
        double[][] m = new double[rows.length][];
        for (int i = 0; i < rows.length; i++) m[i] = vector(rows[i]);
        return m;
    }
    private static double minOf(double[] v) {
        double m = Double.MAX_VALUE;
        for (double x : v) m = Math.min(m, x);
        return m;
    }
    private static String header(int d) {
        StringBuilder b = new StringBuilder();
        for (int i = 1; i <= d; i++) b.append(i > 1 ? "," : "").append("S").append(i);
        return b.toString();
    }
    private static String csv(double[] v) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < v.length; i++) b.append(i > 0 ? "," : "").append(String.format("%.6f", v[i]));
        return b.toString();
    }
    private static String label(double[] v) {
        StringBuilder b = new StringBuilder("(");
        for (int i = 0; i < v.length; i++) b.append(i > 0 ? "; " : "").append(String.format("%.2f", v[i]));
        return b.append(")").toString();
    }
    private static String name(String method) {
        switch (method) {
            case "lsm": return "Longstaff-Schwartz";
            case "brutal": return "Brutal Monte Carlo";
            case "binomial": return "Binomial tree";
            default: return method;
        }
    }
}
