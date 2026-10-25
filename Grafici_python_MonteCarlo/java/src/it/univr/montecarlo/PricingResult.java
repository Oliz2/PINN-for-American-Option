package it.univr.montecarlo;

import java.util.ArrayList;
import java.util.List;

/**
 * Result of one pricing run, with the European price and exercise statistics.
 * The fields are written to the CSV files read by the Python interface (MonteCarloApp).
 *
 * @author Alberto Oliva Medin
 */
public class PricingResult {
    public double price, standardError;                 // American price and its Monte Carlo standard error
    public double europeanPrice, europeanStandardError; // European put on the same paths (lower bound)
    public double exercisedFraction = Double.NaN;   // paths exercised before maturity (Monte Carlo only)
    public double meanExerciseTime = Double.NaN;    // mean exercise time among exercised paths
    public double seconds;                          // computation time
    public double[] boundary;                       // exercise boundary B(t_j) (d = 1 only)
    public double boundaryTimeStep;                 // time step of the boundary grid
    /** Convergence: number of paths, estimate. */
    public final List<double[]> convergence = new ArrayList<>();
    /** For the first paths: exercise date index, or -1 if not exercised before maturity. */
    public int[] sampleStops = new int[0];

    /** Payoff of the put on the minimum of the underlyings: (K - min_i S_i)^+. */
    public static double putOnMin(double strike, double[] s) {
        double m = s[0];
        for (int i = 1; i < s.length; i++) m = Math.min(m, s[i]);
        return Math.max(strike - m, 0.0);
    }
}
