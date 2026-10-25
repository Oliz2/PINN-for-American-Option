package it.univr.montecarlo;

import java.util.SplittableRandom;

/**
 * Black-Scholes model with d correlated underlyings:
 *   log S_i(t + dt) = log S_i(t) + (r - q_i - 1/2 sum_k sigma_ik^2) dt + sqrt(dt) sum_k sigma_ik Z_k
 * with Z_1..Z_d independent standard normals (the covariance of the returns is sigma sigma^T).
 *
 * Paths are generated in antithetic pairs (Z and -Z), one exercise date at a time.
 * The scheme is exact for geometric Brownian motion: there is no discretisation error in the paths,
 * only in the exercise dates.
 *
 * @author Alberto Oliva Medin
 */
public class MultiAssetBlackScholes {

    private final int d;
    private final double[] initialValues;
    private final double[] drift;      // (r - q_i - 1/2 sum_k sigma_ik^2) dt
    private final double[][] vol;      // sigma_ik sqrt(dt)
    private final double dt;
    private final int numberOfDates;

    /**
     * @param initialValues  S_i(0), one per underlying.
     * @param riskFreeRate   r.
     * @param dividendYields q_i, one per underlying.
     * @param sigma          Volatility matrix d x d (same convention as the PINN).
     * @param maturity       T in years.
     * @param numberOfDates  Number of time steps (= exercise dates) between 0 and T.
     */
    public MultiAssetBlackScholes(double[] initialValues, double riskFreeRate, double[] dividendYields,
                                  double[][] sigma, double maturity, int numberOfDates) {
        this.d = initialValues.length;
        if (dividendYields.length != d || sigma.length != d)
            throw new IllegalArgumentException("Dividend yields and sigma matrix must have d = " + d + " rows");
        this.initialValues = initialValues.clone();
        this.numberOfDates = numberOfDates;
        this.dt = maturity / numberOfDates;
        this.drift = new double[d];
        this.vol = new double[d][d];
        for (int i = 0; i < d; i++) {
            if (sigma[i].length != d) throw new IllegalArgumentException("The sigma matrix must be d x d");
            double s2 = 0;   // variance of asset i: sum_k sigma_ik^2
            for (int k = 0; k < d; k++) { s2 += sigma[i][k] * sigma[i][k]; vol[i][k] = sigma[i][k] * Math.sqrt(dt); }
            drift[i] = (riskFreeRate - dividendYields[i] - 0.5 * s2) * dt;
        }
    }

    public int getNumberOfAssets() { return d; }
    public int getNumberOfDates() { return numberOfDates; }
    public double getTimeStep() { return dt; }

    /**
     * Generates an antithetic pair of paths: plus[j][i] and minus[j][i] = S_i at time j * dt.
     * All dates are always generated, so two methods with the same seed see the same paths.
     */
    public void antitheticPair(SplittableRandom random, double[][] plus, double[][] minus) {
        double[] lp = new double[d], lm = new double[d], z = new double[d];
        for (int i = 0; i < d; i++) {
            lp[i] = lm[i] = Math.log(initialValues[i]);
            plus[0][i] = minus[0][i] = initialValues[i];
        }
        for (int j = 1; j <= numberOfDates; j++) {
            // the same normals Z drive both paths, with opposite sign: the pair has lower variance
            for (int k = 0; k < d; k++) z[k] = Gaussian.next(random);
            for (int i = 0; i < d; i++) {
                double shock = 0;
                for (int k = 0; k < d; k++) shock += vol[i][k] * z[k];
                lp[i] += drift[i] + shock;
                lm[i] += drift[i] - shock;
                plus[j][i] = Math.exp(lp[i]);
                minus[j][i] = Math.exp(lm[i]);
            }
        }
    }

    /** All paths in memory (float to save memory): out[j][k * d + i]. */
    public float[][] simulateAll(int numberOfPaths, long seed) {
        if (numberOfPaths % 2 != 0) throw new IllegalArgumentException("The number of paths must be even");
        SplittableRandom random = new SplittableRandom(seed);
        float[][] out = new float[numberOfDates + 1][numberOfPaths * d];
        double[][] plus = new double[numberOfDates + 1][d], minus = new double[numberOfDates + 1][d];
        for (int k = 0; k < numberOfPaths; k += 2) {
            antitheticPair(random, plus, minus);
            for (int j = 0; j <= numberOfDates; j++)
                for (int i = 0; i < d; i++) {
                    out[j][k * d + i] = (float) plus[j][i];
                    out[j][(k + 1) * d + i] = (float) minus[j][i];
                }
        }
        return out;
    }

    /** Standard normal with Marsaglia's polar method (the second value is kept for the next call). */
    static final class Gaussian {
        private static double spare;
        private static boolean hasSpare = false;
        private static SplittableRandom owner = null;

        static double next(SplittableRandom random) {
            if (hasSpare && owner == random) { hasSpare = false; return spare; }
            double u, v, s;
            do {
                u = 2.0 * random.nextDouble() - 1.0;
                v = 2.0 * random.nextDouble() - 1.0;
                s = u * u + v * v;
            } while (s >= 1.0 || s == 0.0);
            double m = Math.sqrt(-2.0 * Math.log(s) / s);
            spare = v * m; hasSpare = true; owner = random;
            return u * m;
        }
    }
}
