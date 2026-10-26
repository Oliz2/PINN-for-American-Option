package it.univr.montecarlo;

import java.util.ArrayList;
import java.util.List;

/**
 * Longstaff-Schwartz (2001) for the American put on the minimum of d underlyings.
 *
 * Pass 1 (rule estimation): on "training" paths one goes backward in time; at each date, among the
 * in-the-money paths, the discounted future cash flow is regressed on a polynomial basis in the
 * standardised variables x_i = (S_i - m_i) / s_i (all monomials of total degree <= basisDegree);
 * exercise where payoff >= estimated continuation value.
 *
 * Pass 2 (pricing): the estimated rule is applied to new, independent paths. The resulting price is
 * biased low (sub-optimal rule) but does not "look into the future".
 *
 * With d = 1 the rule also gives the exercise boundary B(t_j).
 *
 * Reference: F. A. Longstaff and E. S. Schwartz, "Valuing American options by simulation: a simple
 * least-squares approach", The Review of Financial Studies 14(1), 2001.
 *
 * @author Alberto Oliva Medin
 */
public class LongstaffSchwartzMultiAsset extends ForwardPricer {

    private final int basisDegree;              // maximum total degree of the polynomial basis
    private int d;                              // number of underlyings
    private List<int[]> monomials;              // exponents of the basis functions
    private double[][] beta, centers, scales;   // per date j: regression coefficients, means and std devs

    /** @param basisDegree Maximum total degree of the regression polynomials (e.g. 3 or 4). */
    public LongstaffSchwartzMultiAsset(int basisDegree) {
        this.basisDegree = basisDegree;
    }

    /**
     * Pass 1 on paths generated with `seed`, pass 2 on new paths (seed + 1_000_003), then the check of
     * immediate exercise at t = 0 and, for d = 1, the exercise boundary.
     */
    public PricingResult price(MultiAssetBlackScholes model, double[] s0, double strike, double riskFreeRate,
                               int numberOfPaths, long seed, int samplePaths) {
        long start = System.nanoTime();
        estimateRule(model, strike, riskFreeRate, numberOfPaths, seed);
        PricingResult r = price(model, strike, riskFreeRate, numberOfPaths, seed + 1_000_003L, samplePaths);
        exerciseAtZero(r, PricingResult.putOnMin(strike, s0));
        if (d == 1) { r.boundary = boundary(model, strike); r.boundaryTimeStep = model.getTimeStep(); }
        r.seconds = (System.nanoTime() - start) / 1e9;
        return r;
    }

    // Exercise rule of pass 2: exercise if the payoff is at least the estimated continuation value.
    @Override
    protected boolean exercise(int j, double[] s, double ex) {
        return beta[j] != null && ex >= continuation(j, s);
    }

    // ---------------------------------------------------------------------------------------------
    /*
     * Pass 1: backward induction on the training paths. cash[k] is the cash flow of path k under the rule
     * estimated so far, discounted to the current date. At each date j:
     *  1. standardise the prices of the in-the-money paths (mean and std dev per asset);
     *  2. least squares: regress cash on the basis functions phi(x) -> coefficients beta[j];
     *  3. where payoff >= estimated continuation value, the path is exercised: cash[k] = payoff.
     * Only in-the-money paths enter the regression, as in the original paper.
     */
    private void estimateRule(MultiAssetBlackScholes model, double strike, double riskFreeRate,
                              int numberOfPaths, long seed) {
        d = model.getNumberOfAssets();
        final int N = model.getNumberOfDates();
        final double disc = Math.exp(-riskFreeRate * model.getTimeStep());
        monomials = monomials(d, basisDegree);
        final int m = monomials.size();
        beta = new double[N + 1][];
        centers = new double[N + 1][];
        scales = new double[N + 1][];

        float[][] paths = model.simulateAll(numberOfPaths, seed);
        // At maturity every path receives its payoff
        double[] cash = new double[numberOfPaths];
        double[] s = new double[d];
        for (int k = 0; k < numberOfPaths; k++) cash[k] = PricingResult.putOnMin(strike, read(paths[N], k, s));

        for (int j = N - 1; j >= 1; j--) {
            for (int k = 0; k < numberOfPaths; k++) cash[k] *= disc;
            // mean and standard deviation of each S_i over the in-the-money paths
            double[] mean = new double[d], sq = new double[d];
            int itm = 0;
            for (int k = 0; k < numberOfPaths; k++) {
                read(paths[j], k, s);
                if (PricingResult.putOnMin(strike, s) <= 0) continue;
                for (int i = 0; i < d; i++) { mean[i] += s[i]; sq[i] += s[i] * s[i]; }
                itm++;
            }
            if (itm <= 2 * m) continue;          // too few paths: no exercise at this date
            double[] sd = new double[d];
            boolean degenerate = false;
            for (int i = 0; i < d; i++) {
                mean[i] /= itm;
                sd[i] = Math.sqrt(Math.max(sq[i] / itm - mean[i] * mean[i], 0));
                if (sd[i] < 1e-12) degenerate = true;
            }
            if (degenerate) continue;
            centers[j] = mean; scales[j] = sd;

            // Normal equations (A^T A) beta = A^T cash, with A[k][a] = phi_a(x_k) on the in-the-money paths
            double[][] ata = new double[m][m];
            double[] atb = new double[m];
            double[] phi = new double[m];
            for (int k = 0; k < numberOfPaths; k++) {
                read(paths[j], k, s);
                if (PricingResult.putOnMin(strike, s) <= 0) continue;
                basis(j, s, phi);
                for (int a = 0; a < m; a++) {
                    atb[a] += phi[a] * cash[k];
                    for (int b = a; b < m; b++) ata[a][b] += phi[a] * phi[b];
                }
            }
            for (int a = 0; a < m; a++) for (int b = 0; b < a; b++) ata[a][b] = ata[b][a];
            beta[j] = solve(ata, atb);
            // Update the cash flows with the rule of date j
            for (int k = 0; k < numberOfPaths; k++) {
                read(paths[j], k, s);
                double ex = PricingResult.putOnMin(strike, s);
                if (ex > 0 && ex >= continuation(j, s)) cash[k] = ex;
            }
        }
    }

    /** Boundary for d = 1: the largest S below the strike, within the range covered by the paths, where one exercises. */
    private double[] boundary(MultiAssetBlackScholes model, double strike) {
        final int N = model.getNumberOfDates();
        double[] b = new double[N + 1];
        b[N] = strike;
        double[] s = new double[1];
        for (int j = 0; j < N; j++) {
            b[j] = Double.NaN;
            if (beta[j] == null) continue;
            double lo = Math.max(1e-9, centers[j][0] - 3 * scales[j][0]);
            double hi = Math.min(strike, centers[j][0] + 3 * scales[j][0]);
            double found = Double.NaN;
            for (int g = 0; g <= 2000; g++) {
                s[0] = lo + (hi - lo) * g / 2000.0;
                double ex = PricingResult.putOnMin(strike, s);
                if (ex > 0 && ex >= continuation(j, s)) found = s[0];
            }
            b[j] = found;
        }
        return b;
    }

    // Prices of path k at one date (the paths are stored as float[date][path * d + asset]).
    private static double[] read(float[] row, int k, double[] s) {
        for (int i = 0; i < s.length; i++) s[i] = row[k * s.length + i];
        return s;
    }

    // Estimated continuation value at date j: sum_a beta_a phi_a(x).
    private double continuation(int j, double[] s) {
        double[] phi = new double[monomials.size()];
        basis(j, s, phi);
        double c = 0;
        for (int a = 0; a < phi.length; a++) c += beta[j][a] * phi[a];
        return c;
    }

    // Basis functions: monomials of the standardised prices x_i = (S_i - mean_i) / std_i.
    private void basis(int j, double[] s, double[] phi) {
        double[] x = new double[d];
        for (int i = 0; i < d; i++) x[i] = (s[i] - centers[j][i]) / scales[j][i];
        for (int a = 0; a < phi.length; a++) {
            int[] e = monomials.get(a);
            double v = 1;
            for (int i = 0; i < d; i++) for (int p = 0; p < e[i]; p++) v *= x[i];
            phi[a] = v;
        }
    }

    /** All exponents (e_1..e_d) with sum <= degree: 1, x_i, x_i x_k, x_i^2, ... */
    static List<int[]> monomials(int d, int degree) {
        List<int[]> out = new ArrayList<>();
        build(new int[d], 0, degree, out);
        return out;
    }

    // Recursive enumeration: exponent of asset i from 0 to the remaining degree.
    private static void build(int[] e, int i, int left, List<int[]> out) {
        if (i == e.length) { out.add(e.clone()); return; }
        for (int p = 0; p <= left; p++) { e[i] = p; build(e, i + 1, left - p, out); }
        e[i] = 0;
    }

    /** Normal equations with a small regularisation, Gaussian elimination with partial pivoting. */
    private static double[] solve(double[][] A, double[] b) {
        int n = b.length;
        double trace = 0;
        for (int i = 0; i < n; i++) trace += A[i][i];
        double[][] M = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, M[i], 0, n);
            M[i][i] += 1e-10 * trace / n;
            M[i][n] = b[i];
        }
        for (int c = 0; c < n; c++) {
            int p = c;
            for (int r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
            double[] t = M[c]; M[c] = M[p]; M[p] = t;
            for (int r = c + 1; r < n; r++) {
                double f = M[r][c] / M[c][c];
                for (int k = c; k <= n; k++) M[r][k] -= f * M[c][k];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double v = M[i][n];
            for (int k = i + 1; k < n; k++) v -= M[i][k] * x[k];
            x[i] = v / M[i][i];
        }
        return x;
    }
}
