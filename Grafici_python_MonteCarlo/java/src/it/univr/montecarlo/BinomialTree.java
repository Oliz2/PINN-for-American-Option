package it.univr.montecarlo;

/**
 * Cox-Ross-Rubinstein binomial tree for the American put on one underlying (Black-Scholes model).
 *
 *   u = exp(v sqrt(dt)), d = 1/u, p = (exp((r - q) dt) - d) / (u - d)
 *
 * Backward induction: V = max(payoff, discounted expected value). The European price comes from the
 * same tree without early exercise; the tree also gives the exercise boundary.
 * The tree is used only with one underlying (d = 1).
 *
 * @author Alberto Oliva Medin
 */
public class BinomialTree {

    private final int steps;

    /** @param steps Number of time steps of the tree (e.g. 2000). */
    public BinomialTree(int steps) {
        if (steps < 1) throw new IllegalArgumentException("The tree needs at least 1 step");
        this.steps = steps;
    }

    /**
     * Prices the American and the European put at S0.
     *
     * @param tau Time to maturity T - t.
     * @return Prices, exercise boundary and computation time.
     */
    public PricingResult price(double[] s0, double strike, double r, double[] q, double[][] sigma, double tau) {
        if (s0.length != 1) throw new IllegalArgumentException("The binomial tree is available only for d = 1");
        long start = System.nanoTime();
        PricingResult res = price1d(s0[0], strike, r, q[0], vol(sigma), tau);
        res.seconds = (System.nanoTime() - start) / 1e9;
        return res;
    }

    // Volatility of the single asset from the 1 x 1 sigma "matrix" (norm of the first row).
    private static double vol(double[][] sigma) {
        double s2 = 0;
        for (double x : sigma[0]) s2 += x * x;
        return Math.sqrt(s2);
    }

    /*
     * Node i at step n has price S = s0 u^(2i - n). Starting from the payoff at maturity, every step back
     * computes the continuation value (discounted risk-neutral expectation of the two children) and,
     * for the American option, takes the maximum with the payoff. The highest exercised price at each step
     * is the exercise boundary.
     */
    private PricingResult price1d(double s0, double K, double r, double q, double v, double tau) {
        final int N = steps;
        final double dt = tau / N, u = Math.exp(v * Math.sqrt(dt)), dn = 1 / u, disc = Math.exp(-r * dt);
        final double p = (Math.exp((r - q) * dt) - dn) / (u - dn);
        if (!(p > 0 && p < 1))
            throw new IllegalArgumentException("Binomial tree: probability outside (0, 1); increase the number of steps");
        double[] am = new double[N + 1], eu = new double[N + 1];
        for (int i = 0; i <= N; i++) am[i] = eu[i] = Math.max(K - s0 * Math.pow(u, 2 * i - N), 0);
        double[] boundary = new double[N + 1];
        boundary[N] = K;
        for (int n = N - 1; n >= 0; n--) {
            double highest = Double.NaN;
            for (int i = 0; i <= n; i++) {
                double s = s0 * Math.pow(u, 2 * i - n);
                double cont = disc * (p * am[i + 1] + (1 - p) * am[i]);
                double ex = Math.max(K - s, 0);
                if (ex > 0 && ex >= cont) { am[i] = ex; highest = s; }
                else am[i] = cont;
                eu[i] = disc * (p * eu[i + 1] + (1 - p) * eu[i]);
            }
            boundary[n] = highest;
        }
        PricingResult res = new PricingResult();
        res.price = am[0];
        res.europeanPrice = eu[0];
        res.boundary = boundary;
        res.boundaryTimeStep = dt;
        return res;
    }
}
