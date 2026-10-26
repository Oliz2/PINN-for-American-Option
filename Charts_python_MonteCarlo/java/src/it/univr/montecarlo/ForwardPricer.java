package it.univr.montecarlo;

import java.util.SplittableRandom;

/**
 * Forward pass shared by the two Monte Carlo methods: antithetic pairs of paths are simulated and at each
 * date a rule decides whether to exercise. The price is the mean of the discounted cash flows; the
 * standard error is computed on the mean of each pair (independent samples). The European put (no early
 * exercise) is computed on the same paths; it is a lower bound for the American price.
 *
 * @author Alberto Oliva Medin
 */
abstract class ForwardPricer {

    /** true if at this date (index j, underlyings s, payoff ex > 0) the rule exercises. */
    protected abstract boolean exercise(int j, double[] s, double ex);

    protected PricingResult price(MultiAssetBlackScholes model, double strike, double riskFreeRate,
                                  int numberOfPaths, long seed, int samplePaths) {
        final int N = model.getNumberOfDates(), d = model.getNumberOfAssets();
        final double dt = model.getTimeStep();
        SplittableRandom random = new SplittableRandom(seed);
        double[][] plus = new double[N + 1][d], minus = new double[N + 1][d];
        PricingResult res = new PricingResult();
        res.sampleStops = new int[Math.min(samplePaths, numberOfPaths)];

        double sum = 0, sumSq = 0, eu = 0, euSq = 0, exercisedCount = 0, exerciseTimeSum = 0;
        int pairs = numberOfPaths / 2, nextCheckpoint = 8;
        for (int p = 0; p < pairs; p++) {
            model.antitheticPair(random, plus, minus);
            double pair = 0, pairEu = 0;
            for (int side = 0; side < 2; side++) {
                double[][] path = side == 0 ? plus : minus;
                int k = 2 * p + side;
                int stop = -1;
                double cash = 0;
                for (int j = 1; j < N; j++) {
                    double ex = PricingResult.putOnMin(strike, path[j]);
                    if (ex > 0 && exercise(j, path[j], ex)) { stop = j; cash = ex * Math.exp(-riskFreeRate * j * dt); break; }
                }
                double atMaturity = PricingResult.putOnMin(strike, path[N]) * Math.exp(-riskFreeRate * N * dt);
                if (stop < 0) cash = atMaturity;
                else { exercisedCount++; exerciseTimeSum += stop * dt; }
                if (k < res.sampleStops.length) res.sampleStops[k] = stop;
                pair += 0.5 * cash;
                pairEu += 0.5 * atMaturity;
            }
            sum += pair; sumSq += pair * pair; eu += pairEu; euSq += pairEu * pairEu;
            int n = p + 1;
            if (n == nextCheckpoint || n == pairs) {
                res.convergence.add(new double[]{2.0 * n, sum / n});
                nextCheckpoint = (int) Math.ceil(nextCheckpoint * 1.5);
            }
        }
        double m = sum / pairs, me = eu / pairs;
        res.price = m;
        res.standardError = Math.sqrt(Math.max(sumSq / pairs - m * m, 0) / pairs);
        res.europeanPrice = me;
        res.europeanStandardError = Math.sqrt(Math.max(euSq / pairs - me * me, 0) / pairs);
        res.exercisedFraction = exercisedCount / numberOfPaths;
        res.meanExerciseTime = exercisedCount > 0 ? exerciseTimeSum / exercisedCount : Double.NaN;
        res.boundaryTimeStep = dt;
        return res;
    }

    /** At t = 0 one can always exercise immediately: the American price is at least the payoff. */
    static void exerciseAtZero(PricingResult r, double payoff0) {
        if (payoff0 > r.price) {
            r.price = payoff0;
            r.standardError = 0.0;
            r.exercisedFraction = 1.0;          // every path exercises immediately
            r.meanExerciseTime = 0.0;
            java.util.Arrays.fill(r.sampleStops, 0);
        }
    }
}
