package it.univr.montecarlo;

/**
 * "Brutal" Monte Carlo: a naive exercise rule that exercises at the first date on which the option is
 * in the money (payoff > 0), without comparing the payoff with the continuation value.
 *
 * It is wrong on purpose: it gives up the time value of the option and therefore underprices the
 * American option (often even below the European put). It is an example of what not to do.
 * If S0 is already in the money the rule exercises immediately and the price is the payoff.
 *
 * @author Alberto Oliva Medin
 */
public class BrutalMonteCarlo extends ForwardPricer {

    @Override
    protected boolean exercise(int j, double[] s, double ex) {
        return true;   // ex > 0 is already checked: exercise as soon as in the money
    }

    /** Prices the option with the naive rule (same simulation as Longstaff-Schwartz). */
    public PricingResult price(MultiAssetBlackScholes model, double[] s0, double strike, double riskFreeRate,
                               int numberOfPaths, long seed, int samplePaths) {
        long start = System.nanoTime();
        double payoff0 = PricingResult.putOnMin(strike, s0);
        PricingResult r = price(model, strike, riskFreeRate, numberOfPaths, seed, samplePaths);
        if (payoff0 > 0) {      // already in the money today: the rule exercises at t = 0
            r.price = payoff0; r.standardError = 0.0; r.exercisedFraction = 1.0; r.meanExerciseTime = 0.0;
            java.util.Arrays.fill(r.sampleStops, 0);
            r.convergence.clear();
        }
        r.seconds = (System.nanoTime() - start) / 1e9;
        return r;
    }
}
