package it.univr.monteCarloproducts;

import org.apache.commons.math3.random.MersenneTwister;

/**
 * Generates pseudo-random uniform numbers using the Mersenne Twister engine and
 * transforms them into standard normal random variables via Moro's Inverse
 * Transform approximation.
 * 
 */
public class MonteCarloGenerator {

	private final int numberOfSimulation;
	private final MersenneTwister casualNumber;

	// Containers for uniform and standard normal generated values
	private double[] uniformNumber;
	private double[] valuesNormalNumber;

	// Coefficients for Moro's approximation in the central region (0.08 < u < 0.92)
	private static final double A0 = 2.50662823884;
	private static final double A1 = -18.61500062529;
	private static final double A2 = 41.39119773534;
	private static final double A3 = -25.44106049637;
	private static final double B0 = -8.47351093090;
	private static final double B1 = 23.08336743743;
	private static final double B2 = -21.06224101826;
	private static final double B3 = 3.13082909833;

	// Coefficients for Moro's approximation in the tail regions (u <= 0.08 or u >=
	// 0.92)
	private static final double C0 = 0.3374754822726147;
	private static final double C1 = 0.9761690190917186;
	private static final double C2 = 0.1607979714918209;
	private static final double C3 = 0.0276438810333863;
	private static final double C4 = 0.0038405729373609;
	private static final double C5 = 0.0003951896511919;
	private static final double C6 = 0.0000321767881768;
	private static final double C7 = 0.0000002888167364;
	private static final double C8 = 0.00000003960315187;

	/**
	 * Constructs a Monte Carlo random generator for a specified number of samples.
	 * 
	 * @param numberOfSimulation Total number of random values to generate.
	 */
	public MonteCarloGenerator(int numberOfSimulation) {
		this.numberOfSimulation = numberOfSimulation;
		this.casualNumber = new MersenneTwister();
	}

	/**
	 * Generates pseudo-random uniform numbers in the open interval (0, 1) using the
	 * Mersenne Twister engine.
	 * 
	 * @return An array of uniformly distributed numbers.
	 */
	public double[] generateNumberWithMersenneTwister() {
		uniformNumber = new double[numberOfSimulation];

		long max32Bit = 4294967295L; // Maximum 32-bit unsigned integer value of the classical Mersenne Twister

		for (int simulation = 0; simulation < numberOfSimulation; simulation++) {

			double randInt = Integer.toUnsignedLong(casualNumber.nextInt());
			double values = (randInt + 0.5) / (max32Bit + 1.0);
			uniformNumber[simulation] = values;
		}

		return uniformNumber;
	}

	/**
	 * Applies Moro's Inverse Transform algorithm to transform uniform random
	 * numbers into standard normally distributed random variables.
	 * 
	 * @return An array of standard normally distributed numbers.
	 */
	public double[] generateNormalNumber() {
		// Lazy evaluation: ensures uniform random numbers are generated before
		// transformation
		if (uniformNumber == null) {
			generateNumberWithMersenneTwister();
		}

		valuesNormalNumber = new double[numberOfSimulation];
		double normalNumber = 0;

		for (int simulation = 0; simulation < numberOfSimulation; simulation++) {
			double u = uniformNumber[simulation];

			// Central region approximation
			if (u < 0.92 && u > 0.08) {
				double y = u - 0.5;
				double x = y * y;

				// Polynomial evaluation (optimized to avoid multiple Math.pow calls)
				double numerator = A0 + A1 * x + A2 * (x * x) + A3 * (x * x * x);
				double denominator = 1 + B0 * x + B1 * (x * x) + B2 * (x * x * x) + B3 * (x * x * x * x);

				normalNumber = y * (numerator / denominator);

			}
			// Tail regions approximation
			else {
				// Left tail transformation
				if (u <= 0.5) {
					double p = Math.log(-Math.log(u));

					normalNumber = -(C0 + C1 * p + C2 * Math.pow(p, 2) + C3 * Math.pow(p, 3) + C4 * Math.pow(p, 4)
							+ C5 * Math.pow(p, 5) + C6 * Math.pow(p, 6) + C7 * Math.pow(p, 7) + C8 * Math.pow(p, 8));
				}
				// Right tail transformation
				else {
					double p = Math.log(-Math.log(1.0 - u));

					normalNumber = C0 + C1 * p + C2 * Math.pow(p, 2) + C3 * Math.pow(p, 3) + C4 * Math.pow(p, 4)
							+ C5 * Math.pow(p, 5) + C6 * Math.pow(p, 6) + C7 * Math.pow(p, 7) + C8 * Math.pow(p, 8);
				}
			}

			valuesNormalNumber[simulation] = normalNumber;
		}

		return valuesNormalNumber;
	}

	/**
	 * Retrieves the generated uniform random numbers.
	 * 
	 * @return Array of uniform values.
	 */
	public double[] getUniformValue() {
		return uniformNumber;
	}

	/**
	 * Retrieves the generated standard normal random numbers.
	 * 
	 * @return Array of normal values.
	 */
	public double[] getValuesNormalNumber() {
		return valuesNormalNumber;
	}

	/**
	 * Returns the total number of simulations configured.
	 * 
	 * @return Number of simulations.
	 */
	public int getNumberOfSimulation() {
		return numberOfSimulation;
	}
}