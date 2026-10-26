package it.univr.BinomialModel;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import it.univr.utils.*;

public class BinomialModel {
	
	//features of the asset
	private double initialValue;
	private double riskFreeRate;
	private double sigma;
	private double[][] valuesProbabilities;
	private double[][] values;

	//internal storage
	private double upFactor;
	private double downFactor;
	private double riskNeutralProbabilityUp;
	private double riskNeutralProbabilityDown;

	//time discretization
	private int numberOfTimes;
	private double timeHorizon;

	
	
	public BinomialModel(double initialValue, double riskFreeRate, double sigma, int numberOfTimes, double timeHorizon) {
		this.initialValue = initialValue;
		this.riskFreeRate = riskFreeRate;
		this.sigma = sigma;
		this.timeHorizon = timeHorizon;
		this.upFactor = Math.exp(this.sigma*Math.sqrt(timeHorizon/numberOfTimes));
		this.downFactor = 1.0 / this.upFactor;
		this.riskNeutralProbabilityUp = (Math.exp(riskFreeRate * timeHorizon/numberOfTimes) - downFactor)
				/ (upFactor - downFactor);
		this.riskNeutralProbabilityDown = 1.0 - this.riskNeutralProbabilityUp;
		this.numberOfTimes = numberOfTimes;
	}
	
	private void generateValues() {
		values = new double[numberOfTimes][numberOfTimes];
		values[0][0] = initialValue;
		int numberOfDowns;//it will be updated in the for loop
		for (int numberOfMovements = 1; numberOfMovements < numberOfTimes; numberOfMovements++) {
			for (int numberOfUps = 0; numberOfUps <= numberOfMovements; numberOfUps++) {
				numberOfDowns=numberOfMovements-numberOfUps;
				/*
				 * Value of the binomial model when it went numberOfUps times up and numberOfDowns times down.
				 * Note that this is stored in position numberOfDowns! So the first position has all ups and so on
				 */
				values[numberOfMovements][numberOfDowns] = values[0][0] * Math.pow(upFactor, numberOfUps)*
						Math.pow(downFactor, numberOfDowns);
			}
		}
	}
	
	private void generateValuesProbabilities() {
		valuesProbabilities = new double[numberOfTimes][numberOfTimes];
		valuesProbabilities[0][0] = 1;//the first value is deterministic
		int numberOfDowns;//it will be updated in the for loop
		for (int numberOfMovements = 1; numberOfMovements < numberOfTimes; numberOfMovements++) {
			/*
			 * Here we have to take care of the computation of the binomial coefficients. 
			 * We are at time n and start the "internal" for loop with the case when we have k=0 ups.
			 * So we first have binomialCoefficient(n,k)=n!/(k!(n-k)!)=1 
			 */
			double binomialCoefficient = 1;
			for (int numberOfUps = 0; numberOfUps <= numberOfMovements; numberOfUps++) {
				numberOfDowns=numberOfMovements-numberOfUps;
				/*
				 * Probability of having B(0)u^numberOfUps*d^numberOfDowns.
				 * Note that this is stored in position numberOfDowns! So the first position has all ups and so on
				 */
				valuesProbabilities[numberOfMovements][numberOfDowns]
						= binomialCoefficient*Math.pow(riskNeutralProbabilityUp, numberOfUps)
						* Math.pow(riskNeutralProbabilityDown, numberOfDowns);
				/*
				 * Here we update the value of the binomial coeffeicient computing the one
				 * that we will use next, i.e., when we will have one more up: so, if k is
				 * the actual number of ups, we have to compute 
				 * binomialCoefficient(n,k+1)=n!/((k+1)!(n-k-1)!)=n!/(k!(n-k)!)*(n-k)/(k+1).
				 * Since n!/(k!(n-k)!) is the last computed value, we multiply by (n-k) 
				 * (so, by numberOfDowns) and divide by k+1, so, by the current number of ups plus 1.
				 */
				binomialCoefficient = binomialCoefficient * (numberOfDowns)/(numberOfUps+1);
			}
		}
	}
	
	public double[] getValuesAtGivenTimeIndex(int timeIndex) {
		/*
		 * Pay attention: the method generateValues() initializes the array values and sets it. This is
		 * of course needed if we want to get those values. However, we want to do that only once!
		 * So we check if values is null (this means "not yet initialized") and call the method only
		 * in this case. 
		 */
		if (values == null) {
			generateValues();
		}	
		/*
		 * We only return the first timeIndex entries! The others are zero, because the process can take
		 * only timeIndex values at time index timeIndex.
		 */
		return Arrays.copyOfRange(values[timeIndex], 0, timeIndex+1);
	}
	
	public double[] getTransformedValuesAtGivenTimeIndex(int timeIndex, DoubleUnaryOperator transformFunction) {
		//the possible values of the binomial model
		double[] valuesAtGivenTimeIndex = getValuesAtGivenTimeIndex(timeIndex);
		//we return the function applied to this array
		return UsefulMethodsForArrays.applyFunctionToArray(valuesAtGivenTimeIndex, transformFunction);
	}
	
	public double[] getValuesProbabilitiesAtGivenTimeIndex(int timeIndex) {
		/*
		 * Pay attention: the method generateValues() initializes the array valuesProbabilities and sets it.
		 * This is of course needed if we want to get those values. However, we want to do that only once!
		 * So we check if valuesProbabilities is null (this means "not yet initialized") and call the method only
		 * in this case. 
		 */
		if (valuesProbabilities == null) {
			generateValuesProbabilities();
		}
		/*
		 * We only return the first timeIndex entries! The others are zero, because the process can take
		 * only timeIndex values at time index timeIndex.
		 */
		return Arrays.copyOfRange(valuesProbabilities[timeIndex], 0, timeIndex + 1);
	}
	
	public double[] getUpAndDownProbabilities() {
		double[] probabilities = {riskNeutralProbabilityUp,riskNeutralProbabilityDown};
		return probabilities;
	}

	/**
	 * It returns an array representing the discounted conditional expectations at given timeIndex of the
	 * values of (possibly a function of) a binomial model at time timeIndex+1. 
	 * 
	 * @param values, values of (possibly a function of) a binomial model at time timeIndex+1
	 * @param timeIndex, the time index
	 * @return the array of the discounted conditional expectations at timeIndex of binomialValues. 
	 * 			The i-th element is the conditional expectation computed in the case when the underlying
	 * 			has gone down i times.
	 */
	public double[] getConditionalExpectation(double[] values, int timeIndex) {
		//at timeIndex we have timeIndex + 1 values
		double[] conditionalExpectation = new double[timeIndex+1];
		for (int i = 0; i <= timeIndex; i++) {
			/*
			 * computation of the conditional probability at the state with i down. Note that the i-th element
			 * of binomialValues has gone up, because the number of down is still i. 
			 */
			conditionalExpectation[i] = (values[i]*riskNeutralProbabilityUp + values[i + 1]*riskNeutralProbabilityDown)
					/Math.exp(this.riskFreeRate * this.timeHorizon / this.numberOfTimes);
		}
		return conditionalExpectation;
	}
	
	/*
	 * It returns a matrix with the entire process of the stock values
	 */
	public double[][] getValues () {
		if (values == null) {
			generateValues();
		}
		return values;
	}
	
	/*
	 * It returns a matrix with the entire process of the probability for every stock values
	 */
	public double[][] getProbabilities () {
		if (valuesProbabilities == null) {
			generateValuesProbabilities();
		}
		return valuesProbabilities;
	}
	
}