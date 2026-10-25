package it.univr.utils;

import java.util.function.DoubleUnaryOperator;





public class UsefulMethodsForArrays {
	
		/**
		 * It returns the sum of the elements of an array
		 * 
		 * @param array
		 * @return the sum of the elements of the array
		 */
		public static double getSum(double array[]) {
			double sum = 0.0;
			for(double value : array) {
				sum += value;
			}
			return sum;
		}

		
		/**
		 * It returns the average of the elements of an array
		 * 
		 * @param array
		 * @return the sum of the elements of the array
		 */
		public static double getAverage(double array[]) {
			double sum = 0.0;
			for(double value : array) {
				sum += value;
			}
			return sum/array.length;
		}
		/**
		 * It returns the product of two arrays
		 *
		 * @param firstArray
		 * @param secondArray
		 * @return an array representing the element-wise product of the two arrays
		 */
		public static double[] multArrays(double[] firstArray, double[] secondArray) {
			int firstLength = firstArray.length;
			if (firstLength != secondArray.length) {
				throw new IllegalArgumentException("Error: the two arrays must have same length!");
			}
			double[] product = new double[firstLength];
			for (int i = 0; i < firstLength; i++) {
				product[i] = firstArray[i] * secondArray[i];
			}
			return product;
		}

		/**
		 * It returns the sum of two arrays
		 *
		 * @param firstArray
		 * @param secondArray
		 * @return an array representing the element-wise sum of the two arrays
		 */
		public static double[] sumArrays(double[] firstArray, double[] secondArray) {
			int firstLength = firstArray.length;
			if (firstLength != secondArray.length) {
				throw new IllegalArgumentException("Error: the two arrays must have same length!");
			}

			double[] difference = new double[firstLength];
			for (int i = 0; i < firstLength; i++) {
				difference[i] = firstArray[i] + secondArray[i];
			}
			return difference;
		}

		/**
		 * It returns the difference of two arrays
		 *
		 * @param firstArray
		 * @param secondArray
		 * @return an array representing the element-wise difference of the two arrays
		 */
		public static double[] diffArrays(double[] firstArray, double[] secondArray) {
			int firstLength = firstArray.length;
			if (firstLength != secondArray.length) {
				throw new IllegalArgumentException("Error: the two arrays must have same length!");
			}

			double[] difference = new double[firstLength];
			for (int i = 0; i < firstLength; i++) {
				difference[i] = firstArray[i] - secondArray[i];
			}
			return difference;
		}

		/**
		 * It returns the element-wise ratio of two arrays
		 *
		 * @param firstArray
		 * @param secondArray
		 * @return an array representing the element-wise ratio of the two arrays
		 */
		public static double[] ratioArrays(double[] firstArray, double[] secondArray) {
			int firstLength = firstArray.length;
			if (firstLength != secondArray.length) {
				throw new IllegalArgumentException("Error: the two arrays must have same length!");
			}
			double[] ratio = new double[firstLength];
			for (int i = 0; i < firstLength; i++) {
				ratio[i] = firstArray[i] / secondArray[i];
			}
			return ratio;
		}

		/**
		 * It returns an array transformedArray where transformedArray[i] = f(originalArray[i]), with
		 * f given in input as a DoubleUnaryOperator.
		 *  
		 * @param originalArray, 
		 * @param transformFunction, the funtion f above
		 * @return  an array transformedArray where transformedArray[i] = f(originalArray[i])
		 */
		public static double[] applyFunctionToArray(double[] originalArray, DoubleUnaryOperator transformFunction) {
			int lengthOriginalArray = originalArray.length;
			double[] transformedArray = new double[lengthOriginalArray];
			for (int i = 0; i<lengthOriginalArray; i++) {
				//this is how DoubleUnaryOperator work: applyAsDouble is the method of the interface they implement
				//transformedArray[i] = f(originalArray[i) where f is the function represented by transformFunction
				transformedArray[i] = transformFunction.applyAsDouble(originalArray[i]);
			}
			return transformedArray;
		}


		/**
		 * It returns the scalar product of two arrays
		 *
		 * @param firstArray
		 * @param secondArray
		 * @return an array representing the element-wise product of the two arrays
		 */
		public static double getScalarProductTwoArrays(double[] firstArray, double[] secondArray) {
			double[] productOfArrays = multArrays(firstArray, secondArray);
			return getSum(productOfArrays);
		}
		
		
		/**
		 * It returns an array whose values are given by the maximum between the values of the arrays
		 *
		 * @param firstArray
		 * @param secondArray
		 * @return an array whose i-th element is max(firstArray[i], secondArray[i])
		 */
		public static double[] getMaxValuesBetweenTwoArrays(double[] firstArray, double[] secondArray) {
			int firstLength = firstArray.length;
			if (firstLength != secondArray.length) {
				throw new IllegalArgumentException("Error: the two arrays must have same length!");
			}
			double[] maxValues = new double[firstLength];
			for (int i = 0; i < firstLength; i++) {
				maxValues[i] = Math.max(firstArray[i], secondArray[i]);
			}
			return maxValues;
		}

		/**
		 * It returns the biggest element of a one-dimensional array of doubles
		 *
		 * @param vector the one-dimensional array
		 * @return the biggest element of the one-dimensional array
		 */
		public static double getMin(double[] vector) {
			double min = vector[0];
			for (int i = 1; i < vector.length; i++) {
				if (vector[i] < min) {
					min = vector[i];
				}
			}
			return min;
		}

		/**
		 * It returns the biggest element of a one-dimensional array of doubles
		 *
		 * @param vector the one-dimensional array
		 * @return the biggest element of the one-dimensional array
		 */
		public static double getMax(double[] vector) {
			double max = vector[0];
			for (int i = 1; i < vector.length; i++) {
				if (vector[i] > max) {
					max = vector[i];
				}
			}
			return max;
		}
		
		
		/*
		 * Prints a lower triangular matrix where each row i has (i + 1) elements.
		 * 
		 * The method assumes the input matrix represents a tree structure,
		 * such as one used in option pricing models. Each element is printed with
		 * 13 decimal places, and rows are prefixed with the corresponding time step.
		 *
		 */
		public static void printMatrix(double[][] matrix) {
		    for (int i = 0; i < matrix.length; i++) {
		    	System.out.print("t=" + i + ": ");
		        for (int j = 0; j <= i; j++) {
		        	System.out.printf("%10.13f ", matrix[i][j]);
		        }
		        System.out.println();
		    }
		}
		
}

