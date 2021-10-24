
/*
Christopher Tullier
102-58-973
Assignment 2 - MNIST
RNN that learns how to recognize handwritten digits from the MNIST databse
*/
import java.io.*;
import java.util.Scanner;

import javax.naming.spi.DirStateFactory.Result;

import java.util.Arrays;
import java.util.Random;

import background.Array;
import background.MiniBatch;

public class Net {

    static double[] onesArray(int length) {
        double[] onesArray = new double[length];
        for (int i = 0; i < length; i++) {
            onesArray[i] = 1;
        }
        return onesArray;
    }

    // transposes a single array
    static double[][] transpose(double[][] a1) {

        double[][] newArray = new double[a1[0].length][a1.length];

        for (int i = 0; i < a1[0].length; i++) {
            for (int j = 0; j < a1.length; j++) {
                newArray[i][j] = a1[j][i];
            }
        }
        return newArray;
    }

    // https://www.varsitytutors.com/hotmath/hotmath_help/topics/matrix-multiplication
    // dot product two arrays
    static double[] dotProduct(double[][] weights, double[] input) {
        double[] product = new double[weights.length];

        // checks if rows of first matrix and columns of second match
        if (weights[0].length != input.length) {
            System.out.println("cannot dot matrices");
            System.exit(0);
        }

        // dotprod operation
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < input.length; j++) {
                product[i] += weights[i][j] * input[j];
            }
        }
        return product;

    }

    // only used with single dimension vectors
    // array addition, assumes arrays are are perfect rectangles
    static double[] addArrays(double[] a1, double[] a2) {
        // get height and width of arrays
        int len = a1.length;

        // init output array
        double[] arraySum = new double[len];

        // if arrays are addable, add them
        // otherwise return error message and exit
        if (a1.length == a2.length) {
            // adds the arrays together
            for (int i = 0; i < len; i++) {
                arraySum[i] = a1[i] + a2[i];
            }
        } else {
            System.out.println("Cannot add arrays");
            System.exit(0);
        }
        return arraySum;
    }

    // subtract arrays
    static double[] subtractArrays(double[] a1, double[] a2) {
        // get height and width of arrays
        int len = a1.length;

        // init output array
        double[] arrayDiff = new double[len];

        // if arrays are able to, subtract them
        // otherwise return error message and exit
        if (a1.length == a2.length) {
            // adds the arrays together
            for (int i = 0; i < len; i++) {
                arrayDiff[i] = a1[i] - a2[i];
            }
        } else {
            System.out.println("Cannot subtract arrays");
            System.exit(0);
        }
        return arrayDiff;
    }

    static double[] sigmoid(double[] zValue) {
        double[] sigmoidOut = new double[zValue.length];
        for (int i = 0; i < zValue.length; i++) {
            sigmoidOut[i] = (1 / (1 + Math.pow(Math.E, -(zValue[i]))));
        }
        return sigmoidOut;
    }

    // this is a forwardpass, predict is what it's called in tf
    // this input could be a Model class object
    // include all weights, all biases
    // include all training data as the input
    static double[][] predict(double[][][] weights, double[][] biases, double[][] inputs) {
        int lastBiasLength = biases[biases.length - 1].length;

        // initialize final output array
        // inputs.length is how many inputs we have
        // lastBiasLength is how many nodes are in the output (10 for mnist)
        double[][] Y = new double[inputs.length][lastBiasLength];

        // could also use weights.lenth could also be here
        int numLayers = biases.length;

        // loop over all inputs
        for (int i = 0; i < inputs.length; i++) {
            double[] currentInput = inputs[i];

            // loop over desired length of layers
            for (int j = 0; j < numLayers; j++) {
                double[] zValue = addArrays(dotProduct(weights[j], currentInput), biases[j]);
                currentInput = sigmoid(zValue);
            }
            Y[i] = currentInput;
        }
        return Y;
    }

    static void fit(int numEpochs, int learningRate, double[][][] weights, double[][] biases, double[][] inputs,
            double[][] yTrain) {
        int batchSize = 2;
        int Xindex;

        // pending changes
        double[][][] weightsChanges = new double[weights.length][weights[0].length][weights[0][0].length];
        double[][] biasesChanges = new double[biases.length][biases[0].length];

        double[][][] weightsCurrent = weights;
        double[][] biasesCurrent = biases;

        // loop until epochs are done
        for (int i = 0; i < numEpochs; i++) {
            int numBatches = inputs.length / batchSize;
            int lastLayerIndex = weightsChanges.length - 1;
            double[][] batchCurr = new double[batchSize][inputs[0].length];
            double[][] yTrainCurr = new double[batchSize][yTrain[0].length];

            int batchCounter = 0;
            batchCurr = Arrays.copyOfRange(inputs, batchCounter, batchCounter + batchSize);
            yTrainCurr = Arrays.copyOfRange(yTrain, batchCounter, batchCounter + batchSize);

            /// predict on batchSize
            double[][] yCurr = predict(weightsCurrent, biasesCurrent, batchCurr);
            batchCounter += batchSize;

            // last layer weights
            // Big X level
            for (int j = 0; j < batchCurr.length; j++) {
                // X[instance] level
                for (int u = 0; u < weightsCurrent[lastLayerIndex].length; u++) {
                    // neuron level (everyone shares layer term)
                    double errorTerm = yCurr[j][u] * (1 - yCurr[j][u]) * (yTrainCurr[j][u] - yCurr[j][u]);
                    for (int v = 0; v < weightsCurrent[lastLayerIndex][0].length; v++) {
                        // node level (aka weights level of a particular neuron)
                        weightsChanges[lastLayerIndex][u][v] += (learningRate * errorTerm * batchCurr[j][v]);
                    }
                }
            }
            // last layer biases
            for (int j = 0; j < batchCurr.length; j++) {
                // X[instance] level
                for (int u = 0; u < weightsCurrent[lastLayerIndex].length; u++) {
                    // neuron level (everyone shares layer term)
                    double errorTerm = yCurr[j][u] * (1 - yCurr[j][u]) * (yTrainCurr[j][u] - yCurr[j][u]);
                    for (int v = 0; v < weightsCurrent[lastLayerIndex][0].length; v++) {
                        // node level (aka weights level of a particular neuron)
                        weightsChanges[lastLayerIndex][u][v] += (learningRate * errorTerm * batchCurr[j][v]);
                    }
                }
            }

            /// Find gradient using J

            /// update

        }

    }

    public static void main(String[] csv_file_name) throws FileNotFoundException {

        /// for both batches
        double[][] weight1 = { { -0.21, 0.72, -0.25, 1 }, { -0.94, -0.41, -0.47, 0.63 }, { 0.15, 0.55, -0.49, -0.75 } };
        double[][] weight2 = { { 0.76, 0.48, -0.73 }, { 0.34, 0.89, -0.23 } };

        double[] bias1 = { 0.1, -0.36, -0.31 };
        double[] bias2 = { 0.16, -0.46 };

        // min-batch #1
        /// tr;aining case 1
        double[] x1 = { 0, 1, 0, 1 };
        double[] y1 = { 0, 1 };
        /// training case 2
        double[] x2 = { 1, 0, 1, 0 };
        double[] y2 = { 1, 0 };

        // mini-batch #2
        /// training case 1
        double[] x21 = { 0, 0, 1, 1 };
        double[] y21 = { 0, 1 };
        /// training case 2
        double[] x22 = { 1, 1, 0, 0 };
        double[] y22 = { 1, 0 };

        // packing
        double[][] inputs = { x1, x2, x21, x22 };
        double[][] outputs = { y1, y2, y21, y22 };
        double[][][] weights = { weight1, weight2 };
        double[][] biases = { bias1, bias2 };

        /////////////////////////////////////////////////////////
        // Arrays.toString() to print arrays

        // h is a in O'Neal's program
        // double[] h = sigmoid(addArrays(dotProduct(weights[0], inputs[0]),
        // biases[0]));

        // System.out.println(Arrays.toString(h));

        // forward pass

        double[][] finalOut = predict(weights, biases, inputs);

        for (int i = 0; i < inputs.length; i++) {
            System.out.print(Arrays.toString(finalOut[i]));
        }
    }
}
