import java.io.*;
import java.util.Scanner;

import background.Array;
import background.MiniBatch;

public class Net {

    static double[][] onesArray(int height, int width) {
        double[][] onesArray = new double[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                onesArray[i][j] = 1;
            }
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

    // checks to see if two arrays are addable
    // assumes arrays are rectangular
    static Boolean areArraysAddable(double[][] array1, double[][] array2) {
        // gets array height
        int height1 = array1.length;
        int height2 = array2.length;

        // gets array width
        // note: this assumes the row length is consistent
        int width1 = array1[0].length;
        int width2 = array2[0].length;

        // checks to see if the arrays are the same size
        if ((height1 == height2) && (width1 == width2)) {
            return true;
        } else {
            // error if the arrays aren't addable
            // System.out.println("arrays aren't addable");
            // printSize(array1, "a1");
            // printSize(array2, "a2");
            return false;
        }
    }

    // array addition, assumes arrays are are perfect rectangles
    static double[][] addArrays(double[][] a1, double[][] a2) {
        // get height and width of arrays
        int height = a1.length;
        int width = a1[0].length;

        // init output array
        double[][] arraySum = new double[height][width];

        // if arrays are addable, add them
        // otherwise return error message and exit
        if (areArraysAddable(a1, a2)) {
            // adds the arrays together
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    arraySum[i][j] = a1[i][j] + a2[i][j];
                }
            }
        } else {
            System.out.println("Cannot add arrays");
            System.exit(0);
        }
        return arraySum;
    }

    // subtract arrays
    static double[][] subtractArrays(double[][] a1, double[][] a2) {
        // get height and width of arrays
        int height = a1.length;
        int width = a1[0].length;

        // init output array
        double[][] arrayDiff = new double[height][width];

        // if arrays are addable, add them
        // otherwise return error message and exit
        if (areArraysAddable(a1, a2)) {
            // adds the arrays together
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    arrayDiff[i][j] = a1[i][j] - a2[i][j];
                }
            }
        } else {
            System.out.println("Cannot add arrays");
            System.exit(0);
        }
        return arrayDiff;
    }

    // scalar multiplication of elements in two arrays
    static double[][] multiplyScalar(double[][] a1, double[][] a2) {
        // initialize output array
        double[][] mulOutput = new double[a1.length][a1[0].length];

        // multiplication
        if (areArraysAddable(a1, a2)) {
            for (int i = 0; i < a1.length; i++) {
                for (int j = 0; j < a1[0].length; j++) {
                    mulOutput[i][j] = a1[i][j] * a2[i][j];
                }
            }
        } else {
            System.out.println("arrays can't be multiplied scalar-wise");
            System.exit(0);
        }
        return mulOutput;
    }

    // https://www.varsitytutors.com/hotmath/hotmath_help/topics/matrix-multiplication
    // dot product two arrays
    static double[][] dotProduct(double[][] a1, double[][] a2) {
        double[][] product = new double[a1.length][a2[0].length];

        // checks if rows of first matrix and columns of second match
        if (a1[0].length != a2.length) {
            // Array.print(a1, "a1");
            // Array.print(a2, "a2");
            System.out.println("cannot dot matrices");
            System.exit(0);
        }

        // math for dot product
        for (int i = 0; i < a1.length; i++) {
            for (int j = 0; j < a2.length; j++) {
                for (int k = 0; k < a2[0].length; k++) {
                    product[i][k] += a1[i][j] * a2[j][k];
                }
            }
        }
        return product;
    }

    static double[][] sigmoid(double[][] input, double[][] weight, double[][] bias) {
        double[][] sigmoidOut = new double[weight.length][bias[0].length];
        double[][] prod = dotProduct(weight, input);
        double[][] zValue = addArrays(prod, bias);

        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < bias[0].length; j++) {
                sigmoidOut[i][j] = (1 / (1 + Math.pow(Math.E, -(zValue[i][j]))));
            }
        }

        return sigmoidOut;
    }

    // final output layer math
    static double[][] backpropLast(double[][] output, double[][] trainData) {
        double[][] ones = onesArray(trainData.length, trainData[0].length);
        // should I use dot prod to make this simpler?
        return multiplyScalar(multiplyScalar(subtractArrays(output, trainData), output), subtractArrays(ones, output));
    }

    // GradientOfWeights
    static double[][] gradientOfWeights(double[][] gradientBiases, double[][] output) {
        return dotProduct(gradientBiases, transpose(output));
    }

    // everything that's not last
    static double[][] backpropRest(double[][] dLayer, double[][] weights, double[][] prevMyOut) {
        double[][] biasGradient = new double[prevMyOut.length][prevMyOut[0].length];
        for (int k = 0; k < prevMyOut[0].length; k++) {
            for (int j = 0; j < weights[0].length; j++) {
                // could I transpose dLayer to make it simpler and do the dot product
                // this feels kinda sketch
                biasGradient[j][k] += (weights[k][j] * dLayer[k][0] + weights[k + 1][j] * dLayer[k + 1][0])
                        * (prevMyOut[j][0] * (1 - prevMyOut[j][0]));
            }
        }
        return biasGradient;
    }

    static double[][] reviseBias(double[][] bias, double[][] hiddenInput1, double[][] hiddenInput2, double eta) {
        double[][] revisedBias = new double[bias.length][bias[0].length];
        for (int i = 0; i < bias[0].length; i++) {
            for (int j = 0; j < bias.length; j++) {

                revisedBias[j][i] = bias[j][i] - (eta / 2) * (hiddenInput1[j][i] + hiddenInput2[j][i]);
            }
        }
        return revisedBias;
    }

    static double[][] reviseWeights(double[][] originalWeights, double[][] gW1, double[][] gW2, double eta) {
        double[][] revisedWeights = new double[originalWeights.length][originalWeights[0].length];
        for (int i = 0; i < originalWeights[0].length; i++) {
            for (int j = 0; j < originalWeights.length; j++) {
                revisedWeights[j][i] = originalWeights[j][i] - (eta / 2) * (gW1[j][i] + gW2[j][i]);

            }
        }
        return revisedWeights;
    }

    // input training training weights and biases
    // return revised weights and biases
    static double[][][] miniBatch(double[][][] weights, double[][][] biases, double[][][] train) {
        /// Training Case 1
        double[][] myOut = sigmoid(train[0], weights[0], biases[0]);
        double[][] myOut2 = sigmoid(myOut, weights[1], biases[1]);
        double[][] dLayer2 = backpropLast(myOut2, train[1]);
        double[][] gradWeight2 = gradientOfWeights(dLayer2, myOut);
        double[][] dLayer1 = backpropRest(dLayer2, weights[1], myOut);
        double[][] gradWeight1 = gradientOfWeights(dLayer1, train[0]);

        /// Training Case 2
        double[][] my2Out = sigmoid(train[2], weights[0], biases[0]);
        double[][] my2Out2 = sigmoid(my2Out, weights[1], biases[1]);
        double[][] d2layer2 = backpropLast(my2Out2, train[3]);
        double[][] grad2Weight2 = gradientOfWeights(d2layer2, my2Out);
        double[][] d2Layer1 = backpropRest(d2layer2, weights[1], my2Out);
        double[][] grad2Weight1 = gradientOfWeights(d2Layer1, train[2]);

        // TODO: What is eta?
        double eta = 10;

        double[][] rW1 = reviseWeights(weights[0], gradWeight1, grad2Weight1, eta);
        double[][] rB1 = reviseBias(biases[0], dLayer1, d2Layer1, eta);
        /// layer 2 biases and weights
        double[][] rW2 = reviseWeights(weights[1], gradWeight2, grad2Weight2, eta);
        double[][] rB2 = reviseBias(biases[1], dLayer2, d2layer2, eta);

        double[][][] revisedWeightsBiases = { rW1, rB1, rW2, rB2 };

        // Array.print(rW1, "rw1");
        // Array.print(rB1, "rb1");
        // Array.print(rW2, "rw2");
        // Array.print(rB2, "rb2");

        return revisedWeightsBiases;
    }

    static double[][][][] miniBatch(double[][][] weights, double[][][] biases, double[][][][] batches) {
        int numBatches = batches.length;

        for (int i = 0; i < numBatches; i++) {
            System.out.println("Training Batch: " + i);
            double[][][] miniBatch = miniBatch(weights, biases, batches[i]);

            int mbLength = miniBatch.length / 2;

            double[][][] newWeights = new double[mbLength][0][0];
            double[][][] newBiases = new double[mbLength][0][0];

            int weightsTemp = 0;
            int biasesTemp = 0;

            for (int j = 0; j < miniBatch.length; j++) {
                if (j % 2 == 0) {
                    newWeights[weightsTemp] = miniBatch[j];
                    weightsTemp++;
                } else {
                    newBiases[biasesTemp] = miniBatch[j];
                    biasesTemp++;
                }
            }
            weights = newWeights;
            biases = newBiases;
        }
        double[][][][] wb = { weights, biases };
        return wb;
    }

    static void epoch(int numEpochs, double[][][] weights, double[][][] biases, double[][][][] batches) {
        System.out.println("Epoch " + numEpochs + ": ");
        for (int i = 0; i < numEpochs; i++) {
            double[][][][] mbOut = miniBatch(weights, biases, batches);
            weights = mbOut[0];
            biases = mbOut[1];
        }
        Array.print(weights[0], "weights");
        Array.print(weights[1], "weights");
        Array.print(biases[0], "biases");
        Array.print(biases[1], "biases");
    }

    static Scanner csvReader(String fileName) throws FileNotFoundException {
        // https://www.javatpoint.com/how-to-read-csv-file-in-java
        // pulls csv into java, prints it
        // filename: mnist_test.csv

        Scanner csvData = new Scanner(new File(fileName));
        csvData.useDelimiter(",");
        while (csvData.hasNext()) {
            System.out.print(csvData.next());
        }
        csvData.close();
        return csvData;
    }

    // function to combine a bunch of other functions
    // inputs training data
    // returns double[][] with
    // am I going to make a class system for this lol

    public static void main(String[] csv_file_name) {
        /* Epoch 1 */
        /// for both batches
        double[][] weight1 = { { -0.21, 0.72, -0.25, 1 }, { -0.94, -0.41, -0.47, 0.63 }, { 0.15, 0.55, -0.49, -0.75 } };
        double[][] weight2 = { { 0.76, 0.48, -0.73 }, { 0.34, 0.89, -0.23 } };

        double[][] bias1 = { { 0.1 }, { -0.36 }, { -0.31 } };
        double[][] bias2 = { { 0.16 }, { -0.46 } };

        // mini-batch #1
        /// training case 1
        double[][] xTrain1 = { { 0 }, { 1 }, { 0 }, { 1 } };
        double[][] yTrain1 = { { 0 }, { 1 } };
        /// training case 2
        double[][] xTrain2 = { { 1 }, { 0 }, { 1 }, { 0 } };
        double[][] yTrain2 = { { 1 }, { 0 } };

        // mini-batch #2
        /// training case 1
        double[][] x2Train1 = { { 0 }, { 0 }, { 1 }, { 1 } };
        double[][] y2Train1 = { { 0 }, { 1 } };
        /// training case 2
        double[][] x2Train2 = { { 1 }, { 1 }, { 0 }, { 0 } };
        double[][] y2Train2 = { { 1 }, { 0 } };

        // packing
        double[][][] batch1 = { xTrain1, yTrain1, xTrain2, yTrain2 };
        double[][][] batch2 = { x2Train1, y2Train1, x2Train2, y2Train2 };
        double[][][] weights = { weight1, weight2 };
        double[][][] biases = { bias1, bias2 };

        double[][][][] batches = { batch1, batch2 };

        // more packing

        /////////////////////////////////////////////////////////

        // double[][][][] mbOutput = miniBatch(weights, biases, batches);
        // double[][][] rw = mbOutput[0];
        // double[][][] rb = mbOutput[1];

        epoch(6, weights, biases, batches);

    }

}
