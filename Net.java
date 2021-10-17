import java.io.*;
import java.util.Scanner;
import mB.MiniBatch;
import mB.Array;

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
    // TODO: move transpose from main to here
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

    static double[][] revisedBias(double[][] bias, double[][] hiddenInput1, double[][] hiddenInput2, double eta) {
        double[][] revisedBias = new double[bias.length][bias[0].length];
        for (int i = 0; i < bias[0].length; i++) {
            for (int j = 0; j < bias.length; j++) {

                revisedBias[j][i] = bias[j][i] - (eta / 2) * (hiddenInput1[j][i] + hiddenInput2[j][i]);
            }
        }
        return revisedBias;
    }

    // c27 - (eta/2) * (w27+ax27)
    static double[][] revisedWeights(double[][] originalWeights, double[][] gW1, double[][] gW2, double eta) {
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
    static double[][][] miniBatch(double[][][] trainingCases, double[][][] weightsBiases) {
        /// Training Case 1
        // double[][][] revisedWeightsBiases = new
        /// double[weightsBiases.length][weightsBiases[0].length][weightsBiases[0][0].length];
        double[][] myOut = sigmoid(trainingCases[0], weightsBiases[0], weightsBiases[1]);
        double[][] myOut2 = sigmoid(myOut, weightsBiases[2], weightsBiases[3]);
        double[][] dLayer2 = backpropLast(myOut2, trainingCases[1]);
        double[][] gradWeight2 = gradientOfWeights(dLayer2, myOut);
        double[][] dLayer1 = backpropRest(dLayer2, weightsBiases[2], myOut);
        double[][] gradWeight1 = gradientOfWeights(dLayer1, trainingCases[0]);

        /// Training Case 2
        double[][] my2Out = sigmoid(trainingCases[2], weightsBiases[0], weightsBiases[1]);
        double[][] my2Out2 = sigmoid(my2Out, weightsBiases[2], weightsBiases[3]);
        double[][] d2layer2 = backpropLast(my2Out2, trainingCases[3]);
        double[][] grad2Weight2 = gradientOfWeights(d2layer2, my2Out);
        double[][] d2Layer1 = backpropRest(d2layer2, weightsBiases[2], my2Out);
        double[][] grad2Weight1 = gradientOfWeights(d2Layer1, trainingCases[2]);

        double eta = 10;

        double[][] rW1 = revisedWeights(weightsBiases[0], gradWeight1, grad2Weight1, eta);
        double[][] rB1 = revisedBias(weightsBiases[1], dLayer1, d2Layer1, eta);
        /// layer 2 biases and weights
        double[][] rW2 = revisedWeights(weightsBiases[2], gradWeight2, grad2Weight2, eta);
        double[][] rB2 = revisedBias(weightsBiases[3], dLayer2, d2layer2, eta);

        double[][][] revisedWeightsBiases = { rW1, rB1, rW2, rB2 };
        return revisedWeightsBiases;
    }

    static void epoch(double[][][][] packedInfo, int epochNum) {
        System.out.println("Epoch: " + epochNum);
        double[][][] mB = miniBatch(packedInfo[1], packedInfo[0]);
        double[][][] mB2 = miniBatch(packedInfo[2], mB);
        Array.print(mB2[0], "rW1");
        Array.print(mB2[1], "rB1");
        Array.print(mB2[2], "rW2");
        Array.print(mB2[3], "rB2");

        epochNum -= 1;
        double[][][][] pack = { mB2, packedInfo[0], packedInfo[1] };
        if (epochNum > 0) {
            epoch(pack, (epochNum));
        }
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
        double[][] bias1 = { { 0.1 }, { -0.36 }, { -0.31 } };

        double[][] weight2 = { { 0.76, 0.48, -0.73 }, { 0.34, 0.89, -0.23 } };
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
        double[][] x2Train2 = { { 1 }, { 1 }, { 0 }, { 0 } };
        double[][] y2Train2 = { { 1 }, { 0 } };

        // packing
        double[][][] tc1compact = { xTrain1, yTrain1, xTrain2, yTrain2 };
        double[][][] tc2compact = { x2Train1, y2Train1, x2Train2, y2Train2 };
        double[][][] wbcompact = { weight1, bias1, weight2, bias2 };

        // more packing

        double[][][][] packed = { wbcompact, tc1compact, tc2compact };

        /////////////////////////////////////////////////////////

        epoch(packed, 2);

    }

}
