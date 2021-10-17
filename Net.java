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
            Array.print(a1, "array1");
            Array.print(a2, "array2");
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

        Array.print(zValue, "zValue");
        System.out.println();

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
    static double[][] backpropFirst(double[][] dLayer, int layer, double[][] weights, double[][] prevMyOut) {
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

        // mini-batch #1
        // training case 1
        double[][] xTrain1 = { { 0 }, { 1 }, { 0 }, { 1 } };
        double[][] yTrain1 = { { 0 }, { 1 } };

        // training case 2
        double[][] xTrain2 = { { 1 }, { 0 }, { 1 }, { 0 } };
        double[][] yTrain2 = { { 1 }, { 0 } };

        /////////////////////////////////////////////////////////

        // mini-batch #2
        // training case 1
        double[][] x2Train1 = { { 0 }, { 1 }, { 0 }, { 1 } };
        double[][] y2Train1 = { { 0 }, { 1 } };

        // training case 2
        double[][] x2Train2 = { { 1 }, { 0 }, { 1 }, { 0 } };
        double[][] y2Train2 = { { 1 }, { 0 } };

        /////////////////////////////////////////////////////////

        // for both batches
        double[][] weight1 = { { -0.21, 0.72, -0.25, 1 }, { -0.94, -0.41, -0.47, 0.63 }, { 0.15, 0.55, -0.49, -0.75 } };
        double[][] bias1 = { { 0.1 }, { -0.36 }, { -0.31 } };

        double[][] weight2 = { { 0.76, 0.48, -0.73 }, { 0.34, 0.89, -0.23 } };
        double[][] bias2 = { { 0.16 }, { -0.46 } };

        Array.print(xTrain1, "xTrain1");

        Array.print(weight1, "weight1");

        Array.print(bias1, "bias1");

        double[][] myOut = sigmoid(xTrain1, weight1, bias1);
        Array.print(myOut, "myOut");

        double[][] myOut2 = sigmoid(myOut, weight2, bias2);
        Array.print(myOut2, "myOut2");

        // backward pass through layer 2 (the 'output' or 3rd layer) of the network
        double[][] dLayer2 = backpropLast(myOut2, yTrain1);
        Array.print(dLayer2, "dLayer2");

        double[][] gradWeight2 = gradientOfWeights(dLayer2, myOut);
        Array.print(gradWeight2, "gradWeight");

        int currLayer = 2;
        double[][] dLayer1 = backpropFirst(dLayer2, currLayer, weight2, myOut);
        Array.print(dLayer1, "dLayer1");

        double[][] gradWeight1 = gradientOfWeights(dLayer1, xTrain1);
        Array.print(gradWeight1, "gradWeight1");

    }

}
