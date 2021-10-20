
/*
Christopher Tullier
102-58-973
Assignment 2 - MNIST
RNN that learns how to recognize handwritten digits from the MNIST databse
*/

package background;

import java.io.*;
import java.util.Scanner;
import java.util.Random;

public class Network {
    // input array size
    // return array of that size with randomized digits
    public int[] sizes = { 784, 30, 10 };
    public int numLayers = 2;
    public double[] bias = randomArray(784);
    public double[][] weight = randomArray(10, 30);

    public static double[][] randomArray(int x, int y) {
        double[][] randArray = new double[x][y];
        Random r = new Random();

        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                randArray[i][j] = r.nextDouble();
            }
        }
        return randArray;
    }

    public static double[] randomArray(int x) {
        double[] randArray = new double[x];
        Random r = new Random();

        for (int i = 0; i < x; i++) {
            randArray[i] = r.nextDouble();
        }
        return randArray;
    }

    public static double[][] dot(double[][] a1, double[][] a2) {
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

    public static Boolean areArraysAddable(double[][] array1, double[][] array2) {
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
            System.out.println("arrays aren't addable");
            System.exit(0);
            // printSize(array1, "a1");
            // printSize(array2, "a2");
            return false;
        }
    }

    // feedforward
    public double[][] feedForward(double[][] a) {
        double[][] sigmoidOut = new double[this.weight.length][bias.length];
        double[][] prod = dot(this.weight, a);
        double[][] newBias = { this.bias };
        double[][] zValue = addArrays(prod, newBias);

        for (int i = 0; i < this.weight.length; i++) {
            for (int j = 0; j < this.bias.length; j++) {
                sigmoidOut[i][j] = (1 / (1 + Math.pow(Math.E, -(zValue[i][j]))));
            }
        }

        return sigmoidOut;
    }

    public static void SGD(double[][] trainingData, int epochs, int miniBatchSize, double learningRate) {
        System.out.println("Hello there");
        int n = trainingData.length;

        for (int j = 0; j < epochs; j++) {
            // shuffle training data here
            
            // double[][] miniBatch = new double[][];
        }
    }
}
