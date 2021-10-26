
/*
Christopher Tullier
102-58-973
Assignment 2 - MNIST
RNN that learns how to recognize handwritten digits from the MNIST databse
*/
import java.io.*;
import java.util.Scanner;

import java.util.Arrays;

import Network.*;

class Main {
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

        double[] xtra = { 1, 1, 1, 1 };
        double[] ytra = { 0, 0 };
        // packing
        double[][] inputs = { x1, x2, x21, x22, xtra };
        double[][] outputs = { y1, y2, y21, y22, ytra };
        double[][][] weights = { weight1, weight2 };
        double[][] biases = { bias1, bias2 };

        /////////////////////////////////////////////////////////
        // Arrays.toString() to print arrays

        // h is a in O'Neal's program
        // double[] h = sigmoid(addArrays(dotProduct(weights[0], inputs[0]),
        // biases[0]));

        // System.out.println(Arrays.toString(h));

        // forward pass

        // Layer myLayer = new Layer(3, 4);
        // myLayer.printWeights();
        // System.out.println();
        // myLayer.printBiases();
        // System.out.println();

        int[] sizes = { 4, 3, 2 };

        // Model myModel = new Model(sizes);

        // double[][] out = myModel.predict(inputs);

        // for (int i = 0; i < out.length; i++) {
        // System.out.println(Arrays.toString(out[i]));
        // }

        Model excel = new Model(sizes, weights, biases);

        excel.fit(1, 2, 10, inputs, outputs);
        excel.printModel(false);
    }
}
