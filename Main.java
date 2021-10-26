
/*
Christopher Tullier
102-58-973
Assignment 2 - MNIST
RNN that learns how to recognize handwritten digits from the MNIST databse
*/
import java.io.*;
import java.util.Scanner;

import Network.*;

class Main {

    public static void main(String[] csv_file_name) throws FileNotFoundException {
        boolean practice = false;

        if (practice == true) {
            // excel inputs
            double[] x1 = { 0, 1, 0, 1 };
            double[] x2 = { 1, 0, 1, 0 };
            double[] x3 = { 0, 0, 1, 1 };
            double[] x4 = { 1, 1, 0, 0 };
            double[] xtra = { 1, 1, 1, 1 }; // to test non-optimal batch sizes

            // excel desired outputs
            double[] y1 = { 0, 1 };
            double[] y2 = { 1, 0 };
            double[] y3 = { 0, 1 };
            double[] y4 = { 1, 0 };
            double[] ytra = { 0, 0 }; // to test non-optimal batch sizes

            // excel weights
            double[][] w1 = { { -0.21, 0.72, -0.25, 1 }, { -0.94, -0.41, -0.47, 0.63 }, { 0.15, 0.55, -0.49, -0.75 } };
            double[][] w2 = { { 0.76, 0.48, -0.73 }, { 0.34, 0.89, -0.23 } };

            // excel biases
            double[] b1 = { 0.1, -0.36, -0.31 };
            double[] b2 = { 0.16, -0.46 };

            // packing
            double[][] bigX = { x1, x2, x3, x4 };
            double[][] bigY = { y1, y2, y3, y4 };
            double[][][] weightsP = { w1, w2 };
            double[][] biasesP = { b1, b2 };

            int[] sizes = { 4, 3, 2 };
            // instantiate with weights and biases
            Model excel = new Model(sizes, weightsP, biasesP);

            // instantiate with random weights and biases
            // Model excel = new Model(sizes);

            excel.fit(6, 2, 10, bigX, bigY);
            excel.printModel(false);
        }
        /////////////////////////////////////////////////////////

        String test = "data/mnist_test.csv";
        String train = "data/mnist_train.csv";
        double[][] rawCSV = csvReader(test);
        // desired output array
        double[][] desiredOutput = new double[10000][10];

        // separates the starting number from the data
        double[][] numberData = new double[10000][784];

        for (int i = 0; i < numberData.length; i++) {
            for (int j = 0; j < (numberData[0].length); j++) {
                // just the image data
                numberData[i][j] = rawCSV[i][j + 1];
            }
            int value = (int) Math.round(rawCSV[i][0]);

            // sets the value at the position i to 1
            desiredOutput[i][value] = 1.0;
        }

        Net.size(desiredOutput, "desiredoutput");

        int[] sizes = { 784, 30, 10 };
        int epochs = 1;
        int BatchSize = 10;
        int LearningRate = 3;
        Model mnist = new Model(sizes);
        // mnist.fit(epochs, BatchSize, LearningRate, numberData, desiredOutput);
    }

    // https://www.javatpoint.com/how-to-read-csv-file-in-java
    static double[][] csvReader(String fileName) throws FileNotFoundException {
        // pulls csv into java, prints it
        // filename: mnist_test.csv
        double[][] rawData = new double[60000][785];
        Scanner csvData = new Scanner(new File(fileName));
        csvData.useDelimiter(",|\r|\n");
        int x = 0;
        int y = 0;
        while (csvData.hasNext()) {
            String s = csvData.next();
            rawData[x][y] = Double.parseDouble(s) / 255;
            y++;
            if (y == 785) {
                y = 0;
                x++;
            }
        }

        csvData.close();
        return rawData;

    }

}
