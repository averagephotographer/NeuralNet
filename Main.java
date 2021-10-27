
/*
Christopher Tullier
102-58-973
Assignment 2 - MNIST
RNN that learns how to recognize handwritten digits from the MNIST databse
*/
import java.io.*;
import java.util.Scanner;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import Network.*;

class Main {

    public static void main(String[] csv_file_name) throws FileNotFoundException {
        // todo: get user input

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

        // read from csv
        double[][] raw = csvReader(test);

        // initializing arrays
        /// image data
        double[][] X = new double[10000][784];
        /// image classification
        double[][] Y = new double[10000][10];

        for (int i = 0; i < X.length; i++) {
            int index = (int) Math.round(raw[i][0]);
            // image label
            Y[i][index] = 1;

            for (int j = 0; j < (X[0].length); j++) {
                // image data
                X[i][j] = raw[i][j + 1] / 255;
            }
        }

        // Net.size(X, "number data");
        // Net.size(Y, "classification");
        // printNum(X[9998], Y[9998]);

        // variables
        int[] sizes = { 784, 100, 10 };
        int epochs = 6;
        int BatchSize = 10;
        int LearningRate = 3;

        Model mnist = new Model(sizes);
        mnist.fit(epochs, BatchSize, LearningRate, X, Y);

        // saveModel("bestmodel.model", mnist);

        // Model mnist = loadModel("bestmodel.model");
        // mnist.predict(X);
    }

    // https://mkyong.com/java/how-to-read-and-write-java-object-to-a-file/
    static void saveModel(String fileName, Model model) {
        try {
            FileOutputStream f = new FileOutputStream(new File(fileName));
            ObjectOutputStream o = new ObjectOutputStream(f);

            // write objects to file
            o.writeObject(model);

            o.close();
            f.close();
        } catch (FileNotFoundException e) {
            System.out.println(e);
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    static Model loadModel(String fileName) {
        int[] sizes = { 1, 1, 1 };
        Model empty = new Model(sizes);
        try {
            FileInputStream fi = new FileInputStream(new File(fileName));
            ObjectInputStream oi = new ObjectInputStream(fi);

            // Read objects
            Model model = (Model) oi.readObject();

            oi.close();
            fi.close();

            return model;

        } catch (FileNotFoundException e) {
            System.out.println(e);
        } catch (IOException e) {
            System.out.println(e);
        } catch (ClassNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return empty;
    }

    // https://www.javatpoint.com/how-to-read-csv-file-in-java
    static double[][] csvReader(String fileName) throws FileNotFoundException {
        // pulls csv into java, prints it
        double[][] raw = new double[10000][785];
        Scanner csv = new Scanner(new File(fileName));

        csv.useDelimiter(",|\r|\n");
        int x = 0;
        int y = 0;
        while (csv.hasNext()) {
            String s = csv.next();
            // converts from string to double
            raw[x][y] = Double.parseDouble(s);
            y++;
            if (y == 785) {
                y = 0;
                x++;
            }
        }

        csv.close();
        return raw;
    }

    public static void printNum(double[] number, double[] valueArr) {
        int value = 0;
        int counter = 0;

        // get classification
        for (int v = 0; v < valueArr.length; v++) {
            if (valueArr[v] == 1.0) {
                value = v;
                System.out.println("value: " + value);
            }
        }

        // print ascii
        System.out.println("number: " + value);
        for (int i = 1; i < 784; i++) {
            System.out.print(" " + ascii(number[i]));
            counter++;
            if (counter > 27) {
                System.out.println();
                counter = 0;
            }
        }
    }

    static char ascii(double value) {
        int rounded = (int) Math.round((value * 255) / 32);
        switch (rounded) {
        case 0:
            return ' ';
        case 1:
            return '.';
        case 2:
            return ':';
        case 3:
            return ';';
        case 4:
            return 'i';
        case 5:
            return 'I';
        case 6:
            return 'T';
        case 7:
            return 'H';
        case 8:
            return '#';
        }
        return ' ';
    }

}
