import java.io.*;
import java.util.Scanner;
import javax.print.attribute.standard.MultipleDocumentHandling;

public class Net {

    // prints the height x width of an array
    static void printSize(double[][] arr, String name) {
        System.out.println("h x w: " + name);
        System.out.println(arr.length + " x " + arr[0].length);
        System.out.println();
    }

    // prints an array
    static void printArray(double[][] myArray) {

        for (int i = 0; i < myArray.length; i++) {
            for (int j = 0; j < myArray[0].length; j++) {
                System.out.print(myArray[i][j] + "\t");
            }

            System.out.println();
        }
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
            printSize(a1, "array1");
            printSize(a2, "array2");
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

        printSize(zValue, "zValue");
        printArray(zValue);
        System.out.println();

        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < bias[0].length; j++) {
                sigmoidOut[i][j] = (1 / (1 + Math.pow(Math.E, -(zValue[i][j]))));
            }
        }

        return sigmoidOut;

    }

    public static void main(String[] csv_file_name) throws FileNotFoundException {

        // https://www.javatpoint.com/how-to-read-csv-file-in-java
        // pulls csv into java, prints it

        // Scanner csv_scanner = new Scanner(new File("mnist_test.csv"));
        // csv_scanner.useDelimiter(",");
        // while (csv_scanner.hasNext()) {
        // System.out.print(csv_scanner.next());
        // }
        // csv_scanner.close();

        // mini-batch #1
        // training case 1
        double[][] xTrain1 = { { 0 }, { 1 }, { 0 }, { 1 } };
        double[][] yTrain1 = { { 0 }, { 1 } };

        // training case 2
        double[][] xTrain2 = { { 1 }, { 0 }, { 1 }, { 0 } };

        double[][] yTrain2 = { { 1 }, { 0 } };

        // mini-batch #2
        // training case 1
        double[][] x2Train1 = { { 0 }, { 1 }, { 0 }, { 1 } };

        double[][] y2Train1 = { { 0 }, { 1 } };

        // training case 2
        double[][] x2Train2 = { { 1 }, { 0 }, { 1 }, { 0 } };
        double[][] y2Train2 = { { 1 }, { 0 } };

        double[][] weight1 = { { -0.21, 0.72, -0.25, 1 }, { -0.94, -0.41, -0.47, 0.63 }, { 0.15, 0.55, -0.49, -0.75 } };

        double[][] bias1 = { { 0.1 }, { -0.36 }, { -0.31 } };

        double[][] weight2 = { { 0.76, 0.48, -0.73 }, { 0.34, 0.89, -0.23 } };

        double[][] bias2 = { { 0.16 }, { -0.46 } };

        printSize(xTrain1, "xTrain1");
        printArray(xTrain1);
        System.out.println();

        printSize(weight1, "weight1");
        printArray(weight1);
        System.out.println();

        printSize(bias1, "bias1");
        printArray(bias1);
        System.out.println();

        double[][] myOut = sigmoid(xTrain1, weight1, bias1);
        printSize(myOut, "myOut");
        printArray(myOut);

    }
}
