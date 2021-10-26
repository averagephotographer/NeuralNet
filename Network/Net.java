package Network;

import java.util.Random;
import java.text.DecimalFormat;

public class Net {
    public static DecimalFormat numberFormat = new DecimalFormat("0.00##");

    public static void print(double[] arr, String name) {
        // prints the height of an array
        size(arr, name);
        for (int i = 0; i < arr.length; i++) {
            System.out.println(numberFormat.format(arr[i]) + "\t");
        }
        System.out.println();
    }

    public static void print(double[][] arr, String name) {
        // prints the height x width of an array
        size(arr, name);

        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.out.print(numberFormat.format(arr[i][j]) + "\t");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void size(double[] arr, String name) {
        System.out.println();
        System.out.println("h:  " + name);
        System.out.println(arr.length);
        System.out.println();
    }

    public static void size(double[][] arr, String name) {
        System.out.println();
        System.out.println("h x w: " + name);
        System.out.println(arr.length + " x " + arr[1].length);
        System.out.println();
    }

    public static double[] randomArray(int x) {
        double[] randArray = new double[x];
        Random r = new Random();

        for (int i = 0; i < x; i++) {
            randArray[i] = r.nextDouble();
        }
        return randArray;
    }

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

    public static double[] onesArray(int length) {
        double[] onesArray = new double[length];
        for (int i = 0; i < length; i++) {
            onesArray[i] = 1;
        }
        return onesArray;
    }

    public static double[][] transpose(double[][] a1) {
        double[][] newArray = new double[a1[0].length][a1.length];

        for (int i = 0; i < a1[0].length; i++) {
            for (int j = 0; j < a1.length; j++) {
                newArray[i][j] = a1[j][i];
            }
        }
        return newArray;
    }

    static double[] multiplyScalar(double[] a1, double value) {
        // initialize output array
        double[] mulOutput = new double[a1.length];

        // multiplication
        for (int i = 0; i < a1.length; i++) {
            mulOutput[i] = a1[i] * value;
        }
        return mulOutput;
    }

    static double[][] multiplyScalar(double[][] a1, double value) {
        // initialize output array
        double[][] mulOutput = new double[a1.length][a1[0].length];

        // multiplication
        for (int i = 0; i < a1.length; i++) {
            for (int j = 0; j < a1[0].length; j++) {
                mulOutput[i][j] = a1[i][j] * value;
            }
        }
        return mulOutput;
    }

    // https://www.varsitytutors.com/hotmath/hotmath_help/topics/matrix-multiplication
    // dot product two arrays
    public static double[] dotProduct(double[][] weights, double[] input) {
        double[] product = new double[weights.length];

        // checks if rows of first matrix and columns of second match
        if (weights[0].length != input.length) {
            System.out.println("cannot dot matrices");
            System.exit(0);
        }

        // dot prod operation
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < input.length; j++) {
                product[i] += weights[i][j] * input[j];
            }
        }
        return product;
    }

    public static double[] addArrays(double[] a1, double[] a2) {
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

    static double[][] addArrays(double[][] a1, double[][] a2) {
        // get height and width of arrays
        int height = a1.length;
        int width = a1[0].length;

        // init output array
        double[][] arraySum = new double[height][width];

        // if arrays are addable, add them
        // otherwise return error message and exit
        if ((a1.length == a2.length) && (a1[0].length == a2[0].length)) {
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
    public static double[] subtractArrays(double[] a1, double[] a2) {
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

    static double[][] subtractArrays(double[][] a1, double[][] a2) {
        // get height and width of arrays
        int height = a1.length;
        int width = a1[0].length;

        // init output array
        double[][] arraySum = new double[height][width];

        // if arrays can be subtracted
        // otherwise return error message and exit
        if ((a1.length == a2.length) && (a1[0].length == a2[0].length)) {
            // adds the arrays together
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    arraySum[i][j] = a1[i][j] - a2[i][j];
                }
            }
        } else {
            System.out.println("Cannot subtract arrays");
            System.exit(0);
        }
        return arraySum;
    }

    public static double[] sigmoid(double[] zValue) {
        double[] sigmoidOut = new double[zValue.length];
        for (int i = 0; i < zValue.length; i++) {
            sigmoidOut[i] = (1 / (1 + Math.pow(Math.E, -(zValue[i]))));
        }
        return sigmoidOut;
    }

}
