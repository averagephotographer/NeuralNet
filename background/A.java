package background;

import java.text.DecimalFormat;

public class A {
    private double[][] arr;
    // private String name;

    public void setArray(double[][] arr) {
        this.arr = arr;
    }

    public double[][] getArray() {
        return this.arr;
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
            System.out.println("arrays aren't addable");
            System.exit(0);
            // printSize(array1, "a1");
            // printSize(array2, "a2");
            return false;
        }
    }

    // array addition, assumes arrays are are perfect rectangles
    public static double[][] addArrays(double[][] a1, double[][] a2) {
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
    public static double[][] subtractArrays(double[][] a1, double[][] a2) {
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
    public static double[][] multiplyScalar(double[][] a1, double[][] a2) {
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

    public static double[][] transpose(double[][] a1) {

        double[][] newArray = new double[a1[0].length][a1.length];

        for (int i = 0; i < a1[0].length; i++) {
            for (int j = 0; j < a1.length; j++) {
                newArray[i][j] = a1[j][i];
            }
        }
        return newArray;
    }

    public static double[][] dotProduct(double[][] a1, double[][] a2) {
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

    public static void print(double[][] arr, String name) {

        DecimalFormat numberFormat = new DecimalFormat("0.00##");

        // prints the height x width of an array
        System.out.println();
        System.out.print(name + ": ");
        System.out.println(arr.length + " x " + arr[0].length);

        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.out.print(numberFormat.format(arr[i][j]) + "\t");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void main(String[] args) {

    }
}
