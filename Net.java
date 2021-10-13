import java.io.*;
import java.util.Scanner;
import javax.print.attribute.standard.MultipleDocumentHandling;

public class Net {

    // prints the height x width of an array
    static void printSize(int[][] arr, String name) {
        System.out.println("h x w: " + name);
        System.out.println(arr.length + " x " + arr[0].length);
        System.out.println();
    }

    // prints an array
    static void printArray(int[][] myArray) {

        for (int i = 0; i < myArray.length; i++) {
            for (int j = 0; j < myArray[0].length; j++) {
                System.out.print(myArray[i][j] + "\t");
            }

            System.out.println();
        }
    }

    // transposes a single array
    static int[][] transpose(int[][] a1) {

        int[][] newArray = new int[a1[0].length][a1.length];

        for (int i = 0; i < a1[0].length; i++) {
            for (int j = 0; j < a1.length; j++) {
                newArray[i][j] = a1[j][i];
            }
        }
        return newArray;
    }

    // checks to see if two arrays are addable
    // assumes arrays are rectangular
    static Boolean areArraysAddable(int[][] array1, int[][] array2) {
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
    static int[][] addArrays(int[][] a1, int[][] a2) {
        // get height and width of arrays
        int height = a1.length;
        int width = a1[0].length;

        // init output array
        int[][] arraySum = new int[height][width];

        // if arrays are addable, add them
        // otherwise return error message and exit
        if (areArraysAddable(a1, a2)) {
            // adds the arrays together
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    arraySum[i][j] = a1[i][j] + a2[i][j];
                }
            }
            return arraySum;
        } else {
            System.out.println("Cannot add arrays");
            System.exit(0);
            return arraySum;
        }
    }

    // scalar multiplication of elements in two arrays
    static int[][] multiplyScalar(int[][] a1, int[][] a2) {
        // initialize output array
        int[][] mulOutput = new int[a1.length][a1[0].length];

        // math
        if (areArraysAddable(a1, a2)) {
            for (int i = 0; i < a1.length; i++) {
                for (int j = 0; j < a1[0].length; j++) {
                    mulOutput[i][j] = a1[i][j] * a2[i][j];
                }
            }
            return mulOutput;
        } else {
            System.out.println("arrays can't be multiplied scalar-wise");
            System.exit(0);
            return mulOutput;
        }
    }

    // https://www.varsitytutors.com/hotmath/hotmath_help/topics/matrix-multiplication
    // dot product two arrays
    static int[][] dotProduct(int[][] a1, int[][] a2) {
        int[][] product = new int[a1.length][a2[0].length];

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

    public static void main(String[] csv_file_name) throws FileNotFoundException {

        // https://www.javatpoint.com/how-to-read-csv-file-in-java
        // pulls csv into java, prints it

        // Scanner csv_scanner = new Scanner(new File("mnist_test.csv"));
        // csv_scanner.useDelimiter(",");
        // while (csv_scanner.hasNext()) {
        // System.out.print(csv_scanner.next());
        // }
        // csv_scanner.close();

        // test arrays
        int[][] myNums = { { 10, 20, 30, 40 }, { 100, 200, 300, 400 }, { 1000, 2000, 3000, 4000 } };
        int[][] myNums2 = { { 10, 20, 30, 78 }, { 100, 14, 200, 300 }, { 40, 12, 12, 41 } };
        int[][] myNums3 = { { 10, 20, 30, 41 }, { 7, 12, 15, 57 }, { 90, 14, 894, 34 } };
        int[][] tallArray = { { 1, 2, 3, 4, 5, 6, 7, 8, 9 } };
        int[][] simple1 = { { 1, 2, 3 } };
        int[][] simple2 = { { 2, 3, 4 }, { 3, 3, 2 } };
        int[][] simple3 = { { 1, 2 }, { 3, 6 }, { 2, 4 } };

        printArray(transpose(simple1));
        System.out.println();
        printArray(simple3);
        System.out.println();

        System.out.println("\ndot prod: ");
        printArray(dotProduct(transpose(simple1), simple3));

    }
}
