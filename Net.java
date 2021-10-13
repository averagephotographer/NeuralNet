import java.io.*;
import java.util.Scanner;

public class Net {

    static Boolean areArraysAddable(Integer[][] array1, Integer[][] array2) {
        // gets array height
        Integer height1 = array1.length;
        Integer height2 = array2.length;

        // gets array width
        // note: this assumes the row length is consistent
        Integer width1 = array1[0].length;
        Integer width2 = array2[0].length;

        // checks to see if the arrays are the same size
        if ((height1 == height2) && (width1 == width2)) {
            return true;
        } else {
            return false;
        }
    }

    static Integer[][] transpose(Integer[][] a1) {
        Integer[][] newArray = new Integer[a1[0].length][a1.length];
        for (int i = 0; i < a1[0].length; i++) {
            for (int j = 0; j < a1.length; j++) {
                newArray[i][j] = a1[j][i];
            }
        }
        return newArray;
    }

    static Integer[][] dotProduct(Integer[][] a1, Integer[] a2) {
        Integer[][] temp = { { 0, 0 } };
        return temp;
    }

    static Integer testMath(Integer myInt, Integer myInt2) {
        Integer myValue = 2;
        return myValue + myInt;
    }

    // prints arrays
    static void printArray(Integer[][] myArray) {
        for (int i = 0; i < myArray.length; i++) {
            for (int j = 0; j < myArray[0].length; j++) {
                System.out.print(myArray[i][j] + "\t");
            }
            System.out.println();
        }
    }

    // array addition, assumes arrays are are perfect rectangles
    static Integer[][] addArrays(Integer[][] a1, Integer[][] a2) {
        // get height and width of arrays
        Integer height = a1.length;
        Integer width = a1[0].length;

        // init output array
        Integer[][] arraySum = new Integer[height][width];

        // height and width info
        System.out.println("h x w");
        System.out.println(height + " x " + width);
        System.out.println();

        // if arrays are addable, add them
        // otherwise return error message and exit
        if (areArraysAddable(a1, a2)) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    arraySum[i][j] = a1[i][j] + a2[i][j];
                }
            }
        } else {
            System.out.println("matrices are not the same");
            System.exit(0);
        }
        return arraySum;
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

        // do math
        // Integer firstValue = 3;
        // System.out.print(testMath(firstValue));

        // do array
        Integer[][] myNums = { { 10, 20, 30, 40 }, { 100, 200, 300, 400 }, { 1000, 2000, 3000, 4000 } };
        Integer[][] myNums2 = { { 10, 20, 30, 78 }, { 100, 14, 200, 300 }, { 40, 12, 12, 41 } };
        Integer[][] myNums3 = { { 10, 20, 30, 41 }, { 7, 12, 15, 57 }, { 90, 14, 894, 34 } };
        Integer[][] tallArray = { { 1, 2, 3, 4, 5, 6, 7, 8, 9 } };
        // adding arrays
        Integer[][] myFinal = addArrays(myNums2, myNums3);

        // prints output
        printArray(myFinal);
        System.out.println();
        printArray(tallArray);
        System.out.println();
        printArray(transpose(myFinal));
        System.out.println();
        printArray(transpose(tallArray));

    }
}
