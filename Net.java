import java.io.*;
import java.util.Scanner;

public class Net {

    static Integer testMath(Integer myInt, Integer myInt2) {
        Integer myValue = 2;
        return myValue + myInt;
    }

    static Integer[][] addArrays(Integer[][] a1, Integer[][] a2) {
        // gets array height
        Integer column1 = a1.length;
        Integer column2 = a2.length;

        // gets array width
        // note: this assumes the row length is consistent
        Integer row1 = a1[0].length;
        Integer row2 = a2[0].length;

        // init output array
        Integer[][] arraySum = new Integer[column1][row1];
        // checks to see if the arrays are the same size
        if ((column1 == column2) && (row1 == row2)) {
            for (int i = 0; i < column1; i++) {
                for (int j = 0; j < row1; j++) {
                    System.out.println(i + " " + j);
                    arraySum[i][j] = a1[i][j] + a2[i][j];
                }
            }
        } else {
            System.out.println("matrices are not the same");
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
        Integer[][] myNums2 = { { 30 }, { 100, 200, 300 } };
        Integer[][] myNums3 = { { 30 }, { 100 } };

        System.out.print(myNums[1][2] + " " + (myNums.length));
        System.out.println();
        System.out.print(myNums2[1][2] + " " + (myNums2.length));
        System.out.println();
        System.out.print(myNums3[0][0] + " " + (myNums3.length));

        // do array math
        System.out.println();
        addArrays(myNums, myNums2);
        Integer[][] myFinal = addArrays(myNums, myNums);

        System.out.println(myFinal);
        for (int i = 0; i < myFinal.length; i++) {
            for (int j = 0; j < myFinal[0].length; j++) {
                System.out.println(myFinal[i][j]);
            }
            System.out.println();
        }

    }
}
