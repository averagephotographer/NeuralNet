package background;

import java.text.DecimalFormat;

public class Array {
    private double[][] arr;
    // private String name;

    public void setArray(double[][] arr) {
        this.arr = arr;
    }

    public double[][] getArray() {
        return this.arr;
    }

    public static void print(double[][] arr, String name) {

        DecimalFormat numberFormat = new DecimalFormat("0.00##");

        // prints the height x width of an array
        System.out.println();
        System.out.println("h x w: " + name);
        System.out.println(arr.length + " x " + arr[0].length);
        System.out.println();

        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.out.print(numberFormat.format(arr[i][j]) + "\t");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void print(double[] arr, String name) {

        DecimalFormat numberFormat = new DecimalFormat("0.00##");

        // prints the height x width of an array
        System.out.println();
        System.out.println("h: " + name);
        System.out.println(arr.length);
        System.out.println();

        for (int i = 0; i < arr.length; i++) {
            System.out.print(numberFormat.format(arr[i]) + "\t");
        }
        System.out.println();

    }

    public static void size(double[][] arr, String name) {
        System.out.println();
        System.out.println("h x w: " + name);
        System.out.println(arr.length + " x " + arr[1].length);
        System.out.println();
    }

    public static void size(double[] arr, String name) {
        System.out.println();
        System.out.println("h:  " + name);
        System.out.println(arr.length);
        System.out.println();
    }
}
