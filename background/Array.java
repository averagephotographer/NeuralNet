package background;

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

        // prints the height x width of an array
        System.out.println("h x w: " + name);
        System.out.println(arr.length + " x " + arr[0].length);
        System.out.println();

        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.out.print(arr[i][j] + "\t");
            }

            System.out.println();
        }
        System.out.println();
    }

    public static void main(String[] args) {

    }
}
