import background.A;
import background.A.*;

public class Net {

    static double[][] sigmoid(double[][] zValue, double[][] weight, double[][] bias) {
        double[][] output = new double[weight.length][bias[0].length];
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < bias[0].length; j++) {
                output[i][j] = (1 / (1 + Math.pow(Math.E, -(zValue[i][j]))));
            }
        }
        return output;
    }

    static double[][] getZ(double[][] input, double[][] weight, double[][] bias) {
        double[][] dp = A.dotProduct(weight, input);
        double[][] zValue = A.addArrays(dp, bias);
        return zValue;
    }

    public static void epoch(training, weight1, weight2, bias1, bias2) {

        double[][] z = getZ(xTrain1, weight1, bias1);
        double[][] y = sigmoid(z, weight1, bias1);

        double[][] z2 = getZ(y, weight2, bias2);
        double[][] y2 = sigmoid((z2), weight2, bias2);

        A.print(y2, "y2");
    }

    public static void main(String[] args) {

    public double[][] weight1 = { { -0.21, 0.72, -0.25, 1 }, { -0.94, -0.41, -0.47, 0.63 },
            { 0.15, 0.55, -0.49, -0.75 } };
    public double[][] weight2 = { { 0.76, 0.48, -0.73 }, { 0.34, 0.89, -0.23 } };

    public double[][] bias1 = { { 0.1 }, { -0.36 }, { -0.31 } };
    public double[][] bias2 = { { 0.16 }, { -0.46 } };

    // mini-batch #1
    /// training case 1
    public double[][] xTrain1 = { { 0 }, { 1 }, { 0 }, { 1 } };
    public double[][] yTrain1 = { { 0 }, { 1 } };
    /// training case 2
    public double[][] xTrain2 = { { 1 }, { 0 }, { 1 }, { 0 } };
    public double[][] yTrain2 = { { 1 }, { 0 } };

    // mini-batch #2
    /// training case 1
    public double[][] x2Train1 = { { 0 }, { 0 }, { 1 }, { 1 } };
    public double[][] y2Train1 = { { 0 }, { 1 } };
    /// training case 2
    public double[][] x2Train2 = { { 1 }, { 1 }, { 0 }, { 0 } };
    public double[][] y2Train2 = { { 1 }, { 0 } };

    Net myNet = new Net();myNet.epoch();
}}
