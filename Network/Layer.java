package Network;

import java.util.Arrays;

import Network.Net;

public class Layer {
    public double[][] _weights;
    public double[] _biases;
    public double[] _hidden;

    // takes weights, biases
    // M is number of neurons in a layer
    public Layer(double[][] weights, double[] biases, int M) {
        this._weights = weights;
        this._biases = biases;
        this._hidden = new double[M];
    }

    // randomly create weights and biases
    // m = num nodes
    // d = dimensionality of last layer
    public Layer(int M, int D) {
        // randomly init wb
        _weights = Net.randomArray(M, D);
        _biases = Net.randomArray(M);
        _hidden = new double[M];
    }

    public void printWeights() {
        for (int i = 0; i < _weights.length; i++) {
            System.out.println(Arrays.toString(_weights[i]));
        }
        System.out.println();
    }

    public void printBiases() {
        System.out.println(Arrays.toString(_biases));
        System.out.println();
    }

    public double[] call(double[] input) {
        double[] zValue = Net.addArrays(Net.dotProduct(this._weights, input), this._biases);
        _hidden = Net.sigmoid(zValue);
        return _hidden;
    }
}
