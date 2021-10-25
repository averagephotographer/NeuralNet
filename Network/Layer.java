package Network;

import java.util.Arrays;

public class Layer {
    public double[][] _weights;
    public double[] _biases;
    public double[] _hidden;
    public double[][] _weightChanges;
    public double[] _biasChanges;

    // takes weights, biases
    // M is number of neurons in a layer
    public Layer(double[][] weights, double[] biases, int M) {
        this._weights = weights;
        this._biases = biases;
        this._hidden = new double[M];

        this._weightChanges = new double[weights.length][weights[0].length];
        this._biasChanges = new double[biases.length];
    }

    // randomly create weights and biases
    // m = num nodes
    // d = dimensionality of last layer
    public Layer(int M, int D) {
        // randomly init wb
        this._weights = Net.randomArray(M, D);
        this._biases = Net.randomArray(M);
        this._hidden = new double[M];

        this._weightChanges = new double[M][D];
        this._biasChanges = new double[M];
    }

    public void printWeights() {
        for (int i = 0; i < _weights.length; i++) {
            System.out.println(Arrays.toString(_weights[i]));
            System.out.println();
        }
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

    public void holdWeightChanges(double[][] changes) {
        _weightChanges = Net.addArrays(_weightChanges, changes);
    }

    public void holdBiasChanges(double[] changes) {
        _biasChanges = Net.addArrays(_biasChanges, changes);
    }

    public void flush(int batchSize) {
        _weightChanges = Net.multiplyScalar(_weightChanges, 1 / batchSize);
        _biasChanges = Net.multiplyScalar(_biasChanges, 1 / batchSize);

        _weights = Net.addArrays(_weights, _weightChanges);
        _biases = Net.addArrays(_biases, _biasChanges);
    }
}
