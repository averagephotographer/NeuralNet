package Network;

import java.io.Serializable;

public class Layer implements Serializable {
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
    // M = num nodes/neurons
    // D = dimensionality of last layer (# of neurons last layer)
    public Layer(int M, int D) {
        // randomly init wb
        this._weights = Net.randomArray(M, D);
        this._biases = Net.randomArray(M);
        this._hidden = new double[M];

        this._weightChanges = new double[M][D];
        this._biasChanges = new double[M];
    }

    // one forward pass
    public double[] forwardPass(double[] xInput) {
        double[] zValue = Net.addArrays(Net.dotProduct(this._weights, xInput), this._biases);
        _hidden = Net.sigmoid(zValue);
        return _hidden;
    }

    // preps the weights for the next batch
    public void flush(int batchSize) {
        // average the weights and biases over the batch size
        _weightChanges = Net.multiplyScalar(_weightChanges, 1 / Double.valueOf(batchSize));
        _biasChanges = Net.multiplyScalar(_biasChanges, 1 / Double.valueOf(batchSize));

        _weights = Net.subtractArrays(_weights, _weightChanges);
        _biases = Net.subtractArrays(_biases, _biasChanges);

        // reset changes to zero
        resetChanges();
    }

    public void resetChanges() {
        _weightChanges = Net.multiplyScalar(_weightChanges, 0);
        _biasChanges = Net.multiplyScalar(_biasChanges, 0);
    }

    public void printLayer(boolean printState) {
        // print weights
        Net.print(_weights, "Weights");
        // print biases
        Net.print(_biases, "biases");

        // optional print state
        if (printState == true) {
            Net.print(_hidden, "hidden");
        }
        System.out.println();

    }
}
