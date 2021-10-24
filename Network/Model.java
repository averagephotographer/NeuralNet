package Network;

import java.io.LineNumberInputStream;

public class Model {
    int[] _sizes;
    Layer[] _layers;

    public Model(int[] sizes) {
        _sizes = sizes;
        // minus 1 because of the fake layer (already given)
        _layers = new Layer[sizes.length - 1];

        for (int i = 0; i < _sizes.length - 1; i++) {
            // m -> neurons
            int M = _sizes[i + 1];
            // d -> previous layer dimensionality
            int D = _sizes[i];

            _layers[i] = new Layer(M, D);
            _layers[i].printWeights();
        }
    }

    // forward pass is always one x[0]
    // x[0] - y[o]
    public double[] call(double[] input) {
        double[] yCurr = input;
        for (int i = 0; i < _layers.length; i++)
            yCurr = _layers[i].call(yCurr);

        return yCurr;
    }

    // all instances
    // start with hidden layers
    public double[][] predict(double[][] inputs) {
        int lastLayerSize = _sizes[_sizes.length - 1];
        double[][] Y = new double[inputs.length][lastLayerSize];
        // first instance
        for (int i = 0; i < inputs.length; i++) {
            Y[i] = call(inputs[i]);
        }
        return Y;
    }

    public void fit(double[][] input, double[][] output) {
        // Net.predict(_weights, _biases, input)
    }
}
