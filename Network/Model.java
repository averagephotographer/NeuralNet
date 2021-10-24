package Network;

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

    // forward pass
    public void call() {
        // start with hidden layers
    }

    public void fit(double[][] input, double[][] output) {
        // Net.predict(_weights, _biases, input)
    }
}
