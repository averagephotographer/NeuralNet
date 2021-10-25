package Network;

import java.util.Arrays;
import java.util.Currency;

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

    public Model(int[] sizes, double[][][] weights, double[][] biases) {
        // do your stuff
        _sizes = sizes;
        _layers = new Layer[sizes.length - 1];

        for (int i = 0; i < _sizes.length - 1; i++) {
            int M = _sizes[i + 1];
            _layers[i] = new Layer(weights[i], biases[i], M);
        }
    }

    // forward pass is always one x[0]
    // x[0] - y[o]
    public double[] call(double[] xInput) {
        double[] yCurr = xInput;
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

    // loop through each layer and updating the weights and biases
    public void flush(int batchSize) {
        for (int i = 0; i < _layers.length; i++) {
            _layers[i].flush(batchSize);
        }
    }

    // yTrain is the ground truth
    public void fit(int epochs, int batchSize, double lr, double[][] Xtrain, double[][] Ytrain) {

        // last layer and move back

        for (int i = 0; i < epochs; i++) {
            // for every epoch, loop through the entire training dataset

            // initializes to zero
            int batchCounter = 1;

            for (int y = 0; y < Ytrain.length; y++) {
                if (batchCounter <= batchSize) {
                    // gathering pending changes
                    batchCounter++;

                    // forward pass
                    double[] yHat = call(Xtrain[i]);

                    // ground truth instance from big Y
                    double[] Ysingle = Ytrain[y];

                    // last layer terms
                    double[] errorLastLayer;

                    for (int currLayerIndex = _layers.length - 1; currLayerIndex >= 0; currLayerIndex--) {
                        // error term for weights and bias
                        double[] errorTerms = new double[_layers[currLayerIndex]._hidden.length];

                        double[] prevLayersHidden = _layers[currLayerIndex - 1]._hidden;

                        // if current layer is the last layer do special math
                        if (currLayerIndex == _layers.length - 1) {
                            // calculate error term
                            for (int j = 0; j < errorTerms.length; j++) {
                                errorTerms[j] = yHat[j] * (1 - yHat[j]) * (Ysingle[j]);
                                _layers[currLayerIndex]._biasChanges[j] += errorTerms[j] * lr;

                                for (int k = 0; k < _layers[currLayerIndex]._weightChanges[j].length; k++) {
                                    _layers[currLayerIndex]._weightChanges[j][k] += errorTerms[j] * lr
                                            * prevLayersHidden[j];
                                }
                            }
                            // save last layer error

                        } else {
                            // TODO: implement the other weight update rules (non last layers)

                        }

                        // saves last layer error terms for future use
                        errorLastLayer = errorTerms;
                    }

                } else {
                    // once batch are finished, flush
                    flush(batchCounter);
                    // submitting pending changes
                    batchCounter = 0;
                }
            }
        }
    }
}
