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

    // yTrain is the ground truth, everything included
    public void fit(int epochs, int batchSize, double lr, double[][] bigX, double[][] bigY) {

        // for every epoch, loop through the entire training dataset
        for (int i = 0; i < epochs; i++) {

            // initializes to zero
            int batchCounter = 0;

            // for every item in the training data
            for (int y = 0; y < bigY.length; y++) {

                batchCounter++;

                // forward pass
                double[] yHat = call(bigX[y]);

                // ground truth instance from big Y
                double[] Ysingle = bigY[y];

                // last layer terms l+1
                // initialized to a len of 1 because it will be overridden by the last layer
                double[] errorRightLayer = new double[1];

                // start with last layer and move left
                for (int currLayerIndex = _layers.length - 1; currLayerIndex >= 0; currLayerIndex--) {
                    // each node's error term
                    double[] errorTerms = new double[_layers[currLayerIndex]._hidden.length];

                    double[] leftLayersHidden;
                    // the 0th layer is the input layer
                    // not included in _layers because it's fake
                    if (currLayerIndex == 0) {
                        leftLayersHidden = bigX[y];
                    } else {
                        leftLayersHidden = _layers[currLayerIndex - 1]._hidden;
                    }

                    // calculate error terms
                    if (currLayerIndex == _layers.length - 1) {
                        // calculate error term for the last layer
                        for (int j = 0; j < errorTerms.length; j++) {
                            // (aj - yj) * aj * (1-aj)
                            errorTerms[j] = (yHat[j] - Ysingle[j]) * yHat[j] * (1 - yHat[j]);
                        }
                    } else {
                        for (int k = 0; k < errorTerms.length; k++) {
                            double sum = 0;
                            for (int j = 0; j < errorRightLayer.length; j++) {
                                sum += _layers[currLayerIndex + 1]._weights[j][k] * errorRightLayer[j];
                            }
                            errorTerms[k] = sum * _layers[currLayerIndex]._hidden[k]
                                    * (1 - _layers[currLayerIndex]._hidden[k]);
                        }
                    }
                    // accumulate changes
                    for (int j = 0; j < errorTerms.length; j++) {
                        // (aj - yj) * aj * (1-aj)
                        _layers[currLayerIndex]._biasChanges[j] += errorTerms[j] * lr;

                        for (int k = 0; k < _layers[currLayerIndex]._weightChanges[j].length; k++) {
                            _layers[currLayerIndex]._weightChanges[j][k] += errorTerms[j] * lr * leftLayersHidden[k];
                        }
                    }

                    // saves right layer error terms for future use
                    errorRightLayer = errorTerms;
                }
                // once batch are finished, flush
                // if y is the last index, flush the unfinished batch
                if (batchCounter == batchSize || y == bigY.length - 1) {
                    // gathering pending changes

                    // submitting pending changes
                    flush(batchCounter);

                    batchCounter = 0;
                }
            }

        }
    }

    public void printModel(boolean printState) {
        for (int l = 0; l < _layers.length; l++) {
            System.out.println("Layer index: " + l);
            _layers[l].printLayer(printState);
        }
    }
}
