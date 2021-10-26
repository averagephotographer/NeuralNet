package Network;

import java.util.Arrays;
import java.util.Random;

public class Net {
    // for weights
    public static double[][] randomArray(int x, int y) {
        double[][] randArray = new double[x][y];
        Random r = new Random();

        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                randArray[i][j] = r.nextDouble();
            }
        }
        return randArray;
    }

    // for biases
    public static double[] randomArray(int x) {
        double[] randArray = new double[x];
        Random r = new Random();

        for (int i = 0; i < x; i++) {
            randArray[i] = r.nextDouble();
        }
        return randArray;
    }

    public static double[] onesArray(int length) {
        double[] onesArray = new double[length];
        for (int i = 0; i < length; i++) {
            onesArray[i] = 1;
        }
        return onesArray;
    }

    // transposes a single array
    public static double[][] transpose(double[][] a1) {

        double[][] newArray = new double[a1[0].length][a1.length];

        for (int i = 0; i < a1[0].length; i++) {
            for (int j = 0; j < a1.length; j++) {
                newArray[i][j] = a1[j][i];
            }
        }
        return newArray;
    }

    // scalar multiplication of elements in two arrays
    static double[][] multiplyScalar(double[][] a1, double value) {
        // initialize output array
        double[][] mulOutput = new double[a1.length][a1[0].length];

        // multiplication
        for (int i = 0; i < a1.length; i++) {
            for (int j = 0; j < a1[0].length; j++) {
                mulOutput[i][j] = a1[i][j] * value;
            }
        }
        return mulOutput;
    }

    static double[] multiplyScalar(double[] a1, double value) {
        // initialize output array
        double[] mulOutput = new double[a1.length];

        // multiplication
        for (int i = 0; i < a1.length; i++) {
            mulOutput[i] = a1[i] * value;
        }
        return mulOutput;
    }

    // https://www.varsitytutors.com/hotmath/hotmath_help/topics/matrix-multiplication
    // dot product two arrays
    public static double[] dotProduct(double[][] weights, double[] input) {
        double[] product = new double[weights.length];

        // checks if rows of first matrix and columns of second match
        if (weights[0].length != input.length) {
            System.out.println("cannot dot matrices");
            System.exit(0);
        }

        // dotprod operation
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < input.length; j++) {
                product[i] += weights[i][j] * input[j];
            }
        }
        return product;

    }

    // only used with single dimension vectors
    // array addition, assumes arrays are are perfect rectangles
    public static double[] addArrays(double[] a1, double[] a2) {
        // get height and width of arrays
        int len = a1.length;

        // init output array
        double[] arraySum = new double[len];

        // if arrays are addable, add them
        // otherwise return error message and exit
        if (a1.length == a2.length) {
            // adds the arrays together
            for (int i = 0; i < len; i++) {
                arraySum[i] = a1[i] + a2[i];
            }
        } else {
            System.out.println("Cannot add arrays");
            System.exit(0);
        }
        return arraySum;
    }

    static double[][] addArrays(double[][] a1, double[][] a2) {
        // get height and width of arrays
        int height = a1.length;
        int width = a1[0].length;

        // init output array
        double[][] arraySum = new double[height][width];

        // if arrays are addable, add them
        // otherwise return error message and exit
        if ((a1.length == a2.length) && (a1[0].length == a2[0].length)) {
            // adds the arrays together
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    arraySum[i][j] = a1[i][j] + a2[i][j];
                }
            }
        } else {
            System.out.println("Cannot add arrays");
            System.exit(0);
        }
        return arraySum;
    }

    static double[][] subtractArrays(double[][] a1, double[][] a2) {
        // get height and width of arrays
        int height = a1.length;
        int width = a1[0].length;

        // init output array
        double[][] arraySum = new double[height][width];

        // if arrays can be subtracted
        // otherwise return error message and exit
        if ((a1.length == a2.length) && (a1[0].length == a2[0].length)) {
            // adds the arrays together
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    arraySum[i][j] = a1[i][j] - a2[i][j];
                }
            }
        } else {
            System.out.println("Cannot subtract arrays");
            System.exit(0);
        }
        return arraySum;
    }

    // subtract arrays
    public static double[] subtractArrays(double[] a1, double[] a2) {
        // get height and width of arrays
        int len = a1.length;

        // init output array
        double[] arrayDiff = new double[len];

        // if arrays are able to, subtract them
        // otherwise return error message and exit
        if (a1.length == a2.length) {
            // adds the arrays together
            for (int i = 0; i < len; i++) {
                arrayDiff[i] = a1[i] - a2[i];
            }
        } else {
            System.out.println("Cannot subtract arrays");
            System.exit(0);
        }
        return arrayDiff;
    }

    public static double[] sigmoid(double[] zValue) {
        double[] sigmoidOut = new double[zValue.length];
        for (int i = 0; i < zValue.length; i++) {
            sigmoidOut[i] = (1 / (1 + Math.pow(Math.E, -(zValue[i]))));
        }
        return sigmoidOut;
    }

    // this is a forward pass, predict is what it's called in tf
    // this input could be a Model class object
    // include all weights, all biases
    // include all training data as the input
    public static double[][] predict(double[][][] weights, double[][] biases, double[][] inputs) {
        int lastBiasLength = biases[biases.length - 1].length;

        // initialize final output array
        // inputs.length is how many inputs we have
        // lastBiasLength is how many nodes are in the output (10 for mnist)
        double[][] Y = new double[inputs.length][lastBiasLength];

        // could also use weights.lenth could also be here
        int numLayers = biases.length;

        // loop over all inputs
        for (int i = 0; i < inputs.length; i++) {
            double[] currentInput = inputs[i];

            // loop over desired length of layers
            for (int j = 0; j < numLayers; j++) {
                double[] zValue = addArrays(dotProduct(weights[j], currentInput), biases[j]);
                currentInput = sigmoid(zValue);
            }
            Y[i] = currentInput;
        }
        return Y;
    }

    // call - one forward pass

    public static void fit(int numEpochs, int learningRate, double[][][] weights, double[][] biases, double[][] inputs,
            double[][] yTrain) {
        int batchSize = 2;
        // int Xindex;

        // pending changes
        double[][][] weightsChanges = new double[weights.length][weights[0].length][weights[0][0].length];
        double[][] biasesChanges = new double[biases.length][biases[0].length];

        double[][][] weightsCurrent = weights;
        double[][] biasesCurrent = biases;

        // loop until epochs are done
        for (int i = 0; i < numEpochs; i++) {
            // int numBatches = inputs.length / batchSize;
            int lastLayerIndex = weightsChanges.length - 1;
            double[][] batchCurr = new double[batchSize][inputs[0].length];
            double[][] yTrainCurr = new double[batchSize][yTrain[0].length];

            int batchCounter = 0;
            batchCurr = Arrays.copyOfRange(inputs, batchCounter, batchCounter + batchSize);
            yTrainCurr = Arrays.copyOfRange(yTrain, batchCounter, batchCounter + batchSize);

            /// predict on batchSize
            double[][] yCurr = predict(weightsCurrent, biasesCurrent, batchCurr);
            batchCounter += batchSize;

            double[] errorTerms = new double[weightsCurrent[lastLayerIndex].length];

            // last layer weights and biases
            // Big X level
            for (int j = 0; j < batchCurr.length; j++) {
                // X[instance] level
                for (int u = 0; u < weightsCurrent[lastLayerIndex].length; u++) {
                    // neuron level (everyone shares layer term)
                    errorTerms[u] = yCurr[j][u] * (1 - yCurr[j][u]) * (yTrainCurr[j][u] - yCurr[j][u]);
                    biasesChanges[lastLayerIndex][u] += (learningRate * errorTerms[u]);
                    for (int v = 0; v < weightsCurrent[lastLayerIndex][0].length; v++) {
                        // node level (aka weights level of a particular neuron)
                        weightsChanges[lastLayerIndex][u][v] += (learningRate * errorTerms[u] * batchCurr[j][v]);
                    }
                }
            }
            // double[] oldErrorTerms = errorTerms;
            // hidden layer weights and biases
            // Big X level

            // ADD EXTRA LOOP TO UPDATE CURRENT LAYER AND ITERATE OVER REMAINING LAYERS
            // for (int j = 0; j < batchCurr.length; j++) {
            // // X[instance] level
            // for (int u = 0; u < weightsCurrent[lastLayerIndex].length; u++) {
            // // neuron level (everyone shares layer term)
            // double extraSum;
            // for (int z = 0; z < errorTerms.length; z++) {
            // extraSum += weightsCurrent[currentLayer][z][u] * oldErrorTerms[z];
            // }
            // errorTerms[u] = yCurr[j][u] * (1 - yCurr[j][u]) * extraSum;
            // biasesChanges[lastLayerIndex][u] += (learningRate * errorTerms[u]);
            // for (int v = 0; v < weightsCurrent[lastLayerIndex][0].length; v++) {
            // // node level (aka weights level of a particular neuron)
            // weightsChanges[lastLayerIndex][u][v] += (learningRate * errorTerms[u] *
            // batchCurr[j][v]);
            // }
            // }
            // }
            /// Find gradient using J

            /// update

        }
    }
}
