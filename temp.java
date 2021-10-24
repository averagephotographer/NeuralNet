
// scalar multiplication of elements in two arrays
    static double[][] multiplyScalar(double[][] a1, double[][] a2) {
        // initialize output array
        double[][] mulOutput = new double[a1.length][a1[0].length];

        // multiplication
        if (areArraysAddable(a1, a2)) {
            for (int i = 0; i < a1.length; i++) {
                for (int j = 0; j < a1[0].length; j++) {
                    mulOutput[i][j] = a1[i][j] * a2[i][j];
                }
            }
        } else {
            System.out.println("arrays can't be multiplied scalar-wise");
            System.exit(0);
        }
        return mulOutput;
    }


        // math for dot product
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < input.length; j++) {
                for (int k = 0; k < input[0].length; k++) {
                    product[i][k] += weights[i][j] * input[j][k];
                }
            }
        }
        return product;
    }

    static double[][] sigmoid(double[][] input, double[][] weight, double[][] bias) {
        double[][] sigmoidOut = new double[weight.length][bias[0].length];
        double[][] prod = dotProduct(weight, input);
        double[][] zValue = addArrays(prod, bias);

        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < bias[0].length; j++) {
                sigmoidOut[i][j] = (1 / (1 + Math.pow(Math.E, -(zValue[i][j]))));
            }
        }

        return sigmoidOut;
    }

    // last layer backprop
    static double[][] backpropLast(double[][] output, double[][] trainData) {
        double[][] ones = onesArray(trainData.length, trainData[0].length);
        // should I use dot prod to make this simpler?
        return multiplyScalar(multiplyScalar(subtractArrays(output, trainData), output), subtractArrays(ones, output));
    }

    // GradientOfWeights
    static double[][] gradientOfWeights(double[][] gradientBiases, double[][] output) {
        return dotProduct(gradientBiases, transpose(output));
    }

    // backprop for the middle layers
    static double[][] backprop(double[][] dLayer, double[][] weights, double[][] prevMyOut) {
        double[][] biasGradient = new double[prevMyOut.length][prevMyOut[0].length];
        for (int k = 0; k < prevMyOut[0].length; k++) {
            for (int j = 0; j < weights[0].length; j++) {
                // could I transpose dLayer to make it simpler and do the dot product?
                // this feels kinda sketch
                biasGradient[j][k] += (weights[k][j] * dLayer[k][0] + weights[k + 1][j] * dLayer[k + 1][0])
                        * (prevMyOut[j][0] * (1 - prevMyOut[j][0]));
            }
        }
        return biasGradient;
    }

    static double[][] reviseBias(double[][] bias, double[][] hiddenInput1, double[][] hiddenInput2, double eta) {
        double[][] revisedBias = new double[bias.length][bias[0].length];
        for (int i = 0; i < bias[0].length; i++) {
            for (int j = 0; j < bias.length; j++) {

                revisedBias[j][i] = bias[j][i] - (eta / 2) * (hiddenInput1[j][i] + hiddenInput2[j][i]);
            }
        }
        return revisedBias;
    }

    static double[][] reviseWeights(double[][] originalWeights, double[][] gW1, double[][] gW2, double eta) {
        double[][] revisedWeights = new double[originalWeights.length][originalWeights[0].length];
        for (int i = 0; i < originalWeights[0].length; i++) {
            for (int j = 0; j < originalWeights.length; j++) {
                revisedWeights[j][i] = originalWeights[j][i] - (eta / 2) * (gW1[j][i] + gW2[j][i]);

            }
        }
        return revisedWeights;
    }

    // input training training weights and biases
    // return revised weights and biases
    static double[][][] training(double[][][] weights, double[][][] biases, double[][][] train) {

        /// Training Case 1
        double[][] myOut = sigmoid(train[0], weights[0], biases[0]);
        double[][] myOut2 = sigmoid(myOut, weights[1], biases[1]);
        double[][] dLayer2 = backpropLast(myOut2, train[1]);
        double[][] gradWeight2 = gradientOfWeights(dLayer2, myOut);
        double[][] dLayer1 = backprop(dLayer2, weights[1], myOut);
        double[][] gradWeight1 = gradientOfWeights(dLayer1, train[0]);

        /// Training Case 2
        double[][] my2Out = sigmoid(train[2], weights[0], biases[0]);
        double[][] my2Out2 = sigmoid(my2Out, weights[1], biases[1]);
        double[][] d2layer2 = backpropLast(my2Out2, train[3]);
        double[][] grad2Weight2 = gradientOfWeights(d2layer2, my2Out);
        double[][] d2Layer1 = backprop(d2layer2, weights[1], my2Out);
        double[][] grad2Weight1 = gradientOfWeights(d2Layer1, train[2]);

        double learningRate = 10;

        double[][] rW1 = reviseWeights(weights[0], gradWeight1, grad2Weight1, learningRate);
        double[][] rB1 = reviseBias(biases[0], dLayer1, d2Layer1, learningRate);
        /// layer 2 biases and weights
        double[][] rW2 = reviseWeights(weights[1], gradWeight2, grad2Weight2, learningRate);
        double[][] rB2 = reviseBias(biases[1], dLayer2, d2layer2, learningRate);

        double[][][] revisedWeightsBiases = { rW1, rB1, rW2, rB2 };

        return revisedWeightsBiases;
    }

    static double[][][][] miniBatch(double[][][] weights, double[][][] biases, double[][][] data) {
        // changed batch size from 2->10 and the error went from -1000 to -2000
        int miniBatchSize = 10;
        int caseSize = 2;
        int batchLength = miniBatchSize * caseSize;
        int numBatches = data.length / (caseSize * miniBatchSize);

        double[][][][] batches = new double[numBatches][data.length][0][784];
        for (int i = 0; i < data.length; i++) {

            int index1 = 0;
            int index2 = batchLength;
            for (int e = 0; e < (numBatches); e++) {
                // Split cases here into their respective batches
                batches[e] = Arrays.copyOfRange(data, index1, index2);
                index1 += batchLength;
                index2 += batchLength;

            }
        }
        for (int j = 0; j < numBatches; j++) {
            System.out.println("Training Batch: " + j);
            double[][][] revisedWB = training(weights, biases, batches[j]);
            int mbLength = revisedWB.length / 2;

            double[][][] newWeights = new double[mbLength][0][0];
            double[][][] newBiases = new double[mbLength][0][0];

            int weightsTemp = 0;
            int biasesTemp = 0;

            for (int k = 0; k < revisedWB.length; k++) {
                if (k % 2 == 0) {
                    newWeights[weightsTemp] = revisedWB[k];
                    weightsTemp++;
                } else {
                    newBiases[biasesTemp] = revisedWB[k];
                    biasesTemp++;
                }
            }
            weights = newWeights;
            biases = newBiases;
        }

        double[][][][] wb = { weights, biases };
        return wb;
    }

    static double[][][][] epoch(int numEpochs, double[][][] weights, double[][][] biases, double[][][] cases) {
        int count = 1;
        for (int i = 0; i < numEpochs; i++) {
            System.out.println("Epoch " + count + ": ");
            double[][][][] mbOut = miniBatch(weights, biases, cases);
            weights = mbOut[0];
            biases = mbOut[1];
            count++;
        }
        double[][][][] output = { weights, biases };

        Array.print(biases[1], "editBias");
        return output;
    }

    static double[][][] csvReader(String fileName) throws FileNotFoundException {
        // https://www.javatpoint.com/how-to-read-csv-file-in-java
        // pulls csv into java, prints it
        // filename: mnist_test.csv
        double[][][] rawData = new double[60000][785][1];
        Scanner csvData = new Scanner(new File(fileName));
        csvData.useDelimiter(",|\r|\n");
        int x = 0;
        int y = 0;
        while (csvData.hasNext()) {
            String s = csvData.next();
            double[] tempData = { Double.parseDouble(s) };
            rawData[x][y] = tempData;
            y++;
            if (y == 785) {
                y = 0;
                x++;
            }
        }
        int counter = 0;
        // prints last row in the datset

        int trainSize = 59999;
        int testSize = 9999;

        // options: testSize, trainSize
        int size = testSize;

        // prints integers as they come in
        // System.out.println("number: " + rawData[size][0]);
        for (int i = 1; i < 785; i++) {
            // System.out.print(rawData[size][i] + " ");
            counter++;
            if (counter > 27) {
                // System.out.println();
                counter = 0;
            }
        }
        System.out.println();
        csvData.close();
        return rawData;

    }
    // checks to see if two arrays are addable
    // assumes arrays are rectangular
    static Boolean areArraysAddable(double[][] array1, double[][] array2) {
        // gets array height
        int height1 = array1.length;
        int height2 = array2.length;

        // gets array width
        // note: this assumes the row length is consistent
        int width1 = array1[0].length;
        int width2 = array2[0].length;

        // checks to see if the arrays are the same size
        if ((height1 == height2) && (width1 == width2)) {
            return true;
        } else {
            // error if the arrays aren't addable
            System.out.println("arrays aren't addable");
            System.exit(0);
            // printSize(array1, "a1");
            // printSize(array2, "a2");
            return false;
        }
    }
    // fn to take raw csv and take off the classifier row

    // input array size
    // return array of that size with randomized digits
    static double[][] randomArray(int x, int y) {
        double[][] randArray = new double[x][y];
        Random r = new Random();

        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                randArray[i][j] = r.nextDouble();
            }
        }
        return randArray;
    }

    // https://stackoverflow.com/questions/33144667/concatenating-two-arrays-with-alternating-values
    static double[][][] zip(double[][] first, double[][] second) {
        double[][][] output = new double[first.length + second.length][first[0].length + second[0].length][0];
        int index = 0;
        final int minLen = Math.min(first.length, second.length);
        for (int i = 0; i < minLen; i++) {
            double[][] temp1 = { first[i] };
            double[][] temp2 = { second[i] };
            output[index++] = temp1;
            output[index++] = temp2;
        }
        return output;
