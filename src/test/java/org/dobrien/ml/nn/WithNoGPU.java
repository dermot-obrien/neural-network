package org.dobrien.ml.nn;

import org.fjnn.activation.Activation;
import org.fjnn.activation.Sigmoid;
import org.fjnn.activation.SoftMax;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.network.NeuralNetwork;

public class WithNoGPU {

    public static void main(String[] args) {
        // Configure.
        int inputSize  = 10;
        int hiddenSize = 10;
        int outputSize = 2;
        int numberOfHiddenLayers = 3;
        Activation outputActivation = new SoftMax();
        Activation hiddenActivation = new Sigmoid();

        // Create network.
        NeuralNetwork network = new NeuralNetwork(inputSize, outputSize, outputActivation);

        // Hidden layers.
        for(int i=0; i < numberOfHiddenLayers; i++) {
            network.addHiddenLayer(hiddenSize,hiddenActivation);
        }

        // Build network.
        network.build();

        // network.setWeight(...)
        // network.setBias(...)
        // network.compute(inputData)

        float[] inputData = new float[inputSize];
        float[] result = network.compute(inputData);
    }

}
