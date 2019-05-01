package test;

import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.*;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.util.Arrays;
import java.util.Collections;


public class NeuralNetwork {


    public NeuralNetwork() {


        MLDataSet dataSet = Mnist.createDS(Mnist.TRAIN_NAME,60000);
        MLDataSet testSet = Mnist.createDS(Mnist.TEST_NAME,10000);

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true,784));
        network.addLayer(new BasicLayer(new ActivationElliott(), true, 700));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false,10));
        network.getStructure().finalizeStructure();
        network.reset();

        final Train train = new Backpropagation(network,dataSet);
    //    final Train train = new ResilientPropagation(network, dataSet);

        int epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
        } while(epoch < 7);
        train.finishTraining();


        double result = network.calculateError(testSet);


        System.out.println("Eroro_ra je: " + result);

        // test the neural network
        System.out.println("Neural Network Results:");
        int numberCorrect = 0;
        int numberOfValues = 0;

        for(MLDataPair pair: testSet ) {
             MLData output = network.compute(pair.getInput());
             String label = String.valueOf(findLabel(pair.getIdealArray()));
             String resultlabel = String.valueOf(findMax(output.getData()));

            if (label.equals(resultlabel)) {
                numberCorrect++;
            }
            numberOfValues++;
            System.out.println(label + ", " + resultlabel);


//            System.out.println(Arrays.toString(pair.getIdeal().getData())
//                    + ", output=" + Arrays.toString(output.getData()));
        }

        double percentageCorrect = numberCorrect * 100.0 / numberOfValues;
        System.out.println("Number correct: " + numberCorrect + " of " + numberOfValues + " (" + percentageCorrect + ")");

        Encog.getInstance().shutdown();

    }

    private int findLabel(double[] data) {
        for(int i = 0; i < 10; i++){
            if(data[i] == 1d)
                return  i;
        }
        return 10;
    }

    private int findMax(double[] data) {
        double max = data[0];
        int index = 0;
        for(int i = 1; i < 10; i++){
            if(data[i] > max){
                max = data[i];
                index = i;
            }
        }
        return index;
    }

}
