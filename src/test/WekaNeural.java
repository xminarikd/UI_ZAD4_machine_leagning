package test;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.pmml.jaxbbindings.NeuralNetwork;
import weka.filters.supervised.attribute.ClassConditionalProbabilities;
import weka.filters.unsupervised.attribute.Normalize;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import static javafx.scene.input.KeyCode.M;

public class WekaNeural {
    MultilayerPerceptron neuralNetwork = null;

    public void main() throws Exception {

        Instances trainSet;
        Instances testSet;

        System.out.println("Loading dataset...");

        if (!Luncher.EXTEND) {
            trainSet = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\train.arff"))));
            trainSet.setClassIndex(0);
            testSet = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR +  "\\test.arff"))));
            testSet.setClassIndex(0);
        } else {
            trainSet = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\ExTrain.arff"))));
            trainSet.setClassIndex(0);
            testSet = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\ExTest.arff"))));
            testSet.setClassIndex(0);
        }
        Normalize normalizeFilter = new Normalize();
        normalizeFilter.setInputFormat(trainSet);

        testSet = Normalize.useFilter(testSet, normalizeFilter);
        trainSet = Normalize.useFilter(trainSet, normalizeFilter);

       // testSet.get(0).toDoubleArray();


       // System.out.println(testSet.get(0));


        neuralNetwork = new MultilayerPerceptron();
        String backPropOptions =
                "-L " + 0.2 //learning rate
                        + " -M " + 0.1 //momentum
                        + " -N " + 3 //epoch
                        + " -V " + 0 //validation
                        + " -S " + 0 //seed
                        + " -E " + 10 //error
                        + " -H " + "300"; //hidden nodes.
        try {
            neuralNetwork.setOptions(Utils.splitOptions(backPropOptions));
            if (Luncher.TRAIN) {
                long start;
                long end;
                System.out.println("Trainnig....");
                start = System.currentTimeMillis();
                neuralNetwork.buildClassifier(trainSet);

                if (Luncher.EXTEND) weka.core.SerializationHelper.write("Exwekaneural", neuralNetwork);
                else weka.core.SerializationHelper.write("wekaneural", neuralNetwork);

                end = System.currentTimeMillis();
                System.out.println("\tTraining took: " + (end - start) / 1000.0);
            } else {

                if (Luncher.EXTEND)
                    neuralNetwork = (MultilayerPerceptron) weka.core.SerializationHelper.read(Luncher.DIR + "\\Exwekaneural");
                else
                    neuralNetwork = (MultilayerPerceptron) weka.core.SerializationHelper.read(Luncher.DIR + "\\wekaneural");
            }

            System.out.println("Evaluating...");
            Evaluation evaluation = new Evaluation(testSet);
            evaluation.evaluateModel(neuralNetwork, testSet);
            System.out.println("Eroror rate: " + evaluation.errorRate() * 100);
            System.out.println(evaluation.toSummaryString());
            System.out.println("Confusion matrix: " + Arrays.deepToString(evaluation.confusionMatrix()).replaceAll("], ", "]" + System.lineSeparator()));

        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    public MultilayerPerceptron init() throws Exception {
        MultilayerPerceptron nn = new MultilayerPerceptron();
        nn = (MultilayerPerceptron) weka.core.SerializationHelper.read(Luncher.DIR + "\\wekaneural");
        return nn;
    }



}


