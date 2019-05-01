package test;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.pmml.jaxbbindings.NeuralNetwork;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;

import static javafx.scene.input.KeyCode.M;

public class WekaNeural {

    public void main() throws Exception {

        Instances trainSet;
        Instances testSet;

        if(!Luncher.EXTEND) {
            trainSet = new Instances(new BufferedReader(new FileReader(new File("C:\\Users\\minar\\Documents\\UI04v0.2\\train.arff"))));
            trainSet.setClassIndex(0);
            testSet = new Instances(new BufferedReader(new FileReader(new File("C:\\Users\\minar\\Documents\\UI04v0.2\\test.arff"))));
            testSet.setClassIndex(0);
        }
        else{
            trainSet = new Instances(new BufferedReader(new FileReader(new File("C:\\Users\\minar\\Documents\\UI04v0.2\\ExTrain.arff"))));
            trainSet.setClassIndex(0);
            testSet = new Instances(new BufferedReader(new FileReader(new File("C:\\Users\\minar\\Documents\\UI04v0.2\\ExTest.arff"))));
            testSet.setClassIndex(0);
        }
        Normalize normalizeFilter = new Normalize();
        normalizeFilter.setInputFormat(trainSet);

        testSet = Normalize.useFilter(testSet,normalizeFilter);
        trainSet = Normalize.useFilter(trainSet,normalizeFilter);


        MultilayerPerceptron neuralNetwork = new MultilayerPerceptron();
        String backPropOptions =
                "-L "+ 0.2 //learning rate
                +" -M " +0.1 //momentum
                +" -N "+2 //epoch
                +" -V "+0 //validation
                +" -S "+0 //seed
                +" -E "+10 //error
                +" -H "+ "150"; //hidden nodes.
        try {
            neuralNetwork.setOptions(Utils.splitOptions(backPropOptions));
            if(Luncher.TRAIN) {
                long start;
                long end;
                System.out.println("Trainnig....");
                start = System.currentTimeMillis();
                neuralNetwork.buildClassifier(trainSet);

                if(Luncher.EXTEND) weka.core.SerializationHelper.write("Exwekaneural",neuralNetwork);
                else weka.core.SerializationHelper.write("wekaneural",neuralNetwork);

                end = System.currentTimeMillis();
                System.out.println("\tTraining took: " + (end - start) / 1000.0);
            } else {

                if(Luncher.EXTEND) neuralNetwork = (MultilayerPerceptron) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\Exwekaneural");
                else neuralNetwork = (MultilayerPerceptron) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaneural");
            }

            Evaluation evaluation = new Evaluation(trainSet);
            evaluation.evaluateModel(neuralNetwork,testSet);
            System.out.println("Eroror rate: " + evaluation.errorRate() * 100);
            System.out.println("Confusion matrix: " + Arrays.deepToString(evaluation.confusionMatrix()).replaceAll("], ", "]" + System.lineSeparator()));

        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    public MultilayerPerceptron init() throws Exception {
        MultilayerPerceptron nn = new MultilayerPerceptron();
        nn = (MultilayerPerceptron) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaneural");
        return nn;
    }

}


