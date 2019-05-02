package test;

import org.apache.mahout.classifier.ConfusionMatrix;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.common.RandomUtils;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Combi {

    public void init() throws Exception {
        Instances testSetW = new Instances(new BufferedReader(new FileReader(new File("C:\\Users\\minar\\Documents\\UI04v0.2\\test.arff"))));
        testSetW.setClassIndex(0);
        Normalize normalizeFilter = new Normalize();
        normalizeFilter.setInputFormat(testSetW);
        testSetW = Normalize.useFilter(testSetW,normalizeFilter);

        String[] testSetF =  Mnist.createForestDataSet(Mnist.TEST_NAME);
        String descriptor = Forest.buildDescriptor(784);
        Data data = Forest.loadData(testSetF,descriptor);
        evaluate(testSetW);
    }

    public void evaluate(Instances testSetW) throws Exception {

        Random rng = RandomUtils.getRandom();
        MultilayerPerceptron neuralNetwork = (MultilayerPerceptron) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaneural");
        J48 j48 = (J48) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaTree");
        RandomForest forest = (RandomForest) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaForest");
//        Configuration config = new Configuration();
//        Path path = new Path("C:\\Users\\minar\\Documents\\UI04v0.2\\saved-forest_10-trees.txt");
//        DecisionForest forest = DecisionForest.load(config, path);

        try {
            int numberCorrect = 0;
            int numberOfValues = 0;
            int iter = 0;
            int pocetNezhoda = 0;
            int pocetZhoda = 0;
            String cisla = "0,1,2,3,4,5,6,7,8,9";
            List<String> col = Arrays.asList(cisla.split(","));
            ConfusionMatrix confusionMatrix = new ConfusionMatrix(col,"unknown");
            int[] results = null;

            for (int i = 0; i < testSetW.size(); i++) {
//                Instance oneSample = test.get(i);
//                double actualIndex = oneSample.get(0);
                //int actualLabel = test.getDataset().valueOf(0, String.valueOf((int) actualIndex));
                int actualLabel = (int) testSetW.instance(i).toDoubleArray()[0];
                results = new int[3];
                //results[0] = (int) forest.classify(test.getDataset(), rng, oneSample);
                results[0] = (int) forest.classifyInstance(testSetW.instance(i));
                results[1] = (int) neuralNetwork.classifyInstance(testSetW.instance(i));
                results[2] = (int) j48.classifyInstance(testSetW.instance(i));
                int finalLabel = 10;

                if(results[0] == results[1] && results[1] == results[2]) {
                    pocetZhoda++;
                    finalLabel = results[0];
                }
                else if(results[0] == results[1]){
                    finalLabel = results[0];
                }
                else if(results[1] == results[2]){
                    finalLabel = results[1];
                }
                else if(results[0] == results[2]){
                    finalLabel = results[0];
                }
                else {
                    pocetNezhoda++;
                    finalLabel =results[0];
                    //iter = (iter + 1) % 3;
                }

                confusionMatrix.addInstance(String.valueOf(finalLabel),String.valueOf(actualLabel));
                //System.out.println("label = " + label + " actual = " + actualLabel);

                if (finalLabel == actualLabel) {
                    numberCorrect++;
                }
                numberOfValues++;
            }

            double percentageCorrect = numberCorrect * 100.0 / numberOfValues;
            System.out.println("Number of trees: " + " -> Number correct: " + numberCorrect + " of " + numberOfValues + " (" + percentageCorrect + ")");
            System.out.println(Arrays.deepToString(confusionMatrix.getConfusionMatrix()).replaceAll("], ", "]" + System.lineSeparator()));
            System.out.println("Pocet zhod je : " + pocetZhoda + " pocet nezhod je : " + pocetNezhoda);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Error: " + e.getMessage());
        }


    }

}
