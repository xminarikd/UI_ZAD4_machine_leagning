package test;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;

public class RndForest {

    public void init() throws Exception {
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

        int featuresToUse = (int) Math.sqrt(28 * 28);
        RandomForest wekaRF = new RandomForest();
        wekaRF.setNumExecutionSlots(1);
        wekaRF.setMaxDepth(0);
        wekaRF.setNumFeatures(featuresToUse);
        wekaRF.setNumIterations(50); //number of trees


        if(Luncher.TRAIN) {
            long start;
            long end;
            System.out.println("Trainnig....");
            start = System.currentTimeMillis();
            wekaRF.buildClassifier(trainSet);

            if(Luncher.EXTEND) weka.core.SerializationHelper.write("EXwekaForest",wekaRF);
            else weka.core.SerializationHelper.write("wekaForest",wekaRF);

            end = System.currentTimeMillis();
            System.out.println("\tTraining took: " + (end - start) / 1000.0);
        }else {
            if(Luncher.EXTEND) wekaRF = (RandomForest) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\ExwekaForest");
            else wekaRF = (RandomForest) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaForest");
        }

        System.gc();

        Evaluation evaluation = new Evaluation(testSet);
        evaluation.evaluateModel(wekaRF,testSet);
        System.out.println("Eroror rate: " + evaluation.errorRate() * 100 );
        System.out.println("Results:" + evaluation.toSummaryString());
        System.out.println("Confusion matrix: \n" + Arrays.deepToString(evaluation.confusionMatrix()).replaceAll("], ", "]" + System.lineSeparator()));

    }


}
