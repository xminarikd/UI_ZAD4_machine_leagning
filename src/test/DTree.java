package test;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Normalize;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;

public class DTree {

    public void main() throws Exception {
        Instances trainSet;
        Instances testSet;
        System.out.println("Loading dataset...");
        if(!Luncher.EXTEND) {
            trainSet = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\train.arff"))));
            trainSet.setClassIndex(0);
            testSet = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\test.arff"))));
            testSet.setClassIndex(0);
        }
        else{
            trainSet = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\ExTrain.arff"))));
            trainSet.setClassIndex(0);
            testSet = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\ExTest.arff"))));
            testSet.setClassIndex(0);
        }
        Normalize normalizeFilter = new Normalize();
        normalizeFilter.setInputFormat(trainSet);

        testSet = Normalize.useFilter(testSet,normalizeFilter);
        trainSet = Normalize.useFilter(trainSet,normalizeFilter);

        J48 j48 = new J48();
        j48.setUseLaplace(true);
        j48.setCollapseTree(false);
        j48.setUnpruned(true);
        j48.setMinNumObj(3);
        j48.setUseMDLcorrection(true);
        j48.setSeed(0);
        j48.setSubtreeRaising(true);


        if(Luncher.TRAIN) {
            long start;
            long end;
            System.out.println("Trainnig....");
            start = System.currentTimeMillis();
            j48.buildClassifier(trainSet);

            if(Luncher.EXTEND) weka.core.SerializationHelper.write("EXwekaTree",j48);
            else weka.core.SerializationHelper.write("wekaTree",j48);

            end = System.currentTimeMillis();
            System.out.println("\tTraining took: " + (end - start) / 1000.0);
        }else {
            if(Luncher.EXTEND) j48 = (J48) weka.core.SerializationHelper.read(Luncher.DIR + "\\ExwekaTree");
            else j48 = (J48) weka.core.SerializationHelper.read(Luncher.DIR + "\\wekaTree");
        }

        System.out.println("Evaluating...");
        Evaluation evaluation = new Evaluation(testSet);
        evaluation.evaluateModel(j48,testSet);
        System.out.println("Eroror rate: " + evaluation.errorRate() * 100 );
        System.out.println(evaluation.toSummaryString());
        int[][] CM = new int[10][10];
        double[][] CMD = evaluation.confusionMatrix();
        for(int i = 0; i < 10; i++)
            for(int j = 0; j < 10; j++){
                CM[i][j] = (int) CMD[i][j];
            }
        //System.out.println("Confusion matrix: \n" + Arrays.deepToString((evaluation.confusionMatrix())).replaceAll("], ", "]" + System.lineSeparator()));
        System.out.println("Confusion matrix: \n" + Arrays.deepToString(CM).replaceAll("], ", "]" + System.lineSeparator()));

    }



}
