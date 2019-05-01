package test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.ConfusionMatrix;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.DefaultTreeBuilder;
import org.apache.mahout.classifier.df.data.*;
import org.apache.mahout.classifier.df.ref.SequentialBuilder;
import org.apache.mahout.common.RandomUtils;
import org.uncommons.maths.Maths;
import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Forest {

    private DecisionForest buildForest(int numberOfTrees, Data data) {
        int m = (int) Math.floor(Maths.log(2, data.getDataset().nbAttributes()) + 1);

        DefaultTreeBuilder treeBuilder = new DefaultTreeBuilder();
        SequentialBuilder forestBuilder = new SequentialBuilder(RandomUtils.getRandom(), treeBuilder, data.clone());
        treeBuilder.setM(m);

        return forestBuilder.build(numberOfTrees);
    }

    public static String buildDescriptor(int numberOfFeatures) {
        StringBuilder builder = new StringBuilder("L ");
        for (int i = 0; i < numberOfFeatures; i++) {
            builder.append("N ");
        }
        return builder.toString();
    }

    public static Data loadData(String[] sData, String descriptor) throws DescriptorException {
        Dataset dataset = DataLoader.generateDataset(descriptor, false, sData);
        return DataLoader.loadData(dataset, sData);
    }

    public void main(int numberOfTrees) throws DescriptorException {
        String[] trainSet = Mnist.createForestDataSet(Mnist.TRAIN_NAME);
        String[] testSet =  Mnist.createForestDataSet(Mnist.TEST_NAME);
        String descriptor = buildDescriptor(784);

        trainAndtest(numberOfTrees,trainSet,testSet,descriptor);

    }



    private void trainAndtest(int numberOfTrees, String[] trainSet, String[] testSet, String descriptor) throws DescriptorException {

        Data data = loadData(trainSet,descriptor);
        Random rng = RandomUtils.getRandom();
        DecisionForest forest = null;

        if(Luncher.TRAIN) {
            forest = buildForest(numberOfTrees,data);
            try {
                String path = "saved-forest_" + numberOfTrees + "-trees.txt";
                File file = new File(path);
                DataOutputStream dos = new DataOutputStream(new FileOutputStream(file));
                forest.write(dos);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }else {
            try {
                Configuration config = new Configuration();
                Path path = new Path("C:\\Users\\minar\\Documents\\UI04v0.2\\saved-forest_10-trees.txt");
                forest = DecisionForest.load(config, path);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        Data test = DataLoader.loadData(data.getDataset(), testSet);

        try {
            int numberCorrect = 0;
            int numberOfValues = 0;
            String cisla = "0,1,2,3,4,5,6,7,8,9";
            List<String> col = Arrays.asList(cisla.split(","));
            ConfusionMatrix confusionMatrix = new ConfusionMatrix(col,"unknown");


            for (int i = 0; i < test.size(); i++) {
                Instance oneSample = test.get(i);
                double actualIndex = oneSample.get(0);
                int actualLabel = test.getDataset().valueOf(0, String.valueOf((int) actualIndex));

                int classify = (int) forest.classify(test.getDataset(), rng, oneSample);
                //int label = test.getDataset().valueOf(0, String.valueOf((int) classify));
                confusionMatrix.addInstance(String.valueOf( classify),String.valueOf(actualLabel));
                //System.out.println("label = " + label + " actual = " + actualLabel);
                //System.out.println(classify);

                if (classify == actualLabel) {
                    numberCorrect++;
                }
                numberOfValues++;
            }

            double percentageCorrect = numberCorrect * 100.0 / numberOfValues;
            System.out.println("Number of trees: " + numberOfTrees + " -> Number correct: " + numberCorrect + " of " + numberOfValues + " (" + percentageCorrect + ")");
            System.out.println(Arrays.deepToString(confusionMatrix.getConfusionMatrix()).replaceAll("], ", "]" + System.lineSeparator()));

        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Error: " + e.getMessage());
        }
    }

    public double eva(String[] sdata) throws IOException, DescriptorException {
        Random rng = RandomUtils.getRandom();
        Configuration config = new Configuration();
        Path path = new Path("C:\\Users\\minar\\Documents\\UI04v0.2\\saved-forest_10-trees.txt");
        DecisionForest forest = DecisionForest.load(config, path);
        String descriptor = Forest.buildDescriptor(784);
        Data data = Forest.loadData(sdata,descriptor);
        return forest.classify(data.getDataset(),rng,data.get(0));

    }





}
