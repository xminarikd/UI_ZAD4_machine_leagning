package test;

import org.apache.mahout.classifier.df.data.DescriptorException;
import weka.classifiers.trees.RandomForest;

public class Luncher {

    public static final boolean TRAIN = false;
    public static final boolean EXTEND = false;

    public static void main(String[] args) throws Exception {

//        Mnist.extendSet(Mnist.TEST_NAME,"ExTest.csv",10000);
//        Mnist.extendSet(Mnist.TRAIN_NAME,"ExTrain.csv",60000);

//        Mnist.createTreeDataSet("C:\\Users\\minar\\Documents\\UI04v0.2\\ExTest.csv", "ExTestskuska.arff");
//        Mnist.createTreeDataSet("C:\\Users\\minar\\Documents\\UI04v0.2\\ExTrain.csv", "ExTrain.arff");


        WekaNeural wnn = new WekaNeural();
        wnn.main();

//        Forest forest = new Forest();
//        try {
//            forest.main(50);
//        } catch (DescriptorException e) {
//            e.printStackTrace();
//        }

//        DTree tree = new DTree();
//        tree.main();

//        RndForest forest = new RndForest();
//        forest.init();


//        Combi combi = new Combi();
//        combi.init();


    }

}
