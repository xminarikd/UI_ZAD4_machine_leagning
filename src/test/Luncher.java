package test;


public class Luncher {

    public static final boolean TRAIN = false;
    public static final boolean EXTEND = false;
    public static final int MODEL = 1;
    public static String DIR;

    public static void main(String[] args) throws Exception {

        DIR = (System.getProperty("user.dir"));

//        Mnist.extendSet(Mnist.TEST_NAME,"ExTest.csv",10000);
//        Mnist.extendSet(Mnist.TRAIN_NAME,"ExTrain.csv",60000);

//        Mnist.createDataSet("C:\\Users\\minar\\Documents\\UI04v0.2\\ExTest.csv", "ExTest.arff");
//        Mnist.createDataSet("C:\\Users\\minar\\Documents\\UI04v0.2\\ExTrain.csv", "ExTrain.arff");

//        System.out.println(Mnist.rank());

        if(MODEL == 1) {
            WekaNeural wnn = new WekaNeural();
            wnn.main();
        }
        else if(MODEL == 2) {
            DTree tree = new DTree();
            tree.main();
        }
        else if(MODEL == 3) {
            RndForest forest = new RndForest();
            forest.init();
        }
        else if(MODEL == 4) {
            Combi combi = new Combi();
            combi.init();
        }

    }

}
