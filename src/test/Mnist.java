package test;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Mnist {

    public static final String TRAIN_NAME = Luncher.DIR + "\\mnist_train.csv";
    public static final String TEST_NAME = Luncher.DIR + "\\mnist_test.csv";



    public static String rank() throws Exception {
        Instances set;

        if (!Luncher.EXTEND) {
            set = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\train.arff"))));
            set.setClassIndex(0);
        } else {
            set = new Instances(new BufferedReader(new FileReader(new File(Luncher.DIR + "\\ExTrain.arff"))));
            set.setClassIndex(0);
        }
        Ranker ranker = new Ranker();
        AttributeSelection selector = new AttributeSelection();
        InfoGainAttributeEval eva = new InfoGainAttributeEval();
        ranker.setNumToSelect(set.numAttributes() - 1);
        selector.setEvaluator(eva);
        selector.setSearch(ranker);
        selector.SelectAttributes(set);
        selector.rankedAttributes();
        return (selector.toResultsString());
    }


    public static String[] createForestDataSet(String file){
        List<String> list = new ArrayList<String>();

        try {
            DataInputStream in = new DataInputStream(new FileInputStream(file));
            BufferedReader br = new BufferedReader(new InputStreamReader(in));

            String strLine;
            br.readLine(); // discard top one
            while ((strLine = br.readLine()) != null) {
                list.add(strLine);
            }

            in.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Error: " + e.getMessage());
        }
        return list.toArray(new String[list.size()]);

    }

    public static void createDataSet(String file, String name) {
        CSVLoader loader = null;
        Instances instances = null;
        List<String> lab = new ArrayList<>();
        for(int i = 0; i < 10; i++){
            lab.add(String.valueOf(i));
        }
        Attribute atr = new Attribute("Label",lab);

        try {
            loader = new CSVLoader();
            loader.setSource(new File(file));
            instances = loader.getDataSet();
            //instances.replaceAttributeAt(atr,0);
            instances.setClassIndex(0);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        saver.setFile(new File(name));
        saver.writeBatch();

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void extendSet(String path, String name, int size){
        int max = 0;
        int current = 0;
        int[] stlpec;
        List<String> line = null;
        List<String[]> result = new ArrayList<>();
        try {
            System.gc();
            CSVReader reader = new CSVReader(new FileReader(path));
            CSVWriter writer = new CSVWriter(new FileWriter(name),',','\0');
            String[] nextLine = reader.readNext();
            line = new ArrayList<String>(Arrays.asList(nextLine));
            line.add("ex1");
            line.add("ex2");
            line.add("ex3");
            line.add("ex4");
            line.add("ex5");
            writer.writeNext(Arrays.stream(line.toArray()).toArray(String[]::new));
            for(int j = 0; j < size; j++) {
                nextLine = reader.readNext();
                line = new ArrayList<String>(Arrays.asList(nextLine));
                stlpec = new int[28];
                double cx = 0;
                double cy = 0;
                double m = 0;
                int center = 0;
                for(int i = 1; i < line.size(); i++){
                    int y = i / 28;
                    int x = i % 28;
                    int cislo = Integer.valueOf(line.get(i));
                    cx += cislo * x;
                    cy += cislo * y;
                    m += cislo;

                    if(cislo > 0) {
                        current++;
                        stlpec[i % 28] += 1;
                    }

                    if(cislo == 0){
                        if(current > max) max = current;
                        current = 0;
                    }

                    if((i >= 348 && i <= 350) || (i >= 376 && i <= 378) || (i >= 404 && i <= 406)){
                        center += cislo;
                    }

                }

                int poc = 0;
                for(Integer cislo : stlpec){
                    if(cislo > 0) poc++;
                }


                line.add(String.valueOf(poc));
                line.add(String.valueOf(center));
                line.add(String.valueOf(max));
                line.add(String.valueOf((int) ((cx/m) * 100)));
                line.add(String.valueOf((int) ((cy/m) * 100)));

                // if(j > 9996) System.out.println( j + " size : " + line.size());

                //writer.writeNext(Arrays.stream(line.toArray()).toArray(String[]::new));
                result.add(Arrays.stream(line.toArray()).toArray(String[]::new));
                max = current = 0;
            }

//            nextLine = reader.readNext();
//            System.out.println(nextLine);
            result.add(Arrays.stream(line.toArray()).toArray(String[]::new));
//            result.add(Arrays.stream(line.toArray()).toArray(String[]::new));
            writer.writeAll(result);


        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}

