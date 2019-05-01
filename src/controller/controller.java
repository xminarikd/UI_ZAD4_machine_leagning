package controller;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.stage.Stage;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.common.RandomUtils;
import org.datavec.image.loader.NativeImageLoader;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class controller {

    private final int CANVAS_WIDTH = 250;
    private final int CANVAS_HEIGHT = 250;
    public Button bBegin;
    private NativeImageLoader loader;
    private Label lblResult;
    MultilayerPerceptron nn;
    DecisionForest forest = null;
    J48 j48 = null;



    public void loadProject(ActionEvent actionEvent) {

        try{
            Canvas canvas = new Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
            ImageView imgView = new ImageView();
            GraphicsContext ctx = canvas.getGraphicsContext2D();


            loader = new NativeImageLoader(28,28,1,true);
            imgView.setFitHeight(150);
            imgView.setFitWidth(150);
            ctx.setLineWidth(12);
            ctx.setLineCap(StrokeLineCap.BUTT);
            lblResult = new Label();

            HBox hbBottom = new HBox(10,canvas, imgView);
            VBox root = new VBox(5, hbBottom, lblResult);
            hbBottom.setAlignment(Pos.CENTER);
            root.setAlignment(Pos.CENTER);

            Scene scene = new Scene(root, 700, 500);
            Stage stage=new Stage();

            stage.setScene(scene);

            stage.show();
            stage.setTitle("Handwritten Digit Recognition");

            canvas.setOnMousePressed(e -> {
                ctx.setStroke(Color.BLACK);
                ctx.beginPath();
                ctx.moveTo(e.getX(), e.getY());
                ctx.stroke();
            });
            canvas.setOnMouseDragged(e -> {
                ctx.setStroke(Color.BLACK);
                ctx.lineTo(e.getX(), e.getY());
                ctx.stroke();
            });
            canvas.setOnMouseClicked(e -> {
                if (e.getButton() == MouseButton.SECONDARY) {
                    clear(ctx);
                }
            });
            canvas.setOnKeyReleased(e -> {
                if(e.getCode() == KeyCode.ENTER) {
                    BufferedImage scaledImg = getScaledImage(canvas);
                    imgView.setImage(SwingFXUtils.toFXImage(scaledImg, null));
                    try {
                        predictImage(scaledImg);
                        System.out.println("Hello");
                    } catch (Exception e1) {
                        e1.printStackTrace();
                    }
                }
            });
            clear(ctx);
            canvas.requestFocus();

        }catch(Exception ex){
            System.out.println(ex);
        }
    }


    private void clear(GraphicsContext ctx) {
        ctx.setFill(Color.WHITE);
        ctx.fillRect(0, 0, 300, 300);
    }

    private BufferedImage getScaledImage(Canvas canvas) {

        WritableImage writableImage = new WritableImage(CANVAS_WIDTH, CANVAS_HEIGHT);
        canvas.snapshot(null, writableImage);
        Image tmp =  SwingFXUtils.fromFXImage(writableImage, null).getScaledInstance(28, 28, Image.SCALE_SMOOTH);
        BufferedImage scaledImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = scaledImg.getGraphics();
        graphics.drawImage(tmp, 0, 0, null);
        graphics.dispose();
        return scaledImg;
    }

    private void predictImage(BufferedImage img ) throws Exception {
        Random rng = RandomUtils.getRandom();
        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        System.out.println("ok");
        List<double[]> results = new ArrayList<>();
        double[] pole = new double[785];
        for(int i = 0; i < pixels.length; i++){
            if(pixels[i] == -1){
                pole[i] = 0d;

            }
            else if(pixels[i] == 0) pole[i] = 255d;
            else pole[i] = Math.abs(pixels[i]);
        }
        ArrayList<Attribute> attributes = new ArrayList<>();
        List<String> lab = new ArrayList<>();
        for(int i = 0; i < 10; i++){
            lab.add(String.valueOf(i));
        }
        attributes.add(new Attribute("Label",lab));
        for(int i = 1; i <= 28; i++) {
            for (int j = 1; j <= 28; j++) {
                attributes.add(new Attribute(i + "x" + j, 0));
            }
        }

        System.out.println("OK");
        Instances ins = new Instances("mnist_test",attributes,1);
        ins.add(new DenseInstance(1.0,pole));
        ins.setClassIndex(0);



//        Normalize normalizeFilter = new Normalize();
//        normalizeFilter.setInputFormat(i);
//        i = Normalize.useFilter(i,normalizeFilter);

       // System.out.println(ins);

//        stringBuilder.deleteCharAt(stringBuilder.length() - 1);
//
//        String[] sData = new String[] {stringBuilder.toString()};
//
//        Instances i = new Instances();
//        String descriptor = Forest.buildDescriptor(784);
//
//
//        Data data = Forest.loadData(sData, descriptor);
//        Instance oneSample = data.get(0);
//        Forest ff = new Forest();
//        try {
//            double label = ff.eva(sData);
//            System.out.println(label);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }


        //j48 = (J48) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaTree");
        double label = j48.classifyInstance(ins.instance(0));
        double label2 = nn.classifyInstance(ins.instance(0));
        System.out.println(label + " @@@ " + label2);



    }

    public void initialize() throws Exception {
        nn = (MultilayerPerceptron) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaneural");
        j48 = (J48) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaTree");
    }

}
