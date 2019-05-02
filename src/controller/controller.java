package controller;

import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseButton;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.stage.StageStyle;
import org.apache.mahout.common.RandomUtils;
import org.datavec.image.loader.NativeImageLoader;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Controller {

    private final int CANVAS_WIDTH = 250;
    private final int CANVAS_HEIGHT = 250;
    public Button bBegin;
    public Canvas canvas;
    public ImageView imgView;
    public Label flable;
    public Label nlable;
    public Label tlable;
    public Button butt;
    public Label fr;
    public Label nr;
    public Label tr;
    private NativeImageLoader loader;
    private Label lblResult;
    MultilayerPerceptron nn = null;
    RandomForest forest = null;
    J48 j48 = null;


    public void init() {

        try{
            canvas.setHeight(CANVAS_HEIGHT);
            canvas.setWidth(CANVAS_WIDTH);
            GraphicsContext ctx = canvas.getGraphicsContext2D();


            loader = new NativeImageLoader(28,28,1,true);
            imgView.setFitHeight(150);
            imgView.setFitWidth(150);
            ctx.setLineWidth(12);
            ctx.setLineCap(StrokeLineCap.BUTT);
            lblResult = new Label();

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

            clear(ctx);
            canvas.requestFocus();

        }catch(Exception ex){
            ex.printStackTrace();
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


        Instances ins = new Instances("mnist_test",attributes,1);
        ins.add(new DenseInstance(1.0,pole));
        ins.setClassIndex(0);



        int label = (int) j48.classifyInstance(ins.instance(0));
        int label2 = (int) nn.classifyInstance(ins.instance(0));
        int label3 = (int) forest.classifyInstance(ins.instance(0));

        tr.setText(String.valueOf(label));
        nr.setText(String.valueOf(label2));
        fr.setText(String.valueOf(label3));


        System.out.println(label + " @@@ " + label2 + " @@@ " + label3);



    }

    public void enter(ActionEvent actionEvent) {
            BufferedImage scaledImg = getScaledImage(canvas);
            imgView.setImage(SwingFXUtils.toFXImage(scaledImg, null));
            try {
                predictImage(scaledImg);
            } catch (Exception e1) {
                e1.printStackTrace();
            }

    }

    public void initialize() throws Exception {
        nn = (MultilayerPerceptron) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaneural");
        j48 = (J48) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaTree");
        forest = (RandomForest) weka.core.SerializationHelper.read("C:\\Users\\minar\\Documents\\UI04v0.2\\wekaForest");
    }

    public void exit(ActionEvent actionEvent) {
        Platform.exit();
    }

    public void help(ActionEvent actionEvent) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Information Dialog");
        alert.setHeaderText("User's Guide");
        alert.setContentText("You can draw with the left button on the highlighted area. \n Clear board with right button. \n Evaluation with the Predict button.");
        alert.initStyle(StageStyle.UNDECORATED);
        alert.showAndWait();
    }
}
