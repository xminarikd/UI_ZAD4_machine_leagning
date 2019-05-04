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
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.stage.StageStyle;
import org.datavec.image.loader.NativeImageLoader;
import test.Luncher;
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

public class Controller {

    private final int WIDTH = 250;
    private final int HEIGHT = 250;
    public Canvas canvas;
    public ImageView imgView;
    public Button butt;
    public Label fr;
    public Label nr;
    public Label tr;
    private NativeImageLoader loader;
    private Label lblResult;
    private GraphicsContext ctx;
    private MultilayerPerceptron nn = null;
    private RandomForest forest = null;
    private J48 j48 = null;
    private String DIR;

    public void init() {

        try{
            canvas.setHeight(HEIGHT);
            canvas.setWidth(WIDTH);
            ctx = canvas.getGraphicsContext2D();

            loader = new NativeImageLoader(28,28,1,true);
            imgView.setFitHeight(150);
            imgView.setFitWidth(150);
            ctx.setLineWidth(12);
            ctx.setLineCap(StrokeLineCap.ROUND);

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
        WritableImage image;
        Image drawableImage;
        BufferedImage bufferedImage;

        image = new WritableImage(WIDTH, HEIGHT);
        canvas.snapshot(null, image);
        drawableImage = SwingFXUtils.fromFXImage(image, null);
        drawableImage = drawableImage.getScaledInstance(28, 28, Image.SCALE_SMOOTH);
        bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);

        //draw image
        Graphics graphics = bufferedImage.getGraphics();
        graphics.drawImage(drawableImage, 0, 0, null);
        graphics.dispose();

        return bufferedImage;
    }

    private void predict(BufferedImage img ) throws Exception {
        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        List<double[]> results = new ArrayList<>();
        double[] pole = new double[785];
        for(int i = 0; i < pixels.length; i++){
            if(pixels[i] == -1){
                pole[i] = 0d;
            }
            else if(pixels[i] == 0) pole[i] = 255d;
            else if(pixels[i] < 0) pole[i] = (pixels[i] + 128d);
            else pole[i] = pixels[i];
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
                predict(scaledImg);
            } catch (Exception e1) {
                e1.printStackTrace();
            }

    }

    public void initialize() throws Exception {
        DIR = System.getProperty("user.dir");
        nn = (MultilayerPerceptron) weka.core.SerializationHelper.read(DIR + "\\wekaneural");
        j48 = (J48) weka.core.SerializationHelper.read(DIR + "\\wekaTree");
        forest = (RandomForest) weka.core.SerializationHelper.read(DIR + "\\wekaForest");
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

    public void click(MouseEvent mouseEvent) {
        if (mouseEvent.getButton() == MouseButton.SECONDARY) {
            clear(ctx);
        }
    }

    public void dragged(MouseEvent mouseEvent) {
        ctx.setStroke(Color.BLACK);
        ctx.lineTo(mouseEvent.getX(), mouseEvent.getY());
        ctx.stroke();
    }

    public void pressed(MouseEvent mouseEvent) {
        ctx.setStroke(Color.BLACK);
        ctx.beginPath();
        ctx.moveTo(mouseEvent.getX(), mouseEvent.getY());
        ctx.stroke();
    }
}
