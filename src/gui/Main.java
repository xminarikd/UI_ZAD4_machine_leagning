package gui;

import controller.Controller;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.stage.StageStyle;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {

        Controller controller;

        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/gui/windows.fxml"));

        Parent parent = (Parent) fxmlLoader.load();
        controller = fxmlLoader.getController();
        controller.init();

        Stage stage = new Stage();
        stage.setTitle("Digit Recognition");
        stage.initStyle(StageStyle.UNDECORATED);
        stage.setScene(new Scene(parent));
        stage.show();
    }

    public static void main(String[] args){
        launch(args);
    }

}
