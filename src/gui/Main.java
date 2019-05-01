package gui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.stage.StageStyle;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {

        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/gui/windows.fxml"));

        Parent parent = (Parent) fxmlLoader.load();

        Stage stage = new Stage();
        //scene.getStylesheets().add("/styles/Styles.css");
        stage.setTitle("title");
       // stage.setOpacity(1.0);
        stage.initStyle(StageStyle.TRANSPARENT);
        stage.setScene(new Scene(parent));
        stage.show();
    }

    public static void main(String[] args){
        launch(args);
    }

}
