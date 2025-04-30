package edu.cs643;

import java.io.IOException;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.*;

public class WinePredictor {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("Usage: WinePredictor <validation_csv> <model_path>");
            System.exit(1);
        }

        String validationCsv = args[0];
        String modelPath = args[1];

        SparkSession spark = SparkSession.builder().appName("Wine Quality Predictor").getOrCreate();
        Dataset<Row> data = spark.read().option("header", "true").option("inferSchema", "true").csv(validationCsv);

        PipelineModel model = PipelineModel.load(modelPath);
        Dataset<Row> predictions = model.transform(data);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);
        System.out.println("Validation F1 Score: " + f1);

        spark.stop();
    }
}
