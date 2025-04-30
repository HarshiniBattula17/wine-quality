package edu.cs643;

import java.io.IOException;
import java.util.Arrays;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.*;

public class WineTrainer {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("Usage: WineTrainer <input_csv> <model_output_path>");
            System.exit(1);
        }

        String inputCsv = args[0];
        String modelOutputPath = args[1];

        SparkSession spark = SparkSession.builder().appName("Wine Quality Trainer").getOrCreate();
        Dataset<Row> data = spark.read().option("header", "true").option("inferSchema", "true").csv(inputCsv);

        String[] featureCols = Arrays.stream(data.columns())
                .filter(c -> !c.equals("quality"))
                .toArray(String[]::new);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("quality")
                .setOutputCol("label");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("featuresRaw");

        StandardScaler scaler = new StandardScaler()
                .setInputCol("featuresRaw")
                .setOutputCol("features");

        // Try multiple classifiers
        Classifier[] classifiers = new Classifier[]{
            new LogisticRegression().setMaxIter(100).setRegParam(0.01),
            new DecisionTreeClassifier().setMaxDepth(10),
            new RandomForestClassifier().setNumTrees(200).setMaxDepth(10)
        };

        String[] modelNames = {"Logistic Regression", "Decision Tree", "Random Forest"};

        double bestF1 = 0.0;
        PipelineModel bestModel = null;

        for (int i = 0; i < classifiers.length; i++) {
            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                labelIndexer, assembler, scaler, classifiers[i]
            });

            PipelineModel model = pipeline.fit(data);
            Dataset<Row> predictions = model.transform(data);

            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("f1");

            double f1 = evaluator.evaluate(predictions);
            System.out.println(modelNames[i] + " F1 Score: " + f1);

            if (f1 > bestF1) {
                bestF1 = f1;
                bestModel = model;
                bestModel.write().overwrite().save(modelOutputPath);
            }
        }

        System.out.println("Best F1 Score: " + bestF1);
        spark.stop();
    }
}
