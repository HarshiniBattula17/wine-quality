#  Wine Quality Prediction – CS643 Cloud Computing

This project involves building and deploying a parallel machine learning application using Apache Spark to predict wine quality. It includes training multiple classification models and evaluating their performance using F1 Score. The best model is saved and used for predictions on a validation dataset.

---

---

## 🗂 Project Files

- `src/main/java/edu/cs643/WineTrainer.java` – Spark ML training application.
- `src/main/java/edu/cs643/WinePredictor.java` – Spark ML prediction application.
- `Dockerfile` – Docker configuration to containerize the prediction app.
- `CleanedTrainingDataset.csv` – Cleaned dataset used for training the model.
- `ValidationDataset.csv` – Dataset used to test the saved model.
- `target/wine-quality-1.0.jar` – Maven-packaged executable JAR.

---

## 🛠 Tools and Technologies

- Java
- Apache Spark (MLlib)
- Apache Maven
- Docker
- AWS EC2 (Spark cluster setup)
- GitHub & Docker Hub

---

## 🔄 Local Execution Steps

### 1. Build the Project

```bash
mvn clean package
```

### 2. Run Model Training Locally

```bash
spark-submit \
  --class edu.cs643.WineTrainer \
  --master local[*] \
  target/wine-quality-1.0.jar \
  CleanedTrainingDataset.csv \
  wine_model | tee train_output.log
```

### 3. Run Prediction Locally

```bash
spark-submit \
  --class edu.cs643.WinePredictor \
  --master local[*] \
  target/wine-quality-1.0.jar \
  ValidationDataset.csv \
  wine_model | tee predict_output.log
```

---

##  AWS Spark Cluster Execution

### 1. Upload JAR and CSVs to Master

```bash
scp -i ~/.ssh/projectkey.pem target/wine-quality-1.0.jar ubuntu@<master-ip>:~/wine-quality/
scp -i ~/.ssh/projectkey.pem CleanedTrainingDataset.csv ubuntu@<master-ip>:~/wine-quality/
scp -i ~/.ssh/projectkey.pem ValidationDataset.csv ubuntu@<master-ip>:~/wine-quality/
```

### 2. Train on Cluster

```bash
spark-submit \
  --class edu.cs643.WineTrainer \
  --master spark://<master-ip>:7077 \
  wine-quality/wine-quality-1.0.jar \
  wine-quality/CleanedTrainingDataset.csv \
  wine-quality/wine_model | tee train_output.log
```

### 3. Predict on Cluster

```bash
spark-submit \
  --class edu.cs643.WinePredictor \
  --master spark://<master-ip>:7077 \
  wine-quality/wine-quality-1.0.jar \
  wine-quality/ValidationDataset.csv \
  wine-quality/wine_model | tee predict_output.log
```

---

Results:
🔧 Training Performance
The F1 scores for various models on the training dataset were:

Logistic Regression: 0.5995664097102595

Decision Tree: 0.5721479565830936

Random Forest: 0.7173295719472501

Best Model (Random Forest):  0.7173295719472501

 Validation Performance
The selected model was evaluated on the validation dataset, resulting in:

Validation F1 Score: 0.4683180769012569
