FROM openjdk:11

WORKDIR /app

COPY target/wine-quality-1.0.jar wine-quality-1.0.jar
COPY ValidationDataset.csv ValidationDataset.csv
COPY wine_model/ wine_model/

CMD ["java", "-cp", "wine-quality-1.0.jar", "edu.cs643.WinePredictor", "ValidationDataset.csv", "wine_model"]
