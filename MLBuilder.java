/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mlbuilder;

import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;
/**
 *
 * @author Nina
 */
public class MLBuilder {

    /**
     * pomocna funkcija za otvaranje fajla
     * @param path
     * @return 
     */
    public static InputStreamReader getReader(String path){
        try{
            return new InputStreamReader(new FileInputStream(path), "UTF-8");
        }catch(Exception e){
            e.printStackTrace();
        }
        return null;
    }
    
    /**
     * ucitavanje podataka iz csv fajla u RDD
     * @param session
     * @param path
     * @return 
     */
    public static JavaRDD<LabeledPoint> loadData(SparkSession session, String path){
        
        ArrayList<LabeledPoint> data = new ArrayList<>();
        CsvParserSettings settings= new CsvParserSettings();
        settings.getFormat().setLineSeparator("\n");
        
        CsvParser parser = new CsvParser(settings);
        
        parser.beginParsing(getReader(path));
                
        String[] row;
        while((row = parser.parseNext()) != null){
            double label = Double.parseDouble(row[row.length-1]);
            double[] features = new double[row.length-1];
            for(int i = 0; i < row.length-1; i++)
                features[i] = Double.parseDouble(row[i].trim());
            data.add(new LabeledPoint(label, Vectors.dense(features)));
        }
        
        JavaSparkContext jc = JavaSparkContext.fromSparkContext(session.sparkContext());
        
        return jc.parallelize(data);
    }
    
    /**
     * treniranje modela klasifikacije na osnovu trening podataka 
     * @param trainingData
     * @return 
     */
    private static LogisticRegressionModel trainModel(JavaRDD<LabeledPoint> trainingData) {
       LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(trainingData.rdd());
       return model;
    }

    /**
     * izracunavanje tacnosti za date podatke i istrenirani model
     * @param trainingData
     * @param model
     * @return 
     */
    private static double computeAccuracy(JavaRDD<LabeledPoint> trainingData, LogisticRegressionModel model) {
        
        JavaPairRDD<Object, Object> predictionAndLabels = trainingData.mapToPair(p ->
        new Tuple2<>(model.predict(p.features()), p.label()));

        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double accuracy = metrics.accuracy();
        
        return accuracy;
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        // inicijalizacija Sparka i kreiranje sesije
        SparkSession session = SparkSession
                                .builder()
                                .appName("MLBuilder")
                                .master("local")
                                .getOrCreate();
        
        SparkContext context = session.sparkContext();
        
        // ucitavanje podataka za treniranje u RDD
        JavaRDD<LabeledPoint> trainingData = loadData(session, args[0]); 
        
        // kreiranje klasifikacionog modela na osnovu trening podataka
        LogisticRegressionModel model = trainModel(trainingData);
        
        // evaluacija modela nad trening podacima
        double accuracy = computeAccuracy(trainingData, model);
        System.out.println("Accuracy on training data: " + accuracy);
        
        // skladistenje modela
        model.save(context, "C:/spark-models/classification");
        
        session.close();
        
    }
    
}
