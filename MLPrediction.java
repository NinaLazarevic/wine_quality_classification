/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mlprediction;

import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author Nina
 */
public class MLPrediction {
    
    /**
     * Ucitavanje fajla - pomocna funkcija
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
     * ucitavanje podataka iz csv fajla cija je putanja prosledjena kao argument
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
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        // kreiranje sesije i postavljanje osnovnih parametara aplikacije
        SparkSession session = SparkSession
                                .builder()
                                .appName("MLPrediction")
                                .master("local")
                                .getOrCreate();
        
        SparkContext context = session.sparkContext();
        
        // ucitavanje modela
        LogisticRegressionModel model = LogisticRegressionModel.load(context, "C:/spark-models/classification");
        
        // ucitavanje podataka
        JavaRDD<LabeledPoint> testData = loadData(session, args[0]);
        
        // predvidjanje vrednosti na osnovu ucitanih vrednosti: model.predict(vektor atributa)
        // smestanje vrednosti u JavaRDD
        JavaRDD<Double> preds = testData.map(point -> model.predict(point.features()));
                
        // upis rezultata u csv fajl
        writePredictions(preds);
        
        // zatvaranje sesije
        session.close();
    }

    private static void writePredictions(JavaRDD<Double> preds) {
        try {
            FileWriter fw = new FileWriter("D:\\godina\\Skladistenjepodataka\\Projekat\\Moj-Projekat-2\\WineQuality\\Rezultati\\predictions.csv");
            
            
            List<Double> predsDouble = preds.takeOrdered((int)preds.count());
            
            for(Double d : predsDouble){
                fw.write(d.toString() + "\n");
            }
            
            fw.close();
            
        } catch (IOException ex) {
            Logger.getLogger(MLPrediction.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}

