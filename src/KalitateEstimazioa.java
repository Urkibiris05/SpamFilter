import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.File;
import java.io.PrintWriter;
import java.util.Random;

public class KalitateEstimazioa {

    public void holdOut(Instances dataTrain, Instances dataDev, Classifier mlp, String path) throws Exception {
        Evaluation eval = new Evaluation(dataTrain);
        eval.evaluateModel(mlp, dataDev);
        metrikak(eval,dataTrain,path);
    }

    private void metrikak(Evaluation eval, Instances data, String pathOut) throws Exception {
        int spamIdx = data.classAttribute().indexOfValue("spam");
        PrintWriter pw = new PrintWriter(new File(pathOut));
        pw.println("Precision: " + eval.precision(spamIdx));
        pw.println("Recall: " + eval.recall(spamIdx));
        pw.println("F-Score: " + eval.fMeasure(spamIdx));
        pw.close();
    }
}
