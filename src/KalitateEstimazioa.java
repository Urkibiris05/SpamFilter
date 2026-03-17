import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import java.util.Random;

public class KalitateEstimazioa {

    public void ezZintzoa(Instances data, Classifier m) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(m, data);
    }

    public void k_FCV(Instances data, Classifier m) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(m,data,10, new Random(1));
    }

    public void holdOut(Instances dataTrain, Instances dataDev, Classifier m) throws Exception {
        Evaluation eval = new Evaluation(dataTrain);
        eval.evaluateModel(m, dataDev);
    }

    public void repeatedHoldOut(Instances data, Classifier m, int errepikapenak) throws Exception {
        for (int i = 1; i <= errepikapenak; i++) {

        }
    }

    private void metrikak(Evaluation eval, Instances datos) throws Exception {
        int spamIdx = datos.classAttribute().indexOfValue("spam");
        System.out.printf(" Precision:              %.4f\n", eval.precision(spamIdx));
        System.out.printf(" Recall:                 %.4f\n", eval.recall(spamIdx));
        System.out.printf(" F-Measure:              %.4f\n", eval.fMeasure(spamIdx));
    }
}
