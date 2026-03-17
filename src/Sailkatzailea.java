import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.IOException;

public class Sailkatzailea {

    public Instances arffKargatu(String[] args) throws Exception{
        try {
            String inputFile = args[0];
            DataSource source = new DataSource(inputFile);
            Instances data = source.getDataSet();
            if (data.classIndex() == -1){
                data.setClassIndex(data.numAttributes() - 1);
            }
            return data;
        } catch ( IOException e ) {
            System.out.println("ERROREA: arff fitxategia kargatzen.");
            return null;
        }
    }

    public Instances[] HoldOut(int portzentaia, Instances data) throws Exception {
        try{
            Instances[] holdOut = new  Instances[2];
            Resample resample = new Resample();
            resample.setSampleSizePercent(portzentaia);
            resample.setNoReplacement(true);
            resample.setInvertSelection(false);
            resample.setInputFormat(data);
            Instances train = Filter.useFilter(data, resample);
            holdOut[0] = train;
            resample.setInvertSelection(true);
            resample.setInputFormat(data);
            Instances dev = Filter.useFilter(data,resample);
            holdOut[1] = dev;
            return holdOut;
        } catch ( IOException e ) {
            System.out.println("ERROREA: HoldOut egitean.");
            return null;
        }
    }

    public String parametroOptimoaLortu(Instances[] holdOut, int Parametroa){
        Instances train = holdOut[0];
        Instances dev = holdOut[1];
        String balioParametroa = "";
        MultilayerPerceptron estimatzailea = new MultilayerPerceptron();
        return balioParametroa;
    }
}
