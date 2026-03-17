import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class DataProcessor {


    public Instances bektorizatu(Instances rawData) throws Exception {
        Instances bekData = null;
        int wordsToKeep = 1000;

        StringToWordVector filter = new StringToWordVector();

        filter.setLowerCaseTokens(true);
        filter.setOutputWordCounts(true);
        filter.setTFTransform(true);
        filter.setIDFTransform(true);
        //filter.setWordsToKeep(wordsToKeep);

        // Stemmerra (hitzak erro-mailara murrizteko algoritmoa, adib. "running" -> "run")

        // Snowball (ingeleserako oso estandarra), weka paketea independenteki instalatu behar da
        // SnowballStemmer stemmer = new SnowballStemmer();
        // stemmer.setStemmer("english");

        // IteratedLovinsStemmer Weka core-aren barruan dago, kanpoko paketerik behar gabe
        IteratedLovinsStemmer stemmer = new IteratedLovinsStemmer();
        filter.setStemmer(stemmer);

        filter.setStemmer(stemmer);

        // Tokenizatzailea (hitzak nola banatzen ditugun)
        // Lehenespenez, zuriuneak eta puntuazio-ikurrak erabiltzen ditu, oso egokia SMSetarako
        //WordTokenizer tokenizer = new WordTokenizer();
        AlphabeticTokenizer tokenizer = new AlphabeticTokenizer();
        filter.setTokenizer(tokenizer);

        // Stop word-ak (esanahi handirik ematen ez duten hitz arruntak)
        // Wekak automatikoki ezabatuko ditu "the", "a", "an", "in"... bezalako hitzak
        Rainbow stopWords = new Rainbow();
        filter.setStopwordsHandler(stopWords);

        // Datu sorta bektorizatua itzuli
        filter.setInputFormat(rawData);
        bekData = Filter.useFilter(rawData, filter);
        return bekData;
    }

    public void instantziakAztertu(Instances data, String etapaIzena) {

        System.out.println("======================================================");
        System.out.println("📊 DATUEN ANALISIA - ETAPA: " + etapaIzena.toUpperCase());
        System.out.println("======================================================");

        // 1. Informazio Orokorra
        // System.out.println("Dataset-aren izena (Erlazioa): " + data.relationName());
        System.out.println("Instantzia kopuru osoa (SMS): " + data.numInstances());
        System.out.println("Atributuen kopurua (zutabeak): " + data.numAttributes());
        System.out.println("------------------------------------------------------");

        // 2. Klasearen inguruko informazioa (Spam vs Ham)
        if (data.classIndex() != -1) {
            Attribute klasea = data.classAttribute();
            System.out.println("Iragarri beharreko atributua (Klasea): '" + klasea.name() + "' (Indizea: " + data.classIndex() + ")");

            // Klasea Nominala bada (gure kasuan ham/spam), banaketaren kalkulua egingo dugu
            if (klasea.isNominal()) {
                System.out.println("Klase banaketa:");
                AttributeStats estatistikak = data.attributeStats(data.classIndex());
                int[] zenbaketak = estatistikak.nominalCounts;

                for (int i = 0; i < klasea.numValues(); i++) {
                    String klaseaBalioa = klasea.value(i);
                    int kopurua = zenbaketak[i];
                    double ehunekoa = (double) kopurua / data.numInstances() * 100;
                    System.out.printf("  - %s: %d instantzia (%.2f%%)\n", klaseaBalioa, kopurua, ehunekoa);
                }
            }
        } else {
            System.out.println("Klase atributua: DEFINITU GABE (Normala Test Blind bada)");
        }
        System.out.println("------------------------------------------------------");

        // 3. Lehen atributuen laburpena (kontsola ez betetzeko, 600+ badaude)
        System.out.println("Atributu motak:");

        // Mota bakoitzeko zenbat dauden jakiteko zenbagailuak
        int stringKopurua = 0, numerikoKopurua = 0, nominalKopurua = 0;

        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute ezaugarria = data.attribute(i);
            if (ezaugarria.isString()) stringKopurua++;
            else if (ezaugarria.isNumeric()) numerikoKopurua++;
            else if (ezaugarria.isNominal()) nominalKopurua++;
        }

        System.out.println("  - Stringak (Testuak): " + stringKopurua);
        System.out.println("  - Numerikoak (Ad. TF-IDF): " + numerikoKopurua);
        System.out.println("  - Nominalak (Ad. Kategoriak/Klasea): " + nominalKopurua);

        System.out.println("\nLehen 5 atributuen lagina:");
        int muga = Math.min(data.numAttributes(), 5);
        for (int i = 0; i < muga; i++) {
            Attribute ezaugarria = data.attribute(i);
            String mota = ezaugarria.isString() ? "String" :
                    ezaugarria.isNumeric() ? "Numerikoa" :
                            ezaugarria.isNominal() ? "Nominala" : "Bestea";
            System.out.println("  " + (i) + ". " + ezaugarria.name() + " (" + mota + ")");
        }

        if (data.numAttributes() > 5) {
            System.out.println("  ... eta " + (data.numAttributes() - 5) + " atributu gehiago.");
        }
        System.out.println("======================================================\n");
    }

}
