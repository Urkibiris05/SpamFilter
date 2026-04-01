import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

public class SpamFilter {

    private static final DataProcessor dataProcessor = new DataProcessor();
    private static final Sailkatzailea sailkatzailea = new Sailkatzailea();
    private static final KalitateEstimazioa kalitateEstimazioa = new KalitateEstimazioa();
    private static final Iragarpenak iragarpenak = new Iragarpenak();

    public static void main(String[] args) {
        try {
            exekutatuCli(args);
        } catch (IllegalArgumentException e) {
            System.err.println("Errorea: " + e.getMessage());
            inprimatuLaguntza();
            System.exit(1);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void exekutatuCli(String[] args) throws Exception {
        Map<String, String> aukerak = parseArgs(args);

        if (aukerak.containsKey("help") || !aukerak.containsKey("run")) {
            inprimatuLaguntza();
            return;
        }

        String[] komandoak = aukerak.get("run").split(",");
        for (String komandoaRaw : komandoak) {
            String komandoa = komandoaRaw.trim().toLowerCase(Locale.ROOT);
            if (komandoa.isEmpty()) {
                continue;
            }
            exekutatuKomandoa(komandoa, aukerak);
        }
    }

    private static void exekutatuKomandoa(String komandoa, Map<String, String> aukerak) throws Exception {
        switch (komandoa) {
            case "sms2arff":
                exekutatuSms2Arff(aukerak);
                break;
            case "analyze":
                exekutatuAnalyze(aukerak);
                break;
            case "vectorize":
                exekutatuVectorize(aukerak);
                break;
            case "param-search":
                exekutatuParamSearch(aukerak);
                break;
            case "param-search-v2":
                exekutatuParamSearchV2(aukerak);
                break;
            case "sweep":
                exekutatuSweep(aukerak);
                break;
            case "train-optimal":
                exekutatuTrainOptimal(aukerak);
                break;
            case "quality":
                exekutatuQuality(aukerak);
                break;
            case "predict":
                exekutatuPredict(aukerak);
                break;
            default:
                throw new IllegalArgumentException("Komando ezezaguna: " + komandoa);
        }
    }

    private static void exekutatuSms2Arff(Map<String, String> aukerak) throws Exception {
        String txtPath = beharrezkoa(aukerak, "sms2arff.txt");
        String arffPath = beharrezkoa(aukerak, "sms2arff.arff");
        boolean blind = parseBoolean(aukerak.getOrDefault("sms2arff.blind", "false"));
        dataProcessor.sms2Arff(txtPath, arffPath, blind);
    }

    private static void exekutatuAnalyze(Map<String, String> aukerak) throws Exception {
        String dataPath = beharrezkoa(aukerak, "analyze.data");
        String etapaIzena = aukerak.getOrDefault("analyze.stage", "ETAPA");
        dataProcessor.instantziakAztertu(dataPath, etapaIzena);
    }

    private static void exekutatuVectorize(Map<String, String> aukerak) throws Exception {
        String rawDataPath = beharrezkoa(aukerak, "vectorize.raw");
        String bekDataPath = beharrezkoa(aukerak, "vectorize.bek");
        String filterModelPath = beharrezkoa(aukerak, "vectorize.filter");
        boolean isTrain = parseBoolean(aukerak.getOrDefault("vectorize.train", "false"));
        dataProcessor.bektorizatu(rawDataPath, bekDataPath, filterModelPath, isTrain);
    }

    private static void exekutatuParamSearch(Map<String, String> aukerak) throws Exception {
        String rawTrainPath = beharrezkoa(aukerak, "param-search.rawTrain");
        dataProcessor.parametroBilatzailea(rawTrainPath);
    }

    private static void exekutatuParamSearchV2(Map<String, String> aukerak) throws Exception {
        String rawTrainPath = beharrezkoa(aukerak, "param-search-v2.rawTrain");
        dataProcessor.parametroBilatzaileaV2(rawTrainPath);
    }

    private static void exekutatuSweep(Map<String, String> aukerak) throws Exception {
        String[] arff = new String[2];
        arff[0] = beharrezkoa(aukerak, "sweep.trainBek");
        arff[1] = beharrezkoa(aukerak, "sweep.devBek");

        Instances[] instantziak = sailkatzailea.arffKargatu(arff);
        sailkatzailea.parametroOptimoakLortu(instantziak);
    }

    private static void exekutatuTrainOptimal(Map<String, String> aukerak) throws Exception {
        String hl = beharrezkoa(aukerak, "train-optimal.hl");
        String lr = beharrezkoa(aukerak, "train-optimal.lr");
        String m = beharrezkoa(aukerak, "train-optimal.m");
        String[] parametroak = new String[]{hl, lr, m};

        String rawTrainPath = beharrezkoa(aukerak, "train-optimal.train");
        String rawDevPath = aukerak.getOrDefault("train-optimal.dev", "");
        String rawDataPath = beharrezkoa(aukerak, "train-optimal.rawData");
        String bekDataPath = beharrezkoa(aukerak, "train-optimal.bekData");
        String filterModelPath = beharrezkoa(aukerak, "train-optimal.filter");
        String outputPath = beharrezkoa(aukerak, "train-optimal.out");

        sailkatzailea.sailkatzaileaSortu(parametroak, rawTrainPath, rawDevPath, rawDataPath, bekDataPath, filterModelPath, outputPath);
    }

    private static void exekutatuQuality(Map<String, String> aukerak) throws Exception {
        String modelPath = beharrezkoa(aukerak, "quality.model");
        String metricsOutPath = beharrezkoa(aukerak, "quality.out");
        String mode = aukerak.getOrDefault("quality.mode", "holdout").toLowerCase(Locale.ROOT);

        Classifier modeloa = (Classifier) SerializationHelper.read(modelPath);

        if ("srho".equals(mode)) {
            String trainPath = beharrezkoa(aukerak, "quality.train");
            String devPath = beharrezkoa(aukerak, "quality.dev");

            Instances train = kargatuInstantziak(trainPath);
            Instances dev = kargatuInstantziak(devPath);

            int repeats = parseInt(aukerak.getOrDefault("quality.repeats", "10"), "quality.repeats");
            double ratio = parseDouble(aukerak.getOrDefault("quality.ratio", "0.8"), "quality.ratio");
            int seed = parseInt(aukerak.getOrDefault("quality.seed", "42"), "quality.seed");
            String tmpPath = aukerak.getOrDefault("quality.tmp", "src/data/tmp");
            if (ratio <= 0.0 || ratio >= 1.0) {
                throw new IllegalArgumentException("quality.ratio balioak 0 eta 1 artean egon behar du");
            }

            kalitateEstimazioa.stratifiedRepeatedHoldOut(train, dev, repeats, ratio, seed, tmpPath, metricsOutPath, modeloa);
            return;
        }

        String bekTrainPath = beharrezkoa(aukerak, "quality.trainBek");
        String bekDevPath = beharrezkoa(aukerak, "quality.devBek");

        Instances trainBek = kargatuInstantziak(bekTrainPath);
        Instances devBek = kargatuInstantziak(bekDevPath);

        if ("unfair".equals(mode)) {
            kalitateEstimazioa.ezZintzoa(trainBek, devBek, modeloa, metricsOutPath);
        } else {
            kalitateEstimazioa.holdOut(trainBek, devBek, modeloa, metricsOutPath);
            System.out.println("Kalitate metrikak gordeta: " + metricsOutPath);
        }
    }

    private static void exekutatuPredict(Map<String, String> aukerak) throws Exception {
        String test = beharrezkoa(aukerak, "predict.test");
        String modelPath = beharrezkoa(aukerak, "predict.model");
        String iragarpenakOut = beharrezkoa(aukerak, "predict.out");

        Classifier modeloa = (Classifier) SerializationHelper.read(modelPath);
        iragarpenak.Iragarpenak(test, modeloa, iragarpenakOut);
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> parsed = new LinkedHashMap<>();
        int i = 0;

        while (i < args.length) {
            String arg = args[i];
            if (!arg.startsWith("--")) {
                throw new IllegalArgumentException("Argumentu formatu okerra: " + arg);
            }

            String token = arg.substring(2);
            String key;
            String value;

            int eqPos = token.indexOf('=');
            if (eqPos >= 0) {
                key = token.substring(0, eqPos).trim();
                value = token.substring(eqPos + 1).trim();
            } else {
                key = token.trim();
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    value = args[i + 1].trim();
                    i++;
                } else {
                    value = "true";
                }
            }

            if (key.isEmpty()) {
                throw new IllegalArgumentException("Argumentu gakoa hutsik dago");
            }

            parsed.put(key, balioaPathBadaNormalizatu(key, value));
            i++;
        }

        return parsed;
    }

    private static String balioaPathBadaNormalizatu(String key, String value) {
        if (!key.contains("path")
                && !key.endsWith(".txt")
                && !key.endsWith(".arff")
                && !key.endsWith(".bek")
                && !key.endsWith(".model")
                && !key.endsWith(".filter")
                && !key.endsWith(".out")
                && !key.endsWith(".data")
                && !key.endsWith(".test")
                && !key.endsWith(".train")
                && !key.endsWith(".dev")
                && !key.endsWith(".trainBek")
                && !key.endsWith(".devBek")
                && !key.endsWith(".tmp")
                && !key.endsWith(".rawData")
                && !key.endsWith(".bekData")
                && !key.endsWith(".rawTrain")) {
            return value;
        }
        return resolvePath(value);
    }

    private static String resolvePath(String path) {
        Path p = Path.of(path);
        if (!p.isAbsolute()) {
            p = Path.of("").toAbsolutePath().normalize().resolve(p).normalize();
        }
        return p.toString();
    }

    private static String beharrezkoa(Map<String, String> aukerak, String gakoa) {
        String balioa = aukerak.get(gakoa);
        if (balioa == null || balioa.trim().isEmpty()) {
            throw new IllegalArgumentException("Falta da argumentu hau: --" + gakoa);
        }
        return balioa;
    }

    private static boolean parseBoolean(String balioa) {
        String b = balioa.trim().toLowerCase(Locale.ROOT);
        return "true".equals(b) || "1".equals(b) || "bai".equals(b) || "yes".equals(b) || "y".equals(b);
    }

    private static int parseInt(String balioa, String gakoa) {
        try {
            return Integer.parseInt(balioa);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(gakoa + " zenbaki osoa izan behar da: " + balioa);
        }
    }

    private static double parseDouble(String balioa, String gakoa) {
        try {
            return Double.parseDouble(balioa);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(gakoa + " zenbaki hamartarra izan behar da: " + balioa);
        }
    }

    private static void inprimatuLaguntza() {
        System.out.println("Erabilera:");
        System.out.println("  java -jar SpamFilter.jar --run <cmd1,cmd2,...> [--gakoa balioa ...]");
        System.out.println();
        System.out.println("Komandoak:");
        System.out.println("  sms2arff         -> --sms2arff.txt --sms2arff.arff [--sms2arff.blind true|false]");
        System.out.println("  analyze          -> --analyze.data [--analyze.stage ETAPA]");
        System.out.println("  vectorize        -> --vectorize.raw --vectorize.bek --vectorize.filter --vectorize.train true|false");
        System.out.println("  param-search     -> --param-search.rawTrain");
        System.out.println("  param-search-v2  -> --param-search-v2.rawTrain");
        System.out.println("  sweep            -> --sweep.trainBek --sweep.devBek");
        System.out.println("  train-optimal    -> --train-optimal.hl --train-optimal.lr --train-optimal.m --train-optimal.train [--train-optimal.dev] --train-optimal.rawData --train-optimal.bekData --train-optimal.filter --train-optimal.out");
        System.out.println("  quality          -> --quality.model --quality.out --quality.mode holdout|unfair|srho");
        System.out.println("                       holdout/unfair: --quality.trainBek --quality.devBek");
        System.out.println("                       srho: --quality.train --quality.dev [--quality.repeats 10 --quality.ratio 0.8 --quality.seed 42 --quality.tmp src/data/tmp]");
        System.out.println("  predict          -> --predict.test --predict.model --predict.out");
        System.out.println();
        System.out.println("Adibidea (komando bakarra):");
        System.out.println("  --run sms2arff --sms2arff.txt src/data/txt/SMS_SpamCollection.train.txt --sms2arff.arff src/data/arff/SMS_SpamCollection.train.arff --sms2arff.blind false");
        System.out.println();
        System.out.println("Adibidea (komando anitz):");
        System.out.println("  --run sms2arff,vectorize --sms2arff.txt src/data/txt/SMS_SpamCollection.train.txt --sms2arff.arff src/data/arff/SMS_SpamCollection.train.arff --vectorize.raw src/data/arff/SMS_SpamCollection.train.arff --vectorize.bek src/data/arff/SMS_SpamCollection.bektrain.arff --vectorize.filter src/data/model/multiFilter.model --vectorize.train true");
    }

    private static Instances kargatuInstantziak(String path) throws Exception {
        Instances data = new DataSource(path).getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }
}
