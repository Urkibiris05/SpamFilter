import java.io.*;
import java.util.Scanner;

public class SMS2Arff {
    public static void main(String[] args) {

        String inputFile = args[0];
        String outputFile = args[1];
        boolean isTestBlind = Boolean.parseBoolean(args[2]);

        try (Scanner sc = new Scanner(new File(inputFile), "UTF-8");
             PrintWriter pw = new PrintWriter(new File(outputFile), "UTF-8")) {

            pw.println("@relation sms_spam_proiektua\n");
            pw.println("@attribute Text string");
            pw.println("@attribute class_label {ham, spam}\n");
            pw.println("@data");

            while (sc.hasNextLine()) {
                String line = sc.nextLine();
                if (line.trim().isEmpty()) continue;

                String label;
                String text;

                if (isTestBlind) {
                    text = line.trim();
                    label = "?";
                } else {
                    String[] parts = line.split("\t");
                    if (parts.length < 2) continue;
                    label = parts[0].trim();
                    text = parts[1].trim();
                }

                text = text.replace("\"", "'");

                pw.println("\"" + text + "\"," + label);
            }
            System.out.println("Fitxategia ondo sortu da: " + outputFile);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
