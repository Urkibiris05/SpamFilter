public class SpamFilter {

    private static SpamFilter nSpamFilter = null;
    private Bektorizatu bektorizatu;
    private SpamFilter() {
        bektorizatu = new Bektorizatu();
    }

    public static SpamFilter getSpamFilter() {
        if (nSpamFilter == null) {
            nSpamFilter = new SpamFilter();
        }
        return nSpamFilter;
    }

    public void froga(){
        bektorizatu.froga();
    }
}
