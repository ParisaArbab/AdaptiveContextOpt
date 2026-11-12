package aitech.fixture;

/**
 * Standalone compiler-attribution fixture for BOTH caller and target overloads.
 * Synthetic verification only, not a Defects4J bug or accuracy benchmark.
 */
public final class CallerOverloadDemo {
    public static String convert(String value) { return value; }
    public static String convert(Object value) { return value.toString(); }

    public static String use(String value) {
        return convert(value);
    }

    public static String use(Object value) {
        return convert(value);
    }
}
