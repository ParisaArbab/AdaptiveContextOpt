package aitech.fixture;

/**
 * Controlled source-only fixture, NOT a Defects4J bug.
 * Used to measure whether compiler attribution distinguishes overloads.
 */
public final class OverloadDemo {
    public static int translate(String value) {
        return value.length();
    }

    public static int translate(Object value) {
        return value.hashCode();
    }

    public static int run(String value) {
        return translate(value) + translate((Object) value);
    }

    public static OverloadDemo instantiate() {
        return new OverloadDemo();
    }
}
