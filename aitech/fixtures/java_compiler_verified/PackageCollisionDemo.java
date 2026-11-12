package aitech.fixture;

/**
 * Explicit class-qualified same-name overload fixture.
 * Source is controlled, with no benchmark gold data.
 */
public final class PackageCollisionDemo {
    public static int convert(aitech.fixture.first.Date date) { return 1; }
    public static int convert(aitech.fixture.second.Date date) { return 2; }
    public static int use(aitech.fixture.first.Date date) {
        return convert(date);
    }
    public static int use(aitech.fixture.second.Date date) {
        return convert(date);
    }
}
