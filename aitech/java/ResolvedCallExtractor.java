import com.sun.source.tree.*;
import com.sun.source.util.*;
import javax.lang.model.element.*;
import javax.lang.model.type.*;
import javax.tools.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Type-attributed Java call target extraction using javac.
 *
 * Input: directory of complete, compilable .java files (recursively).
 * Output: tab-separated caller owner, caller method, target owner, target
 * method, target parameter types, syntax kind and source path.
 *
 * Only calls with non-null ExecutableElement are emitted as RESOLVED.
 * This is static dispatch resolution, NOT runtime polymorphic target
 * resolution. Compiler diagnostics are emitted to stderr.
 */
public final class ResolvedCallExtractor {
    private static String safe(String s) {
        return s.replace("\t", " ").replace("\n", " ").replace("\r", " ");
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args.length > 2) {
            System.err.println("Usage: ResolvedCallExtractor <complete-source-root> [compile-classpath]");
            System.exit(2);
        }
        Path root = Paths.get(args[0]).toAbsolutePath().normalize();
        List<File> files = new ArrayList<>();
        try (java.util.stream.Stream<Path> paths = Files.walk(root)) {
            paths.filter(p -> p.toString().endsWith(".java"))
                 .sorted().forEach(p -> files.add(p.toFile()));
        }
        if (files.isEmpty()) throw new IllegalArgumentException("No source files found");
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new IllegalStateException("Requires a JDK");
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        StandardJavaFileManager manager = compiler.getStandardFileManager(diagnostics, null, null);
        Iterable<? extends JavaFileObject> units = manager.getJavaFileObjectsFromFiles(files);
        Path classes = Files.createTempDirectory("aitech_javac_classes_");
        List<String> options = new ArrayList<>(Arrays.asList(
            "-proc:none", "-implicit:none", "-d", classes.toString()));
        if (args.length == 2 && !args[1].isEmpty()) {
            options.add("-classpath");
            options.add(args[1]);
        }
        JavacTask task = (JavacTask) compiler.getTask(
            null, manager, diagnostics, options, null, units);
        Iterable<? extends CompilationUnitTree> parsed = task.parse();
        task.analyze();
        Trees trees = Trees.instance(task);
        for (CompilationUnitTree unit : parsed) {
            String relative = root.relativize(Paths.get(unit.getSourceFile().toUri())).toString();
            new TreePathScanner<Void, Void>() {
                private String callerOwner = "";
                private String callerMethod = "";
                private String callerParameters = "";

                @Override public Void visitMethod(MethodTree node, Void p) {
                    TreePath path = getCurrentPath();
                    Element el = trees.getElement(path);
                    String oldOwner = callerOwner, oldMethod = callerMethod;
                    String oldParameters = callerParameters;
                    if (el instanceof ExecutableElement) {
                        callerMethod = el.getSimpleName().toString();
                        List<String> callerParams = new ArrayList<>();
                        for (VariableElement param : ((ExecutableElement) el).getParameters()) {
                            callerParams.add(param.asType().toString());
                        }
                        callerParameters = String.join(",", callerParams);
                        Element encl = el.getEnclosingElement();
                        callerOwner = (encl instanceof TypeElement)
                            ? ((TypeElement) encl).getQualifiedName().toString()
                            : encl.toString();
                    } else {
                        callerMethod = node.getName().toString();
                        callerParameters = "";
                    }
                    try { return super.visitMethod(node, p); }
                    finally {
                        callerOwner = oldOwner;
                        callerMethod = oldMethod;
                        callerParameters = oldParameters;
                    }
                }

                private void emit(TreePath path, String kind) {
                    Element el = trees.getElement(path);
                    if (!(el instanceof ExecutableElement)) {
                        System.out.println("UNRESOLVED\t" + safe(relative) + "\t" + kind);
                        return;
                    }
                    ExecutableElement method = (ExecutableElement) el;
                    Element owner = method.getEnclosingElement();
                    String targetOwner = (owner instanceof TypeElement)
                        ? ((TypeElement)owner).getQualifiedName().toString()
                        : owner.toString();
                    List<String> params = new ArrayList<>();
                    for (VariableElement param : method.getParameters()) {
                        params.add(param.asType().toString());
                    }
                    System.out.println("RESOLVED\t" + safe(relative)
                        + "\t" + safe(callerOwner) + "\t" + safe(callerMethod)
                        + "\t" + kind + "\t" + safe(targetOwner)
                        + "\t" + safe(method.getSimpleName().toString())
                        + "\t" + safe(String.join(",", params))
                        + "\t" + safe(callerParameters));
                }
                @Override public Void visitMethodInvocation(MethodInvocationTree node, Void p) {
                    emit(getCurrentPath(), "METHOD");
                    return super.visitMethodInvocation(node, p);
                }
                @Override public Void visitNewClass(NewClassTree node, Void p) {
                    emit(getCurrentPath(), "CONSTRUCTOR");
                    return super.visitNewClass(node, p);
                }
            }.scan(unit, null);
        }
        long errorCount = diagnostics.getDiagnostics().stream()
            .filter(d -> d.getKind() == Diagnostic.Kind.ERROR).count();
        System.err.println("COMPILER_ERRORS\t" + errorCount);
        manager.close();
        if (errorCount > 0) System.exit(3);
    }
}
