import com.sun.source.tree.*;
import com.sun.source.util.*;
import javax.tools.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Parse independent Java method snippets into AST call expressions.
 *
 * Reports actual syntactic invocations and argument counts, NOT resolved
 * symbols, runtime dispatch targets, or fault causes. Missing types and
 * imports can be tolerated during parse(); malformed snippets are flagged.
 *
 * stdout: FILE<TAB>KIND<TAB>RECEIVER<TAB>NAME<TAB>ARITY
 * stderr: PARSE_ERROR<TAB>FILE<TAB>COUNT
 */
public final class CallAstExtractor {
    private static String safe(String s) {
        return s.replace("\t", " ").replace("\n", " ").replace("\r", " ");
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: CallAstExtractor <source-directory>");
            System.exit(2);
        }
        Path directory = Paths.get(args[0]);
        List<File> sources = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(directory, "*.java")) {
            for (Path source : stream) sources.add(source.toFile());
        }
        sources.sort(Comparator.comparing(File::getName));
        if (sources.isEmpty()) throw new IllegalArgumentException("No Java snippets found");
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new IllegalStateException("JDK compiler is required");
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        StandardJavaFileManager manager = compiler.getStandardFileManager(diagnostics, null, null);
        Iterable<? extends JavaFileObject> files = manager.getJavaFileObjectsFromFiles(sources);
        JavacTask task = (JavacTask) compiler.getTask(
            null, manager, diagnostics, Arrays.asList("-proc:none"), null, files);
        Iterable<? extends CompilationUnitTree> units = task.parse();
        for (CompilationUnitTree unit : units) {
            String filename = Paths.get(unit.getSourceFile().toUri()).getFileName().toString();
            new TreePathScanner<Void, Void>() {
                @Override public Void visitMethodInvocation(MethodInvocationTree node, Void ignored) {
                    ExpressionTree target = node.getMethodSelect();
                    String receiver = "";
                    String method = "";
                    if (target instanceof MemberSelectTree) {
                        MemberSelectTree member = (MemberSelectTree) target;
                        receiver = member.getExpression().toString();
                        method = member.getIdentifier().toString();
                    } else if (target instanceof IdentifierTree) {
                        method = ((IdentifierTree) target).getName().toString();
                    } else {
                        method = target.toString();
                    }
                    System.out.println(filename + "\tMETHOD\t" + safe(receiver)
                        + "\t" + safe(method) + "\t" + node.getArguments().size());
                    return super.visitMethodInvocation(node, ignored);
                }

                @Override public Void visitNewClass(NewClassTree node, Void ignored) {
                    String klass = node.getIdentifier().toString();
                    System.out.println(filename + "\tCONSTRUCTOR\t\t"
                        + safe(klass) + "\t" + node.getArguments().size());
                    return super.visitNewClass(node, ignored);
                }
            }.scan(unit, null);
        }
        Map<String, Integer> errors = new TreeMap<>();
        for (Diagnostic<? extends JavaFileObject> d : diagnostics.getDiagnostics()) {
            if (d.getKind() == Diagnostic.Kind.ERROR && d.getSource() != null) {
                String filename = Paths.get(d.getSource().toUri()).getFileName().toString();
                errors.merge(filename, 1, Integer::sum);
            }
        }
        for (Map.Entry<String, Integer> entry : errors.entrySet()) {
            System.err.println("PARSE_ERROR\t" + entry.getKey() + "\t" + entry.getValue());
        }
        manager.close();
    }
}
