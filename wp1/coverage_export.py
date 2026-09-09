"""Export only the line/test matrix needed by SBFL, without duplicated scopes."""
import json
import sys
from pathlib import Path
from coverage import CoverageData


def export(data_path, output_path, repo_root):
    root = Path(repo_root).resolve()
    data = CoverageData(basename=str(data_path))
    data.read()
    prefix = str(root) + '/'
    files = {}
    def relative(value):
        return value[len(prefix):] if value.startswith(prefix) else value
    for file in sorted(data.measured_files()):
        contexts = {str(line): [relative(context) for context in labels]
                    for line, labels in data.contexts_by_lineno(file).items()}
        files[relative(file)] = {'contexts': contexts}
    Path(output_path).write_text(json.dumps({'files': files}), encoding='utf-8')


if __name__ == '__main__':
    export(*sys.argv[1:])
