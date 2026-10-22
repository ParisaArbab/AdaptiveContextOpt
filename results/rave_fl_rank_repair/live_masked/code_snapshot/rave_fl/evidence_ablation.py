"""Gold-free naming-cue ablation that preserves every assertion line.

Mask ALL candidate method names found inside test/frame method identifiers.
Scenario suffixes, class/file names, line numbers and failure values remain.
This is a lexical-cue ablation, not an independent bug or semantic equivalence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def mask_candidate_names(context, candidate_ids):
    names = sorted({i.split('(', 1)[0].rsplit('.', 1)[-1] for i in candidate_ids}, key=lambda s: (-len(s), s))
    mapping = {name: 'FN_' + hashlib.sha256(name.encode()).hexdigest()[:8] for name in names}
    counts = {name: 0 for name in names}
    def transform_identifier(value):
        # Underscores delimit test-name components; do not replace a substring
        # inside a different Java identifier or mask package/class names.
        pattern = r'(?<![A-Za-z0-9])(' + '|'.join(re.escape(n) for n in names) + r')(?![A-Za-z0-9])'
        def sub(match):
            counts[match.group(1)] += 1
            return mapping[match.group(1)]
        return re.sub(pattern, sub, value) if names else value
    result = []
    for line in context.splitlines(keepends=True):
        if line.startswith('TEST: ') and '::' in line:
            head, method = line.split('::', 1)
            line = head + '::' + transform_identifier(method)
        elif line.startswith('at ') and '(' in line:
            head, suffix = line.split('(', 1)
            if '.' in head:
                cls, method = head.rsplit('.', 1)
                line = cls + '.' + transform_identifier(method) + '(' + suffix
        result.append(line)
    masked = ''.join(result)
    before = [s for s in context.splitlines() if s.startswith('FAILURE:')]
    after = [s for s in masked.splitlines() if s.startswith('FAILURE:')]
    if before != after:
        raise ValueError('Ablation changed assertion evidence')
    return masked, {'kind': 'candidate_names_inside_test_and_frame_methods',
                    'assertion_lines_byte_preserved': True,
                    'changed_occurrences': {k: v for k, v in counts.items() if v},
                    'original_sha256': hashlib.sha256(context.encode()).hexdigest(),
                    'masked_sha256': hashlib.sha256(masked.encode()).hexdigest(),
                    'original_characters': len(context), 'masked_characters': len(masked),
                    'gold_used': False, 'mapping_in_model_prompt': False}


def main():
    from rave_fl.pilot import prepare_case, write_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-directory', type=Path, required=True)
    a = p.parse_args()
    if a.output_directory.exists():
        p.error('Use a new ablation directory')
    manifest = json.loads(a.manifest.read_text())
    prepared = []
    for case in manifest['cases']:
        candidates, context, _ = prepare_case(case)
        text, metadata = mask_candidate_names(context, [c.function_id for c in candidates])
        prepared.append((case, text, metadata))
    cases, metadata_rows = [], []
    for case, text, meta in prepared:
        path = a.output_directory / (case['bug'] + '_masked.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        cases.append({**case, 'evidence_file': str(path)})
        metadata_rows.append({'bug': case['bug'], **meta})
    write_json(a.output_directory / 'manifest.json', {'cases': cases})
    write_json(a.output_directory / 'ablation.json', metadata_rows)
    print(json.dumps(metadata_rows, indent=2))


if __name__ == '__main__':
    main()
