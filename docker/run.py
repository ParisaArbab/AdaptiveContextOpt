"""Prepare the local model, record runtime evidence, then run the unchanged pipeline."""
import json
import os
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

APP_ROOT = Path('/app')
PACKAGES_FILE = Path('/opt/pipeline-packages.txt')


def request(base, endpoint, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + endpoint, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if result.get('error'):
        raise RuntimeError(result['error'])
    return result


def main():
    args = sys.argv[1:]
    if '--help' in args or '-h' in args:
        return subprocess.call(['bash', str(APP_ROOT / 'scripts/run_pipeline.sh'), '--help'])
    # These are fixed here to keep startup/model verification and the run aligned.
    forbidden = ('--llm', '--backend', '--model', '--base-url', '--context-window', '--out-dir')
    if any(a.split('=')[0] in forbidden for a in args):
        raise SystemExit('Set MODEL / CONTEXT_WINDOW in the environment; output directories are automatic.')
    base = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434').rstrip('/')
    model = os.environ.get('MODEL', 'llama3.1:8b')
    context = int(os.environ.get('CONTEXT_WINDOW', '16384'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = APP_ROOT / 'results' / ('docker_' + stamp)
    output.mkdir(parents=True)
    evidence = {'platform': platform.platform(), 'machine': platform.machine(),
                'model': model, 'context_window': context, 'argv': args}
    try:
        evidence['ollama_version'] = request(base, '/api/version')
        models = request(base, '/api/tags').get('models', [])
        if not any(model in (m.get('name'), m.get('model')) for m in models):
            print(f'Pulling {model}; model storage is persistent.', flush=True)
            request(base, '/api/pull', {'model': model, 'stream': False}, timeout=7200)
        evidence['model_details'] = request(base, '/api/show', {'model': model})
        request(base, '/api/generate', {'model': model, 'prompt': 'Reply ready.',
                'stream': False, 'keep_alive': '30m',
                'options': {'num_ctx': context, 'num_predict': 8}}, timeout=600)
        evidence['loaded_models'] = request(base, '/api/ps')
        loaded = [m for m in evidence['loaded_models'].get('models', [])
                  if model in (m.get('name'), m.get('model'))]
        if os.environ.get('REQUIRE_GPU') == '1' and not any(m.get('size_vram', 0) > 0 for m in loaded):
            raise RuntimeError('The model has no GPU allocation; refusing a silently CPU-only GPU run.')
        evidence['packages'] = PACKAGES_FILE.read_text()
        command = ['bash', str(APP_ROOT / 'scripts/run_pipeline.sh'), '--llm', 'ollama', '--model', model,
                   '--base-url', base, '--context-window', str(context), '--local-fallback',
                   '--out-dir', str(output), *args]
        evidence['pipeline_command'] = command
        (output / 'environment.json').write_text(json.dumps(evidence, indent=2))
        code = subprocess.call(command)
        result_file = output / 'wp1_results.json'
        evidence['pipeline_exit_code'] = code
        if code == 0 and '--dry-run' not in args:
            if not result_file.exists():
                raise RuntimeError('Pipeline exited without a result file.')
            result = json.loads(result_file.read_text())
            if result.get('skipped') or not result.get('outcomes'):
                raise RuntimeError('Run has skipped/failed instances or no outcomes; inspect saved traces.')
        (output / 'environment.json').write_text(json.dumps(evidence, indent=2))
        return code
    except Exception as exc:
        evidence['error'] = str(exc)
        (output / 'environment.json').write_text(json.dumps(evidence, indent=2))
        raise


if __name__ == '__main__':
    sys.exit(main())
