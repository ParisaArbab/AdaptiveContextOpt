"""Single zero-inference entry point: audit, test, diagnose, plot, report.

Requires a new result directory. Existing experiment data are read-only.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from rave_fl.independent_preflight import preflight
from rave_fl.pilot import write_json
from rave_fl.v3_audit import run_audits
from rave_fl.v3_diagnostics import run


def run_tests(output):
    output.mkdir(parents=True,exist_ok=True)
    suites=[('RAVE-FL focused suite',[sys.executable,'-m','pytest','-q',*sorted(str(p) for p in Path('tests').glob('test_rave_fl*.py'))]),
            ('Scoped project suite',[sys.executable,'-m','pytest','-q','tests','--ignore=tests/test_pipeline_dependencies.py','--ignore=tests/test_agent4sr_format_finalization.py']),
            ('Known unrelated failures',[sys.executable,'-m','pytest','-q','tests/test_pipeline_dependencies.py','tests/test_agent4sr_format_finalization.py'])]
    results=[]
    for index,(label,cmd) in enumerate(suites):
        proc=subprocess.run(cmd,text=True,capture_output=True)
        log=proc.stdout+proc.stderr
        path=output/f'tests_{index+1}.log';path.write_text(log)
        summary=next((s for s in reversed(log.splitlines()) if re.search(r'\d+ (passed|failed)',s)),'No pytest summary')
        results.append({'label':label,'command':' '.join(cmd),'summary':summary,'returncode':proc.returncode,'log':str(path)})
        if index<2 and proc.returncode:raise RuntimeError('Required suite failed; inspect '+str(path))
        if index==2:
            failures=[s.split(' - ')[0] for s in log.splitlines() if s.startswith('FAILED ')]
            expected=['test_pipeline_dependencies.py::test_fetch_dependency_fails_before_model_preflight',
                      'test_pipeline_dependencies.py::test_existing_instances_do_not_require_dataset_fetcher',
                      'test_agent4sr_format_finalization.py::test_format_only_finalization_uses_seen_entities']
            if proc.returncode and (len(failures)!=3 or not all(any(e in s for s in failures) for e in expected)):
                raise RuntimeError('Known-failure scope changed; inspect '+str(path))
    return {'runs':results}


def pipeline(output,report_dir):
    if output.exists():raise ValueError('Choose a new output directory; preserve earlier datasets')
    subprocess.run(['git','diff','--exit-code','HEAD','--','results/rave_fl_pilot','results/rave_fl_rank_repair','results/rave_fl_phase3'],check=True,capture_output=True)
    output.mkdir(parents=True)
    provenance=report_dir/'provenance';provenance.mkdir(parents=True,exist_ok=True)
    initial=Path('reports/provenance/starting_state.json')
    if initial.exists() and initial.resolve()!=(provenance/'starting_state.json').resolve():shutil.copy2(initial,provenance/'starting_state.json')
    code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['rave_fl','configs','tests','reports/templates'] for p in Path(folder).glob('*') if p.is_file()}
    for folder in ['rave_fl','configs','reports/templates']:
        for p in Path(folder).glob('*'):
            if p.is_file():
                dest=output/'code_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
         'packages':{p:importlib.metadata.version(p) for p in ['pytest','numpy','matplotlib','tree-sitter','tree-sitter-java','lightgbm','scikit-learn']},
         'code_sha256':code,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'command':[sys.executable,'-m','rave_fl.research_pipeline','--output',str(output),'--report-dir',str(report_dir)],
         'new_llm_calls':0,'fresh_provider_tokens':0}
    write_json(output/'environment.json',env);write_json(provenance/'environment.json',env)
    audits=run_audits(provenance/'artifact_audits.json');write_json(output/'audits.json',audits)
    pf=preflight(provenance/'independent_preflight.json');write_json(output/'independent_preflight.json',pf)
    tests=run_tests(output/'logs');write_json(provenance/'test_runs.json',tests);write_json(output/'test_runs.json',tests)
    run(Path('configs/rave_v3_protocol.json'),output/'diagnostics')
    from rave_fl.research_report import generate
    generate(output/'diagnostics',report_dir)
    subprocess.run(['git','diff','--exit-code','HEAD','--','results/rave_fl_pilot','results/rave_fl_rank_repair','results/rave_fl_phase3'],check=True,capture_output=True)
    print(json.dumps({'status':'complete','output':str(output),'reports':str(report_dir),'new_llm_calls':0,
                      'tests':[r['summary'] for r in tests['runs']],'historical_files_preserved':audits['historical_file_count']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--report-dir',type=Path,default=Path('reports'))
    a=p.parse_args();pipeline(a.output,a.report_dir)


if __name__=='__main__':main()
