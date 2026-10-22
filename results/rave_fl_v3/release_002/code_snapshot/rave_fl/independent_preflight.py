"""Resource-only V3 feasibility check. Does not download or invoke a model."""
import json
from pathlib import Path
import shutil
import subprocess
from rave_fl.pilot import write_json


def preflight(output):
    p=json.loads(Path('configs/rave_v3_independent_protocol.json').read_text())
    java=subprocess.run(['java','-version'],text=True,capture_output=True) if shutil.which('java') else None
    cases=[]
    for bug in p['smoke_bugs']:
        checkout=Path('data/rave_fl_checkouts')/bug
        cases.append({'bug':bug,'checkout_present':checkout.exists(),
                      'defects4j_configuration_present':(checkout/'.defects4j.config').exists(),
                      'observed_failure_evidence_present':(checkout/'failing_tests').exists(),
                      'status':'not_executed_missing_prepared_buggy_inputs'})
    blockers=[]
    if not shutil.which('defects4j'):blockers.append('Defects4J CLI unavailable on PATH')
    if any(not c['checkout_present'] or not c['defects4j_configuration_present'] or not c['observed_failure_evidence_present'] for c in cases):
        blockers.append('New preselected buggy checkouts and corresponding failing_tests unavailable')
    r={'execution_kind':'resource_preflight_only','new_llm_calls':0,'provider_tokens':0,
       'defects4j_on_path':shutil.which('defects4j'),'java_version':(java.stderr.strip() if java else None),
       'cases':cases,'status':'blocked_for_fresh_verified_buggy_test_inference',
       'blockers':blockers,
       'saved_smoke_results':'Existing Time-1/Lang-1 ledgers audited separately; not new fresh inference in V3'}
    write_json(output,r);return r
