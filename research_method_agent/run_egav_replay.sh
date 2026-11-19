#!/usr/bin/env bash
# Zero GPU inference. Reuses already collected independent Ollama observations.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="${METHOD_AGENT_OUTPUT:-results/method_discovery_batch00_jsonfix}"
REPORT="${EGAV_REPORT_DIR:-$OUT/evaluation_egav_pilot}"
python research_method_agent/research_method_agent.py evaluate \
  --manifest "$OUT/manifest.json" --observations "$OUT/observations.json" \
  --gold "$OUT/gold_offline.json" --budget "${POLICY_TOKEN_BUDGET:-60000}" \
  --max-verify "${FULL_CAPS:-2}" --output "$REPORT"
python - "$REPORT/results.json" <<'PY'
import json
import sys
from collections import Counter

j = json.load(open(sys.argv[1], encoding='utf-8'))
summary = {x['arm']: x for x in j['summary']}
results = {(r['bug'], r['arm']): r for r in j['results']}
arms = ('l1', 'code_cards', 'risk_full', 'code_cards_risk_full',
        'card_rank_full', 'card_failure_full', 'card_stop_full', 'card_egav')
print('\nEGAV 5-BUG EXPLORATORY REPLAY (NO MODEL CALLS)')
print(f"{'Arm':24} {'n':>2} {'MRR':>9} {'Delta vs cards':>15} {'Mean tokens':>13} {'Full calls':>11}")
for arm in arms:
    s = summary[arm]
    n = s.get('paired_n', 0)
    if not n:
        print(f'{arm:24} INCOMPLETE')
        continue
    paired = [r for (bug, policy), r in results.items()
              if policy == arm and r['status'] == 'complete'
              and results.get((bug, 'code_cards'), {}).get('status') == 'complete']
    delta = sum(r['mrr'] - results[(r['bug'], 'code_cards')]['mrr'] for r in paired) / len(paired)
    calls = sum(r['full_calls'] for r in paired)
    print(f"{arm:24} {n:2d} {s['mean_mrr']:9.4f} {delta:+15.4f} "
          f"{s['mean_tokens']:13,.0f} {calls:11d}")
print('\nPER-BUG FULL TARGETS (NEW ARMS):')
for bug in sorted({r['bug'] for r in j['results']}):
    print('\n', bug)
    for arm in arms[4:]:
        r = results[(bug, arm)]
        names = [x['candidate'].split('::')[-1].split('(')[0] for x in r['actions'] if x['kind'] == 'full']
        print(f"  {arm:20} MRR={r['mrr']:.4f} tokens={r['tokens']:,} "
              f"status={r['status']} Full={names} stop={r.get('policy_stop_reason')}")
print('\nWarning: file-level gold proxy, 5 development bugs, offline replay only.')
print('The saved collection has an incomplete physical token ledger from an earlier crash.')
PY
printf '\nFull report: %s/report.md\n' "$REPORT"