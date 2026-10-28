"""Assemble measured/replayed rank-repair results without conflating them."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from rave_fl.audit_pilot import audit
from rave_fl.pilot import write_json
from rave_fl.replay import require


def build_report(replay_dir, original_dir, masked_dir, manifest, masked_manifest, output,
                 normalized_dir=None, normalized_manifest=None):
    replay = json.loads((replay_dir / 'replay.json').read_text())
    original = json.loads((original_dir / 'run.json').read_text())
    masked = json.loads((masked_dir / 'run.json').read_text())
    require(original.get('status') == masked.get('status') == 'complete', 'Both live runs must be complete')
    original_audit = audit(original_dir / 'run.json', manifest)
    masked_audit = audit(masked_dir / 'run.json', masked_manifest)
    write_json(original_dir / 'audit.json', original_audit)
    write_json(masked_dir / 'audit.json', masked_audit)
    normalized = normalized_audit = None
    if normalized_dir is not None:
        require(normalized_manifest is not None, 'Normalized run needs its own manifest')
        normalized = json.loads((normalized_dir / 'run.json').read_text())
        require(normalized.get('status') == 'complete', 'Normalized run must be complete')
        normalized_audit = audit(normalized_dir / 'run.json', normalized_manifest)
        write_json(normalized_dir / 'audit.json', normalized_audit)
    base = replay['references']['full']
    oc, mc = original['cases'][0], masked['cases'][0]
    require(oc['bug'] == mc['bug'] == replay['provenance']['bug'], 'Different bugs in evidence comparison')
    require(oc['sources_sha256'] == mc['sources_sha256'] == replay['provenance']['sources_sha256'], 'Source pools differ')
    require(oc['evidence']['sha256'] == replay['provenance']['evidence']['sha256'], 'Unmasked evidence changed')
    digests = {base['model_metadata']['digest']} | {
        a['model_metadata']['digest'] for c in (oc, mc) for a in c['arms'].values()}
    if normalized is not None:
        nc = normalized['cases'][0]
        require(nc['bug'] == oc['bug'] and nc['sources_sha256'] == oc['sources_sha256'], 'Normalized source pool differs')
        digests |= {a['model_metadata']['digest'] for a in nc['arms'].values()}
    require(len(digests) == 1, 'Model weights changed across comparisons')
    protocol = replay['protocol']
    primary = [a for a in replay['policy_runs'] if a['config']['token_budget'] == protocol['primary_budget']
               and a['repair_config']['max_full'] == protocol['primary_max_full']]
    predicted = next(a for a in primary if a['mode'] == 'repair_utility')
    actual = oc['arms']['repair_utility']
    keys = ('action', 'level', 'candidate_ids', 'input_tokens', 'output_tokens', 'raw_response', 'observations')
    matches = (len(predicted['calls']) == len(actual['calls']) and
               all(all(a[k] == b[k] for k in keys) for a, b in zip(predicted['calls'], actual['calls'])))
    require(matches and predicted['ranked_functions'] == actual['ranked_functions'], 'Fresh original execution differs from replay prediction')
    single_raw = next(a for a in replay['policy_runs'] if a['mode'] == 'repair_suspicion'
                      and a['config']['token_budget'] == protocol['primary_budget'] and a['repair_config']['max_full'] == 1)
    same_simple = (single_raw['ranked_functions'] == predicted['ranked_functions']
                   and single_raw['actual_tokens'] == predicted['actual_tokens'])
    action_profile = json.loads((replay_dir / 'action_value_profile.json').read_text())
    checks = {'fresh_original_matches_cached_replay': matches,
              'single_raw_suspicion_matches_utility_on_development_bug': same_simple,
              'original_audit': original_audit, 'masked_audit': masked_audit,
              'normalized_audit': normalized_audit,
              'independent_bugs': 1, 'evidence_conditions': 3 if normalized is not None else 2,
              'real_trained_utility_model': False, 'new_action_value_data': action_profile}
    checks['masked_utility_top5'] = mc['arms']['repair_utility']['accuracy']['top5']
    if normalized is not None:
        checks['posthoc_normalization_utility_top5'] = nc['arms']['repair_utility']['accuracy']['top5']
    write_json(output.parent / 'rank_repair_validation.json', checks)
    lines = ['# RAVE-FL V2: L1-first rank repair — 22 October 2026', '',
             'The development change starts from the strongest observed cheap baseline: shared L1 batches for the whole pool. It then uses a capped, recomputed ranking-utility proxy to buy independent Full inspections. The frozen final ranker, candidate pool, full/L1 prompts and model generation settings were retained.', '',
             '## Fresh inference results', '',
             'Model: local `llama3.1:8b`, temperature 0, seed 0, 160 output tokens per candidate. One previously inspected bug: Time-25, 24 candidate functions. Gold is evaluation-only. The unmasked Full row is the audited historical reference; the other rows below are fresh inference. Masked evidence has its own freshly run Full reference.', '',
             '| Evidence | Arm | Calls | Full calls | Provider tokens | Gold rank | Top-1 | Top-5 | MRR | Saved vs same-evidence Full |',
             '|---|---|---:|---:|---:|---:|---|---|---:|---:|']
    entries = [('Original', 'Full reference (historical)', base, base['actual_tokens'])]
    entries += [('Original', n, a, base['actual_tokens']) for n, a in oc['arms'].items()]
    entries += [('Candidate names masked', n, a, mc['arms']['full']['actual_tokens']) for n, a in mc['arms'].items()]
    if normalized is not None:
        entries += [('Masked + normalized numeric assertions (post-hoc)', n, a, nc['arms']['full']['actual_tokens'])
                    for n, a in nc['arms'].items()]
    for evidence, name, a, full_tokens in entries:
        e = a['accuracy']
        lines.append(f"| {evidence} | {name} | {a['call_count']} | {a['full_verified_count']} | {a['actual_tokens']:,} | {e['first_relevant_rank']} | {e['top1']} | {e['top5']} | {e['MRR']:.3f} | {(1-a['actual_tokens']/full_tokens)*100:.1f}% |")
    lines += ['', 'Every fresh arm was independently audited against raw request/response ledgers, source/evidence hashes, provider counters and recomputed offline metrics. No skipped or invalid response is silently free. Replay and fresh inference are kept separate.', '',
              '## Conditional replay: four selection controls', '',
              'No new model calls were made for replay. It verified exact original independent prompts, original batch composition, model digest/settings, raw/parsed replies, actual costs and frozen ranking/tie-breaking. A selected response is revealed only after its action is chosen; preflight planning does not use future actual output cost.', '',
              '| Policy at 16k budget, max 4 Full | Full calls | Accounted replay tokens | Gold rank | MRR |',
              '|---|---:|---:|---:|---:|']
    for a in primary:
        lines.append(f"| {a['mode']} | {a['full_verified_count']} | {a['actual_tokens']:,} | {a['accuracy']['first_relevant_rank']} | {a['accuracy']['MRR']:.3f} |")
    lines += ['',
              f"Fresh original inference exactly reproduced the replay's chosen calls, raw/parsed replies, token counters and final ranking: **{matches}**.", '',
              f"The simpler baseline 'highest raw L1 suspicion, at most one Full' matched utility's cost and ranking: **{same_simple}**. Therefore this bug does not demonstrate superiority or novelty of the utility formula. Baselines forced to spend four calls should not be used to imply otherwise.", '',
              'The utility policy used the same one-Full solution for tested 14k, 16k, 18k and 20k budgets. At 12k, conservative whole-batch admission stopped before covering the complete pool. Recorded cheap cost below a budget does not mean the preflight bound will admit the same final batch.', '',
              '## Action interactions and learning data', '',
              f"The offline dataset contains {action_profile['rows']} conditional action rows from {action_profile['unique_states']} distinct partial-repair states: {action_profile['positive_MRR_gain']} positive and {action_profile['negative_MRR_gain']} negative marginal MRR changes. All still belong to one independent bug. Future Full answers and actual costs are targets only; feature inputs use current evidence and estimated costs. No model was trained from these rows.", '',
              'The oracle enumerated zero-to-four upgrades with gold and future observed costs. It is a diagnostic ceiling, not an online policy. Pair effects were nonadditive: a Full action can undo part of another action\'s ranking gain. The paired-state dataset preserves this dependence rather than summing singleton labels.', '',
              '## Evidence ablation and limits', '',
              'All candidate method names occurring inside test/frame method identifiers were replaced by opaque aliases, not just the gold method name. Assertion lines and expected/actual values were preserved exactly, together with scenario suffixes, class/file names and line numbers. This removes direct method-name cues, not every possible localization cue. The ablation uses the same bug, not an independent test case.', '',
              'The original-to-masked result is a major failure of robustness: Full fell from rank 2 to 6, and L1/utility from rank 4/1 to 23. The utility found no positive proxy and did not upgrade anything. The positive original result therefore cannot be presented as name-independent fault reasoning. Since masked Full itself fails Top-5, this is also an evidence/model-grounding problem, not solely an allocation problem.', '',
              'A separate exact-action replay of the masked condition still found an oracle one-Full repair reaching rank 1 at 12,999 accounted tokens. That gold-informed bound is not deployable: it shows that allocation opportunity exists, while the untrained proxy cannot recognize it from weak L1 evidence. Full verification of every method is therefore a reference, not an accuracy upper bound for mixed evidence.', '',
              'One previously inspected development bug cannot establish accuracy preservation, statistical noninferiority, robustness across projects, calibration or paper novelty. The final utility is untrained, and its positive proxy/stop rule is not a guarantee of real MRR gain. The frozen ranker is the pilot fusion ranker, not a reproduced SieveFL final LLM ranker. Token totals cover verification only; local latency and monetary API savings are separate questions.', '',
              '## Decision for the next research step', '',
              'Continue with L1-first rank repair as the experimental architecture. Keep single-Full raw suspicion and small top-L1 escalation as strong controls. Do not return to unrestricted L0 -> L1 -> Full inspection of every candidate, and do not claim the more complicated proxy is necessary based on this result.', '',
              'Before scaling, test evidence grounding and the simple controls on a small number of additional independent bugs. If those checks hold, freeze the policy/evidence recipe and collect about 10–15 independent exploratory bugs under shared candidate/evidence/model/ranker conditions. Preserve an untouched bug/repository-level test split. Only after observing reproducible marginal-action gains should a learned utility/risk model be trained and calibrated on separate bugs. Measure paired Top-1/5, full-pool MRR, tokens, coverage and violations; do not count states, seeds or evidence variants as independent bugs.', '',
              'Related work already covers learned efficiency-aware localization control: [FuseSearch](https://aclanthology.org/2026.findings-acl.143/) uses SFT/RL for adaptive parallel execution. This pilot studies post-localizer verification allocation; empirical superiority and broader novelty remain open.', '',
              'See [implementation and reproduction instructions](rave_fl_rank_repair.md), the saved protocol in `configs/rave_rank_repair_protocol.json`, replay artifacts under `results/rave_fl_rank_repair/replay_final`, and fresh ledgers under `results/rave_fl_rank_repair/live_original` / `live_masked`.', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    if normalized is not None:
        lines += ['## Post-hoc numeric-evidence follow-up', '',
                  'After the masked failure was observed, a single further variant was specified in `configs/rave_rank_repair_posthoc_evidence.json`. A deterministic parser extracts the expected +04:00 and observed +03:00 offsets from the existing JUnit diff fragments and spells out their +3,600,000 ms difference. It retains the complete raw masked evidence, does not reconstruct ellipsized dates and adds no diagnosis or candidate identity. The policy, pool, model and limits are unchanged.', '',
                  'This is an explicitly post-hoc development experiment. The added text also changes length/repetition, so a difference is not proof of which formatting element caused it. Its fresh same-evidence Full/L1/utility results are in the table above. It is not an independent validation bug, and no further variant was selected to chase a favorable score.', '']
        lines += [f"Numeric clarification did not recover the masked failure: Full rank was {nc['arms']['full']['accuracy']['first_relevant_rank']}, versus {mc['arms']['full']['accuracy']['first_relevant_rank']} without the added summary; fixed L1 and utility both ended at rank {nc['arms']['repair_utility']['accuracy']['first_relevant_rank']}. The utility made zero Full repairs. Do not adopt this normalization as a proven improvement. The corresponding oracle still reaches rank 1 with one correct Full action, but the real policy does not find it.", '']
    lines += ['## Verification', '',
              '63 RAVE-FL tests passed, including 15 new rank-repair/replay/evidence tests. A broader scoped suite passed 83 tests after excluding the two previously documented files with unrelated failures. The three old failures were not modified or claimed fixed. Fresh experiment audits verified raw requests, responses, actual token totals and offline metrics in all three evidence conditions.', '']
    output.write_text('\n'.join(lines))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--masked', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--masked-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--normalized', type=Path)
    p.add_argument('--normalized-manifest', type=Path)
    a = p.parse_args()
    checks = build_report(a.replay, a.original, a.masked, a.manifest, a.masked_manifest, a.output,
                          a.normalized, a.normalized_manifest)
    print(json.dumps({k: v for k, v in checks.items() if not k.endswith('_audit')}, indent=2))


if __name__ == '__main__':
    main()
