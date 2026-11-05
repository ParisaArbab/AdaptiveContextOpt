"""Budget-neutral closure of observable prerequisites, not semantic completeness."""
from dataclasses import replace
from rave_fl.graft.selection import mandatory_ids, package

SUPPORT = frozenset({'REQUIRES', 'PART_OF'})


def prerequisites(graph):
    result = {n.id: set() for n in graph.nodes}
    for edge in graph.edges:
        if edge.relation in SUPPORT:
            result[edge.source].add(edge.target)
    return result


def closure(graph, ids):
    required = prerequisites(graph)
    selected = set(ids)
    if not selected <= set(required):
        raise ValueError('Unknown evidence IDs in closure')
    pending = list(selected)
    while pending:
        for ident in required[pending.pop()] - selected:
            selected.add(ident)
            pending.append(ident)
    return selected


def missing_support(graph, ids):
    selected = set(ids)
    return sorted({(e.source, e.target, e.relation) for e in graph.edges
                   if e.relation in SUPPORT and e.source in selected and e.target not in selected})


def repair_support(graph, selection, counter=None):
    if selection.graph_hash != graph.hash:
        raise ValueError('Selection and graph fingerprints differ')
    if selection.status != 'ready':
        return replace(selection, diagnostics={**selection.diagnostics, 'repair_status': 'input_infeasible'})
    before = set(selection.ids)
    chosen = closure(graph, before)
    protected = set(mandatory_ids(graph, selection.target))
    # Preserve the path contract of the preceding optional pass.
    for path in selection.diagnostics.get('paths', {}).values():
        protected.update(path)
    protected = closure(graph, protected)
    chosen.update(protected)
    requirements = prerequisites(graph)
    removed = []
    def pack(ids):
        return package(graph, selection.target, ids, selection.budget, counter,
                       selection.selector, selection.scores)
    current = pack(chosen)
    while current.status != 'ready':
        alternatives = []
        for ident in sorted(chosen - protected):
            drop = {ident}
            while True:
                extra = {i for i in chosen - drop if requirements[i] & drop}
                if not extra:
                    break
                drop.update(extra)
            if drop & protected:
                continue
            trial = pack(chosen - drop)
            saved = current.input_units - trial.input_units
            if saved > 0:
                loss = sum(selection.scores.get(i, 0.) for i in drop)
                alternatives.append((loss / saved, tuple(sorted(drop)), trial))
        if not alternatives:
            break
        _, drop, current = min(alternatives, key=lambda row: (row[0], row[1]))
        chosen.difference_update(drop)
        removed.extend(drop)
    missing = missing_support(graph, chosen)
    diagnostic = {**selection.diagnostics, 'repair_status': 'ready' if current.status == 'ready' else 'protected_closure_infeasible',
                  'missing_support_before': missing_support(graph, before), 'missing_support_after': missing,
                  'repair_added_ids': sorted(chosen - before), 'repair_removed_ids': sorted(set(removed)),
                  'observable_support_closed': not missing, 'semantic_completeness_guaranteed': False,
                  'support_relations': sorted(SUPPORT), 'repair_budget_delta': 0}
    return replace(current, diagnostics=diagnostic,
                   reason=current.reason if current.status == 'ready' else 'Protected observable dependency closure exceeds input budget')
