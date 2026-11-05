"""Bounded path augmentation with explicit budget/connectivity infeasibility."""
from dataclasses import replace
import heapq
import math
from rave_fl.graft.selection import mandatory_ids,package

PATH_RELATIONS={'OBSERVED_FRAME','STACK_AT','PART_OF','MAY_CALL','REQUIRES','EXECUTES','FAILS_TEST'}


def neighbors(graph,allowed=None):
    allowed=PATH_RELATIONS if allowed is None else set(allowed)
    if 'SEMANTIC' in allowed:raise ValueError('Semantic similarity is not a trusted program evidence path')
    adjacency={n.id:{} for n in graph.nodes}
    for e in graph.edges:
        if e.relation not in allowed:continue
        adjacency[e.source][e.target]=max(adjacency[e.source].get(e.target,0),e.confidence)
        if e.relation=='PART_OF':adjacency[e.target][e.source]=max(adjacency[e.target].get(e.source,0),e.confidence)
    return adjacency


def reachable(adjacency,sources,selected):
    selected=set(selected);seen=set(sources)&selected;pending=list(sorted(seen))
    while pending:
        u=pending.pop()
        for v,w in adjacency[u].items():
            if w>0 and v in selected and v not in seen:seen.add(v);pending.append(v)
    return seen


def shortest_path(adjacency,sources,target,node_costs,max_hops=4,cost_weight=1.):
    if type(max_hops) is not int or not 1<=max_hops<=12 or not math.isfinite(cost_weight) or cost_weight<0:raise ValueError('Invalid bounded path configuration')
    if target not in adjacency or not set(sources)<=set(adjacency):raise ValueError('Unknown path terminal')
    if any(not math.isfinite(v) or v<0 for v in node_costs.values()):raise ValueError('Nonnegative finite path node costs required')
    heap=[(0.,0,s,(s,)) for s in sorted(sources)];heapq.heapify(heap);best={}
    while heap:
        cost,hops,u,path=heapq.heappop(heap)
        if cost>best.get((u,hops),math.inf):continue
        if u==target:return path
        if hops>=max_hops:continue
        for v,weight in sorted(adjacency[u].items()):
            if weight<=0 or v in path:continue
            value=cost+1/(weight+1e-9)+cost_weight*node_costs.get(v,0.)
            if value<best.get((v,hops+1),math.inf):
                best[v,hops+1]=value;heapq.heappush(heap,(value,hops+1,v,path+(v,)))
    return None


def preserve_paths(graph,selection,counter=None,terminals=None,max_targets=3,max_hops=4,tolerance=0.,relevance_weight=.5):
    if selection.graph_hash!=graph.hash:raise ValueError('Selection and graph fingerprints differ')
    if not 1<=max_targets<=10 or not 0<=tolerance<=1 or not 0<=relevance_weight<=1:raise ValueError('Invalid path limits/weights')
    if selection.status!='ready':return replace(selection,diagnostics={**selection.diagnostics,'path_status':'input_infeasible'})
    adjacency=neighbors(graph);sources={n.id for n in graph.nodes if n.kind=='failure'}
    mandatory=set(mandatory_ids(graph,selection.target));chosen=set(selection.ids)
    if terminals is None:
        eligible=[n.id for n in graph.nodes if n.kind in {'code','method'} and n.id not in mandatory]
        ordered=sorted(eligible,key=lambda i:(-selection.scores.get(i,0),i))
        terminals=tuple([graph.targets[selection.target]]+ordered[:max_targets-1])
    terminals=tuple(dict.fromkeys(terminals))
    if not terminals or not set(terminals)<=set(adjacency):raise ValueError('Explicit valid path terminals required')
    base=package(graph,selection.target,mandatory,selection.budget,counter)
    costs={n.id:max(0,package(graph,selection.target,mandatory|{n.id},selection.budget,counter).input_units-base.input_units)/selection.budget for n in graph.nodes}
    paths={};unreachable=[]
    for t in terminals:
        path=shortest_path(adjacency,sources,t,costs,max_hops)
        if path is None:unreachable.append(t)
        else:paths[t]=list(path);chosen.update(path)
    def coverage(ids):return len(reachable(adjacency,sources,ids)&set(terminals))/len(terminals)
    desired=coverage(chosen)
    def objective(ids):return relevance_weight*sum(selection.scores.get(i,0.) for i in ids)+(1-relevance_weight)*coverage(ids)
    removed=[]
    current=package(graph,selection.target,chosen,selection.budget,counter,selection.selector,selection.scores)
    while current.status!='ready':
        options=[]
        for i in sorted(chosen-mandatory-sources):
            trial=chosen-{i}
            if coverage(trial)+1e-12<desired-tolerance:continue
            candidate=package(graph,selection.target,trial,selection.budget,counter)
            saved=current.input_units-candidate.input_units
            if saved>0:options.append(((objective(chosen)-objective(trial))/saved,i,candidate))
        if not options:break
        _,i,candidate=min(options,key=lambda v:(v[0],v[1]));chosen.remove(i);removed.append(i);current=candidate
    result=package(graph,selection.target,chosen,selection.budget,counter,selection.selector,selection.scores)
    diagnostic={**selection.diagnostics,'paths':paths,'path_terminals':list(terminals),'unreachable_terminals':unreachable,
                'path_coverage_before':coverage(selection.ids),'path_coverage_after':coverage(chosen),'path_status':'ready' if result.status=='ready' else 'protected_path_infeasible',
                'path_removed_ids':removed,'path_reachability_is_causality':False,'max_hops':max_hops,'connectivity_tolerance':tolerance}
    return replace(result,reason=result.reason if result.status=='ready' else 'Protected anchors/paths do not fit the input budget',diagnostics=diagnostic)
