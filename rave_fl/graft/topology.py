"""TopoCompress-inspired core; frozen dense+sparse embeddings must be explicit."""
from dataclasses import dataclass,asdict,replace
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from rave_fl.eval.schema import fingerprint
from rave_fl.graft.selection import query_text,pack_scores


def text_hash(text):return hashlib.sha256(text.encode()).hexdigest()

class EmbeddingBundle:
    def __init__(self,value):
        if set(value)-{'schema','model_id','synthetic','texts','provenance'} or value.get('schema')!='graft-embeddings-v1':raise ValueError('Unsupported embedding artifact')
        if not value.get('model_id') or type(value.get('synthetic')) is not bool:raise ValueError('Embedding model identity and synthetic flag required')
        self.value=json.loads(json.dumps(value));self.model_id=value['model_id'];self.synthetic=value['synthetic'];self.dimension=None
        for key,item in value['texts'].items():
            vector=item['dense'];sparse=item['sparse']
            if not vector or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in vector):raise ValueError('Invalid dense embedding')
            if self.dimension is None:self.dimension=len(vector)
            if len(vector)!=self.dimension or sum(x*x for x in vector)<=0:raise ValueError('Inconsistent/zero dense embedding')
            if any(not isinstance(k,str) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for k,v in sparse.items()):raise ValueError('Invalid lexical weights')
            if item.get('truncated',False):raise ValueError('Truncated embedding text is unsupported; extract smaller coherent units')
    @property
    def hash(self):return fingerprint(self.value)
    def lookup(self,text):
        try:item=self.value['texts'][text_hash(text)]
        except KeyError:raise ValueError('Embedding unavailable for exact text/query hash; no synthetic fallback')
        norm=math.sqrt(sum(v*v for v in item['dense']))
        return tuple(x/norm for x in item['dense']),item['sparse']
    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))

@dataclass(frozen=True)
class TopologyConfig:
    mu: float=.5
    acceleration: float=.2
    semantic_weight: float=1.
    sequential_weight: float=1.
    program_weight: float=.25
    eta: float=.85
    redundancy: float=.2
    neighbors: int=5
    max_iterations: int=300
    tolerance: float=1e-9
    typed: bool=False
    propagate: bool=True
    def __post_init__(self):
        if not 0<=self.mu<=1 or not 0<self.eta<1:raise ValueError('Invalid mixture/damping')
        if any(not math.isfinite(v) or v<0 for v in [self.acceleration,self.semantic_weight,self.sequential_weight,self.program_weight,self.redundancy]):raise ValueError('Invalid graph weight')
        if type(self.neighbors) is not int or self.neighbors<1 or self.max_iterations<1 or self.tolerance<=0 or not math.isfinite(self.tolerance):raise ValueError('Invalid topology limits')


def normalize(values):
    low=min(values,default=0.);high=max(values,default=0.)
    if high==low:return [1. if high>0 else 0. for _ in values]
    return [(v-low)/(high-low) for v in values]


def propagate(seed,adjacency,eta=.85,max_iterations=300,tolerance=1e-9):
    if not 0<eta<1 or max_iterations<1 or tolerance<=0:raise ValueError('Invalid propagation configuration')
    if set(seed)!=set(adjacency):raise ValueError('Graph and seed IDs must agree')
    if any(not math.isfinite(v) or v<0 for v in seed.values()):raise ValueError('Nonnegative seed required')
    for u,edges in adjacency.items():
        if set(edges)-set(seed) or any(not math.isfinite(v) or v<0 for v in edges.values()):raise ValueError('Invalid graph transition')
    state=dict(seed)
    for step in range(max_iterations):
        nxt={i:(1-eta)*v for i,v in seed.items()}
        for u,edges in adjacency.items():
            total=sum(edges.values())
            if total==0:nxt[u]+=eta*state[u] # isolated node retains its mass
            else:
                for v,weight in edges.items():nxt[v]+=eta*state[u]*weight/total
        error=max((abs(nxt[i]-state[i]) for i in state),default=0.)
        state=nxt
        if error<=tolerance:return state,{'iterations':step+1,'converged':True,'residual':error}
    raise ValueError('Propagation did not converge within the configured iteration cap')


def topo_select(graph,target,bundle,budget,counter=None,config=None):
    if not isinstance(bundle,EmbeddingBundle):raise TypeError('Explicit dense/sparse embedding bundle required')
    config=config or TopologyConfig();ids=[n.id for n in graph.nodes];vectors={};lexical={}
    for n in graph.nodes:vectors[n.id],lexical[n.id]=bundle.lookup(n.text)
    query_vector,query_sparse=bundle.lookup(query_text(graph,target))
    dot=lambda a,b:max(0.,sum(x*y for x,y in zip(a,b)))
    dense=normalize([dot(vectors[i],query_vector) for i in ids]);sparse=normalize([sum(w*lexical[i].get(t,0) for t,w in query_sparse.items()) for i in ids])
    alignment={i:config.mu*d+(1-config.mu)*l for i,d,l in zip(ids,dense,sparse)}
    groups=defaultdict(list)
    for pos,n in enumerate(graph.nodes):
        if n.kind in {'method','code'}:groups[(n.document,n.owner)].append((n.origin.start_line,pos,n.id))
    acceleration={i:0. for i in ids};adj={i:{} for i in ids}
    for group in groups.values():
        sequence=[i for _,_,i in sorted(group)]
        for a,b in zip(sequence,sequence[1:]):
            adj[a][b]=adj[a].get(b,0)+config.sequential_weight;adj[b][a]=adj[b].get(a,0)+config.sequential_weight
        for a,b,c in zip(sequence,sequence[1:],sequence[2:]):
            acceleration[b]=math.sqrt(sum((2*y-x-z)**2 for x,y,z in zip(vectors[a],vectors[b],vectors[c])))
    acc=normalize([acceleration[i] for i in ids]);seed={i:(1+config.acceleration*k)*alignment[i] for i,k in zip(ids,acc)}
    for i in ids:
        nearest=sorted(((dot(vectors[i],vectors[j]),j) for j in ids if i!=j),key=lambda p:(-p[0],p[1]))[:config.neighbors]
        for sim,j in nearest:
            if sim>0:adj[i][j]=adj[i].get(j,0)+config.semantic_weight*sim
    if config.typed:
        for e in graph.edges:
            if e.relation!='ADJACENT':adj[e.source][e.target]=adj[e.source].get(e.target,0)+config.program_weight*e.confidence
    scores,diagnostic=propagate(seed,adj,config.eta,config.max_iterations,config.tolerance) if config.propagate else (seed,{'iterations':0,'converged':True,'residual':0.})
    result=pack_scores(graph,target,scores,budget,counter,redundancy=lambda i,j:dot(vectors[i],vectors[j]),penalty=config.redundancy,selector='topocompress_typed' if config.typed else 'topocompress_semantic')
    return replace(result,diagnostics={**result.diagnostics,**diagnostic,'embedding_model':bundle.model_id,'embedding_hash':bundle.hash,'embedding_synthetic':bundle.synthetic,
                                      'config':asdict(config),'semantic_acceleration_scope':'source-ordered units within each supplied method; never candidate rank order'})
