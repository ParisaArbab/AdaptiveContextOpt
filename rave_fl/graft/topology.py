"""TopoCompress-inspired core; frozen dense+sparse embeddings must be explicit."""
from dataclasses import dataclass,asdict,replace
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import re
from rave_fl.eval.schema import fingerprint
from rave_fl.graft.selection import query_text,pack_scores


def text_hash(text):return hashlib.sha256(text.encode()).hexdigest()


def embedding_texts(graph):
    return sorted({n.text for n in graph.nodes} | {query_text(graph, t) for t in graph.targets})


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def pinned_revision(value):
    return (
        isinstance(value, str) and bool(value) and value.strip() == value
        and value.lower() not in {'main', 'master', 'latest', 'head'}
    )


BGE_M3_HEAD_FILES = {'sparse_linear.pt', 'colbert_linear.pt'}
BGE_M3_DENSE_DIMENSION = 1024


def is_checkpoint_weight(path):
    return path.suffix == '.safetensors' or (path.suffix == '.bin' and path.name.startswith('pytorch_model'))


def has_bge_m3_files(names):
    names = set(names)
    return (
        {'config.json', *BGE_M3_HEAD_FILES} <= names
        and bool({'tokenizer.json', 'sentencepiece.bpe.model'} & names)
        and any(is_checkpoint_weight(Path(name)) for name in names)
    )


def _unique_json_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError('Duplicate key in embedding artifact')
    return value


class EmbeddingBundle:
    def __init__(self, value):
        keys = {'schema', 'model_id', 'synthetic', 'texts', 'provenance'}
        if not isinstance(value, dict) or set(value) - keys or value.get('schema') != 'graft-embeddings-v1':
            raise ValueError('Unsupported embedding artifact')
        if (not isinstance(value.get('model_id'), str) or not value['model_id'].strip()
                or type(value.get('synthetic')) is not bool):
            raise ValueError('Embedding model identity and synthetic flag required')
        if not isinstance(value.get('texts'), dict) or not value['texts']:
            raise ValueError('Nonempty exact-text embedding map required')
        self.model_id = value['model_id']
        self.synthetic = value['synthetic']
        self.dimension = None
        for key, item in value['texts'].items():
            if not _sha256(key) or not isinstance(item, dict):
                raise ValueError('Invalid exact-text embedding entry')
            vector, sparse = item.get('dense'), item.get('sparse')
            if not isinstance(vector, list) or not vector or any(not _finite_number(x) for x in vector):
                raise ValueError('Invalid dense embedding')
            if self.dimension is None:
                self.dimension = len(vector)
            squared_norm = sum(float(x) * float(x) for x in vector)
            if len(vector) != self.dimension or not 0 < squared_norm < math.inf:
                raise ValueError('Inconsistent/zero/nonfinite dense embedding norm')
            if not isinstance(sparse, dict) or any(
                not isinstance(k, str) or not _finite_number(v) or v < 0 for k, v in sparse.items()
            ):
                raise ValueError('Invalid lexical weights')
            if item.get('truncated', False) is not False:
                raise ValueError('Truncated embedding text is unsupported; extract smaller coherent units')
        self.value = json.loads(json.dumps(value))

    def validate_for_graph(self, graph):
        """Fixed-experiment contract; legacy selection may still use older bundles."""
        expected = {text_hash(t) for t in embedding_texts(graph)}
        entries = self.value['texts']
        if entries.keys() != expected:
            missing, extra = len(expected - entries.keys()), len(entries.keys() - expected)
            raise ValueError(f'Exact-text embedding coverage differs ({missing} missing, {extra} extra); no fallback')
        if self.synthetic:
            return
        provenance = self.value.get('provenance')
        if not isinstance(provenance, dict):
            raise ValueError('Fixed real embeddings require local BGE-M3 export provenance')
        if (not graph.revision or provenance.get('source_revision') != graph.revision
                or provenance.get('graph_sha256') != graph.hash):
            raise ValueError('Embedding graph hash or source revision differs from prepared graph')
        revision = provenance.get('checkpoint_revision')
        files = provenance.get('checkpoint_files')
        if (
            not pinned_revision(revision) or self.model_id != f'BGE-M3:{revision}'
            or not isinstance(files, dict)
            or not all(isinstance(p, str) and _sha256(v) and not Path(p).is_absolute() and '..' not in Path(p).parts
                       for p, v in files.items())
            or not has_bge_m3_files(files)
        ):
            raise ValueError('Fixed real embeddings require a pinned BGE-M3 checkpoint, tokenizer, heads and file hashes')
        if self.dimension != BGE_M3_DENSE_DIMENSION:
            raise ValueError(f'BGE-M3 dense dimension must be {BGE_M3_DENSE_DIMENSION}')
        max_length = provenance.get('max_length')
        longest = provenance.get('max_observed_tokens')
        if (type(max_length) is not int or type(longest) is not int or not 0 < longest <= max_length
                or any(item.get('truncated') is not False for item in entries.values())):
            raise ValueError('Fixed real embeddings require explicit non-truncation and tokenizer lengths')
    @property
    def hash(self):return fingerprint(self.value)
    def lookup(self,text):
        try:item=self.value['texts'][text_hash(text)]
        except KeyError:raise ValueError('Embedding unavailable for exact text/query hash; no synthetic fallback')
        norm=math.sqrt(sum(v*v for v in item['dense']))
        return tuple(x/norm for x in item['dense']),item['sparse']
    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_json_object))


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


def topo_select(graph,target,bundle,budget,counter=None,config=None,*,audit=False):
    if not isinstance(bundle,EmbeddingBundle):raise TypeError('Explicit dense/sparse embedding bundle required')
    nodes=sorted(graph.nodes,key=lambda n:n.id)
    config=config or TopologyConfig();ids=[n.id for n in nodes];vectors={};lexical={}
    for n in nodes:vectors[n.id],lexical[n.id]=bundle.lookup(n.text)
    query_vector,query_sparse=bundle.lookup(query_text(graph,target))
    dot=lambda a,b:max(0.,sum(x*y for x,y in zip(a,b)))
    dense=normalize([dot(vectors[i],query_vector) for i in ids]);sparse=normalize([sum(w*lexical[i].get(t,0) for t,w in query_sparse.items()) for i in ids])
    alignment={i:config.mu*d+(1-config.mu)*l for i,d,l in zip(ids,dense,sparse)}
    groups=defaultdict(list)
    # The builder emits PART_OF edges in AST statement order. Keep this order
    # for statements on the same line even if the node array is reordered.
    source_order={e.source:pos for pos,e in enumerate(graph.edges) if e.relation=='PART_OF'}
    for n in nodes:
        if n.kind in {'method','code'}:
            groups[(n.document,n.owner)].append((n.origin.start_line,n.kind!='method',source_order.get(n.id,len(graph.edges)),n.id))
    acceleration={i:0. for i in ids};adj={i:{} for i in ids}
    for group in groups.values():
        sequence=[i for *_,i in sorted(group)]
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
    if audit:
        diagnostic.update(seed_sha256=fingerprint(seed), adjacency_sha256=fingerprint(adj),
                          seed_mass=math.fsum(seed.values()), score_mass=math.fsum(scores.values()))
    result=pack_scores(graph,target,scores,budget,counter,redundancy=lambda i,j:dot(vectors[i],vectors[j]),penalty=config.redundancy,selector='topocompress_typed' if config.typed else 'topocompress_semantic')
    return replace(result,diagnostics={**result.diagnostics,**diagnostic,'embedding_model':bundle.model_id,'embedding_hash':bundle.hash,'embedding_synthetic':bundle.synthetic,
                                      'config':asdict(config),'semantic_acceleration_scope':'source-ordered units within each supplied method; never candidate rank order'})
