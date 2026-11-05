"""Bounded typed evidence graphs from supplied buggy sources, never gold labels."""
from dataclasses import asdict,dataclass,field
import ast
from collections import defaultdict
import hashlib
import re
import textwrap
from functools import cached_property
from types import MappingProxyType
from rave_fl.eval.schema import CodeEntity,OnlineInstance,fingerprint,relative_path
from rave_fl.eval.representations import java_tree
from rave_fl.graft.evaluation import parse_identity

NODE_KINDS={'failure','frame','method','code','test','value'}
RELATIONS={'PART_OF','ADJACENT','OBSERVED_FRAME','STACK_NEXT','STACK_AT','MAY_CALL','REQUIRES','EXECUTES','SEMANTIC'}

@dataclass(frozen=True)
class Origin:
    path: str
    start_line: int
    end_line: int
    source_hash: str
    extractor: str
    def __post_init__(self):
        if not self.path or self.start_line<1 or self.end_line<self.start_line or not self.extractor:raise ValueError('Evidence provenance required')
        if not re.fullmatch('[a-f0-9]{64}',self.source_hash):raise ValueError('Source SHA-256 required')

@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    text: str
    origin: Origin
    owner: str = ''
    document: str = ''
    defines: tuple = ()
    uses: tuple = ()
    def __post_init__(self):
        object.__setattr__(self,'defines',tuple(self.defines));object.__setattr__(self,'uses',tuple(self.uses))
        if self.kind not in NODE_KINDS or not self.id or not self.text.strip():raise ValueError('Invalid evidence node')

@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    provenance: str
    confidence: float = 1.
    def __post_init__(self):
        if self.relation not in RELATIONS or not self.provenance or not 0<=self.confidence<=1:raise ValueError('Typed edge provenance/confidence required')

@dataclass(frozen=True)
class Graph:
    nodes: tuple
    edges: tuple
    targets: dict
    sources: dict
    language: str
    revision: str
    diagnostics: tuple = ()
    schema: str = 'graft-graph-v1'
    def __post_init__(self):
        object.__setattr__(self,'targets',MappingProxyType(dict(self.targets)))
        object.__setattr__(self,'sources',MappingProxyType({k:MappingProxyType(dict(v)) for k,v in self.sources.items()}))
        object.__setattr__(self,'diagnostics',tuple(MappingProxyType(dict(v)) for v in self.diagnostics))
        ids={n.id for n in self.nodes}
        if self.schema!='graft-graph-v1' or len(ids)!=len(self.nodes):raise ValueError('Duplicate IDs or unsupported graph schema')
        if any(e.source not in ids or e.target not in ids for e in self.edges):raise ValueError('Dangling graph edge')
        if any(i not in ids for i in self.targets.values()) or set(self.targets)!=set(self.sources):raise ValueError('Invalid target mapping')
        for value in self.sources.values():
            if hashlib.sha256(value['source'].encode()).hexdigest()!=value['sha256']:raise ValueError('Graph source hash mismatch')
    def to_dict(self):
        return {'nodes':[asdict(n) for n in self.nodes],'edges':[asdict(e) for e in self.edges],
                'targets':dict(self.targets),'sources':{k:dict(v) for k,v in self.sources.items()},
                'language':self.language,'revision':self.revision,'diagnostics':[dict(v) for v in self.diagnostics],'schema':self.schema}
    @cached_property
    def hash(self):return fingerprint(self.to_dict())
    @cached_property
    def by_id(self):return {n.id:n for n in self.nodes}
    @classmethod
    def from_dict(cls,value):
        value=dict(value);value['nodes']=tuple(Node(**{**n,'origin':Origin(**n['origin'])}) for n in value['nodes']);value['edges']=tuple(Edge(**e) for e in value['edges'])
        return cls(**value)

@dataclass(frozen=True)
class CoverageFact:
    method: str
    test: str
    raw_evidence: str
    origin: Origin
    def __post_init__(self):
        if not self.raw_evidence.strip() or not self.test:raise ValueError('Observed coverage evidence required; ranking scores are not coverage')
        if hashlib.sha256(self.raw_evidence.encode()).hexdigest()!=self.origin.source_hash:raise ValueError('Coverage evidence hash mismatch')


def _java_units(source):
    prefix='class GraftWrapper {\n';data=(prefix+source+'\n}').encode();offset=len(prefix.encode());root=java_tree(data.decode())
    pending=[root];decl=None
    while pending:
        n=pending.pop(0)
        if n.type in {'method_declaration','constructor_declaration'}:decl=n;break
        pending.extend(n.named_children)
    if decl is None or decl.has_error:raise ValueError('Java method AST could not be extracted')
    body=decl.child_by_field_name('body');signature=data[decl.start_byte:body.start_byte if body else decl.end_byte].decode().strip()
    params=decl.child_by_field_name('parameters');param_names=[]
    if params:
        for p in params.named_children:
            name=p.child_by_field_name('name')
            if name:param_names.append(data[name.start_byte:name.end_byte].decode())
    units=[];calls=[]
    for n in body.named_children if body else []:
        text=data[n.start_byte:n.end_byte].decode();nodes=[];stack=[n]
        while stack:
            x=stack.pop();nodes.append(x);stack.extend(x.named_children)
        defs={data[x.child_by_field_name('name').start_byte:x.child_by_field_name('name').end_byte].decode() for x in nodes if x.type=='variable_declarator' and x.child_by_field_name('name')}
        used=set()
        for x in nodes:
            if x.type=='identifier':
                par=x.parent
                if par and par.type=='method_invocation' and par.child_by_field_name('name')==x:continue
                used.add(data[x.start_byte:x.end_byte].decode())
        invocation=[]
        for x in nodes:
            if x.type=='method_invocation':
                name=x.child_by_field_name('name');obj=x.child_by_field_name('object');args=x.child_by_field_name('arguments')
                invocation.append((data[name.start_byte:name.end_byte].decode(),len(args.named_children) if args else 0,data[obj.start_byte:obj.end_byte].decode() if obj else ''))
        start=source.encode()[:max(0,n.start_byte-offset)].count(b'\n')
        end=source.encode()[:max(0,n.end_byte-offset)].count(b'\n')
        units.append((text,start,end,tuple(sorted(defs)),tuple(sorted(used-defs)),invocation))
    return signature,tuple(param_names),units


def _python_units(source):
    source=textwrap.dedent(source);tree=ast.parse(source);decl=tree.body[0]
    if not isinstance(decl,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):raise ValueError('Expected Python callable/class')
    body=decl.body;first=body[0].lineno-1 if body else 1;signature='\n'.join(source.splitlines()[:first]).strip()
    params=tuple(a.arg for a in ast.walk(decl.args) if isinstance(a,ast.arg)) if hasattr(decl,'args') else ()
    units=[]
    for n in body:
        text=ast.get_source_segment(source,n)
        defs=tuple(sorted({x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)}))
        uses=tuple(sorted({x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}-set(defs)))
        # Python dynamic calls remain unresolved; do not invent a call graph.
        units.append((text,n.lineno-1,n.end_lineno-1,defs,uses,[]))
    return signature or source.splitlines()[0],params,units


def build_graph(instance,entities,coverage=(),max_nodes=3000):
    if not isinstance(instance,OnlineInstance) or any(not isinstance(e,CodeEntity) for e in entities):raise TypeError('Typed online inputs required')
    if instance.language not in {'java','python'}:raise ValueError('GRAFT structural graph supports Java/Python')
    nodes=[];edges=[];targets={};sources={};diagnostics=[];method_units={};invocations={}
    def add(kind,text,origin,owner='',document='',defines=(),uses=()):
        ident=kind+'_'+fingerprint([kind,text,asdict(origin),owner])[:20]
        node=Node(ident,kind,text,origin,owner,document,defines,uses)
        if node not in nodes:nodes.append(node)
        if len(nodes)>max_nodes:raise ValueError('Evidence graph exceeds configured node limit')
        return node
    failure=add('failure',instance.evidence.text,Origin(instance.evidence.source or 'inline',1,max(1,len(instance.evidence.text.splitlines())),instance.evidence.sha256,'supplied_text:'+instance.evidence.kind),document='failure')
    for entity in entities:
        if entity.entity_id in targets:raise ValueError('Duplicate candidate identity')
        relative_path(entity.path);h=hashlib.sha256(entity.source.encode()).hexdigest()
        origin=Origin(entity.path,entity.start_line,entity.end_line,h,'supplied_buggy_source')
        try:signature,params,units=(_java_units if instance.language=='java' else _python_units)(entity.source)
        except (ValueError,SyntaxError) as exc:
            signature=entity.source.splitlines()[0];params=();units=[];diagnostics.append({'owner':entity.entity_id,'kind':'parse_error','detail':str(exc)})
        method=add('method',signature,origin,entity.entity_id,entity.path,params)
        targets[entity.entity_id]=method.id;sources[entity.entity_id]={'source':entity.source,'path':entity.path,'sha256':h,'start_line':entity.start_line,'end_line':entity.end_line}
        last=None;unit_nodes=[]
        for text,start,end,defs,uses,calls in units:
            unit=add('code',text,Origin(entity.path,entity.start_line+start,entity.start_line+end,h,'AST_top_level_statement'),entity.entity_id,entity.path,defs,uses)
            edges.append(Edge(unit.id,method.id,'PART_OF','AST containment'))
            if last:
                edges.extend([Edge(last.id,unit.id,'ADJACENT','same method source order'),Edge(unit.id,last.id,'ADJACENT','same method source order')])
            last=unit;unit_nodes.append(unit);invocations[unit.id]=calls
        definitions=defaultdict(list)
        for name in params:definitions[name].append(method)
        for unit in unit_nodes:
            for name in unit.defines:definitions[name].append(unit)
        for unit in unit_nodes:
            for name in unit.uses:
                ds=definitions.get(name,[])
                if len(ds)==1 and ds[0].id!=unit.id and ds[0].origin.start_line<=unit.origin.start_line:
                    edges.append(Edge(unit.id,ds[0].id,'REQUIRES','AST lexical definition, not reaching-definition proof'))
        method_units[entity.entity_id]=unit_nodes
    # Only static unique same-owner/arity call candidates. MAY_CALL is expressly
    # not an observed invocation, runtime data flow or causal dependency.
    identities={}
    if instance.language=='java':
        for ident in targets:
            try:identities[ident]=parse_identity(ident)
            except ValueError:pass
        owners={n.id:n.owner for n in nodes}
        for node_id,calls in invocations.items():
            current=identities.get(owners[node_id])
            if current is None:continue
            for name,arity,receiver in calls:
                matches=[ident for ident,sig in identities.items() if sig.owner==current.owner and sig.name==name and sig.parameters is not None and len(sig.parameters)==arity]
                if receiver not in {'','this',current.owner.rsplit('.',1)[-1]}:continue
                if len(matches)==1:edges.append(Edge(node_id,targets[matches[0]],'MAY_CALL','unique same-owner static name/arity; dynamic dispatch unresolved',.7))
                elif matches:diagnostics.append({'owner':owners[node_id],'kind':'ambiguous_call','name':name,'arity':arity})
    previous=None
    for n,line in enumerate(instance.evidence.text.splitlines() if instance.evidence.kind=='observed_failure' else [],1):
        match=re.search(r'\bat\s+([\w.$<>]+)\(([^():]+):(\d+)\)',line)
        if not match:continue
        frame=add('frame',line.strip(),Origin(instance.evidence.source or 'inline',n,n,instance.evidence.sha256,'stack_frame'),document='failure')
        edges.append(Edge(failure.id,frame.id,'OBSERVED_FRAME','literal supplied stack frame'))
        if previous:edges.append(Edge(previous.id,frame.id,'STACK_NEXT','text stack order; not a direct-call proof'))
        previous=frame;file=match[2];ln=int(match[3]);name=match[1].rsplit('.',1)[-1]
        candidates=[e for e in entities if e.path.rsplit('/',1)[-1]==file and e.start_line<=ln<=e.end_line]
        if len(candidates)==1:
            edges.append(Edge(frame.id,targets[candidates[0].entity_id],'STACK_AT','unique source file/line within supplied candidate span'))
        elif candidates:diagnostics.append({'kind':'ambiguous_frame','text':line})
    for fact in coverage:
        if not isinstance(fact,CoverageFact):raise TypeError('Coverage needs observed, checksummed instrumentation facts')
        if fact.method not in targets:diagnostics.append({'kind':'coverage_outside_pool','method':fact.method});continue
        test=add('test',fact.raw_evidence,fact.origin,document=fact.test)
        edges.append(Edge(test.id,targets[fact.method],'EXECUTES','supplied observed coverage: '+fact.test))
    diagnostics.append({'kind':'coverage_status','detail':'observed facts supplied' if coverage else 'not supplied; ranker scores do not create coverage edges'})
    return Graph(tuple(nodes),tuple(dict.fromkeys(edges)),targets,sources,instance.language,instance.revision,tuple(diagnostics))
