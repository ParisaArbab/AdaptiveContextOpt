"""Versioned offline evaluation; no labels are imported by evidence selection."""
from dataclasses import dataclass
import re
from rave_fl.eval.metrics import evaluate

PROFILES=('legacy_v1','java_strict_v2')

@dataclass(frozen=True)
class MethodIdentity:
    owner: str
    name: str
    parameters: tuple | None
    file: str = ''


def split_parameters(text):
    parts=[];start=0;depth=0
    for n,c in enumerate(text):
        if c in '<([{':depth+=1
        elif c in '>)]}':
            depth-=1
            if depth<0:raise ValueError('Unbalanced parameter syntax')
        elif c==',' and depth==0:parts.append(text[start:n]);start=n+1
    if depth:raise ValueError('Unbalanced parameter syntax')
    return parts+[text[start:]] if text.strip() else []


def parameter_type(text):
    text=re.sub(r'@\w+(?:\.\w+)*(?:\([^()]*\))?\s*','',text).strip()
    text=re.sub(r'\bfinal\s+','',text).replace('...','[]')
    text=re.sub(r'\s*\[\s*\]','[]',text)
    # Erase generic arguments after balanced splitting: Java overload identity
    # uses erasure. Preserve type case and fully qualified conflicting names.
    erased=[];depth=0
    for c in text:
        if c=='<':depth+=1
        elif c=='>':depth-=1
        elif depth==0:erased.append(c)
    text=''.join(erased).strip()
    match=re.fullmatch(r'(.+?)\s+([A-Za-z_$][\w$]*)((?:\[\])*)',text)
    if match:text=match[1]+(match[3] or '')
    text=re.sub(r'\s+','',text).replace('$','.')
    if not re.fullmatch(r'[\w$.]+(?:\[\])*',text):raise ValueError('Unresolved parameter type')
    return text


def parse_identity(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Missing method identity')
    file,sep,value=value.partition('::') if '::' in value else ('','',value)
    value=value.strip().replace('#','.').replace('$','.')
    if '(' not in value:head=value;params=None
    else:
        head,tail=value.split('(',1)
        if not tail.rstrip().endswith(')'):raise ValueError('Malformed method identity')
        params=tuple(parameter_type(p) for p in split_parameters(tail.rstrip()[:-1]))
    owner,dot,name=head.strip().rpartition('.')
    if not dot:owner='';name=head.strip()
    if name=='<init>':name=owner.rsplit('.',1)[-1]
    if not re.fullmatch(r'[\w$]+',name) or not owner and not file:raise ValueError('Unqualified method identity')
    return MethodIdentity(owner,name,params,file)


def type_matches(a,b):
    if a==b:return True
    abase=a.replace('[]','');bbase=b.replace('[]','')
    return a.count('[]')==b.count('[]') and ('.' not in abase or '.' not in bbase) and abase.rsplit('.',1)[-1]==bbase.rsplit('.',1)[-1]


def strict_matches(a,b):
    try:x,y=parse_identity(a),parse_identity(b)
    except ValueError:return False
    return (x.owner==y.owner and x.name==y.name and x.parameters is not None and y.parameters is not None
            and (not x.file or not y.file or x.file==y.file)
            and len(x.parameters)==len(y.parameters) and all(type_matches(u,v) for u,v in zip(x.parameters,y.parameters)))


def evaluate_profile(arm,entities,gold,dataset,profile='java_strict_v2'):
    if profile not in PROFILES:raise ValueError('Unknown evaluation profile')
    result=evaluate(arm,entities,gold,dataset)
    if profile=='legacy_v1':return result
    result['evaluation_profile']=profile
    if not gold.functions:
        result['gold_unavailable_reason']=gold.unavailable_reason or 'No method labels supplied; not zero accuracy'
        return result
    if dataset!='defects4j':return result
    unresolved=[]
    for g in gold.functions:
        try:
            if parse_identity(g).parameters is None:unresolved.append(g)
        except ValueError:unresolved.append(g)
    if unresolved:
        result['method']=None;result['gold_unavailable_reason']='Unresolved Java labels: '+', '.join(unresolved);return result
    if entities and all(e.kind=='file' for e in entities):result['method']=None;return result
    truth=list(dict.fromkeys(gold.functions));ranking=list(dict.fromkeys(arm['ranked_functions']))
    pool=[e.entity_id for e in entities]
    hits=lambda ids:{g for g in truth if any(strict_matches(i,g) for i in ids)}
    relevant=[n for n,i in enumerate(ranking,1) if hits([i])];first=min(relevant) if relevant else None
    recovered=hits(ranking);pool_gold=hits(pool)
    metric={'first_relevant_rank':first,'MRR':1/first if first else 0.,'gold_locations':len(truth),'pool_gold_count':len(pool_gold),
            'candidate_pool_coverage':len(pool_gold)/len(truth),'candidate_pool_any_gold':bool(pool_gold),
            'all_gold_recovered':len(recovered)==len(truth),'all_gold_rank':max(relevant) if len(recovered)==len(truth) else None}
    for k in [1,3,5,10]:
        seen=hits(ranking[:k]);metric.update({f'Hit@{k}':int(bool(seen)),f'Top-{k}':int(bool(seen)),f'Recall@{k}':len(seen)/len(truth)})
    result['method']=metric
    return result
