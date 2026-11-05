"""Explicit local BGE-M3 export. This command loads a model; it never downloads."""
import argparse
import json
import os
from pathlib import Path
import time
from rave_fl.graft.graph import Graph
from rave_fl.graft.topology import EmbeddingBundle,text_hash
from rave_fl.graft.selection import query_text
from rave_fl.eval.schema import file_hash
from rave_fl.pilot import write_json


def export(graph,model_dir,revision,output,device='cpu',batch_size=8,max_length=8192):
    output=Path(output);model_dir=Path(model_dir).resolve()
    if output.exists():raise ValueError('Embedding output already exists')
    if not model_dir.is_dir() or not (model_dir/'config.json').is_file():raise ValueError('Existing complete local BGE-M3 model directory required')
    if not revision or revision in {'main','latest','HEAD'}:raise ValueError('Pin the embedding checkpoint revision')
    # Scope offline mode to this explicit process operation, restoring prior values.
    keys=['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'];previous={k:os.environ.get(k) for k in keys}
    for k in keys:os.environ[k]='1'
    try:
        from transformers import AutoTokenizer
        from FlagEmbedding import BGEM3FlagModel
        tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
        texts=sorted({n.text for n in graph.nodes}|{query_text(graph,t) for t in graph.targets})
        if any(len(tokenizer.encode(t,truncation=False))>max_length for t in texts):raise ValueError('Embedding truncation would occur; reduce coherent source units first')
        start=time.monotonic();model=BGEM3FlagModel(str(model_dir),devices=[device],use_fp16=device.startswith('cuda'))
        encoded=model.encode(texts,batch_size=batch_size,max_length=max_length,return_dense=True,return_sparse=True,return_colbert_vecs=False)
        hashes={str(p.relative_to(model_dir)):file_hash(p) for p in sorted(model_dir.rglob('*')) if p.is_file() and p.suffix in {'.json','.safetensors','.bin'}}
        value={'schema':'graft-embeddings-v1','model_id':'BGE-M3:'+revision,'synthetic':False,
               'texts':{text_hash(t):{'dense':[float(x) for x in d],'sparse':{str(k):float(v) for k,v in s.items()},'truncated':False} for t,d,s in zip(texts,encoded['dense_vecs'],encoded['lexical_weights'])},
               'provenance':{'checkpoint_files':hashes,'device_requested':device,'embedding_seconds_including_load':time.monotonic()-start,'batch_size':batch_size,'max_length':max_length}}
        EmbeddingBundle(value);write_json(output,value)
        return value['provenance']
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--revision',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();print(json.dumps(export(Graph.from_dict(json.loads(Path(a.graph).read_text())),a.model_dir,a.revision,a.output,a.device),indent=2))
