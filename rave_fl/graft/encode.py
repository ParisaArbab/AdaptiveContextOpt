"""Explicit local BGE-M3 export. This command loads a model; it never downloads."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
from rave_fl.graft.graph import Graph
from rave_fl.graft.topology import (
    BGE_M3_HEAD_FILES, EmbeddingBundle, embedding_texts, has_bge_m3_files,
    is_checkpoint_weight, pinned_revision, text_hash,
)
from rave_fl.pilot import write_json


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export(graph,model_dir,revision,output,device='cpu',batch_size=8,max_length=8192):
    output=Path(output);model_dir=Path(model_dir).resolve()
    if output.exists():raise ValueError('Embedding output already exists')
    if not model_dir.is_dir() or not (model_dir/'config.json').is_file():raise ValueError('Existing complete local BGE-M3 model directory required')
    files = {
        p.relative_to(model_dir).as_posix(): p for p in sorted(model_dir.rglob('*'))
        if p.is_file() and p.stat().st_size
        and (is_checkpoint_weight(p) or p.name in BGE_M3_HEAD_FILES or p.suffix in {'.json', '.model', '.txt'})
    }
    if not has_bge_m3_files(files):
        raise ValueError('Local BGE-M3 weights, heads and tokenizer files are required before encoder loading')
    if not pinned_revision(revision):raise ValueError('Pin the embedding checkpoint revision')
    if not graph.revision:raise ValueError('A buggy source revision is required for local embedding export')
    if type(max_length) is not int or max_length<1 or type(batch_size) is not int or batch_size<1:
        raise ValueError('Positive embedding batch size and maximum token length required')
    # Scope offline mode to this explicit process operation, restoring prior values.
    keys=['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'];previous={k:os.environ.get(k) for k in keys}
    for k in keys:os.environ[k]='1'
    try:
        from transformers import AutoTokenizer
        from FlagEmbedding import BGEM3FlagModel
        tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
        texts=embedding_texts(graph)
        longest=max(len(tokenizer.encode(t,truncation=False)) for t in texts)
        if longest>max_length:raise ValueError('Embedding truncation would occur; reduce coherent source units first')
        hashes = {name: checkpoint_hash(path) for name, path in files.items()}
        start=time.monotonic();model=BGEM3FlagModel(str(model_dir),devices=[device],use_fp16=device.startswith('cuda'))
        encoded=model.encode(texts,batch_size=batch_size,max_length=max_length,return_dense=True,return_sparse=True,return_colbert_vecs=False)
        if len(encoded['dense_vecs'])!=len(texts) or len(encoded['lexical_weights'])!=len(texts):raise ValueError('Encoder returned an incomplete text batch')
        packages = {package: version(package) for package in ('FlagEmbedding', 'transformers', 'torch')}
        # Linux CPU wheels append a build tag to the same pinned torch release.
        wheel = packages['torch']
        if wheel.endswith('+cpu'):
            packages['torch'] = wheel.removesuffix('+cpu')
        value={'schema':'graft-embeddings-v1','model_id':'BGE-M3:'+revision,'synthetic':False,
               'texts':{text_hash(t):{'dense':[float(x) for x in d],'sparse':{str(k):float(v) for k,v in s.items()},'truncated':False} for t,d,s in zip(texts,encoded['dense_vecs'],encoded['lexical_weights'])},
               'provenance':{'checkpoint_files':hashes,'checkpoint_revision':revision,
                             'graph_sha256':graph.hash,'source_revision':graph.revision,
                             'device_requested':device,'embedding_seconds_including_load':time.monotonic()-start,
                             'batch_size':batch_size,'max_length':max_length,'max_observed_tokens':longest,
                             'packages':packages,
                             **({'torch_wheel': wheel} if wheel != packages['torch'] else {})}}
        EmbeddingBundle(value).validate_for_graph(graph);write_json(output,value)
        return value['provenance']
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--revision',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();print(json.dumps(export(Graph.from_dict(json.loads(Path(a.graph).read_text())),a.model_dir,a.revision,a.output,a.device),indent=2))
