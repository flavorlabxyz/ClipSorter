"""Offline, reproducible evaluation of frozen labeled embeddings.
Labels affect calibration/evaluation only; held-out labels never enter scoring.
"""
from dataclasses import asdict, replace
import itertools
import hashlib
import json
import random
import resource
import time
import tracemalloc
import numpy as np
from .clustering import cluster, VERSION as CLUSTER_VERSION
from .embedding_cache import EmbeddingCache
from .recommendations import PROFILES, rank_candidates, valid_vectors, VERSION as BANK_VERSION
from ..storage.database import now


def ratio(n, d):
    return n / d if d else None


def metrics(items, assignments, ranked=None):
    tp = fp = fn = 0
    for a, b in itertools.combinations(items, 2):
        same_truth = a['label'] == b['label']
        pa, pb = assignments.get(a['id']), assignments.get(b['id'])
        same_pred = pa is not None and pb is not None and pa == pb
        tp += bool(same_pred and same_truth)
        fp += bool(same_pred and not same_truth)
        fn += bool(same_truth and not same_pred)
    assigned = [c for c in items if assignments.get(c['id']) is not None]
    return {'pairwise_precision': ratio(tp, tp+fp), 'pairwise_recall': ratio(tp,tp+fn),
            'uncertain_fraction': ratio(len(items)-len(assigned),len(items)),
            'suggested_match_accuracy': ratio(sum(assignments[c['id']]==c['label'] for c in assigned),len(assigned)) if ranked is not None else None,
            'top_ranked_accuracy': ratio(sum(ranked.get(c['id'])==c['label'] for c in items),len(items)) if ranked is not None else None,
            'assigned_clips':len(assigned),'evaluated_clips':len(items),'true_positive_pairs':tp,'false_positive_pairs':fp,'missed_pairs':fn}


def predictions(examples, candidates, backend, profile):
    labels = sorted({c['label'] for c in examples})
    groups = {i+2:[c for c in examples if c['label']==label] for i,label in enumerate(labels)}
    names = {i+2:label for i,label in enumerate(labels)}
    result = rank_candidates(groups,candidates,backend,{backend:profile})
    best = {}
    for r in result:
        best.setdefault(r['clip_id'],r)
    assigned = {c['id']:names[best[c['id']]['group_id']] if c['id'] in best and best[c['id']]['evidence']['label']=='suggested' else None for c in candidates}
    ranked = {cid:names[r['group_id']] for cid,r in best.items()}
    return assigned, ranked


def calibrate(examples, calibration, backend, target_precision=.95):
    base = PROFILES[backend]
    if not calibration:
        return base, {'used':False,'reason':'No calibration split; model-specific defaults retained'}
    valid = []
    for threshold in sorted(set([base.threshold,.72,.76,.80,.84,.88,.92,.96])):
        for margin in (.03,.05,.08,.12):
            profile = replace(base,threshold=threshold,margin=margin)
            assigned, ranked = predictions(examples,calibration,backend,profile)
            measured = metrics(calibration,assigned,ranked)
            precision = measured['suggested_match_accuracy']
            if precision is not None and precision >= target_precision:
                valid.append((measured['assigned_clips'],precision,threshold,margin,profile,measured))
    if not valid:
        # No evidence for safe acceptance: abstain rather than lower the floor.
        return replace(base,threshold=1.01), {'used':True,'reason':'No calibration setting met precision target; all predictions abstain','target_precision':target_precision}
    chosen = max(valid,key=lambda v:v[:4])
    return chosen[4], {'used':True,'target_precision':target_precision,'calibration_metrics':chosen[5]}


def validate_manifest(data):
    if data.get('schema') != 1 or not isinstance(data.get('clips'),list):
        raise ValueError('Expected a schema=1 labeled embedding manifest')
    ids=set()
    for c in data['clips']:
        if c['id'] in ids: raise ValueError('Each clip must appear once: no split leakage')
        ids.add(c['id'])
        if c.get('split') not in ('example','calibration','evaluation') or not c.get('label'):
            raise ValueError('Every clip needs label and example/calibration/evaluation split')
    if not any(c['split']=='example' for c in data['clips']) or not any(c['split']=='evaluation' for c in data['clips']):
        raise ValueError('Need example and held-out evaluation clips')


def benchmark(data, target_precision=.95):
    validate_manifest(data)
    examples=[c for c in data['clips'] if c['split']=='example']
    calibration=[c for c in data['clips'] if c['split']=='calibration']
    evaluation=[c for c in data['clips'] if c['split']=='evaluation']
    report={'schema':1,'created_at':now(),'dataset':data.get('name'), 'synthetic':data.get('synthetic',False),
            'manifest_sha256':hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'scorers':{'v1_complete_link':CLUSTER_VERSION,'example_bank':BANK_VERSION},
            'numpy_version':np.__version__,'target_precision':target_precision,
            'scope':'Cached frame scoring only; decode/model extraction measurements are separate',
            'split_counts':{'example':len(examples),'calibration':len(calibration),'evaluation':len(evaluation)},
            'preparation':data.get('preparation'), 'results':[], 'skipped':[]}
    for backend in ('features','clip','hybrid'):
        names=('clip','features') if backend=='hybrid' else (backend,)
        if any(any(name not in c['vectors'] for name in names) for c in data['clips']):
            report['skipped'].append({'backend':backend,'reason':'Missing cached embeddings; no substitute/fabricated model data'});continue
        for name in names:
            versions={c.get('versions',{}).get(name) for c in data['clips']}
            if len(versions)>1 or None in versions:
                raise ValueError(f'Mixed/missing model provenance for {name}')
        profile,calibration_report=calibrate(examples,calibration,backend,target_precision)
        for method in ('v1_complete_link','example_bank'):
            tracemalloc.start();started=time.perf_counter()
            if method=='example_bank':
                assignments, ranked=predictions(examples,evaluation,backend,profile)
                measured=metrics(evaluation,assignments,ranked)
            else:
                items=[]
                for c in examples+evaluation:
                    if backend=='hybrid':
                        vectors=np.concatenate([np.asarray(c['vectors']['clip'])*.75**.5,np.asarray(c['vectors']['features'])*.25**.5],axis=1)
                    else:vectors=c['vectors'][backend]
                    items.append({'id':c['id'],'embeddings':vectors,'metadata':{}})
                result,_=cluster(items,threshold={'features':.89,'clip':.88,'hybrid':.885}[backend])
                measured=metrics(evaluation,{r['clip_id']:r['group'] for r in result})
            elapsed=time.perf_counter()-started;_, peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            report['results'].append({'backend':backend,'method':method,**measured,'scoring_seconds':elapsed,
                'peak_python_allocated_bytes':peak,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'embedding_array_bytes':sum(np.asarray(c['vectors'][name],dtype=np.float32).nbytes for c in data['clips'] for name in names),
                'profile':asdict(profile) if method=='example_bank' else None,'calibration':calibration_report if method=='example_bank' else None})
    return report


def export_labels(service, backends=('clip','features'), prepare=False, seed=17):
    clips=service.db.rows('SELECT * FROM clips WHERE manual_group>1 AND demo=0 ORDER BY id')
    if not clips: raise ValueError('No manually labeled real clips. Assign recipe groups first.')
    rng=random.Random(seed); splits={}
    for group in sorted({c['manual_group'] for c in clips}):
        ids=[c['id'] for c in clips if c['manual_group']==group];rng.shuffle(ids)
        if len(ids)<5: raise ValueError('Label at least five clips per group for example/calibration/evaluation splits')
        n_examples=max(2,len(ids)//3);n_calibration=max(1,len(ids)//4)
        for i,cid in enumerate(ids):splits[cid]='example' if i<n_examples else 'calibration' if i<n_examples+n_calibration else 'evaluation'
    cache=EmbeddingCache(service.cache,service.db);started=time.perf_counter();result=[]
    for c in clips:
        if prepare:
            c=service.prepare_frames(c,'adaptive' if c['sampling_version']=='adaptive-v1' else 'uniform')
        item={"id":c['id'],'filename':c['filename'],'label':str(c['manual_group']),'split':splits[c['id']],
              'fingerprint':c['fingerprint'],'sampling_version':c['sampling_version'],'vectors':{},'versions':{}}
        for backend in backends:
            if prepare:
                v,version=cache.get(c,backend)
                item['vectors'][backend]=v.tolist();item['versions'][backend]=version
            else:
                rows=service.db.rows('SELECT * FROM embeddings WHERE clip_id=? AND backend=? AND fingerprint=?',(c['id'],backend,c['fingerprint']))
                frame_ids = [f['id'] for f in service.db.rows('SELECT id FROM frames WHERE clip_id=? ORDER BY seconds,id', (c['id'],))]
                if rows and rows[0]['sampling_version'] == c['sampling_version'] and rows[0]['model_version'] == cache.expected_version(backend) and json.loads(rows[0]['frame_ids']) == frame_ids:
                    vectors = valid_vectors(json.loads(rows[0]['vectors']))
                    if len(vectors) != len(frame_ids):
                        raise ValueError('Cached frame embeddings are incomplete; rebuild cached analysis')
                    item['vectors'][backend]=vectors.tolist();item['versions'][backend]=rows[0]['model_version']
        result.append(item)
    return {'schema':1,'name':'Manually labeled local recipe groups','synthetic':False,'seed':seed,'clips':result,
            'preparation':{'performed':prepare,'elapsed_seconds':time.perf_counter()-started,'cache':cache.stats,
                           'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           'scope':'Sparse frame preparation and embedding generation/reuse' if prepare else 'Read cached vectors only'}}


def synthetic_fixture():
    """Analytical vectors test stage diversity and counterexamples, NOT model quality."""
    from .embeddings import unit
    basis=np.eye(8)
    clips=[]
    def add(label,split,stage,clip_vectors,feature_vectors):
        clips.append({'id':len(clips)+1,'label':label,'split':split,'stage':stage,
                      'vectors':{'clip':unit(clip_vectors).tolist(),'features':unit(feature_vectors).tolist()},
                      'versions':{'clip':'synthetic-semantic-v1','features':'synthetic-appearance-v1'}})
    for group in range(2):
        a,b=basis[group*2],basis[group*2+1]
        for stage,v in [('mixing',a),('baked',b)]:
            add(str(group),'example',stage,[v,v],[v,v])
        for split in ('calibration','evaluation'):
            add(str(group),split,'mixing to baked',[a*.92+basis[4]*.2,b*.92+basis[5]*.2],[a,b])
            add(str(group),split,'baked',[b,b*.96+basis[6]*.1],[b,b])
            # Similar bowl belongs to the other recipe. CLIP-like hypothetical
            # descriptor favors the wrong recipe; visual corroboration corrects it.
            other=basis[(1-group)*2]
            lookalike=unit(a*.82+other*.90)
            add(str(group),split,'similar bowl, different recipe',[lookalike,lookalike],[a,a])
    for split in ('calibration','evaluation'):
        add('unseen',split,'unknown pan',[basis[7],basis[7]],[basis[7],basis[7]])
    return {'schema':1,'name':'Synthetic multistage and similar-bowl stress fixture','synthetic':True,'clips':clips}
