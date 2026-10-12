"""V1.6 comparisons: independent truth, held-out candidates, explicit denominators."""
import hashlib
import json
import resource
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, replace
import numpy as np
from .evaluation import metrics, validate_manifest, ratio
from .clustering import cluster
from .guarded import guarded_cluster
from .recommendations import rank_candidates, PROFILES, valid_vectors
from .prototypes import rank_prototypes
from ..storage.database import now


def extended_metrics(items, assignments, ranked=None):
    result=metrics(items,assignments,ranked)
    predicted=defaultdict(list);truth=defaultdict(set)
    for c in items:
        group=assignments.get(c['id'])
        if group is not None:predicted[group].append(c);truth[c['label']].add(group)
    contaminated=sum(len(cs) for cs in predicted.values() if len({c['label'] for c in cs})>1)
    result.update(assignment_coverage=ratio(result['assigned_clips'],len(items)),
        cross_recipe_contamination=ratio(contaminated,result['assigned_clips']),
        contaminated_assigned_clips=contaminated,
        recipe_fragmentation=ratio(sum(len(g) for g in truth.values()),len(truth)),
        fragmentation_by_recipe={label:len(truth[label]) for label in sorted({c['label'] for c in items})},
        fully_abstained_recipes=sum(not truth[label] for label in {c['label'] for c in items}))
    return result


def auto_predict(examples, candidates, backend, method, threshold):
    inputs=[]
    for c in examples+candidates:
        if backend=='hybrid':
            # Preserve V1.5's raw-scale hybrid strictly as a historical control.
            vectors=np.concatenate([valid_vectors(c['vectors']['clip'])*.75**.5,valid_vectors(c['vectors']['features'])*.25**.5],axis=1)
        else:vectors=valid_vectors(c['vectors'][backend])
        inputs.append({'id':c['id'],'embeddings':vectors,'metadata':{}})
    rows,_=(cluster if method=='complete' else guarded_cluster)(inputs,threshold=threshold)
    return {r['clip_id']:r['group'] for r in rows}


def guided_predict(examples,candidates,backend,method,profile):
    labels=sorted({c['label'] for c in examples});names={i+2:label for i,label in enumerate(labels)}
    groups={gid:[c for c in examples if c['label']==label] for gid,label in names.items()}
    rows=(rank_candidates if method=='bank' else rank_prototypes)(groups,candidates,backend,{backend:profile})
    best={}
    for row in rows:best.setdefault(row['clip_id'],row)
    assignments={cid:names[row['group_id']] if row['evidence']['label']=='suggested' else None for cid,row in best.items()}
    ranked={cid:names[row['group_id']] for cid,row in best.items()}
    truth={c['id']:c['label'] for c in candidates}
    top10=[]
    for group,label in names.items():
        queue=[row for row in rows if row['group_id']==group][:10]
        correct=sum(truth[row['clip_id']]==label for row in queue)
        top10.append({'recipe':label,'correct':correct,'reviewed':len(queue),'precision':ratio(correct,len(queue))})
    return assignments,ranked,{'per_recipe':top10,'correct':sum(r['correct'] for r in top10),
        'reviewed':sum(r['reviewed'] for r in top10),'precision':ratio(sum(r['correct'] for r in top10),sum(r['reviewed'] for r in top10))}


def benchmark_v16(data,target_precision=.95):
    validate_manifest(data)
    examples=[c for c in data['clips'] if c['split']=='example'];cal=[c for c in data['clips'] if c['split']=='calibration'];evaluation=[c for c in data['clips'] if c['split']=='evaluation']
    warnings=list(data.get('warnings',[]));sessions=defaultdict(set)
    for c in data['clips']:
        if c.get('session'):sessions[c['session']].add(c['split'])
    if not sessions:warnings.append('No filming-session labels; independence of visual content across splits is unverified.')
    if any(len(s)>1 for s in sessions.values()):warnings.append('Some filming sessions span splits; results may overestimate generalization.')
    report={'schema':2,'created_at':now(),'dataset':data.get('name'),'synthetic':data.get('synthetic',False),
        'manifest_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),
        'split_counts':{'example':len(examples),'calibration':len(cal),'evaluation':len(evaluation)},'warnings':warnings,
        'target_calibration_precision':target_precision,'scope':'Frozen frame embeddings; automatic proposals ignore all active/manual assignments. Guided seeds are only the example split.',
        'memory_scope':'tracemalloc excludes some native allocations; RSS is cumulative process high-water mark in KiB. Decode/model preparation is separate.',
        'preparation':data.get('preparation'),'results':[],'skipped':[]}
    for backend in ('features','clip','hybrid'):
        names=('clip','features') if backend=='hybrid' else (backend,)
        if any(name not in c.get('vectors',{}) for c in data['clips'] for name in names):
            report['skipped'].append({'backend':backend,'reason':'Missing embeddings; no substitution'});continue
        for name in names:
            versions={c.get('versions',{}).get(name) for c in data['clips']}
            if None in versions or len(versions)!=1:raise ValueError('Mixed or missing model provenance: '+name)
            for c in data['clips']:valid_vectors(c['vectors'][name])
        default={'features':.89,'clip':.88,'hybrid':.885}[backend]
        for method in ('complete','guarded','bank','prototypes'):
            automatic=method in ('complete','guarded')
            # Report default and calibrated operating points separately.
            for operating_point in ('default','calibrated'):
                threshold=default;profile=PROFILES[backend];calibration=None
                if operating_point=='calibrated' and cal:
                    choices=[]
                    if automatic:
                        for gate in sorted(set([default,.80,.84,.88,.92,.96,.99])):
                            m=extended_metrics(cal,auto_predict(examples,cal,backend,method,gate))
                            if m['pairwise_precision'] is not None and m['pairwise_precision']>=target_precision:choices.append((m['assigned_clips'],m['pairwise_precision'],gate,m))
                        if choices:
                            selected=max(choices,key=lambda c:c[:3]);threshold=selected[2];calibration=selected[3]
                        else:threshold=1.01;calibration={'reason':'No operating point met calibration pair precision; abstaining'}
                    else:
                        for gate in sorted(set([profile.threshold,.72,.80,.88,.94,.98])):
                            for gap in (.03,.06,.12):
                                trial=replace(profile,threshold=gate,margin=gap);a,r,_=guided_predict(examples,cal,backend,method,trial);m=extended_metrics(cal,a,r)
                                if m['suggested_match_accuracy'] is not None and m['suggested_match_accuracy']>=target_precision:choices.append((m['assigned_clips'],m['suggested_match_accuracy'],gate,gap,trial,m))
                        if choices:
                            selected=max(choices,key=lambda c:c[:4]);profile=selected[4];calibration=selected[5]
                        else:profile=replace(profile,threshold=1.01);calibration={'reason':'No operating point met calibration named-group precision; abstaining'}
                elif operating_point=='calibrated':calibration={'reason':'No calibration clips; default retained'}
                tracemalloc.start();started=time.perf_counter()
                if automatic:
                    assignments=auto_predict(examples,evaluation,backend,method,threshold);measured=extended_metrics(evaluation,assignments);top10=None
                else:
                    assignments,ranked,top10=guided_predict(examples,evaluation,backend,method,profile);measured=extended_metrics(evaluation,assignments,ranked)
                elapsed=time.perf_counter()-started;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                report['results'].append({'backend':backend,'method':method,'operating_point':operating_point,**measured,
                    'recommendation_quality_at_10':top10,'threshold':threshold if automatic else None,'review_profile':None if automatic else asdict(profile),
                    'calibration':calibration,'scoring_seconds':elapsed,'peak_python_allocated_bytes':peak,
                    'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'embedding_array_bytes':sum(np.asarray(c['vectors'][name],dtype=np.float32).nbytes for c in data['clips'] for name in names)})
    return report


def stage_fixture():
    """Analytical stage variation and competing lookalikes; never actual CLIP data."""
    clips=[]
    def vectors(angle):
        return [[float(np.cos(np.deg2rad(angle+d))),float(np.sin(np.deg2rad(angle+d))),0.] for d in (-5,5)]
    for recipe,base in [('cake',0),('rolls',180)]:
        for split,stages in [('example',(0,36)),('calibration',(0,18,36)),('evaluation',(0,18,36))]:
            for angle in stages:
                clips.append({'id':len(clips)+1,'label':recipe,'split':split,'session':split,'stage':f'stage {angle}',
                    'vectors':{'clip':vectors(base+angle),'features':vectors(base+angle)},
                    'versions':{'clip':'analytical-stage-v1','features':'analytical-stage-v1'}})
    # Competing visually similar scene between threads has weak/ambiguous support.
    for split in ('calibration','evaluation'):
        clips.append({'id':len(clips)+1,'label':'unknown-pan','split':split,'session':split,'stage':'unseen lookalike',
            'vectors':{'clip':vectors(90),'features':vectors(90)},'versions':{'clip':'analytical-stage-v1','features':'analytical-stage-v1'}})
    return {'schema':1,'name':'Analytical multi-stage threads and unseen pan','synthetic':True,'clips':clips}
