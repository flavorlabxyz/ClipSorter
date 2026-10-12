"""Experimental balanced, diverse example prototypes; evidence stays frame-level."""
from dataclasses import asdict
import numpy as np
from .recommendations import valid_vectors, PROFILES

VERSION='balanced-prototypes-v1'


def diverse_indices(vectors, limit=8, duplicate=.995):
    """Deterministic farthest-point prototypes, with near-duplicate suppression."""
    v=valid_vectors(vectors)
    # Lexical order makes multiplying examples unable to change tie breaking.
    order=sorted(range(len(v)),key=lambda i:tuple(v[i]))
    unique=[]
    for i in order:
        if not unique or max(v[i]@v[unique].T)<duplicate:unique.append(i)
    first=min(unique,key=lambda i:tuple(v[i]))
    selected=[first]
    while len(selected)<min(limit,len(unique)):
        remaining=[i for i in unique if i not in selected]
        choice=min(remaining,key=lambda i:(float(max(v[i]@v[selected].T)),tuple(v[i])))
        selected.append(choice)
    return selected


def rank_prototypes(groups, candidates, backend='clip', profiles=None):
    profile=(profiles or PROFILES)[backend]
    primary='clip' if backend=='hybrid' else backend
    banks={}
    for gid,examples in sorted(groups.items()):
        vectors=[];owners=[]
        for e in sorted(examples,key=lambda x:x['id']):
            for index,v in enumerate(valid_vectors(e['vectors'][primary])):
                vectors.append(v);owners.append((e,index))
        if len(vectors)>=2:
            matrix=np.asarray(vectors);indices=diverse_indices(matrix)
            banks[gid]=(matrix,owners,indices)
    if not banks:return []
    budget=min(8,min(len(b[2]) for b in banks.values()))
    results=[]
    for c in candidates:
        frames=valid_vectors(c['vectors'][primary]);distinct=diverse_indices(frames,limit=len(frames))
        comparisons={}
        for gid,(matrix,owners,indices) in banks.items():
            indices=indices[:budget]
            sims=np.clip(frames@matrix[indices].T,-1,1);nearest=sims.max(axis=1)
            informative=nearest[distinct]
            score=float(.8*informative.mean()+.2*np.quantile(informative,.25))
            comparisons[gid]=(score,nearest,sims,indices,owners)
        ordered=sorted(comparisons,key=lambda g:(-comparisons[g][0],g))
        # Hybrid uses a baseline rank veto, not a weighted sum of incomparable scales.
        visual_scores={}
        if backend=='hybrid':
            vf=valid_vectors(c['vectors']['features'])
            for gid,(_,_,_,indices,owners) in comparisons.items():
                vv=np.asarray([valid_vectors(owners[i][0]['vectors']['features'])[owners[i][1]] for i in indices])
                visual_scores[gid]=float((vf@vv.T).max(axis=1)[distinct].mean())
        for gid in ordered:
            score,nearest,sims,indices,owners=comparisons[gid]
            rivals=[g for g in ordered if g!=gid];rival=rivals[0] if rivals else None
            margin=score-comparisons[rival][0] if rival is not None else None
            support_mask=nearest>=profile.frame_floor
            if rivals:
                rival_frames=np.max([comparisons[g][1] for g in rivals],axis=0)
                support_mask &= nearest-rival_frames>=.02
            coverage=float(support_mask[distinct].mean())
            supported_prototypes=set(sims[distinct].argmax(axis=1)[support_mask[distinct]].tolist())
            reasons=[]
            if score<profile.threshold:reasons.append('Below the model-specific review threshold')
            if coverage<profile.coverage:reasons.append('Insufficient distinct frames favor this group over its competitors')
            if margin is not None and margin<profile.margin:reasons.append('Two plausible groups or a stronger competing group')
            if len(supported_prototypes)<2:reasons.append('Only one distinct example view supports the match; shared background may dominate')
            veto=backend=='hybrid' and any(v>visual_scores[gid]+.04 for g,v in visual_scores.items() if g!=gid)
            if veto:reasons.append('Visual-feature ranking disagrees with the CLIP candidate')
            adequate=not reasons
            if not rivals:reasons.append('No competing group examples are available; separation could not be checked')
            if adequate:reasons.append('Distinct frames match diverse examples; no single shared view determines the result')
            support=[]
            for k,index in enumerate(indices):
                e,frame_index=owners[index];best=int(sims[:,k].argmax())
                support.append({'clip_id':e['id'],'candidate_frame':best,'example_frame':frame_index,
                    'similarity':float(sims[best,k]),'matched_frames':int(np.sum(sims.argmax(axis=1)==k))})
            support=sorted(support,key=lambda r:(-r['matched_frames'],-r['similarity'],r['clip_id']))[:3]
            results.append({'clip_id':c['id'],'group_id':gid,'score':score,'evidence':{
                'label':'suggested' if adequate else 'ambiguous','reasons':reasons,'support':support,
                'frame_coverage':coverage,'frame_count':len(frames),'distinct_frames':len(distinct),
                'supported_frames':int(support_mask.sum()),'example_count':len(groups[gid]),'prototype_budget':budget,
                'rival_group':rival,'rival_score':comparisons[rival][0] if rival is not None else None,'margin':margin,
                'components':{primary:score,**({'visual_rank_score':visual_scores[gid]} if visual_scores else {})},
                'profile':asdict(profile),'version':VERSION}})
    return sorted(results,key=lambda r:(-r['score'],r['clip_id'],r['group_id']))
