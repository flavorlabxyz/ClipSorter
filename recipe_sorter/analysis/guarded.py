"""Experimental average linkage with diameter and per-member support guards."""
import numpy as np
from .similarity import pair_score

VERSION = 'guarded-average-v1'


def guarded_cluster(clips, threshold=.88, margin=.045, floor=None):
    n=len(clips); scores=np.eye(n); visual=np.eye(n);pairs=[]
    floor = threshold-.14 if floor is None else floor
    for i in range(n):
        for j in range(i+1,n):
            score,evidence=pair_score(clips[i],clips[j])
            scores[i,j]=scores[j,i]=score;visual[i,j]=visual[j,i]=evidence['visual']
            pairs.append((clips[i]['id'],clips[j]['id'],score,evidence))
    import heapq
    means=np.zeros((max(1,2*n),max(1,2*n)),dtype=np.float32)
    minima=np.zeros_like(means);means[:n,:n]=visual;minima[:n,:n]=visual
    active={i:(i,) for i in range(n)};heap=[];serial=n
    for i in range(n):
        for j in range(i+1,n):
            if visual[i,j]>=threshold:heapq.heappush(heap,(-float(visual[i,j]),i,j))
    # Weighted updates avoid rescanning all candidate merges on every iteration.
    # Every non-singleton already has a strong companion; a singleton merging
    # at mean >= threshold necessarily has at least one strong cross neighbor.
    while heap:
        negative,a,b=heapq.heappop(heap)
        if a not in active or b not in active:continue
        if means[a,b]<threshold or minima[a,b]<floor:continue
        left,right=active.pop(a),active.pop(b);merged=tuple(sorted(left+right))
        for k,members in active.items():
            mean=(float(means[a,k])*len(left)+float(means[b,k])*len(right))/len(merged)
            low=min(minima[a,k],minima[b,k])
            means[serial,k]=means[k,serial]=mean;minima[serial,k]=minima[k,serial]=low
            if mean>=threshold and low>=floor:heapq.heappush(heap,(-mean,k,serial))
        active[serial]=merged;serial+=1
    groups=[g for _,g in sorted(active.items()) if len(g)>1]
    results=[]
    for i,clip in enumerate(clips):
        ranked=[]
        for gid,members in enumerate(groups):
            peers=[j for j in members if j!=i]
            if peers:ranked.append((float(scores[i,peers].mean()),gid,float(visual[i,peers].min()),float(visual[i,peers].max())))
        ranked.sort(reverse=True);best=ranked[0] if ranked else (0,None,0,0);second=ranked[1] if len(ranked)>1 else (0,None,0,0)
        accepted=best[0]>=threshold-.025 and best[2]>=floor and best[3]>=threshold and best[0]-second[0]>=margin
        reason='grouped' if accepted else 'no_evidence' if not n>1 else 'competing_groups' if second[1] is not None and best[0]-second[0]<margin else 'guarded_visual_support'
        results.append({'clip_id':clip['id'],'group':best[1] if accepted else None,'confidence':best[0],
            'evidence':{'best_candidate':best[1],'second_candidate':second[1],'second_score':second[0],'margin':best[0]-second[0],
                'visual_min':best[2],'threshold':threshold,'floor':floor,'reason_code':reason,
                'reason':'Guarded average agreement across recipe stages' if accepted else 'Withheld: '+reason.replace('_',' '),'version':VERSION}})
    counts={g:sum(r['group']==g for r in results) for g in range(len(groups))}
    for r in results:
        if r['group'] is not None and counts[r['group']]<2:
            r['group']=None;r['evidence'].update(reason_code='singleton',reason='No confident companion remains')
    return results,pairs
