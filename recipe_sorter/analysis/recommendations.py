"""Directed frame-to-example-bank comparisons, independent of source decoders.
Thresholds are review triage defaults, not calibrated confidence probabilities.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .embeddings import unit

VERSION = 'example-bank-v1'


@dataclass(frozen=True)
class Profile:
    threshold: float
    margin: float
    frame_floor: float
    coverage: float = .6


PROFILES = {'clip': Profile(.78, .04, .74), 'features': Profile(.90, .035, .86),
            'hybrid': Profile(.80, .04, .76)}


def valid_vectors(values):
    v = np.asarray(values, dtype=np.float32)
    if v.ndim != 2 or len(v) < 2 or v.shape[1] < 2 or not np.isfinite(v).all() or (not np.isfinite(np.linalg.norm(v,axis=1)).all() or np.any(np.linalg.norm(v, axis=1) < 1e-8)):
        raise ValueError('Invalid frame embeddings: need at least two finite nonzero vectors')
    return unit(v)


def bank_score(candidate, examples, backend):
    """Every candidate frame can match a different stage/example; no all-pairs gate."""
    frames = valid_vectors(candidate['vectors'][backend])
    banks = [valid_vectors(e['vectors'][backend]) for e in examples]
    if not banks or any(b.shape[1] != frames.shape[1] for b in banks):
        raise ValueError('Embedding dimensions/provenance do not match')
    bank = np.concatenate(banks)
    similarities = np.clip(frames @ bank.T, -1, 1)
    nearest = similarities.max(axis=1)
    score = float(.8 * nearest.mean() + .2 * np.quantile(nearest, .25))
    best = similarities.argmax(axis=1)
    offsets = np.cumsum([0] + [len(b) for b in banks])
    supports = []
    for j, example in enumerate(examples):
        i, k = np.unravel_index(np.argmax(similarities[:, offsets[j]:offsets[j+1]]), (len(frames), len(banks[j])))
        supports.append({'clip_id': example['id'], 'candidate_frame': int(i), 'example_frame': int(k),
                         'similarity': float(similarities[i, offsets[j] + k]),
                         'matched_frames': int(np.sum((best >= offsets[j]) & (best < offsets[j+1])))})
    return score, nearest, sorted(supports, key=lambda x: (-x['matched_frames'], -x['similarity'], x['clip_id']))[:3]


def group_score(candidate, examples, backend):
    if backend != 'hybrid':
        score, nearest, support = bank_score(candidate, examples, backend)
        return score, nearest, support, {backend: score}
    clip, cframes, support = bank_score(candidate, examples, 'clip')
    visual, vframes, _ = bank_score(candidate, examples, 'features')
    if len(cframes) != len(vframes):
        raise ValueError('Hybrid scoring requires identical sampled frame positions')
    return .75 * clip + .25 * visual, .75 * cframes + .25 * vframes, support, {'clip': clip, 'features': visual}


def rank_candidates(groups, candidates, backend='clip', profiles=None):
    profile = (profiles or PROFILES)[backend]
    result = []
    for candidate in candidates:
        ranked = []
        for group, examples in sorted(groups.items()):
            examples = [e for e in examples if e['id'] != candidate['id']]
            if examples:
                score, nearest, support, components = group_score(candidate, examples, backend)
                ranked.append((score, group, nearest, support, components, len(examples)))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        for score, group, nearest, support, components, example_count in ranked:
            others = [r for r in ranked if r[1] != group]
            rival_score = others[0][0] if others else None
            separation = score - rival_score if rival_score is not None else None
            coverage = float(np.mean(nearest >= profile.frame_floor))
            adequate = score >= profile.threshold and coverage >= profile.coverage and (separation is None or separation >= profile.margin)
            if backend == 'hybrid' and components['clip'] < PROFILES['clip'].frame_floor:
                adequate = False
            reasons = []
            if backend == 'hybrid' and components['clip'] < PROFILES['clip'].frame_floor:
                reasons.append('Visual appearance is not supported by enough CLIP similarity')
            if score < profile.threshold: reasons.append('Below the model-specific review threshold')
            if coverage < profile.coverage: reasons.append('Too few sampled frames have supporting examples')
            if separation is not None and separation < profile.margin: reasons.append('Another recipe group has similar or stronger evidence')
            if not others: reasons.append('No competing group examples are available yet')
            if example_count == 1: reasons.append('Only one example clip; add other recipe stages')
            if not reasons: reasons.append('Several sampled frames match your examples with a clear lead over other groups')
            result.append({'clip_id': candidate['id'], 'group_id': group, 'score': score,
                           'evidence': {'label': 'suggested' if adequate else 'ambiguous', 'reasons': reasons,
                            'support': support, 'frame_coverage': coverage, 'frame_count': len(nearest),
                            'supported_frames': int(np.sum(nearest >= profile.frame_floor)), 'example_count': example_count,
                            'rival_group': others[0][1] if others else None, 'rival_score': rival_score, 'margin': separation,
                            'components': components, 'profile': asdict(profile), 'version': VERSION}})
    return sorted(result, key=lambda r: (-r['score'], r['clip_id'], r['group_id']))
