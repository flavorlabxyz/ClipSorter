"""Conservative complete-link thread proposals with explicit abstention."""
import numpy as np
from .similarity import pair_score

VERSION = "threads-v3"


def cluster(clips, threshold=.89, margin=.045):
    n = len(clips)
    scores, visual = np.eye(n), np.eye(n)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score, evidence = pair_score(clips[i], clips[j])
            scores[i, j] = scores[j, i] = score
            visual[i, j] = visual[j, i] = evidence["visual"]
            pairs.append((clips[i]["id"], clips[j]["id"], score, evidence))
    # Require all members to resemble one another; avoid single-link chains.
    groups = []
    remaining = set(range(n))
    edges = sorted(((scores[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                    if visual[i, j] >= threshold), reverse=True)
    for _, i, j in edges:
        if i not in remaining or j not in remaining:
            continue
        group = [i, j]
        remaining.difference_update(group)
        for k in sorted(remaining, key=lambda k: (-min(scores[k, g] for g in group), k)):
            if min(visual[k, g] for g in group) >= threshold:
                group.append(k)
                remaining.remove(k)
        groups.append(group)
    results = []
    for i, clip in enumerate(clips):
        candidates = []
        for index, members in enumerate(groups):
            peers = [j for j in members if j != i]
            if peers:
                candidates.append((float(np.mean(scores[i, peers])), index, float(np.min(visual[i, peers]))))
        candidates.sort(reverse=True)
        best = candidates[0] if candidates else (0., None, 0.)
        second = candidates[1] if len(candidates) > 1 else (0., None, 0.)
        accepted = best[2] >= threshold and best[0] - second[0] >= margin
        results.append({"clip_id": clip["id"], "group": best[1] if accepted else None, "confidence": best[0],
                        "evidence": {"best_candidate": best[1], "second_candidate": second[1], "second_score": second[0],
                                     "margin": best[0] - second[0], "visual_min": best[2], "threshold": threshold,
                                     "reason": "Consistent visual evidence across sampled frames" if accepted else "Insufficient visual agreement or competing recipe threads"}})
    # Do not leave a confident singleton after ambiguous members abstain.
    counts = {g: sum(r["group"] == g for r in results) for g in range(len(groups))}
    for result in results:
        if result["group"] is not None and counts[result["group"]] < 2:
            result["group"] = None
            result["evidence"]["reason"] = "No confident companion clip remains"
    return results, pairs
