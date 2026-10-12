"""Evidence providers can be extended without changing decoders or the UI."""
from datetime import datetime
from typing import Protocol
import numpy as np
from .embeddings import unit


class EvidenceProvider(Protocol):
    def __call__(self, left: dict, right: dict) -> dict[str, float]: ...


def recording_evidence(left, right):
    a, b = left.get("metadata", {}), right.get("metadata", {})
    time = 0.0
    try:
        ta, tb = [datetime.fromisoformat((m.get("recorded_at") or "").replace("Z", "+00:00")) for m in (a, b)]
        time = float(np.exp(-abs((ta - tb).total_seconds()) / 1800))
    except (ValueError, KeyError, TypeError):
        pass
    return {"recording_proximity": time, "same_camera": float(bool(a.get("camera")) and a.get("camera") == b.get("camera")),
            "same_dimensions": float(bool(a.get("width")) and (a.get("width"), a.get("height")) == (b.get("width"), b.get("height")))}


def pair_score(left, right, providers=(recording_evidence,)):
    a, b = unit(left["embeddings"]), unit(right["embeddings"])
    matches = a @ b.T
    coverage = float((matches.max(axis=0).mean() + matches.max(axis=1).mean()) / 2)
    center = float(unit(a.mean(axis=0)) @ unit(b.mean(axis=0)))
    visual = max(0., .65 * coverage + .35 * center)
    evidence = {"multi_frame": coverage, "centroid": center, "visual": visual}
    for provider in providers:
        evidence.update(provider(left, right))
    # Metadata only adds a small bonus; it cannot bypass the visual gate.
    score = min(1., visual + .015 * evidence.get("recording_proximity", 0) + .005 * evidence.get("same_camera", 0) + .003 * evidence.get("same_dimensions", 0))
    return score, evidence
