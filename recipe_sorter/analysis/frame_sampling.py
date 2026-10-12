import numpy as np

FRACTIONS = (.1, .3, .5, .7, .9)


def quality(image):
    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
    dy, dx = np.gradient(gray)
    sharpness = float(np.mean(dx * dx + dy * dy))
    black = float(np.mean(gray < 12))
    return sharpness * (1 - black), black


def sample_frames(decoder, source, metadata):
    result = []
    failures = []
    for fraction in FRACTIONS:
        candidates = []
        for offset in (0, .035, -.035):
            seconds = min(max(0, metadata.duration * (fraction + offset)), max(0, metadata.duration - 1 / metadata.fps))
            try:
                image = decoder.frame(source, seconds)
                score, black = quality(image)
                if black < .96:
                    candidates.append((score, seconds, image))
                if score >= 8 and black < .9:
                    break
            except Exception as exc:
                failures.append(str(exc))
        if candidates:
            best = max(candidates, key=lambda c: c[0])
            if best[0] >= .5:
                result.append(best)
    if len(result) < 2:
        raise RuntimeError("Fewer than two usable representative frames; clip needs manual review. " + (failures[-1] if failures else "Frames were black or severely blurred."))
    return result


def adaptive_frames(decoder, source, metadata):
    """Bounded experiment: five base frames + up to four visual-change midpoints.
    Uses low-resolution color/layout descriptors, never a full-frame scan.
    """
    from .embeddings import VisualFeatures
    samples = sample_frames(decoder, source, metadata)
    vectors = VisualFeatures().encode([s[2] for s in samples])
    for i in range(len(samples) - 1):
        if float(vectors[i] @ vectors[i+1]) < .94:
            seconds = (samples[i][1] + samples[i+1][1]) / 2
            try:
                image = decoder.frame(source, seconds)
                sharpness, black = quality(image)
                if sharpness >= .5 and black < .96:
                    samples.append((sharpness, seconds, image))
            except Exception:
                pass  # Keep the already usable base samples.
    return sorted(samples, key=lambda s: s[1])


def diverse_frames(decoder, source, metadata):
    """Reuse the bounded adaptive pool; keep temporal anchors plus diverse views."""
    from .embeddings import VisualFeatures
    pool=adaptive_frames(decoder,source,metadata)
    if len(pool)<=5:return pool
    vectors=VisualFeatures().encode([s[2] for s in pool])
    selected=sorted({0,len(pool)//2,len(pool)-1})
    while len(selected)<min(7,len(pool)):
        remaining=[i for i in range(len(pool)) if i not in selected]
        choice=min(remaining,key=lambda i:(float(max(vectors[i]@vectors[selected].T)),i))
        if len(selected)>=5 and max(vectors[choice]@vectors[selected].T)>=.995:break
        selected.append(choice)
    return [pool[i] for i in sorted(selected)]
