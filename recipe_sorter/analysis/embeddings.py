"""Local embedding providers. Never silently substitute models."""
from abc import ABC, abstractmethod
import hashlib
import json
import os
import numpy as np
from PIL import Image

MODEL_ID = "openai/clip-vit-base-patch32"


def unit(array):
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-9)


class Embedder(ABC):
    version: str
    @abstractmethod
    def encode(self, images: list[Image.Image]) -> np.ndarray: ...


class VisualFeatures(Embedder):
    """Offline color/layout/texture baseline, explicitly not a learned AI model."""
    version = "visual-features-v2"

    def encode(self, images):
        rows = []
        for image in images:
            small = image.convert("RGB").resize((64, 64))
            hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255
            hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(12, 4, 4), range=((0, 1), (0, 1), (0, 1)))
            hist = unit(np.sqrt(hist.ravel()))
            spatial = np.asarray(small.resize((8, 8)), dtype=np.float32).ravel() / 255
            gray = np.asarray(small.convert("L"), dtype=np.float32) / 255
            dy, dx = np.gradient(gray)
            edge, _ = np.histogram(np.arctan2(dy, dx), bins=16, range=(-np.pi, np.pi), weights=np.hypot(dx, dy))
            rows.append(np.concatenate([hist * .85, unit(spatial) * .35, unit(edge) * .2]))
        return unit(rows)


class ClipEmbedder(Embedder):
    def __init__(self, cache):
        configure_model_cache(cache)
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Local CLIP requires: pip install -e '.[ai]'; then recipe-sorter prepare-model") from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dir = cache.path("models/clip-vit-base-patch32")
        if not (model_dir / "config.json").is_file():
            raise RuntimeError("CLIP weights are not installed. Run recipe-sorter prepare-model once; analysis is offline.")
        self.processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True).to(self.device).eval()
        self.version = clip_version(cache)

    def encode(self, images):
        tensors = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            return unit(self.model.get_image_features(**tensors).cpu().numpy())


def create_embedder(name, cache):
    if name == "clip":
        return ClipEmbedder(cache)
    if name == "features":
        return VisualFeatures()
    raise ValueError("Unknown embedding backend")


def configure_model_cache(cache):
    os.environ.update({k: v for k, v in cache.environment().items() if k in ("HF_HOME", "TMPDIR", "XDG_CACHE_HOME")})
    os.environ["TORCH_HOME"] = str(cache.path("models/torch"))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache.path("models/torchinductor"))


def prepare_model(cache):
    configure_model_cache(cache)
    from transformers import CLIPModel, CLIPProcessor
    destination = cache.output("models/clip-vit-base-patch32/config.json").parent
    for cls in (CLIPModel, CLIPProcessor):
        cls.from_pretrained(MODEL_ID, cache_dir=str(cache.path("models/downloads"))).save_pretrained(destination)


def clip_version(cache):
    model_dir = cache.path("models/clip-vit-base-patch32")
    if not (model_dir / "config.json").is_file():
        return None
    signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(model_dir.iterdir()) if p.is_file()]
    return "clip-vit-b32-v1-" + hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
