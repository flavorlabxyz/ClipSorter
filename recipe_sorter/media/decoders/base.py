from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from PIL import Image


class DecoderUnavailable(RuntimeError):
    pass


@dataclass
class Metadata:
    duration: float
    fps: float
    width: int
    height: int
    codec: str
    recorded_at: str | None = None
    camera: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        import math
        if not math.isfinite(self.duration) or not math.isfinite(self.fps) or self.duration <= 0 or self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Clip has invalid or missing duration, frame rate, or dimensions")

    def dict(self):
        return asdict(self)


class MediaDecoder(ABC):
    name: str

    @abstractmethod
    def probe(self, source: Path) -> Metadata: ...

    @abstractmethod
    def frame(self, source: Path, seconds: float, max_size: int = 768) -> Image.Image: ...

    @abstractmethod
    def preview(self, source: Path, relative_output: str, metadata: Metadata) -> Path: ...
