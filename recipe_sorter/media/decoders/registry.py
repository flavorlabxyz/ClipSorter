from .braw import BRAWDecoder
from .ffmpeg import FFmpegDecoder
from ..discovery import EXTENSIONS


def decoder_for(path, cache):
    if path.suffix.lower() == ".braw":
        return BRAWDecoder(cache)
    if path.suffix.lower() in EXTENSIONS:
        return FFmpegDecoder(cache)
    raise ValueError(f"Unsupported media extension: {path.suffix}")
