import io
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from PIL import Image
from .base import MediaDecoder, Metadata, DecoderUnavailable


def number(value, default=0.0):
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError, TypeError):
        return default


def normalize_probe(data: dict) -> Metadata:
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not stream:
        raise ValueError("No video stream found")
    fmt = data.get("format", {})
    tags = {**fmt.get("tags", {}), **stream.get("tags", {})}
    return Metadata(duration=number(stream.get("duration")) or number(fmt.get("duration")),
                    fps=number(stream.get("avg_frame_rate")) or number(stream.get("r_frame_rate")),
                    width=int(stream.get("width", 0)), height=int(stream.get("height", 0)),
                    codec=stream.get("codec_name", "unknown"), recorded_at=tags.get("creation_time"),
                    camera=tags.get("com.apple.quicktime.model") or tags.get("model"), extra=data)


class FFmpegDecoder(MediaDecoder):
    name = "ffmpeg"

    def __init__(self, cache):
        self.cache = cache

    def run(self, args, timeout=180):
        if not shutil.which(args[0]):
            raise DecoderUnavailable(f"{args[0]} is missing. See README Fedora FFmpeg setup.")
        result = subprocess.run(args, capture_output=True, timeout=timeout, cwd=self.cache.root, env=self.cache.environment())
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace")[-2500:])
        return result.stdout

    def probe(self, source):
        self.cache.check_source(source)
        data = self.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(source)])
        return normalize_probe(json.loads(data))

    def frame(self, source, seconds, max_size=768):
        self.cache.check_source(source)
        data = self.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(max(0, seconds)), "-i", str(source),
                         "-map", "0:v:0", "-frames:v", "1", "-vf",
                         f"scale={max_size}:{max_size}:force_original_aspect_ratio=decrease",
                         "-threads", "2", "-f", "image2pipe", "-vcodec", "png", "pipe:1"])
        with Image.open(io.BytesIO(data)) as im:
            return im.convert("RGB")

    def preview(self, source, relative_output, metadata):
        self.cache.check_source(source)
        out = self.cache.output(relative_output)
        # Silent, complete, reduced-resolution proxy. Original is only an input.
        self.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-an",
                  "-vf", "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2,fps=12",
                  "-c:v", "libvpx-vp9", "-b:v", "550k", "-deadline", "realtime", "-cpu-used", "6",
                  "-threads", "2", "-f", "webm", str(out)], timeout=7200)
        return out
