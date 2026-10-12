"""Official SDK bridge; never routes BRAW through FFmpeg for decoding."""
import io
import json
import os
import shutil
import subprocess
from PIL import Image
from .base import MediaDecoder, Metadata, DecoderUnavailable


class BRAWDecoder(MediaDecoder):
    name = "braw-sdk"

    def __init__(self, cache):
        self.cache = cache
        self.bridge = os.environ.get("RCS_BRAW_BRIDGE") or shutil.which("recipe-braw")

    def run(self, *args, timeout=180):
        if not self.bridge:
            raise DecoderUnavailable("BRAW SDK unavailable: install the official Blackmagic RAW Linux SDK, build native/recipe-braw, and set RCS_BRAW_BRIDGE. See README.")
        try:
            p = subprocess.run([self.bridge, *map(str, args)], capture_output=True, timeout=timeout,
                               env=self.cache.environment(), cwd=self.cache.root)
        except OSError as exc:
            raise DecoderUnavailable(f"BRAW bridge could not start: {exc}") from exc
        if p.returncode:
            message = p.stderr.decode(errors="replace")[-2500:]
            if p.returncode == 78:
                raise DecoderUnavailable(message)
            raise RuntimeError(f"Official BRAW SDK: {message}")
        return p.stdout

    def diagnostic(self):
        try:
            return {"available": True, **json.loads(self.run("diagnostic", timeout=15)),
                    "validation": "Requires real BRAW acceptance test on Fedora"}
        except Exception as exc:
            return {"available": False, "detail": str(exc)}

    def probe(self, source):
        self.cache.check_source(source)
        return Metadata(**json.loads(self.run("probe", source)))

    def frame(self, source, seconds, max_size=768):
        self.cache.check_source(source)
        data = self.run("frame", source, max(0, seconds), max_size)
        with Image.open(io.BytesIO(data)) as image:
            result = image.convert("RGB")
        result.thumbnail((max_size, max_size))
        return result

    def preview(self, source, relative_output, metadata):
        self.cache.check_source(source)
        if not shutil.which("ffmpeg"):
            raise DecoderUnavailable("FFmpeg is required to encode SDK-decoded preview frames to WebM")
        out = self.cache.output(relative_output)
        error_file = self.cache.output(relative_output + ".encoder.log")
        bridge_errors = self.cache.output(relative_output + ".sdk.log")
        # One SDK process opens the clip once, sparsely decodes at 6fps to a PPM pipe.
        # FFmpeg only encodes those RGB images; it never opens the .braw file.
        with error_file.open("wb") as enc_log, bridge_errors.open("wb") as sdk_log:
            reader = subprocess.Popen([self.bridge or "recipe-braw", "stream", str(source), "640", "6"],
                                      stdout=subprocess.PIPE, stderr=sdk_log, env=self.cache.environment(), cwd=self.cache.root)
            encoder = None
            try:
                encoder = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "image2pipe",
                    "-framerate", "6", "-vcodec", "ppm", "-i", "pipe:0", "-an", "-vf", "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2", "-c:v", "libvpx-vp9",
                    "-b:v", "500k", "-deadline", "realtime", "-cpu-used", "6", "-threads", "2", "-f", "webm", str(out)],
                    stdin=reader.stdout, stdout=subprocess.DEVNULL, stderr=enc_log, env=self.cache.environment(), cwd=self.cache.root)
                reader.stdout.close()
                encoder.wait(timeout=7200)
                reader.wait(timeout=30)
                if reader.returncode or encoder.returncode:
                    raise RuntimeError("BRAW preview failed. " + bridge_errors.read_text(errors="replace")[-1000:] + error_file.read_text(errors="replace")[-1000:])
            finally:
                for process in (reader, encoder):
                    if process and process.poll() is None:
                        process.kill()
                        process.wait()
        return out
