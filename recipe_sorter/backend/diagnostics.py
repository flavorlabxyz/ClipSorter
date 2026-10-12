import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from ..media.decoders.braw import BRAWDecoder


def diagnostic(cache):
    def version(command):
        if not shutil.which(command):
            return {"available": False, "detail": f"Install {command}; see Fedora instructions"}
        try:
            result = subprocess.run([command, "-version"], capture_output=True, text=True, timeout=10, cwd=cache.root, env=cache.environment())
            return {"available": result.returncode == 0, "detail": result.stdout.splitlines()[0]}
        except Exception as exc:
            return {"available": False, "detail": str(exc)}
    torch_available = importlib.util.find_spec("torch") is not None
    gpu = "Not checked (PyTorch not installed)"
    if torch_available:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (CUDA unavailable)"
    return {"platform": platform.platform(), "distribution": platform.freedesktop_os_release() if sys.platform == "linux" else {},
            "python": sys.version.split()[0], "cpu_threads": os.cpu_count(), "gpu": gpu,
            "ffmpeg": version("ffmpeg"), "ffprobe": version("ffprobe"), "braw": BRAWDecoder(cache).diagnostic(),
            "embedding_backends": {"features": "Available: color/layout/texture baseline; not a learned model",
                "clip": {"dependencies": torch_available and importlib.util.find_spec("transformers") is not None,
                         "weights": cache.path("models/clip-vit-base-patch32/config.json").is_file()}},
            "data_directory": str(cache.root), "database": str(cache.path("library.sqlite3")),
            "source_policy": "Read-only inputs; all organization is virtual"}
