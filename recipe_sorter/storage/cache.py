"""The only writable media destination. Source folders are never outputs."""
import hashlib
import os
from pathlib import Path


def fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()


class Cache:
    def __init__(self, root: Path | None = None):
        default = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "recipe-clip-sorter"
        self.root = Path(root or os.environ.get("RCS_DATA_DIR", default)).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path("tmp").mkdir(exist_ok=True)

    def path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Generated output must stay inside the application data directory")
        return path

    def output(self, relative: str) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def check_source(self, source: Path):
        source = source.expanduser().resolve()
        if source.is_relative_to(self.root) or self.root.is_relative_to(source):
            raise ValueError("Source and application data directories must be separate, non-overlapping trees")

    def environment(self):
        env = os.environ.copy()
        env.update(TMPDIR=str(self.path("tmp")), TMP=str(self.path("tmp")), TEMP=str(self.path("tmp")),
                   HF_HOME=str(self.path("models")), XDG_CACHE_HOME=str(self.path("runtime-cache")))
        return env
