import os
from pathlib import Path

EXTENSIONS = frozenset({".braw", ".mov", ".mp4", ".m4v", ".mkv", ".avi", ".mts", ".mxf", ".webm"})


def discover(folder: Path, recursive: bool = True, on_error=None):
    """Read directory entries; never follow symlinks or mutate sources."""
    folder = folder.expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("Select an existing directory")
    for base, dirs, files in os.walk(folder, followlinks=False, onerror=on_error):
        dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink()) if recursive else []
        for name in sorted(files):
            path = Path(base) / name
            if path.suffix.lower() in EXTENSIONS and not path.is_symlink() and path.is_file():
                yield path
