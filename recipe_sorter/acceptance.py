"""Opt-in real-media acceptance; never labels a mock as validated BRAW."""
import hashlib
import json
import platform
from pathlib import Path
from .storage.cache import fingerprint
from .storage.database import now
from .media.discovery import discover
from .media.decoders.registry import decoder_for
from .analysis.embeddings import create_embedder
from .analysis.similarity import pair_score
from .analysis.clustering import cluster


def snapshot(path):
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    s = path.stat()
    return {'sha256': digest, 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def run_acceptance(source, service, backend):
    source = Path(source).expanduser().resolve(strict=True)
    service.cache.check_source(source)
    if source.suffix.lower() != '.braw':
        raise ValueError('Provide a real .braw source file')
    before = snapshot(source)
    # Snapshot companion files and directory entries too: detect unexpected sidecars.
    siblings = {str(p): snapshot(p) for p in source.parent.iterdir() if p.is_file() and p.suffix.lower() == '.sidecar'}
    names_before = sorted(p.name for p in source.parent.iterdir())
    report = {'source': str(source), 'created_at': now(), 'before': before, 'checks': {}, 'backend': backend,
              'fedora': platform.freedesktop_os_release().get('ID') == 'fedora' if platform.system() == 'Linux' else False,
              'human_checks_pending': ['Inspect sampled thumbnails for reasonable color/exposure', 'Open clip in UI and play/seek proxy', 'Review grouping with other clips from the filming session'],
              'braw_fully_validated': False}
    checks = report['checks']
    try:
        checks['discovered'] = source in list(discover(source.parent, False))
        decoder = decoder_for(source, service.cache)
        checks['backend'] = decoder.name == 'braw-sdk'
        metadata = decoder.probe(source)
        report['metadata'] = metadata.dict()
        checks.update(opens=True, duration=metadata.duration > 0, fps=metadata.fps > 0, resolution=metadata.width > 0 and metadata.height > 0,
                      useful_metadata=bool(metadata.camera or metadata.recorded_at or metadata.extra.get('timecode')))
        cid = service.db.upsert_source(source, fingerprint(source), decoder.name)
        # Use the full production path. Existing library clips also participate.
        service.analyze(backend, force=False)
        clip = service.db.get_clip(cid)
        frames = service.db.rows('SELECT * FROM frames WHERE clip_id=?', (cid,))
        checks['multiple_positions'] = len({f['seconds'] for f in frames}) >= 3
        checks['embeddings'] = len(frames) >= 3 and all(len(json.loads(f['embedding'])) > 0 for f in frames)
        if clip['status'] != 'analyzed':
            raise RuntimeError(clip['error'] or 'BRAW analysis did not complete')
        checks['proposal_persisted'] = bool(service.db.rows('SELECT * FROM proposals WHERE clip_id=?', (cid,)))
        item = {'id': cid, 'metadata': metadata.dict(), 'embeddings': [json.loads(f['embedding']) for f in frames]}
        score, _ = pair_score(item, item)
        checks['similarity'] = score > .99
        checks['clustering'] = len(cluster([item])[0]) == 1
        service.preview(cid)
        proxy = service.db.get_clip(cid)['preview']
        checks['proxy'] = bool(proxy and service.cache.path(proxy).stat().st_size > 0)
        report['clip_id'] = cid
        report['ui_url'] = 'http://127.0.0.1:8765'
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        report['after'] = snapshot(source)
        checks['source_unchanged'] = before == report['after']
        checks['directory_entries_unchanged'] = names_before == sorted(p.name for p in source.parent.iterdir())
        checks['sidecars_unchanged'] = all(Path(p).is_file() and snapshot(Path(p)) == s for p, s in siblings.items())
    report['automated_checks_passed'] = 'error' not in report and bool(checks) and all(checks.values())
    destination = service.cache.output('acceptance/braw-' + now().replace(':', '-') + '.json')
    report['report_path'] = str(destination)
    destination.write_text(json.dumps(report, indent=2))
    return report
