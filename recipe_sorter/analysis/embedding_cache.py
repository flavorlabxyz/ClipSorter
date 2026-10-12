import hashlib
import json
from pathlib import Path
from PIL import Image
from .embeddings import create_embedder, VisualFeatures, clip_version
from .recommendations import valid_vectors
from ..storage.database import now


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class EmbeddingCache:
    """Independent caches retain CLIP and baseline for the same sampled frames."""
    def __init__(self, cache, db):
        self.cache, self.db = cache, db
        self.providers = {}
        self.stats = {'reused': 0, 'encoded': 0}

    def expected_version(self, backend):
        if backend == 'features':
            return VisualFeatures.version
        version = clip_version(self.cache)
        if version:
            return version
        # Cached CLIP can be reviewed without loading torch or having weights installed.
        rows = self.db.rows("SELECT model_version FROM embeddings WHERE backend='clip' ORDER BY updated_at DESC LIMIT 1")
        return rows[0]['model_version'] if rows else None

    def get(self, clip, backend, force=False):
        rows = self.db.rows('SELECT id,path,seconds FROM frames WHERE clip_id=? ORDER BY seconds,id', (clip['id'],))
        ids = [r['id'] for r in rows]
        if len(rows) < 2:
            raise ValueError(f"Analyze {clip['filename']} first: fewer than two cached frames")
        cached = self.db.rows('SELECT * FROM embeddings WHERE clip_id=? AND backend=?', (clip['id'], backend))
        expected = self.expected_version(backend)
        if cached and not force:
            c = cached[0]
            if c['model_version'] == expected and c['fingerprint'] == clip['fingerprint'] and c['sampling_version'] == clip['sampling_version'] and json.loads(c['frame_ids']) == ids:
                vectors = valid_vectors(json.loads(c['vectors']))
                if len(vectors) != len(ids):
                    raise ValueError('Cached frame embeddings are incomplete; rebuild cached analysis')
                self.stats['reused'] += 1
                return vectors, c['model_version']
        if backend not in self.providers:
            self.providers[backend] = create_embedder(backend, self.cache)
        provider = self.providers[backend]
        images = []
        for r in rows:
            with Image.open(self.cache.path(r['path'])) as im:
                images.append(im.convert('RGB'))
        vectors = valid_vectors(provider.encode(images))
        if len(vectors) != len(ids):
            raise ValueError('Model returned the wrong number of frame embeddings')
        with self.db.connect() as db:
            db.execute('INSERT OR REPLACE INTO embeddings VALUES(?,?,?,?,?,?,?,?)',
                       (clip['id'], backend, provider.version, clip['fingerprint'], clip['sampling_version'], json.dumps(ids), json.dumps(vectors.tolist()), now()))
        self.stats['encoded'] += 1
        return vectors, provider.version

    def item(self, clip, backend):
        vectors, versions = {}, {}
        for name in (('clip','features') if backend == 'hybrid' else (backend,)):
            v, version = self.get(clip, name)
            vectors[name], versions[name] = v.tolist(), version
        return {**clip, 'vectors': vectors, 'versions': versions,
                'frames': self.db.rows('SELECT id,seconds FROM frames WHERE clip_id=? ORDER BY seconds,id', (clip['id'],))}


def item_snapshot(db, clip_id, backends, connection=None):
    """Strong derived-data identity + human revision, checked atomically at acceptance."""
    def read(conn):
        c = conn.execute('SELECT id,fingerprint,manual_group,manual_revision,sampling_version,status FROM clips WHERE id=?', (clip_id,)).fetchone()
        if c is None: raise ValueError('Clip is no longer available')
        embeddings = [dict(r) for r in conn.execute('SELECT backend,model_version,fingerprint,sampling_version,frame_ids,vectors FROM embeddings WHERE clip_id=? ORDER BY backend', (clip_id,)) if r['backend'] in backends]
        return {**dict(c), 'embedding_digest': digest(embeddings)}
    if connection is not None:
        return read(connection)
    with db.connect() as conn:
        return read(conn)
