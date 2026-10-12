"""Ordered, transactional schema migrations. The V1 tables/history stay intact."""
import json
import sqlite3

V2 = """
BEGIN EXCLUSIVE;
ALTER TABLE clips ADD COLUMN manual_revision INTEGER NOT NULL DEFAULT 0;
ALTER TABLE clips ADD COLUMN manual_move_id INTEGER;
ALTER TABLE clips ADD COLUMN sampling_version TEXT;
ALTER TABLE corrections ADD COLUMN move_id INTEGER;
ALTER TABLE corrections ADD COLUMN action TEXT NOT NULL DEFAULT 'move';
CREATE TABLE moves(id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, undone_at TEXT, reason TEXT NOT NULL);
CREATE TABLE move_items(move_id INTEGER REFERENCES moves(id), clip_id INTEGER REFERENCES clips(id),
 previous_manual INTEGER REFERENCES groups(id), previous_move_id INTEGER, assigned_group INTEGER REFERENCES groups(id),
 PRIMARY KEY(move_id,clip_id));
CREATE TABLE embeddings(clip_id INTEGER REFERENCES clips(id), backend TEXT NOT NULL, model_version TEXT NOT NULL,
 fingerprint TEXT NOT NULL, sampling_version TEXT NOT NULL, frame_ids TEXT NOT NULL, vectors TEXT NOT NULL,
 updated_at TEXT NOT NULL, PRIMARY KEY(clip_id,backend));
CREATE TABLE recommendation_runs(id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, group_id INTEGER REFERENCES groups(id),
 backend TEXT NOT NULL, version TEXT NOT NULL, context TEXT NOT NULL, summary TEXT NOT NULL);
CREATE TABLE suggestions(id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES recommendation_runs(id),
 clip_id INTEGER REFERENCES clips(id), group_id INTEGER REFERENCES groups(id), score REAL NOT NULL,
 evidence TEXT NOT NULL, snapshot TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending');
CREATE INDEX suggestion_queue ON suggestions(group_id,status,run_id);
CREATE TABLE suggestion_decisions(id INTEGER PRIMARY KEY, suggestion_id INTEGER REFERENCES suggestions(id),
 clip_id INTEGER REFERENCES clips(id), group_id INTEGER REFERENCES groups(id), fingerprint TEXT NOT NULL,
 backend TEXT NOT NULL, decision TEXT NOT NULL, move_id INTEGER REFERENCES moves(id), created_at TEXT NOT NULL);
PRAGMA user_version=2;
COMMIT;
"""


def migrate_v2(db, backup):
    # Once-only pre-upgrade snapshot in application data. Never touches sources.
    if not backup.exists():
        with sqlite3.connect(backup) as destination:
            db.backup(destination)
    db.executescript(V2)
    # Adopt V1 caches with their actual model provenance, not a guessed backend.
    for clip in db.execute('SELECT * FROM clips WHERE analysis_version IS NOT NULL').fetchall():
        version = clip['analysis_version']
        if not version.startswith('sampling-v2:'):
            continue
        model = version.split(':', 1)[1]
        backend = 'clip' if model.startswith('clip-vit-b32-v1-') else 'features' if model == 'visual-features-v2' else None
        if not backend:
            continue
        rows = db.execute('SELECT id,embedding FROM frames WHERE clip_id=? ORDER BY seconds,id', (clip['id'],)).fetchall()
        try:
            from ..analysis.recommendations import valid_vectors
            vectors = valid_vectors([json.loads(r['embedding']) for r in rows]).tolist()
        except (ValueError, TypeError):
            continue
        db.execute("UPDATE clips SET sampling_version='sampling-v2' WHERE id=?", (clip['id'],))
        db.execute('INSERT OR IGNORE INTO embeddings VALUES(?,?,?,?,?,?,?,?)',
                   (clip['id'], backend, model, clip['fingerprint'], 'sampling-v2', json.dumps([r['id'] for r in rows]), json.dumps(vectors), clip['discovered_at']))
