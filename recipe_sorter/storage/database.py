import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE groups(id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'manual', deleted INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
INSERT INTO groups VALUES(1, 'Uncertain', 'uncertain', 0, 'initial');
CREATE TABLE clips(
 id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, filename TEXT NOT NULL, extension TEXT NOT NULL,
 size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, fingerprint TEXT NOT NULL, decoder TEXT NOT NULL,
 metadata TEXT, status TEXT NOT NULL DEFAULT 'discovered', error TEXT, analysis_version TEXT,
 proposed_group INTEGER REFERENCES groups(id) DEFAULT 1, manual_group INTEGER REFERENCES groups(id),
 confidence REAL, evidence TEXT, preview TEXT, demo INTEGER NOT NULL DEFAULT 0, discovered_at TEXT NOT NULL);
CREATE TABLE frames(id INTEGER PRIMARY KEY, clip_id INTEGER REFERENCES clips(id) NOT NULL,
 seconds REAL NOT NULL, path TEXT NOT NULL, quality REAL NOT NULL, embedding TEXT NOT NULL);
CREATE INDEX frames_clip ON frames(clip_id);
CREATE TABLE runs(id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, version TEXT NOT NULL);
CREATE TABLE proposals(id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id), clip_id INTEGER REFERENCES clips(id),
 group_id INTEGER REFERENCES groups(id), confidence REAL, evidence TEXT NOT NULL, fingerprint TEXT NOT NULL);
CREATE TABLE corrections(id INTEGER PRIMARY KEY, clip_id INTEGER REFERENCES clips(id),
 proposed_group INTEGER REFERENCES groups(id), previous_manual_group INTEGER REFERENCES groups(id),
 assigned_group INTEGER REFERENCES groups(id), confidence REAL, evidence TEXT, fingerprint TEXT,
 proposal_id INTEGER REFERENCES proposals(id), created_at TEXT NOT NULL);
CREATE TABLE similarities(run_id INTEGER REFERENCES runs(id), left_id INTEGER REFERENCES clips(id),
 right_id INTEGER REFERENCES clips(id), score REAL NOT NULL, evidence TEXT NOT NULL,
 PRIMARY KEY(run_id, left_id, right_id));
PRAGMA user_version=1;
"""


class Database:
    def __init__(self, cache):
        self.path = cache.path("library.sqlite3")
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 2:
                raise RuntimeError("Database was created by a newer Recipe Clip Sorter")
            if version == 0:
                db.executescript(SCHEMA)
            if version < 2:
                from .migrations import migrate_v2
                migrate_v2(db, cache.output('library.pre-v1.5.sqlite3'))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def rows(self, sql, args=()):
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql, args)]

    def get_clip(self, clip_id):
        result = self.rows("SELECT * FROM clips WHERE id=?", (clip_id,))
        if not result:
            raise ValueError("Clip does not exist")
        return result[0]

    def upsert_source(self, path, fp, backend):
        stat = path.stat()
        with self.connect() as db:
            old = db.execute("SELECT * FROM clips WHERE path=?", (str(path),)).fetchone()
            if old:
                if old["fingerprint"] != fp:
                    db.execute("UPDATE clips SET size=?,mtime_ns=?,fingerprint=?,metadata=NULL,status='discovered',error=NULL,sampling_version=NULL,analysis_version=NULL,preview=NULL,proposed_group=1,confidence=NULL,evidence=NULL WHERE id=?",
                               (stat.st_size, stat.st_mtime_ns, fp, old["id"]))
                    db.execute("DELETE FROM frames WHERE clip_id=?", (old["id"],))
                    db.execute("DELETE FROM embeddings WHERE clip_id=?", (old["id"],))
                return old["id"]
            return db.execute("INSERT INTO clips(path,filename,extension,size,mtime_ns,fingerprint,decoder,discovered_at) VALUES(?,?,?,?,?,?,?,?)",
                              (str(path), path.name, path.suffix.lower(), stat.st_size, stat.st_mtime_ns, fp, backend, now())).lastrowid

    def group(self, name, kind="manual"):
        name = name.strip()
        if not name or len(name) > 100 or name.casefold() == "uncertain":
            raise ValueError("Use a group name of 1–100 characters other than Uncertain")
        with self.connect() as db:
            return db.execute("INSERT INTO groups(name,kind,created_at) VALUES(?,?,?)", (name, kind, now())).lastrowid

    def rename(self, group_id, name):
        name = name.strip()
        if group_id == 1 or not name or len(name) > 100 or name.casefold() == "uncertain":
            raise ValueError("Uncertain is permanent; other names must be 1–100 characters")
        with self.connect() as db:
            if not db.execute("UPDATE groups SET name=? WHERE id=? AND deleted=0", (name, group_id)).rowcount:
                raise ValueError("Group does not exist")

    def delete_group(self, group_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if group_id == 1:
                raise ValueError("Uncertain cannot be deleted")
            if db.execute("SELECT 1 FROM clips WHERE COALESCE(manual_group, proposed_group,1)=?", (group_id,)).fetchone():
                raise ValueError("Only empty virtual groups can be deleted")
            # Tombstone retains training/history references, never deletes media.
            if not db.execute("UPDATE groups SET deleted=1 WHERE id=? AND deleted=0", (group_id,)).rowcount:
                raise ValueError("Group does not exist")

    def assign(self, clip_ids, group_id, expected_revisions=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._assign(db, clip_ids, group_id, expected_revisions)

    def _assign(self, db, clip_ids, group_id, expected_revisions=None, reason="manual"):
        if not clip_ids:
            raise ValueError("Select at least one clip")
        if not db.execute("SELECT 1 FROM groups WHERE id=? AND deleted=0", (group_id,)).fetchone():
            raise ValueError("Group does not exist")
        move = db.execute("INSERT INTO moves(created_at,reason) VALUES(?,?)", (now(), reason)).lastrowid
        for clip_id in sorted(set(clip_ids)):
            c = db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
            if c is None:
                raise ValueError("Clip does not exist")
            if expected_revisions is not None and expected_revisions.get(str(clip_id)) != c["manual_revision"]:
                raise ValueError("A clip was changed in another view. Refresh and try again.")
            db.execute("INSERT INTO move_items VALUES(?,?,?,?,?)", (move, clip_id, c["manual_group"], c["manual_move_id"], group_id))
            self._correction(db, c, group_id, move, "move")
            db.execute("UPDATE clips SET manual_group=?,manual_move_id=?,manual_revision=manual_revision+1 WHERE id=?", (group_id, move, clip_id))
        return move

    def _correction(self, db, c, group, move, action):
        p = db.execute("SELECT id FROM proposals WHERE clip_id=? ORDER BY id DESC LIMIT 1", (c["id"],)).fetchone()
        db.execute("INSERT INTO corrections(clip_id,proposed_group,previous_manual_group,assigned_group,confidence,evidence,fingerprint,proposal_id,created_at,move_id,action) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (c["id"], c["proposed_group"], c["manual_group"], group, c["confidence"], c["evidence"], c["fingerprint"], p[0] if p else None, now(), move, action))

    def undo(self, move_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            move = db.execute("SELECT * FROM moves WHERE id=? AND undone_at IS NULL", (move_id,)).fetchone()
            if not move:
                raise ValueError("This move has already been undone or does not exist")
            items = db.execute("SELECT * FROM move_items WHERE move_id=?", (move_id,)).fetchall()
            for item in items:
                c = db.execute("SELECT * FROM clips WHERE id=?", (item["clip_id"],)).fetchone()
                if c["manual_move_id"] != move_id:
                    raise ValueError("A clip has a newer move. Undo that move first; nothing was changed.")
                destination = item["previous_manual"] if item["previous_manual"] is not None else (c["proposed_group"] or 1)
                if not db.execute("SELECT 1 FROM groups WHERE id=? AND deleted=0", (destination,)).fetchone():
                    raise ValueError("The previous group was deleted. Nothing was changed; move the clip manually.")
                self._correction(db, c, item["previous_manual"], move_id, "undo")
                db.execute("UPDATE clips SET manual_group=?,manual_move_id=?,manual_revision=manual_revision+1 WHERE id=?",
                           (item["previous_manual"], item["previous_move_id"], item["clip_id"]))
            db.execute("UPDATE moves SET undone_at=? WHERE id=?", (now(), move_id))
            # Acceptance remains in history; append the explicit reversal.
            for d in db.execute("SELECT * FROM suggestion_decisions WHERE move_id=? AND decision='accepted'", (move_id,)).fetchall():
                db.execute("INSERT INTO suggestion_decisions(suggestion_id,clip_id,group_id,fingerprint,backend,decision,move_id,created_at) VALUES(?,?,?,?,?,'undone',?,?)",
                           (d["suggestion_id"],d["clip_id"],d["group_id"],d["fingerprint"],d["backend"],move_id,now()))
                db.execute("UPDATE suggestions SET status='undone' WHERE id=?", (d["suggestion_id"],))

    def latest_move(self):
        rows = self.rows("SELECT id,created_at,reason FROM moves WHERE undone_at IS NULL ORDER BY id DESC LIMIT 1")
        return rows[0] if rows else None

    def library(self):
        clips = self.rows("SELECT * FROM clips ORDER BY filename,id")
        frames = self.rows("SELECT id,clip_id,seconds,path,quality FROM frames ORDER BY seconds")
        grouped = {}
        for f in frames:
            grouped.setdefault(f["clip_id"], []).append(f)
        for c in clips:
            c["metadata"] = json.loads(c["metadata"]) if c["metadata"] else {}
            c["evidence"] = json.loads(c["evidence"]) if c["evidence"] else {}
            c["group_id"] = c["manual_group"] or c["proposed_group"] or 1
            c["frames"] = grouped.get(c["id"], [])
        return {"clips": clips, "last_move": self.latest_move(), "groups": self.rows("SELECT * FROM groups WHERE deleted=0 ORDER BY CASE WHEN id=1 THEN 1 ELSE 0 END,id")}
