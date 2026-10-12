import json
import logging
import threading
import uuid
from pathlib import Path
from ..storage.cache import fingerprint
from ..storage.database import now
from ..media.discovery import discover
from ..media.decoders.registry import decoder_for
from ..media.decoders.base import Metadata, DecoderUnavailable
from ..analysis.embeddings import create_embedder
from ..analysis.frame_sampling import sample_frames
from ..analysis.clustering import cluster, VERSION

log = logging.getLogger(__name__)


class Service:
    def __init__(self, cache, database, decoder_factory=decoder_for):
        self.cache, self.db, self.decoder_factory = cache, database, decoder_factory
        self.lock = threading.RLock()
        self.job = {"state": "idle", "message": "Ready", "done": 0, "total": 0, "errors": []}

    def recover_interrupted(self):
        """Call only after acquiring this library's single-process lock."""
        with self.db.connect() as db:
            db.execute("UPDATE clips SET status='failed',error='Analysis interrupted. Retry analysis.' WHERE status='analyzing'")

    def start(self, name, fn, *args):
        with self.lock:
            if self.job["state"] == "running":
                raise ValueError("Another task is running. Wait for it to finish.")
            self.job = {"state": "running", "name": name, "message": name, "done": 0, "total": 0, "errors": []}
        def work():
            try:
                fn(*args)
                self.progress(state="complete", message=f"{name} complete")
            except Exception as exc:
                log.exception("Task failed: %s", name)
                self.progress(state="failed", message=str(exc))
        threading.Thread(target=work, daemon=True, name="media-worker").start()
        return self.snapshot()

    def progress(self, **kwargs):
        with self.lock:
            self.job.update(kwargs)

    def snapshot(self):
        with self.lock:
            return {**self.job, "errors": list(self.job["errors"])}

    def error(self, message):
        with self.lock:
            self.job["errors"].append(message)
        log.warning("%s", message)

    def fail_clip(self, clip_id, exc):
        clip = self.db.get_clip(clip_id)
        status = "braw_unavailable" if isinstance(exc, DecoderUnavailable) and clip["decoder"] == "braw-sdk" else "unsupported" if isinstance(exc, DecoderUnavailable) else "failed"
        with self.db.connect() as db:
            db.execute("UPDATE clips SET status=?,error=?,proposed_group=1,confidence=NULL,evidence=NULL WHERE id=?", (status, str(exc), clip_id))
        self.error(f"{clip['filename']}: {exc}")

    def scan(self, folder, recursive=True):
        folder = Path(folder).expanduser().resolve(strict=True)
        self.cache.check_source(folder)
        paths = list(discover(folder, recursive, on_error=lambda e: self.error(str(e))))
        self.progress(total=len(paths), message="Reading clip metadata")
        for index, path in enumerate(paths):
            clip_id = None
            try:
                self.cache.check_source(path)
                decoder = self.decoder_factory(path, self.cache)
                clip_id = self.db.upsert_source(path, fingerprint(path), decoder.name)
                c = self.db.get_clip(clip_id)
                if not c["metadata"]:
                    meta = decoder.probe(path)
                    if fingerprint(path) != c["fingerprint"]:
                        raise RuntimeError("Source changed while reading. Scan again after recording/copying finishes.")
                    with self.db.connect() as db:
                        db.execute("UPDATE clips SET metadata=?,status='discovered',error=NULL WHERE id=?", (json.dumps(meta.dict()), clip_id))
            except Exception as exc:
                if clip_id:
                    self.fail_clip(clip_id, exc)
                else:
                    self.error(f"{path.name}: {exc}")
            self.progress(done=index + 1, message=f"Scanned {path.name}")

    def prepare_frames(self, clip, sampling="uniform", force=False):
        from ..analysis.frame_sampling import adaptive_frames
        sampling_version = "sampling-v2" if sampling == "uniform" else "adaptive-v1"
        source = Path(clip["path"])
        self.cache.check_source(source)
        if source.is_symlink():
            raise ValueError("Source has become a symlink; refusing analysis")
        self.db.upsert_source(source, fingerprint(source), clip["decoder"])
        clip = self.db.get_clip(clip["id"])
        frames = self.db.rows("SELECT path FROM frames WHERE clip_id=?", (clip["id"],))
        reusable = clip["sampling_version"] == sampling_version and len(frames) >= 2 and all(self.cache.path(f["path"]).is_file() for f in frames)
        if force or not reusable:
            decoder = self.decoder_factory(source, self.cache)
            meta = decoder.probe(source)
            sampler = sample_frames if sampling == "uniform" else adaptive_frames
            samples = sampler(decoder, source, meta)
            records = []
            generation = uuid.uuid4().hex[:12]
            for quality, seconds, image in samples:
                relative = f"frames/{clip['id']}/{clip['fingerprint'][:16]}/{generation}-{len(records)}.jpg"
                image.save(self.cache.output(relative), quality=85)
                records.append((clip['id'], seconds, relative, quality, '[]'))
            if fingerprint(source) != clip['fingerprint']:
                raise RuntimeError("Source changed during sampling. Rescan after recording/copying finishes.")
            with self.db.connect() as db:
                db.execute("DELETE FROM embeddings WHERE clip_id=?", (clip['id'],))
                db.execute("DELETE FROM frames WHERE clip_id=?", (clip['id'],))
                db.executemany("INSERT INTO frames(clip_id,seconds,path,quality,embedding) VALUES(?,?,?,?,?)", records)
                db.execute("UPDATE clips SET metadata=?,sampling_version=? WHERE id=?", (json.dumps(meta.dict()), sampling_version, clip['id']))
        return self.db.get_clip(clip['id'])

    def analyze(self, backend="features", force=False, sampling="uniform"):
        from ..analysis.embedding_cache import EmbeddingCache
        if backend not in ("features", "clip", "hybrid") or sampling not in ("uniform", "adaptive"):
            raise ValueError("Unknown analysis backend or sampling strategy")
        cache = EmbeddingCache(self.cache, self.db)
        clips = self.db.rows("SELECT * FROM clips WHERE demo=0 ORDER BY path")
        if len(clips) > 1500:
            raise ValueError("V1.5 supports up to 1,500 clips per library; use separate data directories for larger sessions")
        self.progress(total=len(clips))
        inputs = []
        for index, old in enumerate(clips):
            self.progress(message=f"Analyzing {old['filename']}")
            try:
                c = self.prepare_frames(old, sampling, force)
                with self.db.connect() as db:
                    db.execute("UPDATE clips SET status='analyzing',error=NULL WHERE id=?", (c['id'],))
                item = cache.item(c, backend)
                primary = 'clip' if backend == 'hybrid' else backend
                version = f"{c['sampling_version']}:{item['versions'][primary]}"
                if fingerprint(Path(c['path'])) != c['fingerprint']:
                    raise RuntimeError("Source changed during analysis; rescan before using results")
                with self.db.connect() as db:
                    db.executemany("UPDATE frames SET embedding=? WHERE id=?", [(json.dumps(v), f['id']) for v, f in zip(item['vectors'][primary], item['frames'], strict=True)])
                    db.execute("UPDATE clips SET status='analyzed',error=NULL,analysis_version=? WHERE id=?", (version, c['id']))
                # Keep the original clustering control for comparison. Hybrid also
                # requires pairwise visual consistency, and is explicitly experimental.
                if backend == 'hybrid':
                    import numpy as np
                    from ..analysis.embeddings import unit
                    vectors = unit(np.concatenate([np.asarray(item['vectors']['clip']) * .75**.5, np.asarray(item['vectors']['features']) * .25**.5], axis=1)).tolist()
                else:
                    vectors = item['vectors'][backend]
                inputs.append({**c, 'metadata': json.loads(c['metadata']), 'embeddings': vectors})
            except Exception as exc:
                self.fail_clip(old['id'], exc)
            self.progress(done=index + 1)
        self.progress(message="Comparing cached evidence; your manual groups stay in place", cache=cache.stats)
        proposals, pairs = cluster(inputs, threshold={'features':.89,'clip':.88,'hybrid':.885}[backend])
        self.save_proposals(inputs, proposals, pairs, f"{backend}:{sampling}:{VERSION}")

    def save_proposals(self, inputs, proposals, pairs, version):
        with self.db.connect() as db:
            run = db.execute("INSERT INTO runs(created_at,version) VALUES(?,?)", (now(), version)).lastrowid
            groups = {}
            used = set()
            # Reuse an existing auto group when a majority of the new members agree.
            candidates = sorted({r["evidence"]["best_candidate"] for r in proposals if r["evidence"]["best_candidate"] is not None})
            for group in candidates:
                member_ids = [p["clip_id"] for p in proposals if p["evidence"]["best_candidate"] == group]
                old = [c["proposed_group"] for c in inputs if c["id"] in member_ids and c["proposed_group"] != 1]
                reuse = max(set(old), key=old.count) if old else None
                if reuse and reuse not in used and old.count(reuse) > len(member_ids) / 2 and db.execute("SELECT 1 FROM groups WHERE id=? AND deleted=0 AND kind='auto'", (reuse,)).fetchone():
                    groups[group] = reuse
                    used.add(reuse)
                else:
                    serial = db.execute("SELECT COALESCE(MAX(id),1) FROM groups").fetchone()[0]
                    groups[group] = db.execute("INSERT INTO groups(name,kind,created_at) VALUES(?,'auto',?)", (f"Recipe thread {serial:02}", now())).lastrowid
            source_by_id = {c["id"]: c for c in inputs}
            for p in proposals:
                group = groups[p["group"]] if p["group"] is not None else 1
                evidence = {**p["evidence"]}
                evidence["best_candidate"] = groups.get(evidence["best_candidate"])
                evidence["second_candidate"] = groups.get(evidence["second_candidate"])
                encoded = json.dumps(evidence)
                db.execute("INSERT INTO proposals(run_id,clip_id,group_id,confidence,evidence,fingerprint) VALUES(?,?,?,?,?,?)",
                           (run, p["clip_id"], group, p["confidence"], encoded, source_by_id[p["clip_id"]]["fingerprint"]))
                db.execute("UPDATE clips SET proposed_group=?,confidence=?,evidence=? WHERE id=?", (group, p["confidence"], encoded, p["clip_id"]))
            # Retain top five pair explanations per clip; compute all pairs, avoid quadratic DB growth.
            keep = set()
            by_clip = {c["id"]: [] for c in inputs}
            for pair in pairs:
                by_clip[pair[0]].append(pair)
                by_clip[pair[1]].append(pair)
            for c in inputs:
                ranked = sorted(by_clip[c["id"]], key=lambda p: p[2], reverse=True)[:5]
                keep.update((p[0], p[1]) for p in ranked)
            db.executemany("INSERT INTO similarities VALUES(?,?,?,?,?)", [(run, a, b, s, json.dumps(e)) for a, b, s, e in pairs if (a, b) in keep])

    def preview(self, clip_id):
        c = self.db.get_clip(clip_id)
        if c["demo"]:
            from ..demo import demo_preview
            return demo_preview(self.cache, self.db, c)
        source = Path(c["path"])
        self.cache.check_source(source)
        if source.is_symlink() or fingerprint(source) != c["fingerprint"]:
            raise ValueError("Source changed since scan. Rescan before previewing.")
        if c["preview"] and self.cache.path(c["preview"]).is_file():
            return
        decoder = self.decoder_factory(source, self.cache)
        meta = Metadata(**json.loads(c["metadata"])) if c["metadata"] else decoder.probe(source)
        relative = f"previews/{clip_id}-{c['fingerprint'][:16]}.webm"
        self.progress(message=f"Building a silent review proxy for {c['filename']}")
        temporary = relative + ".partial"
        path = decoder.preview(source, temporary, meta)
        if fingerprint(source) != c["fingerprint"]:
            raise ValueError("Source changed while generating proxy. Rescan before previewing.")
        path.replace(self.cache.output(relative))
        with self.db.connect() as db:
            db.execute("UPDATE clips SET preview=? WHERE id=?", (relative, clip_id))
