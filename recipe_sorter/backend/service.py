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

    def organization_action(self, action, expected, reset_id=None):
        from ..storage.organization import Organization
        with self.lock:
            if self.job['state'] == 'running':
                raise ValueError('Wait for the current scan, analysis, recommendation or preview job to finish before changing the sorting session.')
            org = Organization(self.db)
            if action == 'reset': result = org.reset(expected)
            elif action == 'restore': result = org.restore(reset_id, expected)
            elif action == 'cleanup': result = org.cleanup(expected)
            else: raise ValueError('Unknown organization operation')
            self.job = {'state':'idle','message':'Organization saved','done':0,'total':0,'errors':[]}
            return result

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
        from ..analysis.frame_sampling import adaptive_frames, diverse_frames
        sampling_version = {"uniform":"sampling-v2","adaptive":"adaptive-v1","diverse":"diverse-v1"}[sampling]
        source = Path(clip["path"])
        self.cache.check_source(source)
        if source.is_symlink():
            raise ValueError("Source has become a symlink; refusing analysis")
        self.db.upsert_source(source, fingerprint(source), clip["decoder"])
        clip = self.db.get_clip(clip["id"])
        frames = self.db.rows("SELECT path,seconds FROM frames WHERE clip_id=?", (clip["id"],))
        reusable = clip["sampling_version"] == sampling_version and len(frames) >= 2 and all(self.cache.path(f["path"]).is_file() for f in frames)
        if force or not reusable:
            decoder = self.decoder_factory(source, self.cache)
            meta = decoder.probe(source)
            sampler = {"uniform":sample_frames,"adaptive":adaptive_frames,"diverse":diverse_frames}[sampling]
            from PIL import Image
            cached = {round(f['seconds'],6):f['path'] for f in frames if self.cache.path(f['path']).is_file()} if not force else {}
            cache_root = self.cache
            class ReuseFrames:
                def frame(self, path, seconds):
                    key=round(seconds,6)
                    if key in cached:
                        with Image.open(cache_root.path(cached[key])) as image:return image.convert('RGB')
                    return decoder.frame(path,seconds)
            samples = sampler(ReuseFrames(), source, meta)
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

    def analyze(self, backend="features", force=False, sampling="uniform", method="complete"):
        generation = self.db.sorting_state()["generation"]
        from ..analysis.embedding_cache import EmbeddingCache
        if backend not in ("features", "clip", "hybrid") or sampling not in ("uniform", "adaptive", "diverse") or method not in ("complete", "guarded"):
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
        from ..analysis.guarded import guarded_cluster
        algorithm = cluster if method == "complete" else guarded_cluster
        proposals, pairs = algorithm(inputs, threshold={'features':.89,'clip':.88,'hybrid':.885}[backend])
        self.save_proposals(inputs, proposals, pairs, f"{backend}:{sampling}:{method}:{VERSION}", generation)

    def save_proposals(self, inputs, proposals, pairs, version, generation=None):
        from collections import Counter
        import numpy as np
        from ..storage.organization import Organization
        generation = generation if generation is not None else self.db.sorting_state()['generation']
        with self.db.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self.db.require_generation(db, generation)
            live = {r['id']:dict(r) for r in db.execute('SELECT * FROM clips')}
            run = db.execute('INSERT INTO runs(created_at,version,generation) VALUES(?,?,?)',(now(),version,generation)).lastrowid
            groups, used = {}, set()
            # Candidates are evidence, not members. Only publish effective groups.
            candidates = sorted({p['group'] for p in proposals if p['group'] is not None and live[p['clip_id']]['manual_group'] is None})
            for cluster_id in candidates:
                member_ids = [p['clip_id'] for p in proposals if p['group']==cluster_id and live[p['clip_id']]['manual_group'] is None]
                old = [live[cid]['proposed_group'] for cid in member_ids if live[cid]['proposed_group'] != 1]
                reuse = max(set(old),key=lambda x:(old.count(x),-x)) if old else None
                if reuse and reuse not in used and old.count(reuse)>len(member_ids)/2 and db.execute("SELECT 1 FROM groups WHERE id=? AND deleted=0 AND kind='auto' AND generation=?",(reuse,generation)).fetchone():
                    groups[cluster_id]=reuse;used.add(reuse)
                else:
                    serial=db.execute('SELECT COALESCE(MAX(id),1) FROM groups').fetchone()[0]
                    groups[cluster_id]=db.execute("INSERT INTO groups(name,kind,created_at,generation) VALUES(?,'auto',?,?)",(f'Recipe thread {serial:02}',now(),generation)).lastrowid
            by_clip = {c['id']:[] for c in inputs}
            for left,right,score,evidence in pairs:
                by_clip[left].append({'clip_id':right,'score':score,'visual':evidence['visual']})
                by_clip[right].append({'clip_id':left,'score':score,'visual':evidence['visual']})
            reasons=Counter()
            for p in proposals:
                c=live[p['clip_id']]; evidence=dict(p['evidence'])
                neighbors=sorted(by_clip[c['id']],key=lambda x:(-x['score'],x['clip_id']))[:5]
                reason = evidence.get('reason_code')
                if not reason:
                    if p['group'] is not None: reason='grouped'
                    elif not neighbors: reason='no_evidence'
                    elif evidence.get('margin',0)<.045 and evidence.get('second_candidate') is not None: reason='competing_groups'
                    elif evidence.get('best_candidate') is None: reason='no_consistent_companions'
                    elif 'companion' in evidence.get('reason','').lower(): reason='singleton'
                    else: reason='weak_visual_agreement'
                if c['manual_group'] is not None: reason='manual_protected'
                evidence.update(reason_code=reason,algorithm_group=p['group'],neighbors=neighbors,
                    best_neighbor=neighbors[0]['clip_id'] if neighbors else None,
                    best_candidate_cluster=evidence.get('best_candidate'),second_candidate_cluster=evidence.get('second_candidate'))
                evidence['best_candidate']=groups.get(evidence.get('best_candidate'))
                evidence['second_candidate']=groups.get(evidence.get('second_candidate'))
                if reason=='manual_protected': evidence['reason']='Your manual decision protects this clip; automatic evidence does not change its group.'
                group=groups.get(p['group'],1)
                db.execute('INSERT INTO proposals(run_id,clip_id,group_id,confidence,evidence,fingerprint) VALUES(?,?,?,?,?,?)',(run,c['id'],group,p['confidence'],json.dumps(evidence),c['fingerprint']))
                db.execute('UPDATE clips SET proposed_group=?,confidence=?,evidence=?,sorting_state=? WHERE id=?',(group,p['confidence'],json.dumps(evidence),'grouped' if group!=1 else 'uncertain',c['id']))
                if group==1 and c['manual_group'] is None: reasons[reason]+=1
            # Retire obsolete automatic groups only, preserving user drop targets.
            for gid in Organization(self.db).empty_ids(db,automatic_only=True):
                db.execute('UPDATE groups SET deleted=1 WHERE id=?',(gid,))
            keep={(min(cid,n['clip_id']),max(cid,n['clip_id'])) for cid,ns in by_clip.items() for n in sorted(ns,key=lambda x:-x['score'])[:5]}
            db.executemany('INSERT INTO similarities VALUES(?,?,?,?,?)',[(run,a,b,score,json.dumps(e)) for a,b,score,e in pairs if (a,b) in keep])
            clips=[dict(r) for r in db.execute('SELECT * FROM clips')]
            scores=[e['visual'] for _,_,_,e in pairs]
            summary={'processed_clips':len(inputs),'total_clips':len(clips),
                'failures':sum(c['status'] in ('failed','unsupported','braw_unavailable') for c in clips),
                'manually_protected':sum(c['manual_group'] is not None for c in clips),
                'automatically_grouped':sum(c['manual_group'] is None and c['proposed_group']!=1 for c in clips),
                'uncertain':sum(c['manual_group'] is None and c['sorting_state']=='uncertain' and c['status']=='analyzed' for c in clips),
                'unsorted':sum(c['manual_group'] is None and c['sorting_state']=='unsorted' for c in clips),
                'uncertainty_reasons':dict(reasons),'visual_score_quantiles':dict(zip(('p10','p50','p90'),map(float,np.quantile(scores,[.1,.5,.9])))) if scores else {},
                'method':version}
            db.execute('UPDATE runs SET summary=? WHERE id=?',(json.dumps(summary),run))
        self.progress(summary=summary)

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
