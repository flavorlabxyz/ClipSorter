"""Persisted, human-reviewed example recommendations. Never auto-assigns."""
import json
from pathlib import Path
from ..analysis.embedding_cache import EmbeddingCache, item_snapshot
from ..analysis.recommendations import rank_candidates, VERSION
from ..storage.cache import fingerprint
from ..storage.database import now


def membership(conn):
    return [dict(r) for r in conn.execute("SELECT c.id,c.manual_group,c.manual_revision FROM clips c JOIN groups g ON c.manual_group=g.id WHERE c.manual_group<>1 AND g.deleted=0 ORDER BY c.id")]


class Recommendations:
    def __init__(self, service):
        self.service, self.db, self.cache = service, service.db, service.cache

    def build(self, group_id, backend='clip', example_ids=None):
        if backend not in ('clip','features','hybrid'):
            raise ValueError('Select CLIP, visual features, or experimental hybrid')
        with self.db.connect() as conn:
            members = membership(conn)
            if not conn.execute('SELECT 1 FROM groups WHERE id=? AND deleted=0 AND id<>1', (group_id,)).fetchone():
                raise ValueError('Choose a recipe group, not Uncertain')
        allowed = {r['id'] for r in members if r['manual_group'] == group_id}
        selected = set(example_ids) if example_ids is not None else allowed
        if not selected or not selected <= allowed:
            raise ValueError('Manually assign example clips to this group first, then choose at least one example')
        seeds = [self.db.get_clip(r['id']) for r in members if r['manual_group'] != group_id or r['id'] in selected]
        candidates = self.db.rows("SELECT * FROM clips WHERE manual_group IS NULL ORDER BY id")
        if len(candidates) + len(seeds) > 1500:
            raise ValueError('Use a library of at most 1,500 clips')
        cache = EmbeddingCache(self.cache, self.db)
        groups, pool, snapshots = {}, [], {}
        backends = ['clip','features'] if backend == 'hybrid' else [backend]
        self.service.progress(total=len(seeds)+len(candidates), message='Preparing example frames')
        for index, c in enumerate(seeds + candidates):
            try:
                if c['demo']:
                    # The demo only uses actual baseline descriptors of illustrations.
                    # It never invents CLIP vectors or advertises real-media inference.
                    if backend != 'features':
                        raise ValueError('Demo recommendations use the visual baseline. Select Visual features; CLIP requires real local footage and model weights.')
                    if not c['sampling_version']:
                        with self.db.connect() as conn:
                            conn.execute("UPDATE clips SET sampling_version='demo-v1' WHERE id=?", (c['id'],))
                        c = self.db.get_clip(c['id'])
                else:
                    strategy = 'adaptive' if c['sampling_version'] == 'adaptive-v1' else 'uniform'
                    c = self.service.prepare_frames(c, strategy)
                item = cache.item(c, backend)
                if not c['demo']:
                    if fingerprint(Path(c['path'])) != c['fingerprint']:
                        raise ValueError('Source changed while comparing; rescan before using results')
                    primary = 'clip' if backend == 'hybrid' else backend
                    with self.db.connect() as conn:
                        conn.executemany('UPDATE frames SET embedding=? WHERE id=?', [(json.dumps(v), f['id']) for v, f in zip(item['vectors'][primary], item['frames'], strict=True)])
                        conn.execute("UPDATE clips SET status='analyzed',error=NULL,analysis_version=? WHERE id=?", (f"{c['sampling_version']}:{item['versions'][primary]}", c['id']))
                snapshot = item_snapshot(self.db, c['id'], backends)
                if c['manual_group'] is not None:
                    groups.setdefault(c['manual_group'], []).append(item)
                else:
                    pool.append(item)
                snapshots[c['id']] = snapshot
            except Exception as exc:
                if not c['demo']:
                    self.service.fail_clip(c['id'], exc)
                if c['id'] in {s['id'] for s in seeds}:
                    raise ValueError(f"Example {c['filename']}: {exc}") from exc
                if c['demo']:
                    self.service.error(f"{c['filename']}: {exc}")
            self.service.progress(done=index+1, message=f"Comparing {c['filename']}")
        # A single run must never compare different model generations.
        versions = {name: {item['versions'][name] for item in pool + [e for values in groups.values() for e in values]} for name in backends}
        if any(len(v) > 1 for v in versions.values()):
            raise ValueError('Mixed model versions in the cache. Re-run analysis with the current installed model.')
        with self.db.connect() as conn:
            negatives = {(r['clip_id'], r['fingerprint']) for r in conn.execute("SELECT clip_id,fingerprint FROM suggestion_decisions WHERE group_id=? AND backend=? AND decision='rejected'", (group_id,backend))}
        pool = [c for c in pool if (c['id'],c['fingerprint']) not in negatives]
        ranked = [r for r in rank_candidates(groups, pool, backend) if r['group_id'] == group_id]
        all_items = {item['id']: item for item in pool + [e for values in groups.values() for e in values]}
        for r in ranked:
            for support in r['evidence']['support']:
                example, candidate = all_items[support['clip_id']], all_items[r['clip_id']]
                support.update(filename=example['filename'], candidate_frame_id=candidate['frames'][support['candidate_frame']]['id'],
                               example_frame_id=example['frames'][support['example_frame']]['id'],
                               candidate_seconds=candidate['frames'][support['candidate_frame']]['seconds'],
                               example_seconds=example['frames'][support['example_frame']]['seconds'])
        context = {'members': members, 'examples': {str(e['id']): snapshots[e['id']] for values in groups.values() for e in values}, 'backends': backends,
                   'versions': {k: sorted(v) for k,v in versions.items()}, 'selected_example_ids': sorted(selected)}
        summary = {'ranked': len(ranked), 'suggested': sum(r['evidence']['label']=='suggested' for r in ranked),
                   'ambiguous': sum(r['evidence']['label']=='ambiguous' for r in ranked), 'cache': cache.stats,
                   'examples': len(selected), 'errors': self.service.snapshot()['errors']}
        with self.db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if membership(conn) != members:
                raise ValueError('Example assignments changed while comparing. Find matches again.')
            self._check_examples(conn, context)
            run = conn.execute('INSERT INTO recommendation_runs(created_at,group_id,backend,version,context,summary) VALUES(?,?,?,?,?,?)',
                               (now(),group_id,backend,VERSION,json.dumps(context),json.dumps(summary))).lastrowid
            conn.execute("UPDATE suggestions SET status='superseded' WHERE group_id=? AND status IN ('pending','skipped')", (group_id,))
            for r in ranked:
                conn.execute('INSERT INTO suggestions(run_id,clip_id,group_id,score,evidence,snapshot) VALUES(?,?,?,?,?,?)',
                             (run,r['clip_id'],group_id,r['score'],json.dumps(r['evidence']),json.dumps(snapshots[r['clip_id']])))
        self.service.progress(recommendation_run=run, recommendation_group=group_id, cache=cache.stats)
        return run

    def _check_examples(self, conn, context):
        if membership(conn) != context['members']:
            raise ValueError('Example groups changed. Find matches again before reviewing these suggestions.')
        for cid, expected in context['examples'].items():
            if item_snapshot(self.db, int(cid), context['backends'], conn) != expected:
                raise ValueError('Example analysis changed. Find matches again.')

    def queue(self, group_id, status='pending'):
        if status not in ('pending','skipped','accepted','rejected','undone'):
            raise ValueError('Unknown suggestion status')
        rows = self.db.rows('SELECT * FROM recommendation_runs WHERE group_id=? ORDER BY id DESC LIMIT 1', (group_id,))
        if not rows:
            return {'run': None, 'suggestions': []}
        run = rows[0]
        run['summary'] = json.loads(run['summary'])
        context = json.loads(run.pop('context'))
        rows = self.db.rows('SELECT s.*,c.filename,c.manual_group FROM suggestions s JOIN clips c ON c.id=s.clip_id WHERE s.run_id=? AND s.status=? ORDER BY s.score DESC,s.clip_id', (run['id'],status))
        with self.db.connect() as conn:
            try:
                self._check_examples(conn, context)
                # Accepted clips must not become examples until a new run. Membership
                # validation below discounts only this run's accepted assignments.
                changed = False
            except ValueError:
                changed = not self._compatible_members(conn, context, run['id'])
            for r in rows:
                r['evidence'] = json.loads(r['evidence'])
                expected = json.loads(r.pop('snapshot'))
                r['stale'] = r['status'] in ('pending','skipped') and (changed or item_snapshot(self.db,r['clip_id'],context['backends'],conn) != expected)
        return {'run': run, 'suggestions': rows}

    def _compatible_members(self, conn, context, run_id):
        # Accepting a suggestion is allowed to add a member without invalidating
        # every remaining card. It is not incorporated into this frozen seed bank.
        accepted = {r['clip_id'] for r in conn.execute("SELECT s.clip_id FROM suggestions s JOIN suggestion_decisions d ON d.suggestion_id=s.id JOIN clips c ON c.id=s.clip_id WHERE s.run_id=? AND s.status='accepted' AND d.decision='accepted' AND c.manual_move_id=d.move_id", (run_id,))}
        if [m for m in membership(conn) if m['id'] not in accepted] != context['members']:
            return False
        return all(item_snapshot(self.db,int(cid),context['backends'],conn)==snapshot for cid,snapshot in context['examples'].items())

    def decide(self, suggestion_id, decision):
        if decision not in ('accepted','rejected','skipped'):
            raise ValueError('Choose accept, reject, or skip')
        with self.db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            s = conn.execute('SELECT * FROM suggestions WHERE id=?', (suggestion_id,)).fetchone()
            if not s or s['status'] not in ('pending','skipped'):
                raise ValueError('Suggestion already reviewed or superseded. Refresh the queue.')
            run = conn.execute('SELECT * FROM recommendation_runs WHERE id=?', (s['run_id'],)).fetchone()
            context = json.loads(run['context'])
            if not self._compatible_members(conn, context, run['id']):
                raise ValueError('Example groups changed. Find matches again.')
            expected = json.loads(s['snapshot'])
            if item_snapshot(self.db,s['clip_id'],context['backends'],conn) != expected or expected['manual_group'] is not None:
                raise ValueError('Clip assignment or analysis changed. Nothing was overwritten; refresh the queue.')
            # Cached evidence remains reviewable offline; accepting needs current source identity.
            if decision == 'accepted':
                for cid in [s['clip_id'], *map(int, context['examples'])]:
                    clip = conn.execute('SELECT * FROM clips WHERE id=?', (cid,)).fetchone()
                    if not clip['demo']:
                        try:
                            path = Path(clip['path'])
                            self.cache.check_source(path)
                            if path.is_symlink() or fingerprint(path) != clip['fingerprint']:
                                raise ValueError('Source changed. Rescan and find matches again.')
                        except OSError as exc:
                            raise ValueError('Source is offline. Reconnect before accepting this suggestion.') from exc
            move = self.db._assign(conn,[s['clip_id']],s['group_id'],reason='recommendation') if decision=='accepted' else None
            conn.execute('INSERT INTO suggestion_decisions(suggestion_id,clip_id,group_id,fingerprint,backend,decision,move_id,created_at) VALUES(?,?,?,?,?,?,?,?)',
                         (s['id'],s['clip_id'],s['group_id'],expected['fingerprint'],run['backend'],decision,move,now()))
            conn.execute('UPDATE suggestions SET status=? WHERE id=?',(decision,s['id']))
            return {'saved': True, 'move_id': move}
