import asyncio
import json
import sqlite3
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np
import httpx
from PIL import Image, ImageDraw
from test_core import Fixture, MockDecoder, snapshot
from recipe_sorter.storage.database import Database, SCHEMA, now
from recipe_sorter.storage.cache import Cache, fingerprint
from recipe_sorter.analysis.embedding_cache import EmbeddingCache
from recipe_sorter.analysis.recommendations import bank_score, rank_candidates, valid_vectors, group_score
from recipe_sorter.analysis.evaluation import benchmark, metrics, synthetic_fixture, validate_manifest, calibrate, export_labels
from recipe_sorter.analysis.frame_sampling import sample_frames, adaptive_frames
from recipe_sorter.backend.service import Service
from recipe_sorter.backend.recommendations import Recommendations
from recipe_sorter.backend.app import create_app


class MoveTests(Fixture):
    def test_migration_backup_cannot_follow_symlink_into_sources(self):
        cache=Cache(self.root/'unsafe-backup');source=self.source('valuable.braw')
        before=snapshot(self.sources)
        cache.path('library.pre-v1.5.sqlite3').symlink_to(source)
        with self.assertRaises(ValueError):Database(cache)
        self.assertEqual(before,snapshot(self.sources))

    def test_migration_preserves_v1_data_and_adopts_cached_embeddings(self):
        cache=Cache(self.root/'v1')
        with sqlite3.connect(cache.path('library.sqlite3')) as conn:
            conn.executescript(SCHEMA)
            conn.execute("INSERT INTO clips(path,filename,extension,size,mtime_ns,fingerprint,decoder,discovered_at,analysis_version,manual_group) VALUES('/original.braw','original.braw','.braw',10,1,'fp','braw-sdk',?,'sampling-v2:clip-vit-b32-v1-abc',1)",(now(),))
            conn.executemany('INSERT INTO frames(clip_id,seconds,path,quality,embedding) VALUES(1,?,?,1,?)',[(1,'a.jpg','[1,0]'),(3,'b.jpg','[0,1]')])
        db=Database(cache)
        self.assertEqual(db.get_clip(1)['manual_group'],1)
        self.assertEqual(db.rows('PRAGMA user_version')[0]['user_version'],2)
        self.assertEqual(db.rows('SELECT * FROM embeddings')[0]['model_version'],'clip-vit-b32-v1-abc')
        self.assertTrue(cache.path('library.pre-v1.5.sqlite3').exists())
        Database(cache)  # idempotent startup
        self.assertEqual(len(db.rows('SELECT * FROM embeddings')),1)

    def test_bulk_undo_survives_restart_and_preserves_history(self):
        ids=[self.add(self.source(n)) for n in ('a.braw','b.mov')];g=self.db.group('Cake')
        before=snapshot(self.sources);move=self.db.assign(ids,g)
        self.db=Database(self.cache);self.db.undo(move)
        self.assertTrue(all(self.db.get_clip(i)['manual_group'] is None for i in ids))
        self.assertEqual(len(self.db.rows('SELECT * FROM corrections')),4)
        self.assertEqual(snapshot(self.sources),before)
        self.assertIsNone(self.db.latest_move())

    def test_undo_chain_and_atomic_conflict(self):
        ids=[self.add(self.source(n)) for n in ('a.mp4','b.mp4')];a=self.db.group('A');b=self.db.group('B')
        first=self.db.assign(ids,a);second=self.db.assign([ids[0]],b)
        with self.assertRaisesRegex(ValueError,'newer move'):self.db.undo(first)
        self.assertEqual(self.db.get_clip(ids[1])['manual_group'],a)
        self.db.undo(second);self.db.undo(first)
        self.assertTrue(all(self.db.get_clip(i)['manual_group'] is None for i in ids))
        with self.assertRaises(ValueError):self.db.undo(first)

    def test_optimistic_move_does_not_overwrite_newer_assignment(self):
        cid=self.add(self.source());a=self.db.group('A');b=self.db.group('B')
        self.db.assign([cid],a, {str(cid):0})
        with self.assertRaisesRegex(ValueError,'another view'):self.db.assign([cid],b,{str(cid):0})
        self.assertEqual(self.db.get_clip(cid)['manual_group'],a)
        self.assertEqual(len(self.db.rows('SELECT * FROM moves')),1)

    def test_deleted_prior_group_prevents_undo_without_partial_changes(self):
        cid=self.add(self.source());a=self.db.group('A');b=self.db.group('B')
        self.db.assign([cid],a);move=self.db.assign([cid],b);self.db.delete_group(a)
        with self.assertRaisesRegex(ValueError,'previous group'):self.db.undo(move)
        self.assertEqual(self.db.get_clip(cid)['manual_group'],b)

    def test_deleted_automatic_group_cannot_be_exposed_by_undo(self):
        cid=self.add(self.source());a=self.db.group('Automatic');b=self.db.group('Reviewed')
        with self.db.connect() as conn:
            conn.execute('UPDATE clips SET proposed_group=? WHERE id=?',(a,cid))
        move=self.db.assign([cid],b);self.db.delete_group(a)
        with self.assertRaisesRegex(ValueError,'previous group'):self.db.undo(move)
        self.assertEqual(self.db.get_clip(cid)['manual_group'],b)


class RecommendationMathTests(unittest.TestCase):
    def item(self,cid,clip,features=None):
        return {'id':cid,'vectors':{'clip':clip,'features':features if features is not None else clip}}

    def test_different_stages_match_different_examples(self):
        mixing=self.item(1,[[1,0,0],[1,0,0]]);baked=self.item(2,[[0,1,0],[0,1,0]])
        transition=self.item(3,[[1,0,0],[0,1,0]])
        score,_,support=bank_score(transition,[mixing,baked],'clip')
        self.assertAlmostEqual(score,1)
        self.assertEqual({r['clip_id'] for r in support},{1,2})
        self.assertLess(bank_score(transition,[mixing],'clip')[0],.6)
        self.assertEqual(rank_candidates({2:[mixing,baked]},[transition])[0]['evidence']['label'],'suggested')

    def test_visually_identical_different_recipes_remain_ambiguous(self):
        a=self.item(1,[[1,0],[1,0]]);b=self.item(2,[[1,0],[1,0]]);c=self.item(3,[[1,0],[1,0]])
        results=rank_candidates({2:[a],3:[b]},[c])
        self.assertTrue(all(r['evidence']['label']=='ambiguous' for r in results))
        self.assertEqual(results[0]['evidence']['margin'],0)

    def test_hybrid_can_resolve_semantic_lookalike_with_visual_corroboration(self):
        a=self.item(1,[[1,0],[1,0]],[[1,0],[1,0]])
        b=self.item(2,[[.8,.6],[.8,.6]],[[0,1],[0,1]])
        candidate=self.item(3,[[.8,.6],[.8,.6]],[[1,0],[1,0]])
        clip=rank_candidates({2:[a],3:[b]},[candidate],'clip')
        hybrid=rank_candidates({2:[a],3:[b]},[candidate],'hybrid')
        self.assertEqual(clip[0]['group_id'],3)  # synthetic semantic false positive
        self.assertEqual(hybrid[0]['group_id'],2)
        self.assertEqual(hybrid[0]['evidence']['label'],'suggested')

    def test_visual_only_match_cannot_bypass_hybrid_semantic_floor(self):
        example=self.item(1,[[1,0],[1,0]],[[1,0],[1,0]])
        candidate=self.item(2,[[0,1],[0,1]],[[1,0],[1,0]])
        self.assertEqual(rank_candidates({2:[example]},[candidate],'hybrid')[0]['evidence']['label'],'ambiguous')

    def test_invalid_vectors_rejected_not_silently_normalized(self):
        for values in ([[0,0],[0,0]],[[float('nan'),1],[1,0]],[1,0],[[1,0]]):
            with self.assertRaises(ValueError):valid_vectors(values)
        np.testing.assert_allclose(valid_vectors([[10,0],[0,100]]),[[1,0],[0,1]])

    def test_model_profiles_are_distinct_and_no_probability_field(self):
        item=self.item(2,[[.82,.57236],[.82,.57236]])
        example=self.item(1,[[1,0],[1,0]])
        self.assertEqual(rank_candidates({2:[example]},[item],'clip')[0]['evidence']['label'],'suggested')
        self.assertEqual(rank_candidates({2:[example]},[item],'features')[0]['evidence']['label'],'ambiguous')
        self.assertNotIn('probability',rank_candidates({2:[example]},[item])[0])


class RecommendationStorageTests(Fixture):
    def setUp(self):
        super().setUp()
        self.service=Service(self.cache,self.db,lambda p,c:MockDecoder(c))
        for i in range(6):self.source(f'{i}.mp4')
        self.service.scan(self.sources);self.service.analyze()
        self.ids=[c['id'] for c in self.db.library()['clips']]
        self.group=self.db.group('My cake')
        self.db.assign(self.ids[:2],self.group)
        self.runner=Recommendations(self.service)

    def build(self):
        self.runner.build(self.group,'features')
        return self.runner.queue(self.group)['suggestions']

    def test_accept_reject_skip_persistence_and_undo(self):
        before=snapshot(self.sources);rows=self.build();self.assertEqual(len(rows),4)
        a,b,c=rows[:3]
        self.runner.decide(a['id'],'skipped')
        self.assertEqual(len(self.runner.queue(self.group,'skipped')['suggestions']),1)
        self.runner.decide(b['id'],'rejected')
        result=self.runner.decide(c['id'],'accepted')
        self.assertEqual(self.db.get_clip(c['clip_id'])['manual_group'],self.group)
        # Acceptance adds membership but does not invalidate other cards in this run.
        self.assertFalse(self.runner.queue(self.group)['suggestions'][0]['stale'])
        self.db.undo(result['move_id'])
        self.assertIsNone(self.db.get_clip(c['clip_id'])['manual_group'])
        self.db=Database(self.cache)
        decisions=self.db.rows('SELECT decision FROM suggestion_decisions ORDER BY id')
        self.assertEqual([d['decision'] for d in decisions],['skipped','rejected','accepted','undone'])
        regenerated=self.build()
        self.assertNotIn(b['clip_id'],{r['clip_id'] for r in regenerated})
        self.assertIn(a['clip_id'],{r['clip_id'] for r in regenerated})
        self.assertEqual(before,snapshot(self.sources))

    def test_manual_move_makes_pending_acceptance_stale(self):
        row=self.build()[0];self.db.assign([row['clip_id']],1)
        with self.assertRaisesRegex(ValueError,'Nothing was overwritten'):self.runner.decide(row['id'],'accepted')
        self.assertEqual(self.db.get_clip(row['clip_id'])['manual_group'],1)
        self.assertTrue(next(r for r in self.runner.queue(self.group)['suggestions'] if r['id']==row['id'])['stale'])

    def test_seed_change_invalidates_queue(self):
        row=self.build()[0];self.db.assign([self.ids[0]],1)
        with self.assertRaisesRegex(ValueError,'Example groups changed'):self.runner.decide(row['id'],'accepted')
        self.assertTrue(self.runner.queue(self.group)['suggestions'][0]['stale'])

    def test_reanalysis_invalidates_old_evidence(self):
        row=self.build()[0];self.service.analyze(force=True)
        with self.assertRaisesRegex(ValueError,'Example groups changed'):self.runner.decide(row['id'],'accepted')

    def test_source_changed_after_recommendation_cannot_be_accepted(self):
        row=self.build()[0];path=Path(self.db.get_clip(row['clip_id'])['path'])
        # Simulate external recording/copy, never an application mutation.
        path.write_bytes(b'externally changed')
        with self.assertRaisesRegex(ValueError,'Source changed'):self.runner.decide(row['id'],'accepted')

    def test_cached_clip_can_be_recommended_without_loading_torch(self):
        with self.db.connect() as conn:
            for row in conn.execute("SELECT * FROM embeddings WHERE backend='features'").fetchall():
                conn.execute('INSERT INTO embeddings VALUES(?,?,?,?,?,?,?,?)',(row['clip_id'],'clip','clip-vit-b32-v1-fixture',row['fingerprint'],row['sampling_version'],row['frame_ids'],row['vectors'],now()))
        with patch('recipe_sorter.analysis.embedding_cache.create_embedder',side_effect=AssertionError('Must reuse cached embeddings')):
            self.runner.build(self.group,'clip')
        self.assertEqual(self.service.snapshot()['cache']['reused'],6)
        self.assertEqual(len(self.runner.queue(self.group)['suggestions']),4)

    def test_subset_examples_must_be_manually_in_target_group(self):
        with self.assertRaisesRegex(ValueError,'Manually assign'):self.runner.build(self.group,'features',[self.ids[-1]])
        self.runner.build(self.group,'features',[self.ids[0]])
        self.assertEqual(self.runner.queue(self.group)['run']['summary']['examples'],1)

    def test_caches_preserved_across_backend_comparison(self):
        cache=EmbeddingCache(self.cache,self.db)
        before=len(self.db.rows("SELECT * FROM embeddings WHERE backend='features'"))
        with patch('recipe_sorter.analysis.embedding_cache.create_embedder') as factory:
            factory.return_value.version='clip-test'
            factory.return_value.encode=lambda images:np.tile([1.,0.],(len(images),1))
            cache.get(self.db.get_clip(self.ids[0]),'clip')
        self.assertEqual(len(self.db.rows("SELECT * FROM embeddings WHERE backend='features'")),before)
        self.assertEqual(len(self.db.rows("SELECT * FROM embeddings WHERE backend='clip'")),1)

    def test_export_omits_stale_cache_without_loading_model(self):
        self.db.assign(self.ids,self.group)
        with self.db.connect() as conn:
            conn.execute("UPDATE embeddings SET sampling_version='obsolete' WHERE clip_id=?",(self.ids[0],))
        with patch('recipe_sorter.analysis.embedding_cache.create_embedder',side_effect=AssertionError('Cache-only export')):
            exported=export_labels(self.service,backends=['features'])
        self.assertEqual(next(c['vectors'] for c in exported['clips'] if c['id']==self.ids[0]),{})
        self.assertTrue(any(c['vectors'] for c in exported['clips']))

    def test_incomplete_cached_vectors_cannot_produce_misaligned_evidence(self):
        cid=self.ids[0]
        with self.db.connect() as conn:
            conn.execute("UPDATE embeddings SET vectors='[[1,0],[1,0]]' WHERE clip_id=?",(cid,))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            EmbeddingCache(self.cache,self.db).get(self.db.get_clip(cid),'features')


class EvaluationTests(unittest.TestCase):
    def test_exact_metrics_and_abstention_denominator(self):
        items=[{'id':1,'label':'A'},{'id':2,'label':'A'},{'id':3,'label':'B'}]
        m=metrics(items,{1:'A',2:None,3:'A'},{1:'A',2:'A',3:'A'})
        self.assertEqual(m['pairwise_precision'],0);self.assertEqual(m['pairwise_recall'],0)
        self.assertAlmostEqual(m['uncertain_fraction'],1/3);self.assertEqual(m['suggested_match_accuracy'],.5)

    def test_reproducible_fixture_and_no_heldout_label_leakage(self):
        data=synthetic_fixture();report=benchmark(data)
        self.assertEqual(len(report['results']),6);self.assertTrue(report['synthetic'])
        # Calibration profiles cannot change when evaluation labels are altered.
        for c in data['clips']:
            if c['split']=='evaluation':c['label']='changed'
        second=benchmark(data)
        self.assertEqual([r['profile'] for r in report['results']],[r['profile'] for r in second['results']])
        for r in report['results']:
            self.assertGreater(r['scoring_seconds'],0)
            if r['method']=='example_bank':self.assertEqual(r['false_positive_pairs'],0)

    def test_missing_real_model_is_reported_as_missing(self):
        data=synthetic_fixture()
        for c in data['clips']:del c['vectors']['clip']
        report=benchmark(data)
        self.assertEqual(len(report['results']),2);self.assertEqual(len(report['skipped']),2)

    def test_duplicate_clip_ids_prevent_split_leakage(self):
        data=synthetic_fixture();data['clips'].append(data['clips'][0])
        with self.assertRaisesRegex(ValueError,'split leakage'):validate_manifest(data)

    def test_adaptive_sampling_bounded_and_finds_visual_change(self):
        class Decoder:
            def frame(self,source,seconds):
                im=Image.new('RGB',(120,80),'red' if seconds<5 else 'blue');d=ImageDraw.Draw(im)
                for x in range(0,120,8):d.line((x,0,x,80),fill='white',width=2)
                return im
        from recipe_sorter.media.decoders.base import Metadata
        meta=Metadata(10,25,120,80,'mock');dec=Decoder()
        uniform=sample_frames(dec,Path('/readonly.mp4'),meta)
        adaptive=adaptive_frames(dec,Path('/readonly.mp4'),meta)
        self.assertEqual(len(uniform),5);self.assertGreater(len(adaptive),5);self.assertLessEqual(len(adaptive),9)
        self.assertTrue(set(s[1] for s in uniform)<=set(s[1] for s in adaptive))


class UpgradeAPITests(Fixture):
    def test_moves_recommendations_and_decisions_through_api(self):
        async def run():
            app=create_app(self.cache.root,True)
            async def heartbeat():
                while True:await asyncio.sleep(.01)
            ticker=asyncio.create_task(heartbeat())
            try:
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',headers={'X-Recipe-Sorter':'local'}) as client:
                        library=(await client.get('/api/library')).json();g=library['groups'][0]['id'];ids=[c['id'] for c in library['clips'][:2]]
                        response=await client.post('/api/assign',json={'clip_ids':ids,'group_id':g,'expected_revisions':{str(cid):0 for cid in ids}})
                        self.assertEqual(response.status_code,200);move=response.json()['move_id']
                        self.assertEqual((await client.post(f'/api/moves/{move}/undo')).status_code,200)
                        await client.post('/api/assign',json={'clip_ids':ids,'group_id':g})
                        response=await client.post('/api/recommendations',json={'group_id':g,'backend':'features','example_ids':ids})
                        self.assertEqual(response.status_code,200)
                        for _ in range(500):
                            job=(await client.get('/api/job')).json()
                            if job['state']!='running':break
                            await asyncio.sleep(.01)
                        self.assertEqual(job['state'],'complete',job)
                        queue=(await client.get(f'/api/recommendations/{g}')).json()['suggestions']
                        self.assertEqual(len(queue),20)
                        accepted=await client.post(f"/api/suggestions/{queue[0]['id']}/decision",json={'decision':'accepted'})
                        self.assertEqual(accepted.status_code,200,accepted.text)
                        duplicate=await client.post(f"/api/suggestions/{queue[0]['id']}/decision",json={'decision':'accepted'})
                        self.assertEqual(duplicate.status_code,400)
            finally:ticker.cancel()
        asyncio.run(run())
