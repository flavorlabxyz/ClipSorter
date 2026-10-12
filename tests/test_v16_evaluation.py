import asyncio
import copy
import unittest
from unittest.mock import patch
import httpx
from test_core import Fixture, MockDecoder, snapshot
from recipe_sorter.backend.app import create_app
from recipe_sorter.backend.service import Service
from recipe_sorter.analysis.comparison import benchmark_v16, stage_fixture, extended_metrics
from recipe_sorter.analysis.evaluation import export_labels


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.report=benchmark_v16(stage_fixture())

    def test_complete_and_guarded_report_stage_tradeoff_separately(self):
        rows={r['method']:r for r in self.report['results'] if r['backend']=='clip' and r['operating_point']=='default'}
        self.assertEqual(len(self.report['results']),24)
        self.assertEqual(rows['guarded']['pairwise_precision'],1)
        self.assertGreater(rows['guarded']['pairwise_recall'],rows['complete']['pairwise_recall'])
        self.assertGreater(rows['guarded']['uncertain_fraction'],0)
        self.assertEqual(rows['guarded']['cross_recipe_contamination'],0)

    def test_automatic_predictions_ignore_manual_and_truth_names(self):
        from recipe_sorter.analysis.comparison import auto_predict
        data=stage_fixture()['clips'];examples=[c for c in data if c['split']=='example'];candidates=[c for c in data if c['split']=='evaluation']
        before=auto_predict(examples,candidates,'clip','guarded',.88)
        changed=copy.deepcopy(candidates)
        for c in changed:c.update(label='wrong',manual_group=999,proposed_group=999)
        self.assertEqual(before,auto_predict(examples,changed,'clip','guarded',.88))

    def test_metrics_penalize_contamination_fragmentation_and_abstention(self):
        items=[{'id':i,'label':l} for i,l in enumerate(['A','A','A','B','B','C'])]
        m=extended_metrics(items,{0:2,1:3,2:None,3:2,4:None,5:None})
        self.assertEqual(m['assignment_coverage'],.5)
        self.assertAlmostEqual(m['cross_recipe_contamination'],2/3)
        self.assertEqual(m['fragmentation_by_recipe'],{'A':2,'B':1,'C':0})
        self.assertEqual(m['fully_abstained_recipes'],1)
        self.assertEqual(m['pairwise_precision'],0)

    def test_review_at_ten_denominators_include_visible_ambiguous_candidates(self):
        for row in self.report['results']:
            q=row['recommendation_quality_at_10']
            if q:
                self.assertEqual(q['reviewed'],14)
                self.assertEqual(q['correct'],6)
                self.assertAlmostEqual(q['precision'],6/14)

    def test_missing_and_mixed_models_never_silently_substitute(self):
        data=stage_fixture()
        for c in data['clips']:del c['vectors']['clip']
        report=benchmark_v16(data)
        self.assertEqual({r['backend'] for r in report['results']},{'features'})
        self.assertEqual(len(report['skipped']),2)
        data=stage_fixture();data['clips'][0]['versions']['clip']='other'
        with self.assertRaisesRegex(ValueError,'provenance'):benchmark_v16(data)


class SamplingAndAPI(Fixture):
    def test_sampling_switch_reuses_cached_base_frames_and_preserves_source(self):
        class Counted(MockDecoder):
            calls=[]
            def frame(self,source,seconds):
                self.calls.append(seconds)
                return super().frame(source,seconds)
        source=self.source();source.chmod(0o444);before=snapshot(self.sources)
        service=Service(self.cache,self.db,lambda p,c:Counted(c));service.scan(self.sources)
        cid=self.db.library()['clips'][0]['id']
        c=service.prepare_frames(self.db.get_clip(cid),'uniform');base=set(Counted.calls);Counted.calls.clear()
        c=service.prepare_frames(c,'diverse')
        self.assertTrue(base.isdisjoint(Counted.calls))
        self.assertGreaterEqual(len(self.db.rows('SELECT * FROM frames')),2)
        self.assertLessEqual(len(self.db.rows('SELECT * FROM frames')),7)
        self.assertEqual(before,snapshot(self.sources));Counted.calls.clear()
        service.prepare_frames(c,'diverse');self.assertEqual(Counted.calls,[])

    def test_truth_split_by_session_survives_reset_and_detects_stale_preparation(self):
        service=Service(self.cache,self.db,lambda p,c:MockDecoder(c))
        for i in range(6):self.source(f'{i}.braw')
        service.scan(self.sources);service.analyze()
        for i,c in enumerate(self.db.library()['clips']):self.db.label(c['id'],'Cake',session=str(i//2))
        manifest=export_labels(service,['features'])
        for session in ('0','1','2'):
            self.assertEqual(len({c['split'] for c in manifest['clips'] if c['session']==session}),1)
        c=self.db.get_clip(1)
        with patch.object(service,'prepare_frames',return_value={**c,'fingerprint':'changed'}):
            with self.assertRaisesRegex(ValueError,'changed during preparation'):export_labels(service,['features'],prepare=True)

    def test_actual_asgi_reset_confirmation_counts_restore_and_labels(self):
        cid=self.add(self.source());group=self.db.group('Cake');self.db.assign([cid],group)
        asyncio.run(self.exercise_api(cid,group))

    async def exercise_api(self,cid,group):
        app=create_app(self.cache.root)
        async def heartbeat():
            while True:await asyncio.sleep(.01)
        ticker=asyncio.create_task(heartbeat());headers={'X-Recipe-Sorter':'local'}
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',headers=headers) as client:
                    label=await client.post(f'/api/clips/{cid}/label',json={'recipe':'Cake','session':'one','stage':'batter'})
                    self.assertEqual(label.status_code,200)
                    preview=(await client.get('/api/organization')).json()
                    self.assertEqual(preview['clips'],1);self.assertEqual(preview['groups'],1)
                    token={k:preview[k] for k in ('generation','revision')}
                    self.assertEqual((await client.post('/api/organization/reset',json=token)).status_code,400)
                    reset=(await client.post('/api/organization/reset',json={**token,'confirmed':True})).json()
                    self.assertEqual((await client.post('/api/organization/reset',json={**token,'confirmed':True})).status_code,400)
                    detail=(await client.get(f'/api/clips/{cid}')).json()
                    self.assertIsNone(detail['manual_group']);self.assertEqual(detail['sorting_state'],'unsorted');self.assertEqual(detail['benchmark_label']['recipe'],'Cake')
                    token=(await client.get('/api/library')).json()['sorting']
                    restore=await client.post('/api/organization/restore/'+str(reset['reset_id']),json={**token,'confirmed':True})
                    self.assertEqual(restore.status_code,200,restore.text)
                    self.assertEqual((await client.get(f'/api/clips/{cid}')).json()['manual_group'],group)
        finally:ticker.cancel()
