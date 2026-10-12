import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageDraw
import asyncio
import httpx
from recipe_sorter.storage.cache import Cache, fingerprint
from recipe_sorter.storage.database import Database
from recipe_sorter.media.discovery import discover
from recipe_sorter.media.decoders.base import Metadata, DecoderUnavailable
from recipe_sorter.media.decoders.registry import decoder_for
from recipe_sorter.media.decoders.ffmpeg import normalize_probe, FFmpegDecoder
from recipe_sorter.media.decoders.braw import BRAWDecoder
from recipe_sorter.analysis.clustering import cluster
from recipe_sorter.analysis.embeddings import VisualFeatures
from recipe_sorter.analysis.similarity import pair_score
from recipe_sorter.analysis.frame_sampling import sample_frames
from recipe_sorter.backend.service import Service
from recipe_sorter.backend.app import create_app


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / 'sources'
        self.sources.mkdir()
        self.cache = Cache(self.root / 'application')
        self.db = Database(self.cache)
    def tearDown(self):
        self.temp.cleanup()
    def source(self, name='clip.braw'):
        p = self.sources / name
        p.write_bytes(b'fixture immutable source\0' * 100)
        return p
    def add(self, path):
        return self.db.upsert_source(path, fingerprint(path), decoder_for(path,self.cache).name)


def snapshot(folder):
    return {str(p.relative_to(folder)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob('*') if p.is_file()}


class DiscoveryTests(Fixture):
    def test_extensions_recursion_and_symlink_safety(self):
        for name in ('A.BRAW','b.MOV','c.mp4','d.txt'): self.source(name)
        nested = self.sources / 'nested'; nested.mkdir(); (nested/'e.braw').write_bytes(b'raw')
        (self.sources/'link.mp4').symlink_to(self.sources/'c.mp4')
        (self.sources/'loop').symlink_to(self.sources, target_is_directory=True)
        before = snapshot(self.sources)
        self.assertEqual({p.name for p in discover(self.sources)}, {'A.BRAW','b.MOV','c.mp4','e.braw'})
        self.assertEqual(len(list(discover(self.sources,False))), 3)
        self.assertEqual(snapshot(self.sources), before)
    def test_fingerprint_invalidation(self):
        path = self.source(); a = fingerprint(path)
        self.assertEqual(a, fingerprint(path))
        s = path.stat(); os.utime(path, ns=(s.st_atime_ns, s.st_mtime_ns + 1000))
        self.assertNotEqual(a, fingerprint(path))
    def test_cache_cannot_escape_or_overlap(self):
        for relative in ('../source.mp4','/tmp/escape','.'):
            with self.assertRaises(ValueError): self.cache.output(relative)
        (self.cache.root/'escape').symlink_to(self.sources, target_is_directory=True)
        with self.assertRaises(ValueError): self.cache.output('escape/new.jpg')
        for path in (self.cache.root,self.root,self.cache.root/'somewhere'):
            with self.assertRaises(ValueError): self.cache.check_source(path)
        self.cache.check_source(self.sources)
    def test_decoder_selection_is_explicit(self):
        self.assertIsInstance(decoder_for(Path('A.BRAW'),self.cache), BRAWDecoder)
        self.assertIsInstance(decoder_for(Path('A.mov'),self.cache), FFmpegDecoder)
        with self.assertRaises(ValueError): decoder_for(Path('x.txt'),self.cache)
    def test_braw_unavailable_not_ffmpeg_fallback(self):
        source = self.source()
        decoder = BRAWDecoder(self.cache); decoder.bridge = None
        with self.assertRaisesRegex(DecoderUnavailable, 'official Blackmagic RAW'):
            decoder.probe(source)
        self.assertFalse(decoder.diagnostic()['available'])
    def test_metadata_normalization(self):
        meta = normalize_probe({'streams':[{'codec_type':'audio'},{'codec_type':'video','width':1920,'height':1080,'avg_frame_rate':'0/0','r_frame_rate':'30000/1001','codec_name':'h264','tags':{'creation_time':'2026-01-01T12:00:00Z'}}], 'format':{'duration':'10.5'}})
        self.assertAlmostEqual(meta.fps,29.97003,places=4)
        self.assertEqual(meta.duration,10.5)
        self.assertEqual(meta.recorded_at,'2026-01-01T12:00:00Z')
        with self.assertRaises(ValueError): normalize_probe({'streams':[]})
        with self.assertRaises(ValueError): Metadata(float('nan'),25,1,1,'x')


class StorageTests(Fixture):
    def test_assignments_persist_without_overwriting_proposal(self):
        cid = self.add(self.source()); g = self.db.group('Cake')
        self.db.assign([cid],g)
        reopened = Database(self.cache)
        self.assertEqual(reopened.get_clip(cid)['manual_group'],g)
        self.assertEqual(reopened.get_clip(cid)['proposed_group'],1)
        correction = reopened.rows('SELECT * FROM corrections')[0]
        self.assertEqual(correction['proposed_group'],1)
        self.assertEqual(correction['assigned_group'],g)
        self.assertIsNotNone(correction['created_at'])
        self.db.assign([cid],1)
        self.assertEqual(len(self.db.rows('SELECT * FROM corrections')),2)
        self.db.delete_group(g)
        self.assertEqual(len(self.db.rows('SELECT * FROM groups WHERE deleted=0')),1)
    def test_groups_delete_empty_only_and_uncertain_permanent(self):
        cid = self.add(self.source()); g = self.db.group('Bread')
        self.db.rename(g,'Rolls'); self.db.assign([cid],g)
        with self.assertRaises(ValueError): self.db.delete_group(g)
        with self.assertRaises(ValueError): self.db.delete_group(1)
        with self.assertRaises(ValueError): self.db.rename(1,'Lost')
        with self.assertRaises(ValueError): self.db.group('  ')
    def test_bulk_assignment_transaction_rolls_back(self):
        cid = self.add(self.source()); g = self.db.group('Cake')
        with self.assertRaises(ValueError): self.db.assign([cid,999],g)
        self.assertIsNone(self.db.get_clip(cid)['manual_group'])
        self.assertEqual(self.db.rows('SELECT * FROM corrections'),[])
    def test_changes_invalidate_only_derived_analysis(self):
        path=self.source();cid=self.add(path);g=self.db.group('Review');self.db.assign([cid],g)
        with self.db.connect() as db: db.execute("UPDATE clips SET status='analyzed',analysis_version='old',preview='old.webm' WHERE id=?",(cid,))
        stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000));self.add(path)
        c=self.db.get_clip(cid)
        self.assertEqual(c['status'],'discovered');self.assertEqual(c['manual_group'],g)
        self.assertIsNone(c['preview']);self.assertIsNone(c['analysis_version'])


class ClusterTests(unittest.TestCase):
    def item(self,i,vector,time='2026-01-01T12:00:00Z'):
        return {'id':i,'embeddings':[vector,vector], 'metadata':{'recorded_at':time,'camera':'Camera','width':1920,'height':1080}}
    def test_interleaved_recipes_use_visuals(self):
        clips=[self.item(1,[1,0,0]),self.item(2,[0,1,0]),self.item(3,[1,0,0]),self.item(4,[0,1,0]),self.item(5,[0,0,1])]
        results,pairs=cluster(clips);r={x['clip_id']:x for x in results}
        self.assertIsNotNone(r[1]['group']);self.assertEqual(r[1]['group'],r[3]['group']);self.assertEqual(r[2]['group'],r[4]['group']);self.assertNotEqual(r[1]['group'],r[2]['group']);self.assertIsNone(r[5]['group']);self.assertEqual(len(pairs),10)
    def test_singleton_and_empty(self):
        self.assertEqual(cluster([]),([],[]))
        self.assertIsNone(cluster([self.item(1,[1,0])])[0][0]['group'])
    def test_competing_threads_abstain(self):
        a=[1,0]; b=[.75,.6614378]; bridge=[.935414,.353553]
        clips=[self.item(1,a),self.item(2,a),self.item(3,b),self.item(4,b),self.item(5,bridge)]
        r={x['clip_id']:x for x in cluster(clips,threshold=.89,margin=.06)[0]}
        self.assertIsNone(r[5]['group'])
        self.assertIsNotNone(r[5]['evidence']['second_candidate'])
    def test_all_frames_contribute(self):
        a=self.item(1,[1,0]);b=self.item(2,[1,0]);b['embeddings']=[[1,0],[0,1]]
        self.assertLess(pair_score(a,b)[0],.95)
    def test_embeddings_normalized_and_deterministic(self):
        a=Image.new('RGB',(64,64),'red');b=Image.new('RGB',(64,64),'blue')
        vectors=VisualFeatures().encode([a,b]);self.assertTrue(np.allclose(np.linalg.norm(vectors,axis=1),1));self.assertLess(float(vectors[0]@vectors[1]),.5)


class MockDecoder:
    name='braw-sdk'
    frame_calls=0
    def __init__(self,cache): self.cache=cache
    def probe(self,source): return Metadata(10.,25.,1920,1080,'Mock RAW',camera='Mock camera')
    def frame(self,source,seconds,max_size=768):
        type(self).frame_calls+=1
        im=Image.new('RGB',(160,100),'#657048');d=ImageDraw.Draw(im)
        for x in range(0,160,10): d.rectangle((x,0,x+4,100),fill='#cabc89')
        return im
    def preview(self,source,relative,metadata):
        out=self.cache.output(relative);out.write_bytes(b'mock preview');return out


class AnalysisTests(Fixture):
    def setUp(self):
        super().setUp();MockDecoder.frame_calls=0
        self.service=Service(self.cache,self.db,lambda p,c:MockDecoder(c))
    def test_read_only_sources_incremental_analysis_and_corrections(self):
        self.source('one.braw');self.source('two.mov')
        before=snapshot(self.sources)
        for p in self.sources.iterdir():p.chmod(0o444)
        self.service.scan(self.sources);self.service.analyze()
        self.assertEqual({c['status'] for c in self.db.library()['clips']},{'analyzed'})
        cid=self.db.library()['clips'][0]['id'];g=self.db.group('My correction');self.db.assign([cid],g)
        count=MockDecoder.frame_calls;self.service.analyze();self.assertEqual(count,MockDecoder.frame_calls)
        self.service.preview(cid)
        self.assertEqual(before,snapshot(self.sources));self.assertEqual(self.db.get_clip(cid)['manual_group'],g)
        self.assertGreaterEqual(len(self.db.rows('SELECT * FROM proposals WHERE clip_id=?',(cid,))),2)
        self.service.analyze(force=True);self.assertGreater(MockDecoder.frame_calls,count)
        self.assertEqual(before,snapshot(self.sources))
    def test_missing_cache_regenerated(self):
        self.source();self.service.scan(self.sources);self.service.analyze();count=MockDecoder.frame_calls
        path=self.db.rows('SELECT path FROM frames LIMIT 1')[0]['path'];self.cache.path(path).unlink()
        self.service.analyze();self.assertGreater(MockDecoder.frame_calls,count)
    def test_per_file_failure_continues(self):
        self.source('a.braw');self.source('b.mp4')
        class Faulty(MockDecoder):
            def probe(self,source):
                if source.suffix=='.braw':raise DecoderUnavailable('SDK unavailable')
                return super().probe(source)
        service=Service(self.cache,self.db,lambda p,c:Faulty(c));service.scan(self.sources);service.analyze()
        states={c['filename']:c['status'] for c in self.db.library()['clips']}
        self.assertEqual(states,{'a.braw':'braw_unavailable','b.mp4':'analyzed'})
    def test_black_frames_are_not_accepted(self):
        class Black(MockDecoder):
            def frame(self,*args,**kwargs):return Image.new('RGB',(30,30))
        with self.assertRaisesRegex(RuntimeError,'Fewer than two'):
            sample_frames(Black(self.cache),self.source(),Metadata(10,25,100,100,'mock'))
    def test_changed_model_version_recomputes(self):
        self.source();self.service.scan(self.sources);self.service.analyze();count=MockDecoder.frame_calls
        with patch.object(VisualFeatures,'version','new-version'):
            self.service.analyze()
        # V1.5 changes only embeddings when the model changes, reusing frames.
        self.assertEqual(MockDecoder.frame_calls,count)
        self.assertEqual(self.db.rows('SELECT model_version FROM embeddings')[0]['model_version'],'new-version')


class APITests(Fixture):
    def test_demo_api_and_security(self):
        asyncio.run(self.exercise_api())

    async def exercise_api(self):
        app=create_app(self.cache.root,demo=True)
        # Keep a timer scheduled: restricted Cloud sandboxes can suppress the
        # event loop's cross-thread socket wakeup. This does not mock any API.
        async def heartbeat():
            while True:
                await asyncio.sleep(.01)
        ticker=asyncio.create_task(heartbeat())
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
                    data=(await client.get('/api/library')).json();self.assertTrue(data['demo']);self.assertEqual(len(data['clips']),22)
                    headers={'X-Recipe-Sorter':'local'}
                    group=(await client.post('/api/groups',json={'name':'Review'},headers=headers)).json()['id']
                    cid=data['clips'][0]['id']
                    self.assertEqual((await client.post('/api/assign',json={'clip_ids':[cid],'group_id':group},headers=headers)).status_code,200)
                    detail=(await client.get(f'/api/clips/{cid}')).json();self.assertEqual(detail['manual_group'],group);self.assertEqual(len(detail['corrections']),1)
                    self.assertEqual((await client.post('/api/groups',json={'name':'Blocked'})).status_code,403)
                    self.assertEqual((await client.post('/api/groups',json={'name':'Blocked'},headers={**headers,'Origin':'http://evil.example'})).status_code,403)
                    self.assertEqual((await client.get('/api/library',headers={'Host':'evil.example'})).status_code,403)
                    self.assertEqual((await client.post('/api/scan',json={'folder':str(self.sources)},headers=headers)).status_code,400)
                    self.assertEqual((await client.get('/api/frames/'+str(data['clips'][0]['frames'][0]['id']))).status_code,200)
                    self.assertEqual((await client.get('/static/../../library.sqlite3')).status_code,404)
                    self.assertIn('frame-ancestors',(await client.get('/')).headers['Content-Security-Policy'])
        finally:
            ticker.cancel()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required for real decode test')
class FFmpegIntegrationTests(Fixture):
    def test_real_decode_proxy_cache_and_source_safety(self):
        source=self.sources/'color pattern.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=12','-t','2','-c:v','mpeg4','-threads','1',str(source)],check=True)
        before=snapshot(self.sources);source.chmod(0o444)
        service=Service(self.cache,self.db);service.scan(self.sources);service.analyze()
        c=self.db.library()['clips'][0];self.assertEqual(c['status'],'analyzed',c.get('error'));self.assertEqual(len(c['frames']),5)
        service.preview(c['id']);c=self.db.get_clip(c['id']);self.assertTrue(self.cache.path(c['preview']).stat().st_size>0)
        p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(self.cache.path(c['preview']))],capture_output=True,text=True,check=True)
        self.assertGreater(float(json.loads(p.stdout)['format']['duration']),1.8)
        with patch.object(FFmpegDecoder,'frame',side_effect=AssertionError('Unchanged source should reuse frames')):service.analyze()
        self.assertEqual(before,snapshot(self.sources))


if __name__=='__main__':unittest.main()
