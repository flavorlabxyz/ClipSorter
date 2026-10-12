import json
import sqlite3
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_core import Fixture, MockDecoder, snapshot
from recipe_sorter.storage.database import Database
from recipe_sorter.storage.organization import Organization
from recipe_sorter.backend.service import Service
from recipe_sorter.backend.recommendations import Recommendations
from recipe_sorter.analysis.guarded import guarded_cluster
from recipe_sorter.analysis.clustering import cluster
from recipe_sorter.analysis.prototypes import rank_prototypes
from recipe_sorter.analysis.evaluation import export_labels


class OrganizationTests(Fixture):
    def setUp(self):
        super().setUp()
        self.service=Service(self.cache,self.db,lambda p,c:MockDecoder(c))
        for i in range(8):self.source(f'{i}.braw').chmod(0o444)
        self.service.scan(self.sources);self.service.analyze()
        self.ids=[c['id'] for c in self.db.library()['clips']]
        self.a=self.db.group('Cake');self.b=self.db.group('Pasta')
        self.first=self.db.assign(self.ids[:2],self.a);self.db.assign(self.ids[2:4],self.b)
        self.empty=self.db.group('Future recipe')
        self.runner=Recommendations(self.service);self.runner.build(self.a,'features')
        self.rejected=self.runner.queue(self.a)['suggestions'][0]
        self.runner.decide(self.rejected['id'],'rejected')
        self.pending=self.runner.queue(self.a)['suggestions'][0]
        for cid in self.ids:self.db.label(cid,'Cake' if cid<5 else 'Pasta',split='example' if cid<3 else 'evaluation')
        self.org=Organization(self.db)

    def token(self):return self.db.sorting_state()

    def test_reset_preserves_caches_sources_history_truth_and_restart(self):
        before=snapshot(self.sources);frames=self.db.rows('SELECT * FROM frames');embeddings=self.db.rows('SELECT * FROM embeddings')
        proposals=self.db.rows('SELECT * FROM proposals');corrections=self.db.rows('SELECT * FROM corrections');labels=self.db.rows('SELECT * FROM benchmark_labels')
        clips_before=self.db.rows('SELECT id,path,metadata,preview,fingerprint,status FROM clips')
        result=self.service.organization_action('reset',self.token())
        self.assertTrue(self.cache.path(result['backup']).is_file())
        with sqlite3.connect(self.cache.path(result['backup'])) as backup:
            self.assertEqual(backup.execute('PRAGMA quick_check').fetchone()[0],'ok')
            self.assertEqual(backup.execute('SELECT COUNT(*) FROM clips WHERE manual_group IS NOT NULL').fetchone()[0],4)
        self.db=Database(self.cache);lib=self.db.library()
        self.assertEqual(len(lib['groups']),1);self.assertIsNone(lib['last_move']);self.assertIsNone(lib['analysis_summary'])
        self.assertTrue(all(c['manual_group'] is None and c['proposed_group']==1 and c['sorting_state']=='unsorted' for c in lib['clips']))
        for table,old in [('frames',frames),('embeddings',embeddings),('proposals',proposals),('corrections',corrections),('benchmark_labels',labels)]:self.assertEqual(self.db.rows('SELECT * FROM '+table),old)
        self.assertEqual(clips_before,self.db.rows('SELECT id,path,metadata,preview,fingerprint,status FROM clips'))
        self.assertEqual(snapshot(self.sources),before)
        self.assertTrue(Organization(self.db).preview()['restore']['available'])

    def test_old_undo_suggestions_and_proposal_commits_inactive(self):
        generation=self.token()['generation'];self.org.reset(self.token())
        with self.assertRaises(ValueError):self.db.undo(self.first)
        with self.assertRaisesRegex(ValueError,'session changed'):self.runner.decide(self.pending['id'],'accepted')
        with self.assertRaisesRegex(ValueError,'session changed'):self.service.save_proposals([],[],[],'old-worker',generation)
        self.assertIsNone(self.runner.queue(self.a)['run'])
        self.assertEqual(len(self.db.library()['groups']),1)

    def test_new_suggestions_and_rejections_after_restore(self):
        reset=self.org.reset(self.token());self.org.restore(reset['reset_id'],self.token())
        self.runner.build(self.a,'features')
        self.assertIn(self.rejected['clip_id'],[r['clip_id'] for r in self.runner.queue(self.a)['suggestions']])
        self.assertIsNone(self.db.latest_move())
        with self.assertRaises(ValueError):self.db.undo(self.first)

    def test_genuine_unassigned_candidates_after_reset(self):
        self.org.reset(self.token());group=self.db.group('New recipe');self.db.assign(self.ids[:2],group)
        self.runner.build(group,'features')
        self.assertEqual(len(self.runner.queue(group)['suggestions']),6)

    def test_restore_conflicts_with_subsequent_manual_work(self):
        reset=self.org.reset(self.token());self.db.group('New work')
        before=self.db.library()
        with self.assertRaisesRegex(ValueError,'subsequent work'):self.org.restore(reset['reset_id'],self.token())
        self.assertEqual(self.db.library(),before);self.assertTrue(self.org.preview()['restore']['conflict'])

    def test_restore_previous_organization_offline_and_preserve_new_truth(self):
        before=[(c['id'],c['manual_group'],c['proposed_group']) for c in self.db.library()['clips']]
        with self.db.connect() as conn:conn.execute("UPDATE clips SET path='/offline/'||filename")
        reset=self.org.reset(self.token());self.db.label(self.ids[0],'Verified cake label')
        self.org.restore(reset['reset_id'],self.token());self.db=Database(self.cache)
        self.assertEqual(before,[(c['id'],c['manual_group'],c['proposed_group']) for c in self.db.library()['clips']])
        self.assertEqual(self.db.rows('SELECT recipe FROM benchmark_labels WHERE clip_id=?',(self.ids[0],))[0]['recipe'],'Verified cake label')
        with self.assertRaises(ValueError):self.org.restore(reset['reset_id'],self.token())

    def test_backup_failure_and_mid_transaction_failure_are_atomic(self):
        before=self.db.library();token=self.token()
        with patch.object(Organization,'backup',side_effect=ValueError('Backup failed')):
            with self.assertRaisesRegex(ValueError,'Backup failed'):self.org.reset(token)
        self.assertEqual(self.db.library(),before)
        with self.db.connect() as conn:
            conn.execute("CREATE TRIGGER injected_reset_failure BEFORE UPDATE OF sorting_state ON clips WHEN NEW.sorting_state='unsorted' BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.org.reset(token)
        self.assertEqual(self.db.library(),before)
        self.assertEqual(self.db.rows('SELECT * FROM sorting_resets'),[])

    def test_backup_directory_failure_is_actionable_and_atomic(self):
        before=self.db.library()
        (self.cache.root/'backups').write_text('blocked')
        with self.assertRaisesRegex(ValueError,'Backup failed'):self.org.reset(self.token())
        self.assertEqual(self.db.library(),before)

    def test_job_conflict_and_stale_confirmation_are_refused(self):
        ready=threading.Event();release=threading.Event()
        def work():ready.set();release.wait(5)
        self.service.start('Paused job',work);self.assertTrue(ready.wait(1))
        try:
            with self.assertRaisesRegex(ValueError,'current'):self.service.organization_action('reset',self.token())
        finally:release.set()
        for _ in range(100):
            if self.service.snapshot()['state']!='running':break
            time.sleep(.01)
        token=self.token();self.db.rename(self.a,'Cake renamed')
        with self.assertRaisesRegex(ValueError,'confirmation'):self.org.reset(token)

    def test_two_resets_from_same_confirmation_only_one_commits(self):
        token=self.token();results=[]
        def reset():
            try:results.append(self.org.reset(token))
            except ValueError:results.append('conflict')
        threads=[threading.Thread(target=reset) for _ in range(2)]
        for t in threads:t.start()
        for t in threads:t.join(5);self.assertFalse(t.is_alive())
        self.assertEqual(results.count('conflict'),1)
        self.assertEqual(len(self.db.rows('SELECT * FROM sorting_resets')),1)

    def test_cleanup_counts_full_library_and_retains_history(self):
        before=self.org.preview();self.assertEqual(before['empty_groups'],1)
        result=self.org.cleanup(self.token());self.assertEqual(result['removed'],1)
        groups={g['id'] for g in self.db.library()['groups']}
        self.assertIn(self.a,groups);self.assertIn(self.b,groups);self.assertNotIn(self.empty,groups);self.assertIn(1,groups)
        self.assertEqual(self.db.rows('SELECT deleted FROM groups WHERE id=?',(self.empty,))[0]['deleted'],1)
        with self.assertRaises(ValueError):self.db.assign([self.ids[0]],self.empty)

    def test_direct_group_to_group_bulk_move_undo_noop_and_reanalysis(self):
        before=snapshot(self.sources);token=self.token();moves=len(self.db.rows('SELECT * FROM moves'))
        self.assertIsNone(self.db.assign(self.ids[:2],self.a))
        self.assertEqual(self.token(),token);self.assertEqual(len(self.db.rows('SELECT * FROM moves')),moves)
        move=self.db.assign([self.ids[0],self.ids[2]],self.b)
        self.assertEqual(len(self.db.rows('SELECT * FROM move_items WHERE move_id=?',(move,))),1)
        self.db.undo(move);self.assertEqual(self.db.get_clip(self.ids[0])['manual_group'],self.a)
        self.db.assign(self.ids[:2],self.b);self.service.analyze();self.db=Database(self.cache)
        self.assertTrue(all(self.db.get_clip(cid)['manual_group']==self.b for cid in self.ids[:2]))
        self.assertEqual(before,snapshot(self.sources))

    def test_reanalysis_does_not_accumulate_empty_auto_groups(self):
        for _ in range(3):self.service.analyze()
        visible=self.db.library()['groups'];automatic=[g for g in visible if g['kind']=='auto']
        self.assertEqual(len(automatic),1);self.assertIn(self.empty,{g['id'] for g in visible})
        self.db.assign(self.ids,self.a)
        for _ in range(2):self.service.analyze()
        self.assertEqual([g for g in self.db.library()['groups'] if g['kind']=='auto'],[])
        self.assertEqual(self.db.analysis_summary()['summary']['manually_protected'],8)

    def test_truth_export_after_reset_does_not_use_active_manual_assignments(self):
        before=export_labels(self.service,['features'])
        self.org.reset(self.token());after=export_labels(self.service,['features'])
        self.assertEqual(before['clips'],after['clips'])


class RevisedAlgorithmTests(unittest.TestCase):
    def item(self,cid,angle):
        a=np.deg2rad(angle);v=[np.cos(a),np.sin(a),0.]
        return {'id':cid,'embeddings':[v,v],'vectors':{'clip':[v,v],'features':[v,v]},'metadata':{}}

    def test_guarded_average_joins_stages_complete_link_splits(self):
        clips=[self.item(i,a) for i,a in enumerate((0,15,30),1)]
        old,_=cluster(clips,threshold=.88);new,_=guarded_cluster(clips,threshold=.88)
        self.assertLess(sum(r['group'] is not None for r in old),3)
        self.assertEqual(sum(r['group'] is not None for r in new),3)
        self.assertEqual(len({r['group'] for r in new}),1)

    def test_guarded_average_does_not_chain_across_recipes(self):
        clips=[self.item(i,a) for i,a in enumerate((0,15,30,45,60,75,90),1)]
        rows,_=guarded_cluster(clips,threshold=.88)
        groups={r['group'] for r in rows if r['group'] is not None}
        self.assertGreater(len(groups),1)
        for group in groups:
            angles=[(r['clip_id']-1)*15 for r in rows if r['group']==group]
            self.assertLessEqual(max(angles)-min(angles),45)

    def test_prototypes_ignore_duplicate_examples_and_repeated_background(self):
        a=self.item(1,0);a['vectors']['clip']=[[1,0,0],[0,1,0]]
        b=self.item(2,90);b['vectors']['clip']=[[0,0,1],[0,-1,0]]
        c=self.item(3,0);c['vectors']['clip']=[[1,0,0],[0,1,0]]
        base=rank_prototypes({2:[a],3:[b]},[c])
        duplicates=[{**a,'id':100+i} for i in range(30)]
        other=rank_prototypes({2:[a,*duplicates],3:[b]},[c])
        self.assertEqual([(r['group_id'],r['score']) for r in base],[(r['group_id'],r['score']) for r in other])
        self.assertEqual(base[0]['evidence']['label'],'suggested')
        c['vectors']['clip']=[[1,0,0]]*5
        rows=rank_prototypes({2:[a],3:[b]},[c]);self.assertEqual(rows[0]['evidence']['label'],'ambiguous')
        self.assertIn('Only one',str(rows[0]['evidence']['reasons']))

    def test_two_plausible_groups_and_no_competitors_reported(self):
        a=self.item(1,0);a['vectors']['clip']=[[1,0,0],[0,1,0]]
        c={**a,'id':3}
        rows=rank_prototypes({2:[a],3:[{**a,'id':2}]},[c])
        self.assertTrue(all(r['evidence']['label']=='ambiguous' for r in rows))
        self.assertIn('No competing',str(rank_prototypes({2:[a]},[c])[0]['evidence']['reasons']))
