#!/usr/bin/env python3
"""V1.6 rendered-browser acceptance on a populated NON-demo library.
Run: pip install -e '.[browser]'; python -m playwright install chromium
     python scripts/browser_acceptance.py
Requires FFmpeg, localhost sockets and Chromium. Sources are synthesized once,
made read-only, and hashed before/after. No user's library is opened.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recipe_sorter.storage.cache import Cache
from recipe_sorter.storage.database import Database
from recipe_sorter.backend.service import Service
from recipe_sorter.backend.recommendations import Recommendations

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'.test-data').mkdir(exist_ok=True)
ROOT_DATA=Path(tempfile.mkdtemp(prefix='browser-v16-',dir=ROOT/'.test-data'))
# Fail quickly if this host cannot run a web application.
with socket.socket() as s:
    s.bind(('127.0.0.1',0));PORT=s.getsockname()[1]
URL=f'http://127.0.0.1:{PORT}'
SOURCES=ROOT_DATA/'sources';SOURCES.mkdir()
CACHE=Cache(ROOT_DATA/'app-data');DB=Database(CACHE);SERVICE=Service(CACHE,DB)
for i in range(8):
    path=SOURCES/f'clip-{i}.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',f'testsrc2=size=160x96:rate=10:duration=2','-c:v','mpeg4','-q:v','6',str(path)],check=True,env=CACHE.environment(),cwd=CACHE.root)
    path.chmod(0o444)

def source_snapshot():
    return {p.name:(p.stat().st_size,p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()) for p in SOURCES.iterdir()}

BEFORE=source_snapshot()
SERVICE.scan(SOURCES);SERVICE.analyze()
IDS=[c['id'] for c in DB.library()['clips']]
A=DB.group('Recipe A');B=DB.group('Recipe B');EMPTY=DB.group('Empty target')
DB.assign(IDS[:2],A);DB.assign(IDS[2:4],B)
RUNNER=Recommendations(SERVICE);RUNNER.build(A,'features')
OLD=RUNNER.queue(A)['suggestions'];RUNNER.decide(OLD[0]['id'],'rejected')
OLD_PENDING=OLD[1]['id'];OLD_MOVE=DB.latest_move()['id']
LABEL=IDS[0];DB.label(LABEL,'Known cake',session='session-one')
for i in range(20):DB.group(f'Scroll destination {i:02}')
SCROLL=DB.group('Last scroll target')
FRAMES=DB.rows('SELECT * FROM frames');EMBEDDINGS=DB.rows('SELECT * FROM embeddings')


def read(path='/api/library'):
    with urllib.request.urlopen(URL+path,timeout=4) as response:return json.load(response)


def start(log):
    proc=subprocess.Popen([sys.executable,'-m','recipe_sorter','--data-dir',str(CACHE.root),'serve','--port',str(PORT)],cwd=ROOT,stdout=log,stderr=log)
    for _ in range(100):
        if proc.poll() is not None:raise RuntimeError('Server exited; inspect server.log')
        try:read();return proc
        except OSError:time.sleep(.1)
    proc.terminate();raise RuntimeError('Server startup timed out')


def stop(proc):
    if proc is None:return
    proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()


def effective(cid):return next(c['group_id'] for c in read()['clips'] if c['id']==cid)


server=None
with (ROOT_DATA/'server.log').open('w') as log:
    try:
        server=start(log)
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=os.environ.get('RCS_BROWSER_EXECUTABLE'))
            page=browser.new_page(viewport={'width':1500,'height':980});errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)));page.goto(URL)
            expect(page.locator('.clip-card')).to_have_count(8)
            # Direct A -> B, never through Uncertain.
            page.locator(f'[data-drag-clip="{IDS[0]}"]').drag_to(page.locator(f'#groupNav [data-drop-group="{B}"]'))
            expect(page.locator(f'section[data-drop-group="{B}"] .clip-card')).to_have_count(3)
            assert effective(IDS[0])==B
            page.locator('#undoMove').click()
            expect(page.locator(f'section[data-drop-group="{A}"] .clip-card')).to_have_count(2)
            # Same-group no-op does not add undo history.
            move=read()['last_move']
            page.locator(f'[data-drag-clip="{IDS[0]}"]').drag_to(page.locator(f'#groupNav [data-drop-group="{A}"]'))
            expect(page.locator('#toast')).to_contain_text('Nothing changed')
            assert read()['last_move']==move
            # Multi-selection includes hidden clips; drag in a filtered grouped view.
            for cid in IDS[:2]:page.locator(f'[data-select="{cid}"]').check()
            page.locator('#groupedTab').click();page.locator('#search').fill('clip-0.mp4')
            expect(page.locator('#selectedCount')).to_contain_text('1 hidden')
            page.locator(f'[data-drag-clip="{IDS[0]}"]').drag_to(page.locator(f'#groupNav [data-drop-group="{B}"]'))
            expect(page.locator('#bulk')).to_be_hidden()
            assert all(effective(cid)==B for cid in IDS[:2])
            page.reload();page.locator('#undoMove').click()
            expect(page.locator(f'section[data-drop-group="{A}"] .clip-card')).to_have_count(2)
            # All footage and workspace destinations; sample controls must not start dragging.
            page.locator('#flatTab').click();page.locator(f'[data-drag-clip="{IDS[0]}"]').drag_to(page.locator('#uncertainNav'))
            expect(page.locator('#toast')).to_contain_text('moved')
            assert effective(IDS[0])==1
            page.locator('#undoMove').click();page.locator('#workspaceTab').click()
            page.locator(f'[data-drag-clip="{IDS[0]}"]').drag_to(page.locator(f'section[data-drop-group="{B}"] .group-header'))
            expect(page.locator(f'section[data-drop-group="{B}"] .clip-card')).to_have_count(3)
            page.locator('#undoMove').click()
            page.locator('.frame-dots button').first.click()
            assert not page.locator('body').evaluate("e=>e.classList.contains('is-dragging')")
            # Real mouse drag holds at the sidebar's lower edge for auto-scroll.
            handle=page.locator(f'[data-drag-clip="{IDS[0]}"]');handle.scroll_into_view_if_needed();box=handle.bounding_box()
            page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2);page.mouse.down();page.mouse.move(box['x']+25,box['y']+15,steps=8)
            nav=page.locator('#groupNav');rect=nav.bounding_box()
            page.mouse.move(rect['x']+rect['width']/2,rect['y']+rect['height']-4,steps=20)
            page.wait_for_function("document.getElementById('groupNav').scrollTop>0",timeout=10000)
            target=page.locator(f'#groupNav [data-drop-group="{SCROLL}"]')
            for _ in range(50):
                box=target.bounding_box()
                if box and box['y']+box['height']<=rect['y']+rect['height']:break
                page.wait_for_timeout(100)
            box=target.bounding_box();page.mouse.move(box['x']+20,box['y']+box['height']/2,steps=8);page.mouse.up()
            expect(page.locator('#toast')).to_contain_text('moved');assert effective(IDS[0])==SCROLL
            page.locator('#undoMove').click()
            # Cleanup while a filter hides ALL members of B must retain B.
            page.locator('#search').fill('clip-0.mp4');page.locator('#cleanupGroups').click()
            expected=read('/api/organization')['empty_groups']
            expect(page.locator('#organizationCounts')).to_contain_text(str(expected))
            page.locator('#organizationConfirm').check();page.locator('#organizationSubmit').click()
            expect(page.locator('#organizationDialog')).not_to_be_visible()
            assert B in {g['id'] for g in read()['groups']}
            assert EMPTY not in {g['id'] for g in read()['groups']}
            # Cancel and confirm reset; every source/cache survives.
            count=len(read()['groups'])-1
            page.locator('#resetSorting').click();expect(page.locator('#organizationCounts')).to_contain_text(f'8 clips and {count}')
            page.locator('[data-close=organizationDialog]').last.click()
            assert len(read()['groups'])-1==count
            # Actual backup-path failure in this disposable application directory.
            backup_dir=CACHE.root/'backups'
            assert not backup_dir.exists()
            backup_dir.write_text('Injected backup-path failure')
            page.locator('#resetSorting').click();page.locator('#organizationConfirm').check();page.locator('#organizationSubmit').click()
            expect(page.locator('#organizationError')).to_contain_text('Backup failed')
            assert len(read()['groups'])-1==count and effective(IDS[0])==A
            page.locator('[data-close=organizationDialog]').last.click();backup_dir.unlink()
            # A second client changes organization after confirmation was shown.
            page.locator('#resetSorting').click()
            result=page.evaluate("async ()=>(await fetch('/api/groups',{method:'POST',headers:{'Content-Type':'application/json','X-Recipe-Sorter':'local'},body:JSON.stringify({name:'Concurrent target'})})).status")
            assert result==200
            page.locator('#organizationConfirm').check();page.locator('#organizationSubmit').click()
            expect(page.locator('#organizationError')).to_contain_text('changed since confirmation')
            assert effective(IDS[0])==A
            page.locator('[data-close=organizationDialog]').last.click()
            page.locator('#resetSorting').click();page.locator('#organizationConfirm').check();page.locator('#organizationSubmit').click()
            expect(page.locator('#organizationDialog')).not_to_be_visible()
            library=read();assert all(c['manual_group'] is None and c['sorting_state']=='unsorted' and c['group_id']==1 for c in library['clips'])
            assert len(library['groups'])==1 and library['last_move'] is None
            assert DB.rows('SELECT * FROM frames')==FRAMES and DB.rows('SELECT * FROM embeddings')==EMBEDDINGS
            page.screenshot(path=str(ROOT_DATA/'after-reset.png'),full_page=True)
            # Old actions fail at the actual API boundary after reset.
            for route,body in [(f'/api/moves/{OLD_MOVE}/undo',{}),(f'/api/suggestions/{OLD_PENDING}/decision',{'decision':'accepted'})]:
                result=page.evaluate("async ([url,body])=>{let r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Recipe-Sorter':'local'},body:JSON.stringify(body)});return r.status}",[route,body]);assert result==400
            stop(server);server=start(log);page.reload()
            expect(page.locator('.clip-card')).to_have_count(8);assert len(read()['groups'])==1
            # Offline organization restoration uses the persisted snapshot.
            page.locator('#restoreSorting').click();page.locator('#organizationConfirm').check();page.locator('#organizationSubmit').click()
            expect(page.locator('#organizationDialog')).not_to_be_visible();assert effective(IDS[0])==A
            # New generation makes historical rejections inactive for the same group ID.
            page.locator(f'[data-recommend="{A}"]').click();page.locator('#recommendBackend').select_option('features');page.locator('#findMatches').click()
            expect(page.locator('.recommendation-card').first).to_be_visible(timeout=30000)
            assert OLD[0]['clip_id'] in {s['clip_id'] for s in read(f'/api/recommendations/{A}')['suggestions']}
            page.locator('[data-close=recommendDialog]').click()
            # Another reset followed by a new group must prevent restoration.
            page.locator('#resetSorting').click();page.locator('#organizationConfirm').check();page.locator('#organizationSubmit').click()
            expect(page.locator('#organizationDialog')).not_to_be_visible()
            page.locator('#newGroup').click();page.locator('#groupName').fill('Subsequent work');page.locator('#groupForm button[type=submit]').click()
            expect(page.get_by_role('heading',name='Subsequent work',exact=True)).to_be_visible()
            page.locator('#restoreSorting').click();expect(page.locator('#organizationSubmit')).to_be_disabled();expect(page.locator('#organizationError')).to_contain_text('conflicts')
            page.locator('[data-close=organizationDialog]').last.click()
            page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(ROOT_DATA/'mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert not errors,errors
            browser.close()
        assert source_snapshot()==BEFORE
        assert DB.rows('SELECT recipe FROM benchmark_labels WHERE clip_id=?',(LABEL,))[0]['recipe']=='Known cake'
        print(json.dumps({'passed':True,'source_unchanged':True,'artifacts':str(ROOT_DATA)},indent=2))
    finally:stop(server)
