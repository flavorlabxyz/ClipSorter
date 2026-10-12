#!/usr/bin/env python3
"""Workstation-only Playwright V1.5 workflow test; no real footage is scanned.
Install: pip install -e '.[browser]'; python -m playwright install chromium
Run: python scripts/browser_smoke.py
Requires local sockets, Chromium sandbox/process support, and FFmpeg.
Optional RCS_BROWSER_EXECUTABLE selects an installed Chromium binary.
Artifacts remain in a unique .test-data/browser-smoke-* application data tree.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from playwright.sync_api import sync_playwright, expect

root = Path(__file__).resolve().parents[1]
(root / '.test-data').mkdir(exist_ok=True)
data = Path(tempfile.mkdtemp(prefix='browser-smoke-', dir=root / '.test-data'))
with socket.socket() as port_socket:
    port_socket.bind(('127.0.0.1', 0))
    port = port_socket.getsockname()[1]
url = f'http://127.0.0.1:{port}'


def read(path='/api/library'):
    with urllib.request.urlopen(url + path, timeout=3) as response:
        return json.load(response)


def start(log):
    process = subprocess.Popen([sys.executable, '-m', 'recipe_sorter', '--data-dir', str(data), 'serve', '--demo', '--port', str(port)], cwd=root, stdout=log, stderr=log)
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError('Server exited; inspect ' + str(data / 'server.log'))
        try:
            read()
            return process
        except OSError:
            time.sleep(.1)
    stop(process)
    raise RuntimeError('Server did not start within ten seconds')


def stop(process):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


server = None
with (data / 'server.log').open('w') as log:
    try:
        server = start(log)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=os.environ.get('RCS_BROWSER_EXECUTABLE'))
            page = browser.new_page(viewport={'width':1440,'height':1000})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            expect(page.locator('.clip-card')).to_have_count(22)
            expect(page.locator('.sorting-board > .uncertain')).to_be_visible()
            page.screenshot(path=str(data / 'desktop.png'), full_page=True)

            # Create and rename a group through the actual forms.
            page.locator('#newGroup').click()
            page.locator('#groupName').fill('Smoke review')
            page.locator('#groupForm button[type=submit]').click()
            expect(page.get_by_role('heading', name='Smoke review', exact=True)).to_be_visible()
            group = next(g['id'] for g in read()['groups'] if g['name']=='Smoke review')
            page.locator(f'[data-rename="{group}"]').click()
            page.locator('#groupName').fill('Cake review')
            page.locator('#groupForm button[type=submit]').click()
            expect(page.get_by_role('heading', name='Cake review', exact=True)).to_be_visible()

            ids = [c['id'] for c in read()['clips'] if c['group_id']==1][:2]
            assert len(ids)==2
            for cid in ids:
                page.locator(f'[data-select="{cid}"]').check()
            page.locator(f'[data-drag-clip="{ids[0]}"]').drag_to(page.locator(f'#groupNav [data-drop-group="{group}"]'))
            expect(page.locator('#bulk')).to_be_hidden()
            assert all(c['manual_group']==group for c in read()['clips'] if c['id'] in ids)
            page.reload()
            expect(page.locator('#undoMove')).to_be_enabled()
            page.locator('#undoMove').click()
            expect(page.locator(f'.uncertain [data-select="{ids[0]}"]')).to_be_visible()
            assert all(c['manual_group'] is None for c in read()['clips'] if c['id'] in ids)

            # Workspace group drop, then the keyboard-accessible move alternative.
            page.locator(f'[data-drag-clip="{ids[0]}"]').drag_to(page.locator(f'section[data-drop-group="{group}"] .group-header'))
            expect(page.locator(f'section[data-drop-group="{group}"] .clip-card')).to_have_count(1)
            page.locator(f'[data-select="{ids[1]}"]').check()
            page.locator('#bulkGroup').select_option(str(group))
            page.locator('#bulkMove').focus()
            page.keyboard.press('Enter')
            expect(page.locator(f'section[data-drop-group="{group}"] .clip-card')).to_have_count(2)

            # Demo recommendations compute the actual baseline on illustrations.
            page.locator(f'[data-recommend="{group}"]').click()
            expect(page.locator('#exampleChoices input')).to_have_count(2)
            expect(page.locator('#recommendBackend')).to_have_value('features')
            page.locator('#findMatches').click()
            expect(page.locator('.recommendation-card').first).to_be_visible(timeout=30000)
            for decision in ('skipped','rejected','accepted'):
                button=page.locator(f'[data-decision="{decision}"]').first
                suggestion=button.get_attribute('data-suggestion')
                button.click()
                expect(page.locator(f'[data-suggestion-card="{suggestion}"]')).to_have_count(0)
            for status in ('skipped','rejected','accepted'):
                page.locator('#recommendStatus').select_option(status)
                expect(page.locator('.recommendation-card')).to_have_count(1)
            page.screenshot(path=str(data / 'recommendations.png'), full_page=True)
            page.locator('[data-close=recommendDialog]').click()

            # Undo accepted recommendation using the keyboard; preserve its history.
            page.locator('#workspaceTab').focus()
            page.keyboard.press('Control+z')
            expect(page.locator(f'section[data-drop-group="{group}"] .clip-card')).to_have_count(2)
            assert len(read(f'/api/recommendations/{group}?status=undone')['suggestions'])==1

            # Search, samples and cached video preview remain functional.
            name=next(c['filename'] for c in read()['clips'] if c['id']==ids[0])
            page.locator('#search').fill(name)
            expect(page.locator('.clip-card')).to_have_count(1)
            page.locator('.thumb-button').click()
            expect(page.locator('#sampleStrip button')).to_have_count(5)
            page.locator('#sampleStrip button').first.click()
            page.locator('#buildPreview').click()
            expect(page.locator('#viewer video')).to_be_visible(timeout=30000)
            page.locator('[data-close=clipDialog]').click()
            page.locator('#search').fill('')

            # Empty-only group deletion and responsive view.
            page.locator('#newGroup').click()
            page.locator('#groupName').fill('Delete me')
            page.locator('#groupForm button[type=submit]').click()
            empty=page.locator('.group-section').filter(has=page.get_by_role('heading',name='Delete me',exact=True))
            empty.locator('[data-delete]').click()
            expect(page.get_by_role('heading',name='Delete me',exact=True)).to_have_count(0)
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(data / 'mobile.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')

            # Stop and reopen the actual server using the same SQLite library.
            stop(server)
            server = start(log)
            page.reload()
            expect(page.locator('.clip-card')).to_have_count(22)
            assert all(c['manual_group']==group for c in read()['clips'] if c['id'] in ids)
            for status in ('skipped','rejected','undone'):
                assert len(read(f'/api/recommendations/{group}?status={status}')['suggestions'])==1
            assert not errors, errors
            browser.close()
        print(json.dumps({'passed':True,'artifacts':str(data)},indent=2))
    finally:
        stop(server)
