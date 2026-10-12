"""Illustrative data, never imported by production analysis or decoders."""
import json
import math
import random
from datetime import datetime, timedelta, timezone
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .storage.database import now
from .media.decoders.ffmpeg import FFmpegDecoder


def illustration(thread, stage, sample):
    rng = random.Random(thread * 1000 + stage * 20 + sample)
    w, h = 640, 400
    pixels = np.random.default_rng(thread * 1000 + stage * 20 + sample).normal(0, 2, (h, w, 1))
    base = np.array([(80, 86, 77), (138, 116, 88), (69, 76, 72), (104, 100, 92)][thread])
    im = Image.fromarray(np.uint8(np.clip(base + pixels, 0, 255)))
    d = ImageDraw.Draw(im)
    # Linen and board lend each synthetic thread a consistent setting.
    d.polygon([(0, 280), (120, 230), (235, 400), (0, 400)], fill=(181, 174, 151))
    for i in range(14):
        d.line([(i * 17, 280), (i * 17 + 70, 400)], fill=(166, 159, 139), width=2)
    if thread == 1:
        d.rounded_rectangle((70, 35, 558, 364), radius=22, fill=(174, 132, 86))
        for i in range(24):
            y = rng.randint(45, 355)
            d.line([(85, y), (545, y + rng.randint(-8, 8))], fill=(157, 116, 76), width=1)
        for i in range(6):
            x, y = 140 + (i % 3) * 125, 132 + (i // 3) * 125
            color = (226, 191, 132) if stage < 3 else (178, 111, 45)
            d.ellipse((x - 46, y - 36, x + 48, y + 38), fill=(75, 57, 37))
            d.ellipse((x - 46, y - 43, x + 46, y + 28), fill=color)
            d.arc((x - 29, y - 31, x + 25, y + 12), 220, 345, fill=(248, 216, 163), width=5)
            for _ in range(12):
                px, py = x + rng.randint(-30, 30), y + rng.randint(-24, 12)
                d.line((px, py, px + 8, py - 5), fill=(78, 94, 45), width=2)
    else:
        cx, cy = 318 + sample * 2, 202
        dark = thread == 2
        d.ellipse((cx - 155, cy - 129, cx + 169, cy + 144), fill=(40, 43, 37))
        d.ellipse((cx - 166, cy - 142, cx + 166, cy + 142), fill=(40, 42, 37) if dark else (213, 209, 187))
        d.ellipse((cx - 145, cy - 122, cx + 145, cy + 122), fill=(24, 29, 25) if dark else (159, 160, 141))
        food = (225, 199, 127) if thread == 0 else (168, 68, 34) if thread == 2 else (214, 198, 164)
        d.ellipse((cx - 132, cy - 110, cx + 132, cy + 110), fill=food)
        if thread == 2:
            d.rounded_rectangle((462, 173, 630, 209), radius=12, fill=(33, 36, 31))
            for _ in range(38):
                x, y = rng.randint(cx - 105, cx + 105), rng.randint(cy - 85, cy + 85)
                if ((x - cx) / 122) ** 2 + ((y - cy) / 98) ** 2 < 1:
                    d.ellipse((x - 14, y - 10, x + 16, y + 11), fill=rng.choice([(188, 69, 40), (212, 106, 52), (201, 121, 76)]))
                    if rng.random() < .35:
                        d.ellipse((x, y, x + 18, y + 8), fill=(67, 100, 49))
        else:
            for n in range(7):
                d.arc((cx - 105 + n * 10, cy - 85 + n * 10, cx + 108 - n * 9, cy + 88 - n * 9), 5, 290, fill=(236, 216, 167), width=3)
            d.line((391, 153, 557, 43), fill=(125, 86, 51), width=15)
            d.ellipse((371, 135, 422, 189), fill=(177, 134, 76))
        if thread == 0:
            for x, y in [(72, 90), (130, 63)]:
                d.ellipse((x - 30, y - 23, x + 30, y + 23), fill=(219, 185, 64))
                d.ellipse((x - 23, y - 18, x + 23, y + 18), outline=(243, 225, 145), width=3)
    d.text((18, 376), 'ILLUSTRATIVE DEMO / NOT SOURCE FOOTAGE', fill=(237, 232, 214))
    return im


def seed_demo(cache, db):
    if db.rows('SELECT id FROM clips LIMIT 1'):
        return
    groups = [db.group(name, 'auto') for name in ['Lemon cake', 'Rosemary rolls', 'Tomato skillet']]
    base = datetime(2026, 9, 28, 9, 12, tzinfo=timezone.utc)
    with db.connect() as conn:
        run = conn.execute('INSERT INTO runs(created_at,version) VALUES(?,?)', (now(), 'demo-illustrative-v1')).lastrowid
        for i in range(22):
            thread = 0 if i < 7 else 1 if i < 13 else 2 if i < 18 else 3
            group = groups[thread] if thread < 3 else 1
            ext = '.braw' if i % 4 != 3 else '.mov'
            filename = f'A001_{i + 1:04}_0928{ext}'
            duration = 14 + (i * 19 % 82)
            confidence = round(.89 + (i % 5) * .019, 3) if thread < 3 else .53 + (i % 3) * .055
            metadata = {'duration': duration, 'fps': 25., 'width': 6144, 'height': 3456, 'codec': 'Blackmagic RAW (demo)' if ext == '.braw' else 'ProRes (demo)',
                        'recorded_at': (base + timedelta(minutes=(i * 7) % 83)).isoformat(), 'camera': 'Pocket Cinema Camera 6K · demo', 'extra': {'illustrative': True}}
            evidence = {'reason': 'Illustrative demo proposal, not real analysis', 'best_candidate': groups[i % 3], 'second_candidate': groups[(i + 1) % 3], 'second_score': .51, 'margin': .15}
            cid = conn.execute('INSERT INTO clips(path,filename,extension,size,mtime_ns,fingerprint,decoder,metadata,status,analysis_version,proposed_group,confidence,evidence,demo,discovered_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)',
                (f'/DEMO_ONLY/September kitchen/{filename}', filename, ext, duration * 28_000_000, 0, f'demo-{i}', 'demo', json.dumps(metadata), 'analyzed', 'demo-illustrative-v1', group, confidence, json.dumps(evidence), now())).lastrowid
            conn.execute('INSERT INTO proposals(run_id,clip_id,group_id,confidence,evidence,fingerprint) VALUES(?,?,?,?,?,?)', (run, cid, group, confidence, json.dumps(evidence), f'demo-{i}'))
            for sample in range(5):
                relative = f'demo/frames/{cid}-{sample}.jpg'
                illustration(thread, i % 7, sample).save(cache.output(relative), quality=87)
                conn.execute('INSERT INTO frames(clip_id,seconds,path,quality,embedding) VALUES(?,?,?,?,?)', (cid, duration * (.1 + .2 * sample), relative, 20., '[]'))


def demo_preview(cache, db, clip):
    if clip['preview'] and cache.path(clip['preview']).is_file():
        return
    frame = db.rows('SELECT path FROM frames WHERE clip_id=? ORDER BY seconds LIMIT 1', (clip['id'],))[0]['path']
    relative = f"demo/previews/{clip['id']}.webm"
    FFmpegDecoder(cache).run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-loop', '1', '-i', str(cache.path(frame)), '-t', '5',
        '-vf', 'scale=640:-2', '-r', '6', '-c:v', 'libvpx-vp9', '-b:v', '400k', '-deadline', 'realtime', '-cpu-used', '6', '-threads', '2', str(cache.output(relative))])
    with db.connect() as conn:
        conn.execute('UPDATE clips SET preview=? WHERE id=?', (relative, clip['id']))
