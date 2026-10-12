"""Atomic virtual organization. Backup and snapshots never access source media."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from .database import now


class Organization:
    def __init__(self, database):
        self.db, self.cache = database, database.cache

    def state(self, conn):
        return dict(conn.execute('SELECT generation,revision FROM sorting_state WHERE id=1').fetchone())

    def check(self, conn, expected):
        if self.state(conn) != expected:
            raise ValueError('Organization changed since confirmation. Review the current counts and try again.')

    def empty_ids(self, conn, automatic_only=False):
        return [r[0] for r in conn.execute("SELECT g.id FROM groups g WHERE g.id<>1 AND g.deleted=0 AND (?=0 OR g.kind='auto') AND NOT EXISTS(SELECT 1 FROM clips c WHERE COALESCE(c.manual_group,c.proposed_group,1)=g.id)", (int(automatic_only),))]

    def preview(self):
        with self.db.connect() as conn:
            state = self.state(conn)
            last = conn.execute('SELECT id,created_at,after_generation,after_revision,restored_at FROM sorting_resets ORDER BY id DESC LIMIT 1').fetchone()
            restore = dict(last) if last else None
            if restore:
                restore['available'] = not restore['restored_at'] and state == {'generation':restore['after_generation'], 'revision':restore['after_revision']}
                restore['conflict'] = not restore['available'] and not restore['restored_at']
            return {**state, 'clips':conn.execute('SELECT COUNT(*) FROM clips').fetchone()[0],
                    'groups':conn.execute('SELECT COUNT(*) FROM groups WHERE deleted=0 AND id<>1').fetchone()[0],
                    'empty_groups':len(self.empty_ids(conn)), 'restore':restore}

    def backup(self, purpose):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        relative = f'backups/{purpose}-{stamp}-{uuid.uuid4().hex[:8]}.sqlite3'
        # Caller holds BEGIN IMMEDIATE. A separate read connection copies the
        # committed WAL snapshot without backing up a connection holding a write tx.
        try:
            target = self.cache.output(relative)
            with sqlite3.connect(self.db.path) as source, sqlite3.connect(target) as destination:
                source.backup(destination)
                if destination.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('Backup integrity check failed')
        except Exception as exc:
            raise ValueError(f'Backup failed; organization was not changed: {exc}') from exc
        return relative

    def reset(self, expected):
        with self.db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self.check(conn, expected)
            snapshot = {'clips':[dict(r) for r in conn.execute('SELECT id,manual_group,proposed_group,confidence,evidence,sorting_state FROM clips')],
                        'groups':[dict(r) for r in conn.execute('SELECT * FROM groups WHERE id<>1 AND deleted=0')]}
            backup = self.backup('before-sorting-reset')
            generation = expected['generation'] + 1
            conn.execute('UPDATE sorting_state SET generation=?,revision=revision+1 WHERE id=1',(generation,))
            conn.execute('UPDATE groups SET deleted=1 WHERE id<>1 AND deleted=0')
            conn.execute("UPDATE clips SET manual_group=NULL,proposed_group=1,confidence=NULL,evidence=NULL,manual_move_id=NULL,manual_revision=manual_revision+1,sorting_state='unsorted'")
            revision = self.state(conn)['revision']
            reset_id = conn.execute('INSERT INTO sorting_resets(created_at,before_generation,after_generation,after_revision,backup_path,snapshot) VALUES(?,?,?,?,?,?)',
                (now(),expected['generation'],generation,revision,backup,json.dumps(snapshot))).lastrowid
            return {'saved':True,'reset_id':reset_id,'backup':backup,'generation':generation}

    def restore(self, reset_id, expected):
        with self.db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self.check(conn, expected)
            row = conn.execute('SELECT * FROM sorting_resets WHERE id=?',(reset_id,)).fetchone()
            if not row or row['restored_at'] or expected != {'generation':row['after_generation'],'revision':row['after_revision']}:
                raise ValueError('Cannot restore: subsequent work or another reset conflicts. Nothing was overwritten. Recover the backup into a separate data directory if needed.')
            snapshot = json.loads(row['snapshot'])
            backup = self.backup('before-sorting-restore')
            generation = expected['generation']+1
            conn.execute('UPDATE sorting_state SET generation=?,revision=revision+1 WHERE id=1',(generation,))
            for g in snapshot['groups']:
                conn.execute('UPDATE groups SET name=?,kind=?,deleted=0,generation=? WHERE id=?',(g['name'],g['kind'],generation,g['id']))
            for c in snapshot['clips']:
                conn.execute('UPDATE clips SET manual_group=?,proposed_group=?,confidence=?,evidence=?,sorting_state=?,manual_move_id=NULL,manual_revision=manual_revision+1 WHERE id=?',
                    (c['manual_group'],c['proposed_group'],c['confidence'],c['evidence'],c['sorting_state'],c['id']))
            conn.execute('UPDATE sorting_resets SET restored_at=?,restore_backup=? WHERE id=?',(now(),backup,reset_id))
            return {'saved':True,'backup':backup,'generation':generation}

    def cleanup(self, expected):
        with self.db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self.check(conn, expected)
            ids = self.empty_ids(conn)
            conn.executemany('UPDATE groups SET deleted=1 WHERE id=?',[(i,) for i in ids])
            return {'saved':True,'removed':len(ids)}
